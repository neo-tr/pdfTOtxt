from __future__ import annotations

import argparse
import json
import logging
import re
from statistics import median
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from tqdm import tqdm

# ---------------------------- Константы под эвристики ----------------------------
SUP_OFFSET_FACTOR = 0.28    # порог для определения надстрочных относительно базы
SUB_OFFSET_FACTOR = 0.25    # порог для определения подстрочных
COLUMN_GAP_RATIO = 0.18     # относительная ширина разрыва между для определения двух колонок
LARGE_FONT_FACTOR = 1.25    # порог для большого шрифта (возможно загоовок)
MIN_LINE_LEN = 2            # минимальная длина строки для некоторых эвристик (игнор определенных мест)

# ---------------------------- Утилиты ----------------------------------------------
_tag_re = re.compile(r"<[^>]+>")

def _strip_tags(s: str) -> str:
    """
    Убдаляю HTML-тип теги (<sup>, <sub> что водим)
    Используется для чистых вычислений (детекция авторов)
    """
    if not s:
        return ""
    try:
        import html as _html
        s = _html.unescape(s)
    except Exception:
        pass
    return _tag_re.sub("", s).strip()

# Регулярка для токенизации чисел
_NUM_PATTERN = re.compile(
    r"(?<!\w)([+-]?(?:\d{1,3}(?:[,\u00A0]\d{3})*|\d+)(?:[.,]\d+)?(?:[eE][+-]?\d+)?%?)(?!\w)"
)

def _tokenize_numbers(text: str, num_map: Dict[str, str], start_index: int = 1) -> str:
    """
    Заменяет числа токенами __NUM_n__ и наполняет num_map
    """
    counter = start_index + len(num_map)

    def repl(m):
        nonlocal counter
        val = m.group(1)
        token = f"__NUM_{counter}__"
        num_map[token] = val
        counter += 1
        return token

    return _NUM_PATTERN.sub(repl, text)

def _normalize_numbers_in_text(text: str) -> str:
    """
    Лёгкая нормализация чисел, если не делаем токенизацию:
    - удаляем NBSP,
    - удаляем запятые, похожие на тысячные (1,000 -> 1000).
    Не пытаемся менять локали/десятичные запятые сложно
    """
    t = text.replace("\u00A0", "")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    return t

def _page_median_font_size(block: dict) -> float:
    """
    Хелпер, должен вернуть медиану по размеру шрифтов в блоке и 0.0
    """
    sizes = []
    for line in block.get("lines", []):
        for s in line.get("spans", []):
            size = s.get("size", 0)
            if size and size > 0:
                sizes.append(size)
    return median(sizes) if sizes else 0.0

def _join_lines_preserve_words(lines: List[str]) -> str:
    """
        Соединяем строки, сохраняя переносы слов через дефис
    """
    out: List[str] = []
    prev = ""
    for L in lines:
        Ls = (L or "").strip()
        if not Ls:
            if prev: # это нашли разделитель параграфов
                out.append(prev)
                prev = ""
            continue
        if prev.endswith("-"): # это перенос слова через дефис, убираем дефис и сцепляем в одно слово
            prev = prev[:-1] + Ls
        else:
            if prev:
                prev = prev + " " + Ls # склеиваем с пробелом
            else:
                prev = Ls # начинаем новый
    if prev:
        out.append(prev)
    return "\n\n".join(out)

# ---------------------------- Эвристика для авторов или афилированности к чему-то ----------------------------
_email_re = re.compile(r"\S+@\S+")
_affil_keywords = {
    "department", "university", "institute", "laboratory", "school", "center", "centre",
    "college", "research", "correspondence", "address", "key laboratory", "physics", "materials",
    "russia","china", "usa", "united states", "japan", "germany", "beijing", "shanghai"
}
_address_markers = re.compile(r"\b(road|street|lane|rd|ave|boulevard|park|city|province|zip|postcode|room|floor)\b", re.I)

def looks_like_author_or_affil(clean_line: str, page_num: int, y: float, page_height: float) -> bool:
    """
    Эвристика для определения строк авторов или аффилиаций
    Возвращает True, если строка выглядит как автор/affiliation (чтобы не маркировать её дальше)
    """
    if not clean_line or not clean_line.strip():
        return False
    s = clean_line.strip()
    if _email_re.search(s):
        return True
    low = s.lower()
    for kw in _affil_keywords:
        if kw in low:
            return True
    if _address_markers.search(low):
        return True
    # много запятых и заглавных токенов, это список авторов
    comma_count = s.count(",")
    tokens = [t for t in re.split(r"\s+", s) if t]
    cap_count = sum(1 for t in tokens if t and t[0].isupper() and re.search(r"[A-Za-z]", t))
    if comma_count >= 1 and cap_count >= 2:
        return True
    # верх страницы (первая страница) и много заглавных токенов, вероятно заголовок или авторы
    if page_num == 0 and y < 0.20 * page_height and cap_count >= 3:
        return True
    # пометки *, †, ‡ рядом с именами
    if ("*" in s or "†" in s or "‡" in s) and (comma_count >= 1 or cap_count >= 2):
        return True
    return False

# ---------------------------- Эвристика для TITLE ----------------------------

def detect_title_lines_on_first_page(
    page_dict: dict,
    page_height: float,
    page_width: Optional[float] = None,
    top_frac: float = 0.45, # рассматриваем верхние 45% страницы как зону, где может быть заголовок или авторы
    debug: bool = False,
) -> Tuple[List[float], Optional[str], Optional[Tuple[float, float]]]:
    """
    Детекция заголовка
    Возвращает: (title_ys, title_text, (min_y, max_y)) или ([], None, None)

    title_ys - список координат y строк в заголовке
    title_text - объединённый текст заголовка
    (min_y, max_y) - диапазон вертикальных координат заголовка
    """
    def log(msg: str): # печатаем отладку только когда debug=True
        if debug:
            print("DEBUG:", msg)

    # -------------------- собрать строки верхней части --------------------
    rows = []
    blocks = page_dict.get("blocks", []) # структура fitz get_text("dict")
    for bi, b in enumerate(blocks):
        if b.get("type", 0) != 0: # в PyMuPDF type==0 это текстовый блок, другие могут быть рисунки
            continue
        for line in b.get("lines", []):
            y = line["bbox"][1]  # верхняя координата линии (bbox: [x0, y0, x1, y1]
            if y > top_frac * page_height: # работаем только в верхней зоне, выше y, ниже зона
                continue
            """
            texts (тексты из spans) 
            sizes (размеры шрифтов) 
            xcent (центры каждого span по X)
            x_center (медиана центров по X)
            """
            texts = []
            sizes = []
            xcent = []
            for s in line.get("spans", []):
                t = (s.get("text") or "").strip()
                if t:
                    texts.append(t)
                sz = s.get("size", 0)
                if sz and sz > 0:
                    sizes.append(sz)
                try:
                    xcent.append((s["bbox"][0] + s["bbox"][2]) / 2)
                except Exception:
                    pass
            if not texts:
                continue
            rows.append({
                "y": y,
                "text": " ".join(texts),
                "size_med": median(sizes) if sizes else 0.0,
                "x_center": (median(xcent) if xcent else None),
                "block": bi
            })

    if not rows:
        log("no candidate rows in top_frac")
        return [], None, None

    rows = sorted(rows, key=lambda r: r["y"])  # Сортируем rows сверху вниpз
    page_w = page_width or (rows[0].get("x_center", 0) * 2 if rows else 600)
    # если page_width не указан, пытаемся восстановить через x_center первой строки
    # сли и это не доступно - 600 пикселей по умолчанию
    page_center = page_w / 2.0

    sizes = [r["size_med"] for r in rows if r["size_med"] > 0]
    global_med = median(sizes) if sizes else 0.0
    max_size = max(sizes) if sizes else 0.0
    log(f"rows={len(rows)} global_med={global_med:.2f} max_size={max_size:.2f}")  # global_med и max_size - используются для нормализации размеров шрифтов в скоринге

    # -------------------- Вспомогательные детекторы --------------------
    def is_author_affil_line_strict(txt: str) -> bool:
        """
        Более сильная версия детекции авторов/аффилиаций, использующая email, arxiv/doi, ключевые слова и т.д.
        Используется для исключения строк при выборе стартовой строки заголовка
        """
        if not txt or not txt.strip():
            return False
        t = txt.strip()
        low = t.lower()
        if _email_re.search(t):
            return True
        if re.search(r"\barxiv\b|arxiv:|doi\b|doi:", t, re.I):
            return True
        if low.startswith("dated:") or low.startswith("submitted") or low.startswith("received"):
            return True
        for kw in ("department", "university", "institute", "laboratory", "school", "centre", "center", "college", "address"):
            if kw in low:
                return True
        comma_count = t.count(",")
        cap_tokens = sum(1 for tk in re.split(r"\s+", t) if tk and tk[0].isupper())
        if comma_count >= 1 and cap_tokens >= 2:
            return True
        # суперскрипты или индексы рядом с буквами, это авторы
        if re.search(r"[\*\†\‡\§\u00B9\u00B2\u00B3\u2070-\u2079]", t) and re.search(r"[A-Za-zА-Яа-я]", t):
            return True
        # цифры + запятые или афиляционные слова, авторы или ифилированнаые места
        if re.search(r"\d", t) and (comma_count >= 1 or any(kw in low for kw in ("university","department","institute"))):
            return True
        # короткие строки с 2-4 слов, все с заглавной, вероятное имя (будем осторожны и не сразу маркировать)
        words = [w for w in re.split(r"\s+", t) if w]
        if 1 < len(words) <= 4:
            cap_prop = sum(1 for w in words if w and w[0].isupper()) / len(words)
            if cap_prop >= 0.75 and comma_count == 0:
                # не считать автоматически автором, дальше логика расширения решит
                return False
        return False

    # похоже ли содержимое строки на продолжение заголовка, словарь
    cont_prepositions = {"above","below","with","by","via","for","in","on","through","under","using","implemented","from","to","of","and","the"}
    unit_pattern = re.compile(r"\b(K|T|Hz|GHz|cm|mm|nm|meV|eV)\b", re.I)
    def is_likely_title_continuation(txt: str) -> bool:
        if not txt:
            return False
        words = [w for w in re.split(r"\s+", txt) if w]
        if not words:
            return False
        first = words[0].strip("()[]").lower()
        if first in cont_prepositions:
            return True
        if words[0] and words[0][0].islower():
            return True
        if unit_pattern.search(txt) and re.search(r"\d", txt):
            return True
        if len(words) <= 3 and re.match(r"^[A-Za-z\-]{3,30}$", txt):
            return True
        if first in {"implemented","using","a","an","the","pressure","method","approach","implemented"}:
            return True
        return False

    def is_probable_name_line(txt: str) -> bool:
        """
        Если строка короткая и большинство слов начинается с заглавной буквы, возможно имя авторов
        """
        words = [w for w in re.split(r"\s+", txt) if w]
        if not words:
            return False
        if len(words) <= 4:
            cap_prop = sum(1 for w in words if w and w[0].isupper()) / len(words)
            if cap_prop >= 0.75 and not re.search(r"\d|@|department|university|institute", txt, re.I):
                return True
        return False

    # -------------------- Скоринг кандидатов старта заголовка --------------------
    candidates = []
    """
    size_score (вес 2.0), большие шрифты часто заголовок
    center_score (вес 1.6), центрированность повышает шансы 
    wc_score (вес 0.8), достаточное число слов (заголовок обычно больше 4 слов)
    bad_tokens (вес 0.6), если строка содержит слова, типичные для секции abstract или introduction, снижает вероятность
    """
    for r in rows:
        txt = r["text"].strip()
        if not txt:
            continue
        if is_author_affil_line_strict(txt):
            log(f"skip-as-author-affil-candidate: '{txt[:80]}'")
            continue
        size_score = (r["size_med"] / global_med) if global_med > 0 else 1.0
        size_score = max(0.1, min(size_score, 3.0))
        center_score = 0.6
        if r.get("x_center") is not None:
            center_dist = abs(r["x_center"] - page_center) / (page_w/2)
            center_score = max(0.0, 1.0 - center_dist)
        word_count = len([w for w in re.split(r"\s+", txt) if w])
        wc_score = 1.0 if word_count >= 4 else 0.6
        bad_tokens = 1.0
        if re.search(r"\b(abstract|keywords|copyright|introduction)\b", txt, re.I):
            bad_tokens = 0.0
        score = (2.0 * size_score) + (1.6 * center_score) + (0.8 * wc_score) + (0.6 * bad_tokens)
        candidates.append((score, r))
        log(f"CAND score={score:.2f} sz={r['size_med']:.1f} center={center_score:.2f} wc={word_count} -> '{txt[:80]}'")

    if not candidates:
        log("no scored candidates, fallback first non-author line")
        for r in rows:
            if not is_author_affil_line_strict(r["text"]):
                return [r["y"]], r["text"], (r["y"], r["y"])
        return [], None, None

    candidates_sorted = sorted(candidates, key=lambda t: t[0], reverse=True)
    best_score, best_row = candidates_sorted[0]
    log(f"best start candidate score={best_score:.2f}: '{best_row['text'][:120]}'") # берём строку с максимальным score

    # -------------------- расширение блока заголовка (вниз и вверх) --------------------
    title_block = [best_row]
    try:
        idx = rows.index(best_row)
    except ValueError:
        idx = 0
    base_size = best_row["size_med"] or global_med or 1.0

    """
    Логика расширения вверх и ввниз
    Остановка, если: 
    - встречаем мета-информацию (arXiv/doi/abstract)
    - следующая строка явно авторили афиляция, если не выглядит как продолжение
    - следующая строка выглядит как имя
    - размер ниже base_size * 0.6 и нет явного продолжения
    - строка сильно смещена влево/вправо относительно первой строки заголовка, обычно это не часть заголовка
    
    Почему встраиваем логику расширения, заголовок может занимать несколько строк и он должен быть собран,
    при этом нужно не задеть авторов/параграфы/сноски
    """

    # вниз
    i = idx + 1
    while i < len(rows):
        r = rows[i]; txt = r["text"].strip()
        # остановка на мета
        if re.search(r"\barXiv\b|arXiv:|doi\b|doi:|\babstract\b", txt, re.I):
            log(f"stop down: meta -> '{txt[:80]}'")
            break
        # остановка на афиляции пока это не выглядит как продолжение
        if is_author_affil_line_strict(txt):
            if is_likely_title_continuation(txt):
                log(f"allow down: continuation despite author-like -> '{txt[:80]}'")
            else:
                log(f"stop down: author/affil -> '{txt[:80]}'")
                break
        # вероятно строка имени, обычно остановка (если предыдущая строка не заканчивается дефисом или продолжением)
        if is_probable_name_line(txt):
            prev_txt = title_block[-1]["text"]
            if prev_txt.endswith("-") or is_likely_title_continuation(txt):
                log(f"include down: short name-like because prev endswith '-' or continuation -> '{txt[:80]}'")
            else:
                log(f"stop down: probable name -> '{txt[:80]}'")
                break
        # остановка по размеру
        sz = r["size_med"] or global_med
        if sz < base_size * 0.6 and not is_likely_title_continuation(txt):
            log(f"stop down: size drop too large: {sz:.1f} < {base_size*0.6:.1f} ('{txt[:60]}')")
            break
        # остановка по выравниванию или продолжение
        xc_ok = True
        xc_diff = 0.0
        if best_row.get("x_center") and r.get("x_center"):
            xc_diff = abs(r["x_center"] - best_row["x_center"]) / max(1.0, best_row["x_center"])
            xc_ok = (xc_diff <= 0.6)
        join_ok = is_likely_title_continuation(txt) or len(re.split(r"\s+", txt)) <= 6 or title_block[-1]["text"].endswith("-")
        if not xc_ok and not join_ok:
            log(f"stop down: alignment/continuation fail xc_diff={round(xc_diff,2)} join_ok={join_ok} -> '{txt[:60]}'")
            break
        log(f"extend down: adding '{txt[:80]}' (size {sz:.1f})")
        title_block.append(r)
        i += 1

    # вверх, логика идентична
    j = idx - 1
    while j >= 0:
        r = rows[j]; txt = r["text"].strip()
        if re.search(r"\barXiv\b|arXiv:|doi\b|doi:|\babstract\b", txt, re.I):
            log(f"stop up: meta -> '{txt[:80]}'")
            break
        if is_author_affil_line_strict(txt):
            if is_likely_title_continuation(txt):
                log(f"allow up: continuation despite author-like -> '{txt[:80]}'")
            else:
                log(f"stop up: author/affil -> '{txt[:80]}'")
                break
        if is_probable_name_line(txt):
            log(f"stop up: probable name above -> '{txt[:80]}'")
            break
        sz = r["size_med"] or global_med
        if sz < base_size * 0.6 and not is_likely_title_continuation(txt):
            log(f"stop up: size too small -> '{txt[:60]}'")
            break
        xc_ok = True
        xc_diff = 0.0
        if best_row.get("x_center") and r.get("x_center"):
            xc_diff = abs(r["x_center"] - best_row["x_center"]) / max(1.0, best_row["x_center"])
            xc_ok = (xc_diff <= 0.6)
        join_ok = is_likely_title_continuation(best_row["text"]) or len(re.split(r"\s+", txt)) <= 6
        if not xc_ok and not join_ok:
            log(f"stop up: alignment fail xc_diff={round(xc_diff,2)} -> '{txt[:60]}'")
            break
        log(f"extend up: prepending '{txt[:80]}' (size {sz:.1f})")
        title_block.insert(0, r)
        j -= 1

    # -------------------- соберём итоговый вариант --------------------
    def join_lines(lines):
        """
        Объединяем линии, учитывая перенос через дефис, возвращаем список y и диапазон
        Возвращаемое значение используется далее при маркировке строк <TITLE>...</TITLE> в основной функции
        """
        out = []
        prev = ""
        for L in lines:
            s = (L["text"] or "").strip()
            if not s:
                continue
            if prev.endswith("-"):
                prev = prev[:-1] + s
            else:
                if prev:
                    prev = prev + " " + s
                else:
                    prev = s
        if prev:
            out.append(prev)
        return "\n".join(out)

    title_text = join_lines(title_block).strip()
    title_ys = [r["y"] for r in title_block]
    if not title_ys:
        return [], None, None
    min_y, max_y = min(title_ys), max(title_ys)
    log(f"DET title lines={len(title_block)} y=[{min_y:.1f}..{max_y:.1f}] -> '{title_text[:300]}'")
    return title_ys, title_text, (min_y, max_y)




# ---------------------------- Основной конвертер pdf -> text ----------------------------
def convert_pdf_to_text(
    pdf_path: str,
    tokenize_nums: bool = False,
    return_mapping: bool = True,
    debug: bool = False,
) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    Главная функция конвертации PDF -> текст
    - По умолчанию не токенизирует числа
    - Помечает заголовок на первой странице тегом <TITLE>...</TITLE>
    Возвращает (full_text, num_map|None)

    pdf_path - путь к файлу
    tokenize_nums - включить замену чисел на токены
    return_mapping - вернуть num_map при tokenize_nums
    debug - включить встраиваемые debug
    """
    logger = logging.getLogger("pdf2text")
    doc = fitz.open(pdf_path) # открываем PDF через PyMuPDF
    num_map: Dict[str, str] = {} if tokenize_nums else {}

    # предварительно обнаружим title-строки на первой странице
    # загружаем первую страницу и вызываем функцию детекции заголовка,получаем title_ys, detected_title_text и title_range
    title_ys: List[float] = []
    detected_title_text: Optional[str] = None
    if len(doc) > 0:
        first = doc.load_page(0)
        first_dict = first.get_text("dict")
        title_ys, detected_title_text, title_range = detect_title_lines_on_first_page(first_dict, first.rect.height, first.rect.width, debug=debug)


    pages_output: List[str] = []  # хранит готовый текст по страницам

    for p in tqdm(range(len(doc)), desc="Parsing pages"):
        page = doc.load_page(p)
        w, h = page.rect.width, page.rect.height
        j = page.get_text("dict")  # получаем словарное представление страницы
        blocks_all = [b for b in j.get("blocks", []) if b.get("type", 0) == 0]

        # разделим full-width (не менее 85% ширины) и кандидатов для колоночного анализа
        full_width_thresh = 0.85 * w
        full_blocks = []
        candidates = []
        for b in blocks_all:
            bw = b["bbox"][2] - b["bbox"][0]
            if bw >= full_width_thresh:
                full_blocks.append(b)
            else:
                candidates.append(b)

        """
        Пытаемся найти наибольший промежуток между средними X центрами блоков,
        если он достаточно большой (по отношению к ширине страницы), то это разделитель колонок
        
        split_x, средняя координата между соседними x-центрами
        """
        x_centers = [((b["bbox"][0] + b["bbox"][2]) / 2) for b in candidates]
        two_columns = False
        split_x = None
        if len(x_centers) >= 2:
            sorted_x = sorted(x_centers)
            gaps = [sorted_x[i + 1] - sorted_x[i] for i in range(len(sorted_x) - 1)]
            if gaps:
                max_gap = max(gaps)
                max_gap_idx = gaps.index(max_gap)
                if max_gap > COLUMN_GAP_RATIO * w:
                    two_columns = True
                    split_x = (sorted_x[max_gap_idx] + sorted_x[max_gap_idx + 1]) / 2

        """
        Если найдено split_x, делим кандидатов на лево/право по x-центру, затем сортируем по вертикали (bbox[1]) в каждой колонке
        
        full_blocks тоже сортируем по вертикали, они будут вставлены между колонками для логичного порядка чтения
        """
        if two_columns and split_x is not None:
            left = []
            right = []
            for b in candidates:
                xc = (b["bbox"][0] + b["bbox"][2]) / 2
                if xc <= split_x:
                    left.append(b)
                else:
                    right.append(b)
            columns = [sorted(left, key=lambda b: b["bbox"][1]), sorted(right, key=lambda b: b["bbox"][1])]
        else:
            columns = [sorted(candidates, key=lambda b: b["bbox"][1])]

        full_blocks = sorted(full_blocks, key=lambda b: b["bbox"][1])

        page_lines: List[Tuple[int, float, str]] = []

        def _process_block(block: dict, col_order: int):
            """
            Проход по линиям блока: формируем line_with_tags (с <sup>/<sub>) и собираем в page_lines
            При debug можно добавлять комментарии.
            """
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                # базовый уровень и медианный размер шрифта этого line/spans
                centers = [(s["bbox"][1] + s["bbox"][3]) / 2 for s in spans]
                baseline = median(centers) if centers else 0.0
                font_sizes = [s.get("size", 0) for s in spans]
                font_sizes_nonzero = [fs for fs in font_sizes if fs and fs > 0]
                median_font = median(font_sizes_nonzero) if font_sizes_nonzero else 0.0

                assembled_parts: List[str] = []
                for s in spans:
                    txt = s.get("text", "") or ""
                    if txt.strip() == "":
                        continue
                    center = (s["bbox"][1] + s["bbox"][3]) / 2
                    size = s.get("size", median_font) or median_font or 0.0
                    if (baseline - center) > SUP_OFFSET_FACTOR * size:
                        assembled_parts.append(f"<sup>{txt}</sup>")
                    elif (center - baseline) > SUB_OFFSET_FACTOR * size:
                        assembled_parts.append(f"<sub>{txt}</sub>")
                    else:
                        assembled_parts.append(txt)
                if not assembled_parts:
                    continue

                line_with_tags = " ".join(assembled_parts)
                clean_line = _strip_tags(line_with_tags)

                # TITLE тэг с использованием title_range + fallback по тексту
                is_title = False

                if p == 0:
                    # 1) диапазон Y заголовка
                    if title_range:
                        min_ty, max_ty = title_range
                        y0 = line["bbox"][1]
                        tol_y = max(12.0, 0.6 * median_font)
                        if (min_ty - tol_y) <= y0 <= (max_ty + tol_y):
                            is_title = True

                    # 2) fallback, текстовое совпадение (если вдруг Y не поймался)
                    if not is_title and detected_title_text:
                        # чистим для сравнения
                        cl = clean_line.lower()
                        dt = detected_title_text.lower()
                        # если строка - часть цели заголовка
                        if dt.startswith(cl) or cl in dt:
                            is_title = True

                # Не метим авторов/аффилиации/арxiv
                if is_title:
                    if not looks_like_author_or_affil(clean_line, p, line["bbox"][1], h) and not re.search(
                            r"\barXiv\b|doi:|arXiv:", clean_line, re.I):
                        line_with_tags = f"<TITLE>{line_with_tags}</TITLE>"

                # debug-пометки
                if debug:
                    dbg = []
                    if is_title:
                        dbg.append("title")
                    if looks_like_author_or_affil(clean_line, p, line["bbox"][1], h):
                        dbg.append("author_affil")
                    if dbg:
                        line_with_tags = f"{line_with_tags} <!--DEBUG:{','.join(dbg)}-->"

                page_lines.append((col_order, line["bbox"][1], line_with_tags))

        # порядок обработки: left col (0), full-width (1), right col (2) — чтобы чтение было понятным
        if len(columns) == 2:
            for b in columns[0]:
                _process_block(b, col_order=0)
            for b in full_blocks:
                _process_block(b, col_order=1)
            for b in columns[1]:
                _process_block(b, col_order=2)
        else:
            for b in columns[0]:
                _process_block(b, col_order=0)
            for b in full_blocks:
                _process_block(b, col_order=1)

        # сортируем по вертикали и колонке, затем соединяем
        page_lines_sorted = sorted(page_lines, key=lambda t: (round(t[1], 2), t[0]))
        lines_text = [tup[2] for tup in page_lines_sorted]
        page_text = _join_lines_preserve_words(lines_text)
        pages_output.append(page_text)

    # объединяем страницы и делаем базовую нормализацию пробелов/переносов
    full_text = "\n\n".join(pages_output)
    full_text = re.sub(r"(?<!\n)\n(?!\n)", " ", full_text)
    full_text = re.sub(r" {2,}", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    # обработка чисел: либо токенизация, либо лёгкая нормализация
    if tokenize_nums:
        full_text = _tokenize_numbers(full_text, num_map, start_index=1)
    else:
        full_text = _normalize_numbers_in_text(full_text)
        num_map = None

    return full_text, num_map

# ---------------------------- Метаданные ----------------------------
def extract_title_authors_abstract(pdf_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Максимальное извлечение title/authors/abstract из первой страницы
    Вернуть (title, authors, abstract) - возможны None
    """
    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        return None, None, None
    page = doc.load_page(0)
    j = page.get_text("dict")
    blocks = [b for b in j.get("blocks", []) if b.get("type", 0) == 0]
    rows = []
    for b in blocks:
        for line in b.get("lines", []):
            y = line["bbox"][1]
            texts = []
            sizes = []
            for s in line.get("spans", []):
                t = s.get("text", "").strip()
                if t:
                    texts.append(t)
                sizes.append(s.get("size", 0))
            if texts:
                size_med = median([x for x in sizes if x and x > 0]) if sizes else 0
                rows.append({"y": y, "text": " ".join(texts), "size_med": size_med})
    if not rows:
        return None, None, None

    rows_sorted = sorted(rows, key=lambda r: r["y"])
    sizes = [r["size_med"] for r in rows_sorted if r["size_med"] > 0]
    global_med = median(sizes) if sizes else 0
    title_candidates = [r for r in rows_sorted[:12] if r["size_med"] >= 1.2 * global_med] if global_med > 0 else []
    if title_candidates:
        title = " ".join(r["text"] for r in title_candidates)
    else:
        top_texts = [r["text"] for r in rows_sorted[:4] if r["text"]]
        title = " ".join(top_texts[:2]) if top_texts else None

    # authors: строки сразу после title зоны
    authors = None
    if title:
        title_ys = set(r["y"] for r in title_candidates) if title_candidates else {rows_sorted[0]["y"]}
        idx = 0
        for i, r in enumerate(rows_sorted):
            if r["text"] and (r["text"] in title or r["y"] in title_ys):
                idx = i
        potential = []
        for r in rows_sorted[idx + 1 : idx + 6]:
            t = r["text"]
            if not t:
                continue
            if "," in t or " and " in t.lower() or "department" in t.lower() or re.search(r"\bUniversity\b|\bInstitute\b|\bLaboratory\b", t, re.I):
                potential.append(t)
        if potential:
            authors = " ".join(potential)
        else:
            if idx + 1 < len(rows_sorted):
                authors = rows_sorted[idx + 1]["text"]

    # abstract, если будет начинаться с этого ключевого слова
    abstract = None
    abstract_idx = None
    for i, r in enumerate(rows_sorted):
        if re.match(r"\s*Abstract\b", r["text"], re.I):
            abstract_idx = i
            break
    if abstract_idx is not None:
        lines = []
        for r in rows_sorted[abstract_idx + 1 :]:
            if not r["text"]:
                break
            if re.match(r"\s*(Keywords|1\.|Introduction)\b", r["text"], re.I):
                break
            lines.append(r["text"])
        if lines:
            abstract = " ".join(lines).strip()

    return title, authors, abstract

# ---------------------------- CLI ----------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="PDF -> clean text converter (preserves <sup>/<sub>")
    p.add_argument("pdf", help="input PDF file")
    p.add_argument("--out", default="parsed_output.txt", help="output text file")
    p.add_argument("--mapout", default="num_map.json", help="output JSON mapping numbers (only if --tokenize-numbers)")
    p.add_argument("--metaout", help="output metadata JSON (title/authors/abstract)")
    p.add_argument("--tokenize-numbers", action="store_true", help="replace numbers with tokens __NUM_i__ and save mapping")
    p.add_argument("--debug", action="store_true", help="include inline debug hints")
    return p.parse_args()

def main():
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("pdf2text")
    logger.info("Converting PDF -> text: %s", args.pdf)

    text, num_map = convert_pdf_to_text(args.pdf, tokenize_nums=args.tokenize_numbers, return_mapping=True, debug=args.debug)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("Wrote text -> %s", args.out)

    if args.tokenize_numbers and num_map:
        with open(args.mapout, "w", encoding="utf-8") as f:
            json.dump(num_map, f, ensure_ascii=False, indent=2)
        logger.info("Wrote number mapping -> %s", args.mapout)

    if args.metaout is not None and not args.no_meta:
        title, authors, abstract = extract_title_authors_abstract(args.pdf)
        meta = {"title": title, "authors": authors, "abstract": abstract}
        with open(args.metaout, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("Wrote metadata -> %s", args.metaout)

if __name__ == "__main__":
    main()
