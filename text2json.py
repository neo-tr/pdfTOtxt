import argparse
import json
import re
import sys
from pathlib import Path


# ------------------------ REGEX ------------------------

TITLE_RE = re.compile(r"<TITLE>(.*?)</TITLE>", re.I | re.S)

ARXIV_RE = re.compile(r"\barXiv:\S+", re.I)

TAGS_RE = re.compile(r"<[^>]+>")

TEMP_K_RE = re.compile(
    r"(?:~|≈|about|around|above|below|over|nearly|roughly)?\s*(\d+(?:\.\d+)?)\s*K\b",
    re.I
)

PRESSURE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:gpa|kbar|bar)\b",
    re.I
)

# PRESSURE_WORD_RE = re.compile(r"pressure", re.I)

UNCONVENTIONAL_RE = re.compile(
    r"\b(antiferromagnet|mott\s+insulator)\b",
    re.I
)

EXPERIMENT_RE = re.compile(
    r"""
    \b(
        experiment|experimental|
        measure(d|ments)?|observ(ed|ations)?|
        transport\s+measurement|
        arpes|stm|spectroscopy|
        thin\s+film|film\s+grown|
        sample|synthes(i|e)zed
    )\b
    """,
    re.I | re.X
)

THEORY_RE = re.compile(
    r"""
    \b(
        theory|theoretical|
        model(ing)?|
        calculation(s)?|calculated|
        dft|ab\s+initio|first[-\s]?principles|
        mean[-\s]?field|
        simulation|
        hamiltonian
    )\b
    """,
    re.I | re.X
)
DIM_2D_RE = re.compile(
    r"\b(2d|two[-\s]?dimensional|single[-\s]?layer|monolayer|thin\s+film|ultrathin)\b",
    re.I
)

BULK_RE = re.compile(
    r"\b(bulk|three[-\s]?dimensional|3d|single\s+crystal)\b",
    re.I
)


# ------------------------ ФИЛЬТР 1ОЙ СТРАНИЦЫ ------------------------

FORBIDDEN_KEYWORDS = [
    "qubit",
    "josephson",
    "transmon",
    "fluxon",
    "squid",
    "majorana",
    "diode",
    "duality",
    "ads-cft",
]


def first_page_contains_forbidden(text: str) -> bool:
    """
    Проверяет первую страницу (первый абзац в txt)
    """
    first_page = text.split("\n\n", 1)[0].lower()
    return any(word in first_page for word in FORBIDDEN_KEYWORDS)


# ------------------------ TITLE ------------------------

def extract_title(text: str) -> str | None:
    matches = TITLE_RE.findall(text)
    if not matches:
        return None

    cleaned_parts = []
    for m in matches:
        t = TAGS_RE.sub("", m).strip()
        if t:
            cleaned_parts.append(t)

    if not cleaned_parts:
        return None

    return " ".join(cleaned_parts)


# ------------------------ ARXIV ID ------------------------

def extract_arxiv_id(text: str) -> str | None:
    m = ARXIV_RE.search(text)
    return m.group(0) if m else None

# ------------------------ T <sub>c</sub> 56 K ------------------------

def extract_tc_K(text: str) -> float | None:
    """
    Извлекает максимальную температуру в Кельвинах (Tc)
    """
    matches = TEMP_K_RE.findall(text)
    if not matches:
        return None

    values = []
    for m in matches:
        try:
            values.append(float(m))
        except ValueError:
            continue

    return max(values) if values else None

# ------------------------ Pressure ------------------------

def contains_pressure(text: str) -> bool:
    """
    True - статью убираем
    """
    if PRESSURE_RE.search(text):
        return True
    # if PRESSURE_WORD_RE.search(text):
    #     return True
    return False

# ------------------------ Тэг ------------------------

def extract_unconventional(text: str) -> bool:
    """
    True, если статья про unconventional superconductivity
    """
    return bool(UNCONVENTIONAL_RE.search(text))

# ------------------------ Тип ------------------------

def extract_article_type(text: str) -> str:
    """
    Возвращает: experiment | theory | hybrid
    """
    has_exp = bool(EXPERIMENT_RE.search(text))
    has_theory = bool(THEORY_RE.search(text))

    if has_exp and has_theory:
        return "hybrid"
    if has_exp:
        return "experiment"
    if has_theory:
        return "theory"

    return ""

# ------------------------ Dimension ------------------------

def extract_dimensionality(text: str) -> str:
    """
    Возвращает: "2D", "Bulk" или ""
    """
    has_2d = bool(DIM_2D_RE.search(text))
    has_bulk = bool(BULK_RE.search(text))

    if has_2d and has_bulk:
        return "mix of dimension"

    if has_2d:
        return "2D"

    if has_bulk:
        return "Bulk"

    return "Bulk"

# ------------------------ Материал ------------------------
# CHEM_ELEMENTS = {
#     "H","He","Li","Be","B","C","N","O","F","Ne",
#     "Na","Mg","Al","Si","P","S","Cl","Ar",
#     "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
#     "Ga","Ge","As","Se","Br","Kr",
#     "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
#     "In","Sn","Sb","Te","I","Xe",
#     "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy",
#     "Ho","Er","Tm","Yb","Lu",
#     "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
#     "Tl","Pb","Bi","Po","At","Rn",
#     "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf",
#     "Es","Fm","Md","No","Lr",
#     "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn",
#     "Nh","Fl","Mc","Lv","Ts","Og"
# }
#
# SC_CONTEXT = {
#     "superconduct",
#     "Tc",
#     "transition temperature",
#     "pairing",
#     "gap",
#     "film",
#     "bulk",
#     "monolayer",
#     "system",
#     "heterostructure",
#     "interface",
# }
#
# LIGHT_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S"}
#
# COMMON_BINARIES = {
#     "CN", "ON", "NO", "CO", "PC", "PN", "SN",
#     "UV", "IR", "RF"
# }
#
# ELEMENT_RE = r"(?:{})".format("|".join(sorted(CHEM_ELEMENTS, key=len, reverse=True)))
#
# SUB = r"<sub>[^<]+</sub>"
# SUP = r"<sup>[^<]+</sup>"
#
# INDEX = rf"(?:\s*{SUB}|\s*{SUP}|\d+)"
#
# FORMULA_CORE = rf"""
# {ELEMENT_RE}
# (?:{ELEMENT_RE}|{INDEX}|[-xXδ±])*
# """
#
# LEFT_BOUNDARY  = r"(?<![A-Za-z])"
# RIGHT_BOUNDARY = r"(?![A-Za-z])"
#
# MATERIAL_RE = re.compile(
#     rf"""
#     {LEFT_BOUNDARY}
#     (?P<material>
#         {FORMULA_CORE}
#         (?:\s*/\s*{FORMULA_CORE})*
#     )
#     {RIGHT_BOUNDARY}
#     """,
#     re.X
# )
#
#
# ELEMENT_TOKEN_RE = re.compile(ELEMENT_RE)
#
# def extract_elements(formula: str) -> set[str]:
#     return set(ELEMENT_TOKEN_RE.findall(formula))
#
# def count_sc_context(text: str, material: str, window: int = 80) -> int:
#
#     count = 0
#
#     for m in re.finditer(re.escape(material), text):
#         start = max(0, m.start() - window)
#         end = min(len(text), m.end() + window)
#         context = text[start:end].lower()
#
#         if any(k in context for k in SC_CONTEXT):
#             count += 1
#
#     return count
#
#
# def is_real_material(text: str, formula: str) -> bool:
#     # простая химическая валидность
#     if not is_valid_material_formula(formula):
#         return False
#
#     # должен встретиться хотя бы 1 раз в SC-контексте
#     sc_hits = count_sc_context(text, formula)
#
#     return sc_hits > 0
#
#
# def is_valid_material_formula(formula: str) -> bool:
#     elements = extract_elements(formula)
#
#     if len(elements) < 2:
#         return False
#
#     if formula in COMMON_BINARIES:
#         return False
#
#     if all(el in LIGHT_ELEMENTS for el in elements):
#         return False
#
#     return True
#
# def normalize_sub_sup(s: str) -> str:
#     return re.sub(
#         r"<(sub|sup)>\s*(.*?)\s*</\1>",
#         r"<\1>\2</\1>",
#         s
#     )
#
# def extract_materials(text: str, with_counts: bool = True, main_only: bool = True):
#
#     materials: dict[str, int] = {}
#
#     for m in MATERIAL_RE.finditer(text):
#         mat = re.sub(r"\s+", " ", m.group("material")).strip()
#         mat = normalize_sub_sup(mat)
#
#         if not is_valid_material_formula(mat):
#             continue
#
#         count = count_sc_context(text, mat)
#         if count <= 0:
#             continue
#
#         materials[mat] = materials.get(mat, 0) + count
#
#     if not materials:
#         return {} if with_counts else []
#
#     if main_only:
#         main_material = max(materials, key=materials.get)
#         if with_counts:
#             return {main_material: materials[main_material]}
#         else:
#             return [main_material]
#
#     if with_counts:
#         return materials
#     else:
#         return sorted(materials.keys())


import re

# --- 1. Элементы ---
CHEM_ELEMENTS = {
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy",
    "Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf",
    "Es","Fm","Md","No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn",
    "Nh","Fl","Mc","Lv","Ts","Og"
}

# сортировка важна!
ELEMENT_PATTERN = "|".join(sorted(CHEM_ELEMENTS, key=len, reverse=True))

# --- 2. HTML индексы ---
SUB = r"<sub>\d+[a-zA-Z\-xXδ]*</sub>"
SUP = r"<sup>\d+[+\-]?</sup>"

INDEX = rf"(?:{SUB}|{SUP}|\d+)"

# --- 3. основной блок формулы ---
# FORMULA_UNIT = rf"(?:{ELEMENT_PATTERN})(?:\s*{INDEX})?"
FORMULA_UNIT = rf"""
(?:{ELEMENT_PATTERN})(?:\s*{INDEX})?
| δ
| x
| y
"""

FORMULA_PATTERN = rf"""
{FORMULA_UNIT}
(?:\s*[-+xXδ±]?\s*{FORMULA_UNIT})*
"""

# гетероструктуры
FULL_FORMULA_PATTERN = rf"""
(?<![A-Za-z0-9])
(
    {FORMULA_PATTERN}
    (?:\s*/\s*{FORMULA_PATTERN})*
)
"""

def expand_formula(text: str, start: int, end: int) -> str:
    i = end
    n = len(text)

    depth = 0

    while i < n:
        c = text[i]

        # --- СКОБКИ ---
        if c == "(":
            depth += 1
            i += 1
            continue

        elif c == ")":
            # если скобка лишняя в старте — стоп
            if depth == 0:
                break

            depth -= 1
            i += 1
            continue

        # --- HTML ТЕГИ  ---
        if text[i:i + 5].lower() in {"<sub>", "<sup>"}:
            tag = text[i:i + 5].lower()
            close_tag = "</sub>" if tag == "<sub>" else "</sup>"

            j = i + 5
            while j < n and text[j:j + len(close_tag)].lower() != close_tag:
                j += 1

            if j < n:
                j += len(close_tag)
                i = j
                continue
            else:
                break

        # --- РАЗРЕШЁННЫЕ СИМВОЛЫ ---
        if c.isdigit():
            i += 1
            continue

        if c in "-+/=−":  # минус точка
            i += 1
            continue

        if c in {"δ", "x", "y"}:
            i += 1
            continue


        # # греческие буквы (кроме δ)
        if c in {"Γ", "Δ", "Λ", "Ω"}:
            break

        # --- ЭЛЕМЕНТ ---
        if c.isupper():
            # допускаем элемент: Fe, Cu, Sr
            if i + 1 < n and text[i+1].islower():
                i += 2
            else:
                i += 1
            continue

        if text[i] in {" ", "\n"}:
            i += 1
            continue

        if text[i] in {"-", "+", "−"}:
            i += 1
            continue

        # остальное — стоп
        break

    return text[start:i]

MATERIAL_RE = re.compile(FULL_FORMULA_PATTERN, re.X)

def restore_subscripts(text: str) -> str:
    return re.sub(
        r"([A-Z][a-z]?)(\d+)",
        r"\1<sub>\2</sub>",
        text
    )

def normalize_formula_indices(s: str) -> str:
    # --- 1. объединяем 2 - x → 2-x внутрь sub ---
    s = re.sub(
        r"<sub>(\d+)</sub>-([a-z])",
        r"<sub>\1-\2</sub>",
        s
    )

    # --- 2. Mx → M<sub>x</sub> ---
    s = re.sub(
        r"([A-Z])([xyz])\b",
        r"\1<sub>\2</sub>",
        s
    )

    # --- 3. O<sub>8</sub>+δ → O<sub>8+δ</sub> ---
    s = re.sub(
        r"<sub>(\d+)</sub>\+δ",
        r"<sub>\1+δ</sub>",
        s
    )

    # O<sub>7</sub><sub>δ</sub> → O<sub>7-δ</sub>
    s = re.sub(
        r"<sub>(\d+)</sub>\s*<sub>(δ)</sub>",
        r"<sub>\1-δ</sub>",
        s
    )

    return s

# --- 4. нормализация ---
def normalize_text(text: str) -> str:
    text = re.sub(r"<(sub|sup)>\s*(.*?)\s*</\1>", r"<\1>\2</\1>", text)

    text = re.sub(r"([A-Za-z])\s+(<sub>\d+</sub>)", r"\1\2", text)

    text = re.sub(r"(</sub>|</sup>)\s+([A-Z])", r"\1\2", text)

    text = re.sub(r"([A-Z][a-z]?)\s+(\d+)", r"\1\2", text)

    # Bi 2 → Bi2
    text = re.sub(r"([A-Z][a-z]?)\s+(\d+)", r"\1\2", text)

    # 2 Sr → 2Sr
    text = re.sub(r"(\d)\s+([A-Z])", r"\1\2", text)

    # 2 − x → 2-x
    text = re.sub(r"([0-9])\s*[−-]\s*([a-zA-Z])", r"\1-\2", text)

    # x M x → xMx
    text = re.sub(r"\b([a-z])\s+([A-Z])\s+([a-z])\b", r"\1\2\3", text)

    # O 8 → O8
    text = re.sub(r"([A-Z])\s+(\d+)", r"\1\2", text)

    # + δ → +δ
    text = re.sub(r"\+\s+δ", r"+δ", text)

    text = re.sub(r"\s+\)", ")", text)

    text = restore_subscripts(text)

    text = re.sub(r"\s+", " ", text)

    return text


# --- 5. извлечение элементов ---
ELEMENT_TOKEN_RE = re.compile(ELEMENT_PATTERN)

def extract_elements(formula: str) -> set[str]:
    return set(ELEMENT_TOKEN_RE.findall(formula))


# --- 6. фильтр формул ---
# def looks_like_real_formula(formula: str) -> bool:
#     # должен быть индекс или модификатор
#     if not re.search(r"\d|<sub>|<sup>|[-xXδ]", formula):
#         return False
#
#     elements = extract_elements(formula)
#
#     # минимум 2 элемента
#     if len(elements) < 2:
#         return False
#
#     # запрещаем чистые аббревиатуры типа BCS
#     if re.fullmatch(r"[A-Z]{2,4}", formula):
#         return False
#
#     return True

def looks_like_real_formula(formula: str) -> bool:
    # убираем HTML
    clean = re.sub(r"<[^>]+>", "", formula)

    # --- 1. должны быть элементы ---
    elements = extract_elements(clean)
    if len(elements) < 2:
        return False

    # --- 2. запрет на слова ---
    if re.search(r"[A-Z]{3,}", clean):
        return False

    # --- 3. запрет FIG, TABLE и т.д.
    if re.search(r"\b(FIG|TABLE|EQUATION)\b", clean, re.I):
        return False

    # --- инициалы ---
    if re.search(r"(?:[A-Z]\.\s*){2,}", formula):
        return False

    # # --- физика ---
    if "=" in formula:
        return False

    # # --- запятая ---
    if "," in formula:
        return False

    # --- 4. должен быть индекс или модификатор
    if not re.search(r"\d|δ|x|<sub>|<sup>", formula):
        return False

    # --- 5. не должно быть обычных слов
    if re.search(r"[a-z]{3,}", clean):
        return False

    # если индекс 0 → почти всегда мусор
    if re.search(r"<sub>0</sub>", formula):
        return False

    return True

import math

MATERIAL_CONTEXT = {
    "system",
    "compound",
    "material",
    "crystal",
    "film",
    "sample",
    "superconductor",
}

BAD_CONTEXT = {
    "plane",
    "layer",
    "surface",
    "interface",
}


def position_score(start: int, text_len: int) -> float:
    return 1.0 - (start / text_len)


def context_features(text: str, start: int, end: int):
    window = text[max(0, start - 80): end + 80].lower()

    has_good = any(k in window for k in MATERIAL_CONTEXT)
    has_bad = any(k in window for k in BAD_CONTEXT)

    return has_good, has_bad


def score_formula(formula: str, matches, text: str) -> float:
    total_score = 0

    for m in matches:
        start, end = m.span()

        score = 0

        # позиция (самый важный сигнал)
        score += 2.0 * position_score(start, len(text))

        # контекст
        has_good, has_bad = context_features(text, start, end)

        if has_good:
            score += 2.5

        if has_bad:
            score -= 2.0

        # базовый вклад
        score += 0.3

        total_score += score

    # небольшой вклад частоты
    total_score += 0.5 * math.log(1 + len(matches))

    return total_score

def dump_pdf_context(text: str, max_matches: int = 20):
    with open("pdf_debug.txt", "w", encoding="utf-8") as f:
        count = 0

        for m in MATERIAL_RE.finditer(text):
            start, end = m.span(1)

            # широкий контекст (важно!)
            left = max(0, start - 300)
            right = min(len(text), end + 300)

            snippet = text[left:right]

            f.write("="*100 + "\n")
            f.write(f"MATCH: {m.group(1)}\n\n")
            f.write("CONTEXT:\n")
            f.write(snippet + "\n")

            count += 1
            if count >= max_matches:
                break

# --- 7. извлечение материалов ---
def extract_materials(
    text: str,
    with_counts: bool = True,
    main_only: bool = False
):
    text = re.sub(r"</?TITLE>", "", text)
    text = normalize_text(text)

    # собираем все матчи по формуле
    material_matches = {}

    for m in MATERIAL_RE.finditer(text):
        # formula = m.group(1).strip()
        start, end = m.span(1)
        dump_pdf_context(text)
        formula = expand_formula(text, start, end).strip()

        formula = normalize_formula_indices(formula)

        if not looks_like_real_formula(formula):
            continue

        material_matches.setdefault(formula, []).append(m)

    if not material_matches:
        return {} if with_counts else []

    # считаем score для каждой формулы
    scores = {
        formula: score_formula(formula, matches, text)
        for formula, matches in material_matches.items()
    }

    # сортируем по score
    sorted_items = sorted(scores.items(), key=lambda x: -x[1])

    if main_only:
        main_material = sorted_items[0][0]

        if with_counts:
            return {main_material: len(material_matches[main_material])}
        else:
            return [main_material]

    if with_counts:
        return {
            formula: len(material_matches[formula])
            for formula, _ in sorted_items
        }
    else:
        return [formula for formula, _ in sorted_items]

# ------------------------ Дебаевская частота ------------------------

NUMBER_RE = r"\d+(?:\.\d+)?"

DEBYE_SYMBOL_RE = r"""
(?:Debye\s+frequenc(?:y|ies)) |
(?:ω|\\omega)\s*
(?:<sub>\s*(?:D|Db)\s*</sub>|_(?:D|Db))
"""

DEBYE_VALUE_RE = re.compile(
    rf"""
    (
        {NUMBER_RE}      # число перед
        \s*
        {DEBYE_SYMBOL_RE}
    )
    |
    (
        {DEBYE_SYMBOL_RE}
        \s*
        {NUMBER_RE}      # число после
    )
    """,
    re.X | re.I
)

def extract_debye_frequency(text: str) -> list[float]:
    values = []

    for m in DEBYE_VALUE_RE.finditer(text):
        # число в совпадении
        nums = re.findall(r"\d+(?:\.\d+)?", m.group(0))
        if nums:
            values.append(float(nums[0]))

    return values


# ------------------------ PAYLOAD ------------------------

def extract_payload_stub(text: str) -> dict:
    return {
        "material": extract_materials(text, with_counts=False, main_only=True),
        "tc_K": extract_tc_K(text),
        "dimensionality": extract_dimensionality(text),
        "type": extract_article_type(text),
        # "pressure": None,           # Используется во 2ом фильтре
        "unconventional": extract_unconventional(text),
        "debye_frequency": extract_debye_frequency(text),
    }


def extract_vector_stub(text: str):
    """
    Под векторизацию
    """
    return None


# ------------------------ ОБРАБОТКА ФАЙЛА ------------------------

def process_file(txt_path: Path, out_dir: Path):
    text = txt_path.read_text(encoding="utf-8", errors="ignore")

    # фильтр 1: пропускаем файл целиком
    if first_page_contains_forbidden(text):
        return None, None

    # фильтр 2: давление (вся статья)
    if contains_pressure(text):
        return None, None

    title = extract_title(text)
    arxiv_id = extract_arxiv_id(text)

    data = {
        "id": arxiv_id,
        "title": title,
        "payload": extract_payload_stub(text),
        "vector": extract_vector_stub(text),
    }

    out_path = out_dir / (txt_path.stem + ".json")
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return out_path, data


# ------------------------ CLI ------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert parsed PDF text files into structured JSON."
    )
    parser.add_argument("input", help="Файл .txt или директория с .txt")
    parser.add_argument("--out-dir", "-o", default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    inp = Path(args.input)

    if not inp.exists():
        print("Input not found.", file=sys.stderr)
        return

    if inp.is_file() and inp.suffix.lower() == ".txt":
        files = [inp]
    elif inp.is_dir():
        files = sorted(inp.glob("*.txt"))
    else:
        print("Input must be .txt file or directory", file=sys.stderr)
        return

    out_dir = Path(args.out_dir) if args.out_dir else (
        inp.parent if inp.is_file() else inp
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        try:
            out_path, data = process_file(f, out_dir)
            if out_path is None:
                if args.debug:
                    print(f"Skipped (forbidden keywords): {f}")
                continue

            if args.debug:
                print(f"Processed: {f} → {out_path}")
                print(json.dumps(data, ensure_ascii=False, indent=2))

        except Exception as e:
            print(f"Error processing {f}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
