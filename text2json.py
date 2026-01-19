"""
Конвертирует текстовые файлы (после pdf2text.py) в JSON формата:
"""

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
    Проверяет первую страницу (первый абзац)
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
        return ""

    if has_2d:
        return "2D"

    if has_bulk:
        return "Bulk"

    return ""

# ------------------------ Материал ------------------------
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

SC_CONTEXT = {
    "superconduct",
    "Tc",
    "transition temperature",
    "pairing",
    "gap",
    "film",
    "bulk",
    "monolayer",
    "system",
    "heterostructure",
    "interface",
}

LIGHT_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S"}

COMMON_BINARIES = {
    "CN", "ON", "NO", "CO", "PC", "PN", "SN",
    "UV", "IR", "RF"
}

ELEMENT_RE = r"(?:{})".format("|".join(sorted(CHEM_ELEMENTS, key=len, reverse=True)))

SUB = r"<sub>[^<]+</sub>"
SUP = r"<sup>[^<]+</sup>"

INDEX = rf"(?:\s*{SUB}|\s*{SUP}|\d+)"

FORMULA_CORE = rf"""
{ELEMENT_RE}
(?:{ELEMENT_RE}|{INDEX}|[-xXδ±])*
"""

LEFT_BOUNDARY  = r"(?<![A-Za-z])"
RIGHT_BOUNDARY = r"(?![A-Za-z])"

MATERIAL_RE = re.compile(
    rf"""
    {LEFT_BOUNDARY}
    (?P<material>
        {FORMULA_CORE}
        (?:\s*/\s*{FORMULA_CORE})*
    )
    {RIGHT_BOUNDARY}
    """,
    re.X
)


ELEMENT_TOKEN_RE = re.compile(ELEMENT_RE)

def extract_elements(formula: str) -> set[str]:
    return set(ELEMENT_TOKEN_RE.findall(formula))

def count_sc_context(text: str, material: str, window: int = 80) -> int:

    count = 0

    for m in re.finditer(re.escape(material), text):
        start = max(0, m.start() - window)
        end = min(len(text), m.end() + window)
        context = text[start:end].lower()

        if any(k in context for k in SC_CONTEXT):
            count += 1

    return count


def is_real_material(text: str, formula: str) -> bool:
    # простая химическая валидность
    if not is_valid_material_formula(formula):
        return False

    # должен встретиться хотя бы 1 раз в SC-контексте
    sc_hits = count_sc_context(text, formula)

    return sc_hits > 0


def is_valid_material_formula(formula: str) -> bool:
    elements = extract_elements(formula)

    if len(elements) < 2:
        return False

    if formula in COMMON_BINARIES:
        return False

    if all(el in LIGHT_ELEMENTS for el in elements):
        return False

    return True

def normalize_sub_sup(s: str) -> str:
    return re.sub(
        r"<(sub|sup)>\s*(.*?)\s*</\1>",
        r"<\1>\2</\1>",
        s
    )

def extract_materials(
    text: str,
    with_counts: bool = True,
    main_only: bool = True
):

    materials: dict[str, int] = {}

    for m in MATERIAL_RE.finditer(text):
        mat = re.sub(r"\s+", " ", m.group("material")).strip()
        mat = normalize_sub_sup(mat)

        if not is_valid_material_formula(mat):
            continue

        count = count_sc_context(text, mat)
        if count <= 0:
            continue

        materials[mat] = materials.get(mat, 0) + count

    if not materials:
        return {} if with_counts else []

    if main_only:
        main_material = max(materials, key=materials.get)
        if with_counts:
            return {main_material: materials[main_material]}
        else:
            return [main_material]

    if with_counts:
        return materials
    else:
        return sorted(materials.keys())


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
        "material": extract_materials(text, with_counts=True, main_only=False),
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
