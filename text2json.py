"""
Конвертирует текстовые файлы (после pdf2text.py) в JSON формата:

{
  "id": "...",
  "title": "..."
}

Ищет только:
  <TITLE>...</TITLE>
  arXiv:...
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ------------------------ REGEX ------------------------

# Ищем <TITLE>...<TITLE> (их может быть несколько)
TITLE_RE = re.compile(r"<TITLE>(.*?)</TITLE>", re.I | re.S)

# Ищем arXiv ID
ARXIV_RE = re.compile(r"\barXiv:\S+", re.I)

# Убираем HTML-подобные теги внутри TITLE
TAGS_RE = re.compile(r"<[^>]+>")


# ------------------------ экстракт TITLE ------------------------

def extract_title(text: str) -> str | None:
    """
    Извлекает ВСЕ <TITLE>...</TITLE> и склеивает в одну строку
    """

    matches = TITLE_RE.findall(text)
    if not matches:
        return None

    cleaned_parts = []

    for m in matches:
        # убираем теги <sub>, <sup>
        t = TAGS_RE.sub("", m).strip()
        if t:
            cleaned_parts.append(t)

    if not cleaned_parts:
        return None

    # Склеиваем в одно предложение
    return " ".join(cleaned_parts)


# ------------------------ ARXIV ID ------------------------

def extract_arxiv_id(text: str) -> str | None:
    """
    Возвращает первый найденный arXiv ID
    """
    m = ARXIV_RE.search(text)
    if not m:
        return None
    return m.group(0)


# ------------------------ Обработка одного файла ------------------------

def process_file(txt_path: Path, out_dir: Path):
    text = txt_path.read_text(encoding="utf-8", errors="ignore")

    title = extract_title(text)
    arxiv_id = extract_arxiv_id(text)

    data = {
        "id": arxiv_id,
        "title": title,
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
        description="Convert text files (parsed PDFs) into JSONs with id/title."
    )
    parser.add_argument("input", help="Файл .txt или директория с .txt файлами")
    parser.add_argument("--out-dir", "-o", default=None,help="куда сохранять .json (по умолчанию рядом с входным)")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    inp = Path(args.input)

    if not inp.exists():
        print("Input not found.", file=sys.stderr)
        return

    # Один файл или директория
    if inp.is_file() and inp.suffix.lower() == ".txt":
        files = [inp]
    elif inp.is_dir():
        files = sorted(inp.glob("*.txt"))
    else:
        print("Input must be .txt file or directory", file=sys.stderr)
        return

    # Выходная директория
    out_dir = Path(args.out_dir) if args.out_dir else (
        inp.parent if inp.is_file() else inp
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        try:
            out_path, data = process_file(f, out_dir)
            if args.debug:
                print(f"Processed: {f} → {out_path}")
                print(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Error processing {f}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
