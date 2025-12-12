import argparse
import subprocess
import sys
from pathlib import Path
import shutil


def process_pdf(pdf: Path, out_dir: Path, converter_script: Path, parser_script: Path,
                meta_dir: Path | None, no_meta: bool, json_dir):
    base = pdf.stem

    # где сохранить txt
    out_txt = out_dir / f"{base}.txt"

    # metaout только если meta_dir указан
    if meta_dir:
        meta_file = meta_dir / f"{base}_meta.json"
    else:
        meta_file = None


    # ---------------------------- PDF → TXT -----------------------------
    cmd = [
        sys.executable,
        str(converter_script),
        str(pdf),
        "--out", str(out_txt)
    ]

    if meta_file and not no_meta:
        cmd += ["--metaout", str(meta_file)]

    if no_meta:
        cmd.append("--no-meta")

    subprocess.run(cmd, check=True)


    # ---------------------------- TXT → JSON -----------------------------
    cmd = [
        sys.executable,
        str(parser_script),
        str(out_txt),
        "--out-dir", str(json_dir)
    ]

    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("pdf_dir", help="directory containing PDF files")
    p.add_argument("--out-dir", required=True, help="directory for txt + json output")
    p.add_argument("--converter", default="pdf2text.py", help="PDF → TXT script")
    p.add_argument("--parser", default="text2json.py", help="TXT → JSON script")

    # если указано, пишем
    p.add_argument("--metaout", help="directory for metadata JSONs")

    # удалить txt после окончания
    p.add_argument("--delete-txt", action="store_true")

    p.add_argument("--no-meta", action="store_true")

    args = p.parse_args()

    pdf_dir = Path(args.pdf_dir)

    converter_script = Path(args.converter)
    parser_script = Path(args.parser)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON всегда лежат в поддиректории out_dir/json
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    # если указал директорию для меты
    if args.metaout:
        meta_dir = Path(args.metaout)
        meta_dir.mkdir(parents=True, exist_ok=True)
    else:
        meta_dir = None

    # обрабатываем без подпапок
    pdfs = sorted(pdf_dir.glob("*.pdf"))

    for pdf in pdfs:
        process_pdf(
            pdf=pdf,
            out_dir=out_dir,
            converter_script=converter_script,
            parser_script=parser_script,
            meta_dir=meta_dir,
            no_meta=args.no_meta,
            json_dir=json_dir
        )


    # ---------------------------- DELETE TXT если флаг -----------------------------
    if args.delete_txt:
        for txt in out_dir.glob("*.txt"):
            txt.unlink()

        # если пусто, можно удалить саму папку
        if not any(out_dir.iterdir()):
            try:
                out_dir.rmdir()
            except:
                pass


if __name__ == "__main__":
    main()
