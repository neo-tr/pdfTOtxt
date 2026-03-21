import argparse
import json
import logging

import boto3
import psycopg2
from psycopg2.extras import execute_batch

from pdf2text import convert_pdf_to_text
from text2json import (
    extract_payload_stub,
    extract_title,
    extract_arxiv_id,
    first_page_contains_forbidden,
    contains_pressure,
)

# ---------------------------- S3 ----------------------------

def get_s3_client(endpoint, key, secret):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )


def list_pdfs(s3, bucket, prefix=""):
    continuation_token = None

    while True:
        kwargs = {
            "Bucket": bucket,
            "MaxKeys": 1000,
            "Prefix": prefix,
        }

        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        resp = s3.list_objects_v2(**kwargs)

        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".pdf"):
                yield key

        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break

def load_pdf_bytes(s3, bucket, key) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

from pathlib import Path

def key_to_arxiv_id(key: str) -> str:
    name = Path(key).stem

    # старый стиль: supr-con_9609001 в supr-con/9609001
    if "_" in name and name.split("_")[0].count("-") > 0:
        prefix, rest = name.split("_", 1)
        return f"{prefix}/{rest}"

    return name


# ---------------------------- PostgreSQL ----------------------------

def get_pg_conn(host, db, user, password, port):
    return psycopg2.connect(
        host=host,
        dbname=db,
        user=user,
        password=password,
        port=port,
    )

def update_payload(conn, rows):
    with conn.cursor() as cur:
        execute_batch(
            cur,
            """
            UPDATE arxiv_paper
            SET payload = %s::jsonb
            WHERE arxiv_id = %s
            """,
            [(payload, arxiv_id) for arxiv_id, payload in rows],
            page_size=50,
        )
    conn.commit()


def update_status(conn, arxiv_id, status):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO arxiv_processing_status (arxiv_id, status)
            VALUES (%s, %s)
            ON CONFLICT (arxiv_id)
            DO UPDATE SET
                status = EXCLUDED.status,
                updated_at = now()
        """, (arxiv_id, status))
    conn.commit()


# ---------------------------- Нормализация ----------------------------

def normalize_payload(p):
    return {
        "tc_K": p.get("tc_K"),
        "type": p.get("type"),
        "material": p.get("material") or {},
        "dimensionality": p.get("dimensionality"),
        "unconventional": p.get("unconventional"),
        "debye_frequency": p.get("debye_frequency") or [],
    }


# ---------------------------- Обработка ----------------------------

def process_pdf_bytes(pdf_bytes: bytes):

    text, _ = convert_pdf_to_text(pdf_bytes)

    payload = extract_payload_stub(text)

    return {
        "text": text,
        "payload": normalize_payload(payload),
        "id": extract_arxiv_id(text),
        "title": extract_title(text),
    }


# ---------------------------- MAIN ----------------------------

def main():
    parser = argparse.ArgumentParser()

    # --- S3 ---
    parser.add_argument("--s3-endpoint", required=True)
    parser.add_argument("--s3-key", required=True)
    parser.add_argument("--s3-secret", required=True)
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--s3-prefix", default="pdf/")

    # --- Postgres ---
    parser.add_argument("--pg-host", required=True)
    parser.add_argument("--pg-db", required=True)
    parser.add_argument("--pg-user", required=True)
    parser.add_argument("--pg-password", required=True)
    parser.add_argument("--pg-port", default=5432)

    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    # print(args)

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("orchestrator")

    # init
    s3 = get_s3_client(args.s3_endpoint, args.s3_key, args.s3_secret)
    conn = get_pg_conn(args.pg_host, args.pg_db, args.pg_user, args.pg_password, args.pg_port)

    buffer = []
    processed = 0

    for key in list_pdfs(s3, args.s3_bucket, args.s3_prefix):

        arxiv_id = key_to_arxiv_id(key)

        try:
            log.info(f"Processing: {key} → {arxiv_id}")

            update_status(conn, arxiv_id, "new")

            pdf_bytes = load_pdf_bytes(s3, args.s3_bucket, key)

            text, _ = convert_pdf_to_text(pdf_bytes)

            # --- фильтры ---
            if first_page_contains_forbidden(text):
                log.info(f"FAILED (forbidden): {arxiv_id}")
                update_status(conn, arxiv_id, "filtered_firstPage")
                continue

            if contains_pressure(text):
                log.info(f"FAILED (pressure): {arxiv_id}")
                update_status(conn, arxiv_id, "filtered_pressure")
                continue

            payload = normalize_payload(extract_payload_stub(text))

            buffer.append((
                arxiv_id,
                json.dumps(payload),
            ))

            # --- update ---
            if len(buffer) >= args.batch_size:
                update_payload(conn, buffer)

                for aid, _ in buffer:
                    update_status(conn, aid, "done")

                log.info(f"Updated batch: {len(buffer)}")
                buffer.clear()

            processed += 1

            if args.limit and processed >= args.limit:
                break

        except Exception as e:
            log.exception(f"Error processing {key}: {e}")
            update_status(conn, arxiv_id, "failed")

    if buffer:
        update_payload(conn, buffer)

        for aid, _ in buffer:
            update_status(conn, aid, "done")

        log.info(f"Updated final batch: {len(buffer)}")

    conn.close()
    log.info("Done")


if __name__ == "__main__":
    main()
