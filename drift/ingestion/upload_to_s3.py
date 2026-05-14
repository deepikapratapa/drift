"""
drift/ingestion/upload_to_s3.py

Converts raw REES46 CSV files to Parquet and uploads to S3.
Parquet is columnar — dramatically faster for Athena queries and
cheaper (you pay per byte scanned).
"""

import os
import boto3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("S3_BUCKET_NAME")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

DTYPE_MAP = {
    "event_time": str,
    "event_type": "category",
    "product_id": "int32",
    "category_id": "int64",
    "category_code": "category",
    "brand": "category",
    "price": "float32",
    "user_id": "int64",
    "user_session": str,
}


def csv_to_parquet(csv_path: Path) -> Path:
    """Read CSV, cast dtypes, write Parquet beside it."""
    print(f"Reading {csv_path.name} ...")
    df = pd.read_csv(csv_path, dtype=DTYPE_MAP)

    # Parse event_time properly
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)

    # Add partition columns for Athena efficiency
    df["year"] = df["event_time"].dt.year.astype("int16")
    df["month"] = df["event_time"].dt.month.astype("int8")

    parquet_path = csv_path.with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"  Wrote {parquet_path.name} — {len(df):,} rows, "
          f"{parquet_path.stat().st_size / 1e6:.1f} MB")
    return parquet_path


def upload_to_s3(parquet_path: Path, s3_key: str) -> None:
    """Upload a local Parquet file to S3."""
    s3 = boto3.client("s3", region_name=REGION)
    print(f"Uploading {parquet_path.name} → s3://{BUCKET}/{s3_key} ...")
    s3.upload_file(
        Filename=str(parquet_path),
        Bucket=BUCKET,
        Key=s3_key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )
    print(f"  Done.")


def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}. "
              "Download 2019-Oct.csv and 2019-Nov.csv from Kaggle.")
        return

    for csv_path in csv_files:
        # Convert to Parquet locally
        parquet_path = csv_to_parquet(csv_path)

        # Upload under a clean S3 prefix
        month_slug = csv_path.stem.lower().replace(" ", "-")  # e.g. 2019-oct
        s3_key = f"raw/{month_slug}/{parquet_path.name}"
        upload_to_s3(parquet_path, s3_key)

    print("\nAll files uploaded. Run athena setup next.")


if __name__ == "__main__":
    main()
