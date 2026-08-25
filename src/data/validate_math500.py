import json
from pathlib import Path


DATA_PATH = Path("data/raw/math500_test.jsonl")

REQUIRED_FIELDS = {
    "problem",
    "solution",
    "answer",
}


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATA_PATH}"
        )

    records = []

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            records.append(record)

    print(f"Total records: {len(records)}")

    if not records:
        raise ValueError("Dataset is empty.")

    missing_fields = []

    for index, record in enumerate(records):
        missing = REQUIRED_FIELDS - record.keys()

        if missing:
            missing_fields.append(
                {
                    "index": index,
                    "missing": sorted(missing),
                }
            )

    if missing_fields:
        raise ValueError(
            f"Records with missing fields: {missing_fields[:10]}"
        )

    print("Required fields: OK")

    print("\nFirst problem:")
    print(records[0]["problem"])

    print("\nGround-truth answer:")
    print(records[0]["answer"])

    print("\nDataset validation: PASSED")


if __name__ == "__main__":
    main()