from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "HuggingFaceH4/MATH-500"
OUTPUT_DIR = Path("data/raw")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME)

    print(dataset)

    for split_name, split in dataset.items():
        output_path = OUTPUT_DIR / f"math500_{split_name}.jsonl"

        split.to_json(
            output_path,
            orient="records",
            lines=True,
        )

        print(
            f"Saved {split_name}: "
            f"{len(split)} problems -> {output_path}"
        )


if __name__ == "__main__":
    main()