
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".venv", ".git", "__pycache__"}

def main():
    csv_files = []

    for path in ROOT.rglob("*.csv"):
        if not any(part in SKIP_DIRS for part in path.parts):
            csv_files.append(path)

    if not csv_files:
        print(f"No CSV files found under: {ROOT}")
        print("Copy your existing project CSV files into data/ or results/.")
        return

    for path in sorted(csv_files):
        print("\n" + "=" * 80)
        print(f"FILE: {path.relative_to(ROOT)}")

        try:
            df = pd.read_csv(path, nrows=5)
            full_df = pd.read_csv(path)

            print(f"Rows: {len(full_df)}")
            print(f"Columns: {list(full_df.columns)}")
            print("\nSample:")
            print(df.head(3).to_string(index=False, max_colwidth=90))

            columns = [str(c).lower().strip() for c in full_df.columns]

            original_cols = [
                c for c in columns
                if "original" in c or "source_text" in c
            ]
            simplified_cols = [
                c for c in columns
                if "simplif" in c or "target_text" in c
            ]
            rating_cols = [
                c for c in columns
                if any(term in c for term in
                       ("human_score", "human_rating", "annotator",
                        "ground_truth", "gold_score", "lmp_rating"))
            ]

            print(f"\nPossible original-text columns: {original_cols}")
            print(f"Possible simplified-text columns: {simplified_cols}")
            print(f"Possible human-rating columns: {rating_cols}")

        except Exception as exc:
            print(f"Could not read this CSV: {exc}")

if __name__ == "__main__":
    main()
