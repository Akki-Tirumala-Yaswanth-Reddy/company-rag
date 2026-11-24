"""Render RAGAS result CSVs as readable tables.

Usage (from v2/):
    uv run python evals/show.py                      # every evals/*.csv
    uv run python evals/show.py evals/ragas_golden.csv
    uv run python evals/show.py --full               # also print question, answer and reference text
"""
import sys
import textwrap
from pathlib import Path

import pandas as pd

SCORES = {
    "llm_context_precision_with_reference": "ctx_prec",
    "context_recall": "ctx_recall",
    "faithfulness": "faithful",
    "answer_relevancy": "relevancy",
}


def show(path: Path, full: bool) -> None:
    df = pd.read_csv(path)
    scores = df[list(SCORES)].rename(columns=SCORES)

    table = scores.round(2)
    table.insert(0, "question", df["user_input"].str.slice(0, 55) + "...")
    table.index = pd.RangeIndex(1, len(df) + 1, name="#")

    print(f"\n=== {path} ({len(df)} questions) ===")
    print(table.to_string(na_rep="NaN"))
    print("\naverage (NaN ignored):")
    print(scores.mean().round(2).to_string())
    print("NaN count:", scores.isna().sum().to_dict())

    if full:
        for n, row in df.iterrows():
            print(f"\n--- #{n + 1} ---")
            for label in ("user_input", "response", "reference"):
                print(f"{label}:\n{textwrap.indent(str(row[label]), '  ')}")


if __name__ == "__main__":
    args = sys.argv[1:]
    full = "--full" in args
    paths = [Path(a) for a in args if a != "--full"] or sorted(Path("evals").glob("*.csv"))
    for p in paths:
        show(p, full)
