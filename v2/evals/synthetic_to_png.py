"""Render evals/synthetic_raw.csv as a PNG table.

Usage (from v2/):
    uv run python -m evals.synthetic_to_png
    uv run python -m evals.synthetic_to_png evals/synthetic_raw.csv evals/synthetic_raw.png
"""
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "evals/synthetic_raw.csv"
DST = sys.argv[2] if len(sys.argv) > 2 else "evals/synthetic_raw.png"

# column title -> (csv column, characters per line before wrapping, relative width)
COLUMNS = {
    "#": (None, 3, 0.03),
    "Question": ("user_input", 55, 0.30),
    "Reference answer": ("reference", 80, 0.45),
    "Persona": ("persona_name", 14, 0.10),
    "Type": ("synthesizer_name", 12, 0.12),
}
LINE_H = 0.19    # inches per line of text
PAD = 0.25       # extra inches per row
FIG_W = 18

df = pd.read_csv(SRC)
df["synthesizer_name"] = (
    df["synthesizer_name"].str.replace("_query_synthesizer", "").str.replace("_", " ")
)

cells, n_lines = [], []
for i, row in df.iterrows():
    wrapped = []
    for title, (col, width, _) in COLUMNS.items():
        text = str(i + 1) if col is None else str(row[col])
        wrapped.append(textwrap.fill(text, width))
    cells.append(wrapped)
    n_lines.append(max(c.count("\n") + 1 for c in wrapped))

row_h = [n * LINE_H + PAD for n in n_lines]
header_h = 0.5
fig_h = header_h + sum(row_h)

fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
ax.axis("off")
table = ax.table(
    cellText=cells,
    colLabels=list(COLUMNS),
    colWidths=[w for _, _, w in COLUMNS.values()],
    cellLoc="left",
    loc="upper left",
    bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(9)

total = fig_h
for (r, c), cell in table.get_celld().items():
    cell.set_height((header_h if r == 0 else row_h[r - 1]) / total)
    cell.set_edgecolor("#cccccc")
    cell.PAD = 0.02
    if r == 0:
        cell.set_facecolor("#2f3e4e")
        cell.set_text_props(color="white", weight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#f4f6f8")

fig.savefig(DST, dpi=150, bbox_inches="tight")
print(f"Saved {DST} ({len(df)} rows)")
