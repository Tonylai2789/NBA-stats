"""
Create the AP Stats data display artifacts from the cached player-game data.

Default output:
  - prints the LaTeX table to the terminal
  - displays the scatterplot with matplotlib

The script does not re-scrape any website. It uses the CSV files already stored
in bbref_cache/ and recomputes the previous-5-games workload variable.
"""
from __future__ import annotations

import argparse
import os
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
LOCAL_DEPS = ROOT / ".python_deps"
if LOCAL_DEPS.exists() and any(LOCAL_DEPS.glob(f"**/*.{sys.implementation.cache_tag}-*.so")):
    sys.path.insert(0, str(LOCAL_DEPS))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

try:
    import matplotlib

    if not sys.stdout.isatty() and "MPLBACKEND" not in os.environ:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing or incompatible Python packages. In VS Code, run these commands "
        "from /Users/tonylai/Desktop/NBA_stats first:\n\n"
        "  python3 -m venv .venv\n"
        "  .venv/bin/python -m pip install -r requirements.txt\n"
        "  .venv/bin/python make_data_display.py\n"
    ) from exc

CACHE = ROOT / "bbref_cache"
WINDOW = 5
RAW_TABLE_ROWS = 12


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fix_mojibake(text: object) -> object:
    if not isinstance(text, str):
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text


def ascii_name(text: object) -> object:
    if not isinstance(text, str):
        return text
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def load_player_names() -> pd.DataFrame:
    qualifiers = []
    for path in sorted(CACHE.glob("_qualifiers_*.csv")):
        df = pd.read_csv(path)
        df["season"] = int(path.stem.split("_")[-1])
        df["name"] = df["name"].map(fix_mojibake)
        df["name"] = df["name"].map(ascii_name)
        qualifiers.append(df[["player_id", "season", "name"]])
    return pd.concat(qualifiers, ignore_index=True).drop_duplicates()


def load_game_logs() -> pd.DataFrame:
    parts = []
    for path in sorted(CACHE.glob("*.csv")):
        if path.name.startswith("_qualifiers_"):
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        if {"Date", "MP", "GmSc", "player_id", "season"}.issubset(df.columns):
            parts.append(df[["Date", "MP", "GmSc", "player_id", "season"]])

    games = pd.concat(parts, ignore_index=True)
    games["MP"] = pd.to_numeric(games["MP"], errors="coerce")
    games["GmSc"] = pd.to_numeric(games["GmSc"], errors="coerce")
    games = games.dropna(subset=["Date", "MP", "GmSc"])

    names = load_player_names()
    games = games.merge(names, on=["player_id", "season"], how="inner")
    return games


def add_recent_workload(games: pd.DataFrame) -> pd.DataFrame:
    games = games.sort_values(["player_id", "season", "Date"]).reset_index(drop=True)
    col = f"MP_prev_{WINDOW}"
    games[col] = (
        games.groupby(["player_id", "season"])["MP"]
        .transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=WINDOW).sum())
    )
    return games.dropna(subset=[col]).reset_index(drop=True)


def create_scatterplot(data: pd.DataFrame) -> plt.Figure:
    x_col = f"MP_prev_{WINDOW}"
    x = data[x_col].to_numpy()
    y = data["GmSc"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    fit_x = np.linspace(x.min(), x.max(), 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=7, alpha=0.11, color="#2f6f9f", linewidths=0)
    ax.plot(
        fit_x,
        intercept + slope * fit_x,
        color="#c0392b",
        linewidth=2.3,
        label=f"Least-squares line: y = {intercept:.2f} + {slope:.4f}x",
    )
    ax.set_xlabel(f"Minutes played in previous {WINDOW} games")
    ax.set_ylabel("Game Score in current game")
    ax.set_title("Recent NBA Workload vs. Current-Game Performance")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def save_scatterplot(data: pd.DataFrame) -> Path:
    fig = create_scatterplot(data)
    out = ROOT / "recent_workload_scatterplot.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def create_raw_data_table(data: pd.DataFrame) -> str:
    x_col = f"MP_prev_{WINDOW}"
    preferred_ids = [
        "jamesle01",
        "curryst01",
        "tatumja01",
        "gilgesh01",
        "doncilu01",
        "jokicni01",
        "bookede01",
        "edwaran01",
        "butleji01",
        "brunsja01",
        "halibty01",
        "duranke01",
    ]
    sample = (
        data[data["player_id"].isin(preferred_ids)]
        .sort_values(["season", "player_id", "Date"])
        .groupby("player_id", as_index=False)
        .head(1)
        .head(RAW_TABLE_ROWS)
        .copy()
    )
    if len(sample) < RAW_TABLE_ROWS:
        sample = data.sort_values(["season", "player_id", "Date"]).head(RAW_TABLE_ROWS).copy()

    sample["Date"] = sample["Date"].dt.strftime("%Y-%m-%d")
    sample["MP"] = sample["MP"].map(lambda v: f"{v:.1f}")
    sample[x_col] = sample[x_col].map(lambda v: f"{v:.1f}")
    sample["GmSc"] = sample["GmSc"].map(lambda v: f"{v:.1f}")

    columns = [
        ("name", "Player"),
        ("season", "Season"),
        ("Date", "Date"),
        ("MP", "MP"),
        (x_col, "Prev. 5 MP"),
        ("GmSc", "Game Score"),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sample of raw player-game records used to build the workload data set.}",
        r"\label{tab:raw-data-sample}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        " & ".join(label for _, label in columns) + r" \\",
        r"\hline",
    ]
    for _, row in sample.iterrows():
        values = [latex_escape(row[col]) for col, _ in columns]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def save_raw_data_table(data: pd.DataFrame) -> Path:
    out = ROOT / "raw_data_table.tex"
    out.write_text(create_raw_data_table(data), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the LaTeX raw-data table and display the workload scatterplot."
    )
    parser.add_argument(
        "--save-files",
        action="store_true",
        help="Also write recent_workload_scatterplot.png and raw_data_table.tex to this folder.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open/display the matplotlib figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_game_logs()
    data = add_recent_workload(raw)
    table = create_raw_data_table(data)
    fig = create_scatterplot(data)

    print(f"Loaded {len(raw):,} raw player-game rows.")
    print(f"Usable rows after previous-{WINDOW}-game workload calculation: {len(data):,}.")
    print("\nLaTeX table:\n")
    print(table)

    if args.save_files:
        plot_path = ROOT / "recent_workload_scatterplot.png"
        table_path = ROOT / "raw_data_table.tex"
        fig.savefig(plot_path, dpi=140)
        table_path.write_text(table, encoding="utf-8")
        print(f"Scatterplot saved to {plot_path.name}.")
        print(f"LaTeX raw-data table saved to {table_path.name}.")

    if args.no_show:
        plt.close(fig)
    else:
        print("Displaying scatterplot. Close the plot window to finish the script.")
        plt.show()


if __name__ == "__main__":
    main()
