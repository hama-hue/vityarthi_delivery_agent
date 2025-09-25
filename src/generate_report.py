"""
src/generate_report.py

Auto-generates a PDF report for the Vityarthi Delivery Agent project.

Steps:
1. Load experiment results (CSV).
2. Insert results table into template.
3. Embed plots (from plot_results.py).
4. Export Markdown → PDF using pypandoc.

Usage (from project root):
    python src/generate_report.py --results results.csv --plots plots --out report.pdf

Requires:
    pip install pypandoc pandas
    (and ensure Pandoc is installed: https://pandoc.org/)
"""

import argparse
import os
import pandas as pd
import pypandoc


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to Markdown table."""
    return df.to_markdown(index=False)


def generate_report(results_file: str, plots_dir: str, out_file: str):
    # Load CSV
    df = pd.read_csv(results_file)

    # Create results table
    table_md = dataframe_to_markdown(df)

    # Build report content
    content = f"""
# Vityarthi Delivery Agent – Assignment Report

## 1. Introduction
This report presents the implementation and evaluation of search-based planners
(UCS, A*, Local Search) for a delivery agent navigating static and dynamic obstacles.

## 2. Algorithms
- **UCS**: guarantees optimal path, but slow.
- **A***: efficient using admissible heuristics.
- **Local Search**: hill climbing with random restarts, useful for dynamic replanning.

## 3. Experiments
We tested planners on the given maps using the simulation framework.

### 3.1 Results Table
{table_md}

### 3.2 Plots
Steps vs Planner:

![Steps vs Planner]({os.path.join(plots_dir, "steps_vs_planner.png")})

Collisions vs Planner:

![Collisions vs Planner]({os.path.join(plots_dir, "collisions_vs_planner.png")})

Planning Time vs Planner:

![Planning Time vs Planner]({os.path.join(plots_dir, "planning_time_vs_planner.png")})

## 4. Discussion
- **UCS** is correct but scales poorly.
- **A*** balances speed and accuracy.
- **Local Search** replans quickly and adapts to dynamic maps.

## 5. Conclusion
A* is best for static maps, while Local Search helps in dynamic environments.
Future work: multi-agent coordination, larger maps, advanced heuristics.
"""

    # Export to DOCX instead of PDF (easier, no LaTeX required)
    out_docx = out_file.replace(".pdf", ".docx")
    print(f"Exporting report to {out_docx} ...")
    pypandoc.convert_text(
        content,
        "docx",        # output format
        format="md",   # input format
        outputfile=out_docx,
        extra_args=["--standalone"],
    )
    print("Report saved:", out_docx)
    print("\n👉 Open the DOCX file in Word (or LibreOffice/Google Docs) and export as PDF.")



def main():
    parser = argparse.ArgumentParser(description="Auto-generate PDF report")
    parser.add_argument("--results", required=True, help="Path to results.csv")
    parser.add_argument("--plots", required=True, help="Directory with plot PNGs")
    parser.add_argument("--out", default="report.pdf", help="Output PDF file")
    args = parser.parse_args()

    generate_report(args.results, args.plots, args.out)


if __name__ == "__main__":
    main()
