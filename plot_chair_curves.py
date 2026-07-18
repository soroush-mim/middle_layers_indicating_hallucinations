"""Score every caption log with CHAIR and plot hallucination vs verbosity.

For each ``*.jsonl`` in the log dir this runs the CHAIR evaluator, then:
  * prints a table (incl. Len) and writes chair_summary.csv
  * plots C_S vs Recall and C_I vs Recall, with each foreground variant drawn
    as a connected curve (points ordered by alpha) and Greedy / Head-guide as
    reference markers.

The point of plotting against Recall (a verbosity proxy) is to compare methods
at *matched verbosity*: a method only genuinely reduces hallucination if it sits
below another curve at the same Recall, rather than by simply describing less.

Uses the cached ``chair.pkl`` evaluator if present (fast); otherwise builds one
from --coco_path.

Example:
    python plot_chair_curves.py --log-dir ./log/llava-1.5 --coco_path "$ANNODIR"
"""

import argparse
import csv
import glob
import os
import pickle
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chair import CHAIR


def parse_variant(fname):
    """Map a log filename to (family, base, alpha) for grouping/labelling."""
    base = float(re.search(r"_base([0-9.]+)", fname).group(1)) if "_base" in fname else 0.0
    m_alpha = re.search(r"_alpha([0-9.]+)", fname)
    alpha = float(m_alpha.group(1)) if m_alpha else None
    binary = "_binary" in fname
    covnorm = "_covnorm" in fname

    if "head_guided" in fname:
        return "Head-guide (paper)", None, alpha
    if "foreground_guided" not in fname:
        return "Greedy (baseline)", None, None

    if base > 0:
        family = f"Enrichment+foreground (base={base})"
    else:
        tags = []
        if binary:
            tags.append("binary")
        if covnorm:
            tags.append("covnorm")
        family = "Foreground" + ("+" + "+".join(tags) if tags else " (fractional)")
    return family, base, alpha


def load_evaluator(cache, coco_path):
    if cache and os.path.exists(cache):
        print(f"loaded evaluator from cache: {cache}")
        return pickle.load(open(cache, "rb"))
    if not coco_path:
        raise SystemExit(f"No cache at {cache!r}; pass --coco_path to build one.")
    print("building CHAIR evaluator from scratch...")
    evaluator = CHAIR(coco_path)
    if cache:
        pickle.dump(evaluator, open(cache, "wb"))
    return evaluator


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", default="./log/llava-1.5")
    ap.add_argument("--pattern", default="*.jsonl")
    ap.add_argument("--cache", default="chair.pkl")
    ap.add_argument("--coco_path", default=None, help="only needed if no --cache exists")
    ap.add_argument("--out-dir", default="./chair_plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    evaluator = load_evaluator(args.cache, args.coco_path)

    files = sorted(glob.glob(os.path.join(args.log_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No files matching {args.pattern} in {args.log_dir}")

    rows = []
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            out = evaluator.compute_chair(fpath, "image_id", "caption")
        except Exception as exc:  # noqa: BLE001 - skip malformed logs, keep going
            print(f"skip {fname}: {exc}")
            continue
        m = {k: v * 100 for k, v in out["overall_metrics"].items()}  # match print_metrics
        family, base, alpha = parse_variant(fname)
        rows.append(
            {
                "file": fname,
                "family": family,
                "base": base,
                "alpha": alpha,
                "C_S": m["CHAIRs"],
                "C_I": m["CHAIRi"],
                "Recall": m["Recall"],
                "Prec": m["Precision"],
                "F1": m["F1"],
                "Len": m["Len"],
            }
        )

    # ---- table + csv ---------------------------------------------------------
    rows.sort(key=lambda r: (r["family"], r["alpha"] if r["alpha"] is not None else -1))
    hdr = f"{'family':32} {'base':>4} {'alpha':>5} {'C_S':>6} {'C_I':>6} {'Rec':>6} {'Prec':>6} {'F1':>6} {'Len':>6}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        base = "-" if r["base"] is None else f"{r['base']:g}"
        alpha = "-" if r["alpha"] is None else f"{r['alpha']:g}"
        print(
            f"{r['family']:32} {base:>4} {alpha:>5} {r['C_S']:>6.1f} {r['C_I']:>6.1f} "
            f"{r['Recall']:>6.1f} {r['Prec']:>6.1f} {r['F1']:>6.1f} {r['Len']:>6.1f}"
        )

    csv_path = os.path.join(args.out_dir, "chair_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}")

    # ---- plots ---------------------------------------------------------------
    families = [f for f in dict.fromkeys(r["family"] for r in rows)]
    ref = {"Greedy (baseline)", "Head-guide (paper)"}
    cmap = plt.get_cmap("tab10")
    curve_families = [f for f in families if f not in ref]
    colors = {f: cmap(i % 10) for i, f in enumerate(curve_families)}

    ylabels = {
        "C_S": "C_S (sentence hallucination) $\\downarrow$",
        "C_I": "C_I (instance hallucination) $\\downarrow$",
    }
    # Each x-axis is a verbosity measure: read hallucination off at matched x.
    xaxes = {
        "Recall": "Recall (verbosity proxy) $\\uparrow$",
        "Len": "Mean caption length (verbosity) $\\uparrow$",
    }

    for xkey, xlabel in xaxes.items():
        for metric in ("C_S", "C_I"):
            fig, ax = plt.subplots(figsize=(8, 6))
            for fam in curve_families:
                pts = sorted(
                    (r for r in rows if r["family"] == fam),
                    key=lambda r: r["alpha"] if r["alpha"] is not None else 0,
                )
                ax.plot([p[xkey] for p in pts], [p[metric] for p in pts], "-o",
                        color=colors[fam], label=fam)
                for p in pts:
                    if p["alpha"] is not None:
                        ax.annotate(f"{p['alpha']:g}", (p[xkey], p[metric]),
                                    textcoords="offset points", xytext=(4, 4),
                                    fontsize=7, color=colors[fam])
            # reference points + guide lines
            for r in rows:
                if r["family"] == "Greedy (baseline)":
                    ax.scatter(r[xkey], r[metric], marker="s", s=90, color="black",
                               zorder=5, label="Greedy (baseline)")
                if r["family"] == "Head-guide (paper)":
                    ax.scatter(r[xkey], r[metric], marker="*", s=220, color="red",
                               zorder=5, label="Head-guide (paper)")
                    ax.axhline(r[metric], ls="--", lw=0.8, color="red", alpha=0.5)
                    ax.axvline(r[xkey], ls="--", lw=0.8, color="red", alpha=0.5)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabels[metric])
            ax.set_title(f"{metric} vs {xkey} — lower-left of the Head-guide lines wins\n(point labels = alpha)")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            out = os.path.join(args.out_dir, f"{metric.lower()}_vs_{xkey.lower()}.png")
            fig.tight_layout()
            fig.savefig(out, dpi=130)
            plt.close(fig)
            print(f"saved {out}")


if __name__ == "__main__":
    main()
