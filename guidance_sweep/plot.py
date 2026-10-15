"""Standalone matplotlib figures from audited tables; no model imports."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def nondominated(rows):
    return sorted([r for r in rows if not any(q["gen_ppl"] <= r["gen_ppl"] and q["entropy"] >= r["entropy"] and
        (q["gen_ppl"] < r["gen_ppl"] or q["entropy"] > r["entropy"]) for q in rows)], key=lambda r: r["entropy"])


def separate_labels(fig, labels):
    # Move only colliding labels; connector lines keep their sampled coordinates clear.
    for _ in range(12):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        changed = False
        boxes = []
        for label in labels:
            box = label.get_window_extent(renderer).expanded(1.08, 1.1)
            if any(box.overlaps(other) for other in boxes):
                x, y = label.get_position()
                label.set_position((x + 3, y + 7))
                changed = True
            boxes.append(box)
        if not changed:
            break


def main():
    directory = Path("guidance_sweep/results")
    rows = json.loads((directory / "per_seed.json").read_text())
    aggregates = json.loads((directory / "aggregate.json").read_text())
    colors = {"C": "#2265a8", "D": "#bd4437", "A_measured_local": "#27805c"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    for sampler, title in [("primary", "32 SDE steps · gamma 1.5"), ("secondary", "64 SDE steps · gamma 1.0")]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey="row", gridspec_kw={"height_ratios": [3, 1.15]})
        labels = []
        for column, arm in enumerate(("C", "D")):
            ax, diag = axes[0, column], axes[1, column]
            screening = [r for r in rows if r["arm"] == arm and r["sampler"] == sampler and r["stage"] != "confirmation"]
            confirmation = [r for r in aggregates if r["arm"] in (arm, "A_measured_local") and r["sampler"] == sampler and r["stage"] == "confirmation"]
            valid = [r for r in screening if r["gen_ppl"] is not None and r["entropy"] is not None]
            front = nondominated(valid)
            ax.plot([r["entropy"] for r in front], [r["gen_ppl"] for r in front], color=colors[arm], alpha=.3, lw=1, linestyle="--")
            for index, row in enumerate(sorted(valid, key=lambda r: r["sc_cfg"])):
                marker = "^" if row["outside_trained_range"] else "o"
                ax.scatter(row["entropy"], row["gen_ppl"], marker=marker, facecolors="none", edgecolors=colors[arm], s=42, zorder=3)
                labels.append(ax.annotate(f"{row['sc_cfg']:g}", (row["entropy"], row["gen_ppl"]),
                    xytext=(5, 5 if index % 2 else -11), textcoords="offset points", fontsize=8, color=colors[arm],
                    arrowprops={"arrowstyle": "-", "color": colors[arm], "alpha": .4, "lw": .5}))
                if row["repeated_4gram_fraction"] is not None:
                    diag.scatter(row["entropy"], row["repeated_4gram_fraction"], marker=marker, facecolors="none", edgecolors=colors[arm], s=35)
            confirmation_front = []
            for row in confirmation:
                x, y = row["entropy"]["mean"], row["gen_ppl"]["mean"]
                if x is None or y is None:
                    continue
                name, scale = row["arm"], row["sc_cfg"]
                color = colors[name]
                marker = "s" if name == "A_measured_local" else ("^" if row["outside_trained_range"] else "D")
                seeds = [r for r in rows if r["stage"] == "confirmation" and r["arm"] == name and r["sampler"] == sampler and r["sc_cfg"] == scale]
                ax.scatter([r["entropy"] for r in seeds], [r["gen_ppl"] for r in seeds], s=13, color=color, alpha=.45, zorder=2)
                ax.errorbar(x, y, xerr=row["entropy"]["sample_sd"], yerr=row["gen_ppl"]["sample_sd"], fmt=marker,
                    markersize=7, color=color, capsize=3, lw=1, zorder=5)
                label = "A local · 3" if name == "A_measured_local" else f"{scale:g} confirm"
                labels.append(ax.annotate(label, (x, y), xytext=(7, 11), textcoords="offset points", color=color, weight="bold", fontsize=9,
                    arrowprops={"arrowstyle": "-", "color": color, "alpha": .5, "lw": .6}))
                repetition = row["repeated_4gram_fraction"]["mean"]
                if repetition is not None:
                    diag.errorbar(x, repetition, xerr=row["entropy"]["sample_sd"], yerr=row["repeated_4gram_fraction"]["sample_sd"],
                        fmt=marker, markersize=6, color=color, capsize=3, lw=1)
                if name == arm:
                    confirmation_front.append({"entropy": x, "gen_ppl": y})
            front = nondominated(confirmation_front)
            ax.plot([r["entropy"] for r in front], [r["gen_ppl"] for r in front], color=colors[arm], alpha=.5, lw=1)
            if sampler == "primary":
                ax.scatter(5.15, 24.08, marker="*", s=170, color="#292929", zorder=6)
                labels.append(ax.annotate("A published external", (5.15, 24.08), xytext=(-120, 15), textcoords="offset points", fontsize=9,
                    arrowprops={"arrowstyle": "-", "color": "#292929", "lw": .6}))
            else:
                ax.text(.02, .98, "Published A (24.08, 5.15) uses 32 steps;\nno external 64-step point is available.", transform=ax.transAxes, va="top", fontsize=9,
                        bbox={"boxstyle": "round,pad=.4", "fc": "white", "ec": "#dddddd", "alpha": .9})
            ax.set_title(f"Arm {arm}", color=colors[arm], weight="bold")
            ax.set_yscale("log")
            for panel in (ax, diag):
                panel.axvspan(5.10, 5.20, color="#d3c18d", alpha=.15, zorder=0)
                panel.grid(alpha=.18)
                panel.margins(x=.12, y=.18)
            missing = len(screening) - len(valid)
            if missing:
                ax.text(.02, .03, f"{missing} nonfinite screening points retained in tables", transform=ax.transAxes, fontsize=8)
            diag.set_ylim(-.03, 1.03)
            diag.set_xlabel("Mean sample unigram entropy (post-cap GPT-2 tokens)")
        axes[0, 0].set_ylabel("GenPPL (log scale; lower is better)")
        axes[1, 0].set_ylabel("Repeated 4-gram fraction")
        legend = [Line2D([], [], marker="o", color="gray", markerfacecolor="none", linestyle="none", label="Screening, seed 10 / n=250"),
                  Line2D([], [], marker="D", color="gray", linestyle="none", label="Confirmation mean ± sample SD (3 seeds)"),
                  Line2D([], [], marker="^", color="gray", markerfacecolor="none", linestyle="none", label="Outside trained SC range [.5, 5]"),
                  Line2D([], [], marker="s", color=colors["A_measured_local"], linestyle="none", label="Local A anchor, SC=3")]
        fig.suptitle(f"ELF C/D guidance study · {title}", fontsize=16)
        fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .045))
        fig.text(.5, .015, "Labels are SC-CFG scales. All measured points retained. Frontier segments are visual summaries; interpolation was not sampled.\n"
                 "Shading: predefined entropy neighborhood [5.10, 5.20]. Error bars reflect inference variation only; one training seed per model.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .12, 1, .95))
        separate_labels(fig, labels)
        for extension in ("png", "svg", "pdf"):
            fig.savefig(directory / f"{sampler}.{extension}", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
