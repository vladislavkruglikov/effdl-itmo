#!/usr/bin/env python3
"""Графики из исходных CSV; запуск из текущей рабочей папки отчёта."""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/mistral_nemo_h100_5rep.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    args = parser.parse_args()
    groups = defaultdict(list)
    sources = set()
    with args.input.open() as source:
        for row in csv.DictReader(source):
            sources.add(row["test_data"])
            groups[row["kind"], row["method"]].append(
                (float(row["runtime_ms"]), float(row["op_mem_mb"]))
            )

    if len(sources) != 1:
        raise ValueError("В CSV должна быть ровно одна модель")
    model_key = sources.pop()
    model_title = {"mistral-nemo": "Mistral NeMo", "phi3.5": "Phi 3.5 Mini"}[model_key]
    file_prefix = {"mistral-nemo": "mistral_nemo", "phi3.5": "phi35"}[model_key]

    methods = ["cce", "torch_compile", "baseline"]
    names = ["CCE", "torch.compile", "PyTorch baseline"]
    colors = ["#2563eb", "#0d9488", "#64748b"]
    kinds = ["loss-fw", "loss-bw", "loss-fw-bw"]
    titles = ["Forward", "Backward", "Forward + backward"]
    for kind in kinds:
        for method in methods:
            if len(groups[kind, method]) != 5:
                raise ValueError(f"Ожидалось 5 повторов: {kind}, {method}")

    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.edgecolor": "#cbd5e1",
        "text.color": "#0f172a", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#334155",
    })

    for metric, file_name in [(0, "runtime"), (1, "memory")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
        max_runtime = max(statistics.mean(v[0] for v in values) + statistics.stdev(v[0] for v in values) for values in groups.values())
        time_limit = max_runtime * 1.3
        for ax, kind, title in zip(axes, kinds, titles):
            values = [[v[metric] for v in groups[kind, m]] for m in methods]
            means = [statistics.mean(v) for v in values]
            ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
            ax.set_yticks(range(3), names)
            ax.set_ylim(2.55, -0.55)
            ax.tick_params(axis="y", length=0)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
            if metric == 0:
                sd = [statistics.stdev(v) for v in values]
                ax.barh(range(3), means, color=colors, height=0.48,
                        xerr=sd, capsize=4, error_kw={"ecolor": "#0f172a"})
                ax.set_xlim(0, time_limit)
                ax.set_xlabel("Время, мс (среднее ± SD)")
                for i, (value, deviation) in enumerate(zip(means, sd)):
                    ax.text(value + deviation + time_limit * 0.018, i, f"{value:.2f}", va="center", fontsize=10)
            else:
                ax.set_xscale("log")
                ax.set_xlim(0.1, 70000)
                ax.set_xticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "10³", "10⁴"])
                ax.minorticks_off()
                ax.set_xlabel("Память, MiB (лог. шкала)")
                for i, (value, color) in enumerate(zip(means, colors)):
                    ax.scatter(value, i, s=100, color=color, zorder=3)
                    label = f"{value:,.2f}".replace(",", " ")
                    ax.annotate(label, (value, i), xytext=(0, 13), textcoords="offset points",
                                va="bottom", ha="center", fontsize=10)
        fig.suptitle(model_title + " на H100: " + ("время выполнения" if metric == 0 else "дополнительная память операции"),
                     fontsize=19, fontweight="bold", x=0.04, ha="left", y=0.98)
        fig.text(0.04, 0.88, "8 192 токена, BF16; 5 повторов; 50 итераций на метод и режим; меньше — лучше",
                 fontsize=11, color="#475569")
        footnote = ("Горизонтальные отрезки: выборочное SD между пятью повторами."
                    if metric == 0 else
                    "Точки: среднее пяти повторов. Шкала логарифмическая; показана память операции, а не всей модели.")
        fig.text(0.04, 0.045, footnote, fontsize=10, color="#475569")
        fig.subplots_adjust(left=0.115, right=0.98, top=0.73, bottom=0.2, wspace=0.7)
        fig.savefig(args.output / f"{file_prefix}_{file_name}.png", dpi=180, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
