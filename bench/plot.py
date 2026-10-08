#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Draw the README's and HISTORY's figures from the published raw records.

    uv run bench/plot.py            # writes docs/img/*.svg

The README figures read `bench/results/<date>-<round>/`, so no figure can carry a number that is not in this
repository, and each prints what it drew so the values can be checked against the round's write-up. The history
figure takes one value per served configuration from the write-ups named in HISTORY below. Style and palette follow
the Qwen3.8-Flash-Next repository's bench/plot.py.
"""

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
OUT = ROOT / "docs" / "img"
SERVED = RESULTS / "2026-10-07-r882b-glm53-swap-agent" / "XA"  # the configuration README.md describes

DECODE, AGG, PREFILL = "#0969da", "#cf222e", "#8250df"
plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.edgecolor": "#d8dee4",
    "axes.labelcolor": "#57606a",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.color": "#57606a",
    "ytick.color": "#57606a",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
})

# (round, c1 score tok/s or None, sum over 4 streams tok/s or None, write-up in bench/results/)
HISTORY = [
    ("R858", None, 68.6, "r858-glm53-audition.md"),      # MTP depth 1, 104 experts per layer on the CPU, dynamic
    ("R860", 46.2, None, "r860-glm53-chain.md"),         # MTP off, 104 on the CPU, dynamic
    ("R864", 48.4, None, "r864-glm53-levers.md"),        # 96 on the CPU, dynamic
    ("R869", 57.5, None, "r869-glm53-hotset.md"),        # static placement from broad counts
    ("R873", None, 86.3, "r873-glm53-c4.md"),            # static broad placement with the agent overlay
    ("R882b", 59.9, 109.9, "r882b-glm53-swap-agent.md"),  # exchange swaps with the agent overlay (served)
]


def save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, format="svg", bbox_inches="tight", metadata={"Title": caption})
    plt.close(fig)


def annotate(ax, xs, ys, color, fmt="{:.1f}", dy=7):
    for x, y in zip(xs, ys):
        ax.annotate(fmt.format(y), (x, y), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=8.5, color=color)


def style(ax):
    ax.grid(axis="y", color="#eaeef2")
    ax.set_axisbelow(True)


def served():
    """c1 by kind (median over runs), concurrency summaries (median over rounds) and cold prefill points."""
    kinds = {}
    for line in open(SERVED / "c1.jsonl"):
        r = json.loads(line)
        kinds.setdefault(r["kind"], []).append(r["tps"])
    kinds = {k: st.median(v) for k, v in kinds.items()}
    conc, prefill = {}, []
    for line in open(SERVED / "measure.jsonl"):
        r = json.loads(line)
        u = r.get("usage") or {}
        if str(r.get("tag", "")).startswith("prefill-") and (u.get("prompt_tokens_details") or {}).get("cached_tokens") == 0:
            prefill.append((u["prompt_tokens"], r.get("engine_prefill_tps") or u["prompt_tokens"] / u["prompt_time"]))
        if r.get("phase") == "decode-summary":
            for s in r["summaries"]:
                conc[s["c"]] = (s["ss_per_stream_tps_median"], s["ss_agg_tps_median"])
    return kinds, conc, sorted(prefill)


def figure_decode_concurrency(conc):
    # Two panels rather than a twin axis, as in the Flash-Next repository: the sum rises while the per-stream rate falls.
    xs = sorted(conc)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for a, idx, color, title, ylabel in (
            (ax, 1, AGG, "Decode, sum over the streams", "decode tokens per second, sum over streams"),
            (ax2, 0, DECODE, "Decode rate per stream", "decode tokens per second, one stream (median)")):
        ys = [conc[c][idx] for c in xs]
        a.plot(xs, ys, marker="o", markersize=5, color=color, linewidth=2)
        annotate(a, xs, ys, color)
        a.set_title(title)
        a.set_xlabel("concurrent streams")
        a.set_ylabel(ylabel)
        a.set_ylim(0, max(ys) * 1.25)
        a.set_xticks(xs)
        style(a)
    fig.suptitle("Decode rate after the first token against concurrency, served configuration (R882b)", fontsize=11,
                 fontweight="bold")
    print("decode by concurrency (R882b, median of 3 rounds):", {c: tuple(round(v, 1) for v in conc[c]) for c in xs})
    save(fig, "decode-concurrency.svg", "Decode rate after the first token against concurrency, sum over streams and per stream")


def figure_c1_by_kind(kinds):
    order = ["code", "prose", "chat", "html", "edit"]
    vals = [kinds[k] for k in order]
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.barh(order[::-1], vals[::-1], color=DECODE, height=0.55)
    for y, v in enumerate(vals[::-1]):
        ax.annotate(f"{v:.1f}", (v, y), textcoords="offset points", xytext=(5, -3), fontsize=8.5, color=DECODE)
    ax.set_title("Single-stream decode by content kind, served configuration (R882b)")
    ax.set_xlabel("decode tokens per second, median of 2 runs")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(axis="x", color="#eaeef2")
    ax.set_axisbelow(True)
    print("c1 by kind (R882b):", {k: round(kinds[k], 1) for k in order}, "mean", round(st.mean(vals), 1))
    save(fig, "c1-by-kind.svg", "Single-stream decode by content kind")


def figure_prefill(points):
    # The x axis spans the 262,144-token window, so longer prompts measured later extend the same figure.
    toks, rate = [t for t, _ in points], [v for _, v in points]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.plot(toks, rate, marker="o", color=PREFILL, linewidth=2)
    annotate(ax, toks, rate, PREFILL, fmt="{:,.0f}", dy=-16)
    ax.set_title("Cold prefill rate against prompt length, served configuration (R882b)")
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("prompt tokens per second, prefill")
    ax.set_xlim(0, 262144)
    ax.set_xticks(range(0, 262145, 65536), ["0"] + [f"{t // 1024}k" for t in range(65536, 262145, 65536)])
    ax.set_ylim(0, max(rate) * 1.25)
    style(ax)
    print("prefill (R882b, engine-timed, cold):", [(t, round(v)) for t, v in points])
    save(fig, "prefill.svg", "Cold prefill rate against prompt length")


def figure_history():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for a, idx, color, title in ((ax, 1, DECODE, "One stream, c1 score"), (ax2, 2, AGG, "Four streams, sum of the stream rates")):
        rows = [h for h in HISTORY if h[idx] is not None]
        names, vals = [h[0] for h in rows], [h[idx] for h in rows]
        a.bar(names, vals, color=color, width=0.5)
        for i, v in enumerate(vals):
            a.annotate(f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5, color=color)
        a.set_title(title)
        a.set_ylabel("decode tokens per second")
        a.set_ylim(0, max(vals) * 1.2)
        style(a)
    fig.suptitle("Served configurations, 2026-10-06 to 2026-10-07 (docs/HISTORY.md)", fontsize=11, fontweight="bold")
    print("history:", [(h[0], h[1], h[2]) for h in HISTORY])
    save(fig, "history.svg", "Served configurations over time, one stream and four streams")


if __name__ == "__main__":
    kinds, conc, prefill = served()
    figure_decode_concurrency(conc)
    figure_c1_by_kind(kinds)
    figure_prefill(prefill)
    figure_history()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.svg"))))
