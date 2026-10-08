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
SERVED = RESULTS / "2026-10-08-r914-glm53-promote-combo" / "CF"  # the configuration README.md describes (R914 arm CF)
MEMORY = RESULTS / "2026-10-08-r900-glm53-memory-layout" / "memory.json"  # memory layout of the served configuration (R900)
PREFILL_SRC = RESULTS / "2026-10-07-r882b-glm53-swap-agent" / "XA"  # last cold-prefill measurement (previous configuration)

DECODE, AGG, PREFILL, AGG_ONE = "#0969da", "#cf222e", "#8250df", "#f0a8ad"
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

# (round, c1 score tok/s or None, sum over 4 streams tok/s or None, c4 method, write-up in bench/results/)
# c4 method "one": every stream the same prompt (R858 to R883); "distinct": code/prose/chat/html per stream (R911 on).
# The two are not comparable: R899 measured 104.8 (one prompt) against 95.7 (distinct) on the same configuration.
HISTORY = [
    ("R858", None, 68.6, "one", "r858-glm53-audition.md"),      # MTP depth 1, 104 experts per layer on the CPU, dynamic
    ("R860", 46.2, None, None, "r860-glm53-chain.md"),          # MTP off, 104 on the CPU, dynamic
    ("R864", 48.4, None, None, "r864-glm53-levers.md"),         # 96 on the CPU, dynamic
    ("R869", 57.5, None, None, "r869-glm53-hotset.md"),         # static placement from broad counts
    ("R873", None, 86.3, "one", "r873-glm53-c4.md"),            # static broad placement with the agent overlay
    ("R882b", 59.9, 109.9, "one", "r882b-glm53-swap-agent.md"),  # exchange swaps with the agent overlay
    ("R911 re-run", 58.0, 97.4, "distinct", "r911-glm53-mtpcap-dynamic.md"),  # the R882b configuration again, mean of D0 and D1
    ("R914", 64.2, 87.7, "distinct", "r914-glm53-promote-combo.md"),   # MTP depth 1 at c1 only, 104 on the CPU (served)
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
    for line in open(SERVED / "dec.jsonl"):
        r = json.loads(line)
        if r.get("phase") == "decode-summary":
            for s in r["summaries"]:
                assert s.get("distinct"), "the README concurrency figure uses distinct prompts per stream"
                conc[s["c"]] = (s["ss_per_stream_tps_median"], s["ss_agg_tps_median"])
    for line in open(PREFILL_SRC / "measure.jsonl"):
        r = json.loads(line)
        u = r.get("usage") or {}
        if str(r.get("tag", "")).startswith("prefill-") and (u.get("prompt_tokens_details") or {}).get("cached_tokens") == 0:
            prefill.append((u["prompt_tokens"], r.get("engine_prefill_tps") or u["prompt_tokens"] / u["prompt_time"]))
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
    fig.suptitle("Decode rate after the first token against concurrency, distinct prompts per stream, served configuration (R914)",
                 fontsize=11, fontweight="bold")
    print("decode by concurrency (R914 CF, distinct prompts, median of 2 rounds):", {c: tuple(round(v, 1) for v in conc[c]) for c in xs})
    save(fig, "decode-concurrency.svg", "Decode rate after the first token against concurrency, sum over streams and per stream")


def figure_c1_by_kind(kinds):
    order = ["code", "prose", "chat", "html", "edit"]
    vals = [kinds[k] for k in order]
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.barh(order[::-1], vals[::-1], color=DECODE, height=0.55)
    for y, v in enumerate(vals[::-1]):
        ax.annotate(f"{v:.1f}", (v, y), textcoords="offset points", xytext=(5, -3), fontsize=8.5, color=DECODE)
    ax.set_title("Single-stream decode by content kind, MTP depth 1, served configuration (R914)")
    ax.set_xlabel("decode tokens per second, median of 2 runs")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(axis="x", color="#eaeef2")
    ax.set_axisbelow(True)
    print("c1 by kind (R914 CF):", {k: round(kinds[k], 1) for k in order}, "mean", round(st.mean(vals), 1))
    save(fig, "c1-by-kind.svg", "Single-stream decode by content kind")


def figure_prefill(points):
    # The x axis spans the 262,144-token window, so longer prompts measured later extend the same figure.
    toks, rate = [t for t, _ in points], [v for _, v in points]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.plot(toks, rate, marker="o", color=PREFILL, linewidth=2)
    annotate(ax, toks, rate, PREFILL, fmt="{:,.0f}", dy=-16)
    ax.set_title("Cold prefill rate against prompt length, previous configuration (R882b, 2026-10-07)")
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("prompt tokens per second, prefill")
    ax.set_xlim(0, 262144)
    ax.set_xticks(range(0, 262145, 65536), ["0"] + [f"{t // 1024}k" for t in range(65536, 262145, 65536)])
    ax.set_ylim(0, max(rate) * 1.25)
    style(ax)
    print("prefill (R882b, engine-timed, cold):", [(t, round(v)) for t, v in points])
    save(fig, "prefill.svg", "Cold prefill rate against prompt length")


def figure_history():
    from matplotlib.patches import Patch
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for a, idx, title in ((ax, 1, "One stream, c1 score"), (ax2, 2, "Four streams, sum of the stream rates")):
        rows = [h for h in HISTORY if h[idx] is not None]
        names, vals = [h[0] for h in rows], [h[idx] for h in rows]
        colors = [DECODE if idx == 1 else (AGG if h[3] == "distinct" else AGG_ONE) for h in rows]
        a.bar(names, vals, color=colors, width=0.5)
        for i, v in enumerate(vals):
            a.annotate(f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5,
                       color=colors[i] if idx == 1 else AGG)
        a.set_title(title)
        a.set_ylabel("decode tokens per second")
        a.set_ylim(0, max(vals) * (1.2 if idx == 1 else 1.45))  # room for the legend above the bars
        style(a)
    ax2.legend(handles=[Patch(color=AGG_ONE, label="same prompt in every stream"),
                        Patch(color=AGG, label="a different prompt per stream")], fontsize=8, loc="upper left", frameon=False)
    fig.suptitle("Served configurations, 2026-10-06 to 2026-10-08 (docs/HISTORY.md)", fontsize=11, fontweight="bold")
    print("history:", [(h[0], h[1], h[2], h[3]) for h in HISTORY])
    save(fig, "history.svg", "Served configurations over time, one stream and four streams")


# Memory categories in drawing order, one colour each (routed experts share one hue on both bars).
MEMORY_GROUPS = [
    ("routed experts", "#0969da", ("routed experts on GPU", "CPU experts")),
    ("other weights (attention, shared experts, lm_head, MTP layer)", "#8250df",
     ("attention, norms, other weights", "shared experts", "lm_head", "MTP layer")),
    ("KV pool", "#1a7f37", ("KV pool",)),
    ("embedding table", "#bf8700", ("embedding table",)),
    ("vision tower", "#cf222e", ("vision tower",)),
    ("recurrent-state cache", "#e16f24", ("recurrent-state cache",)),
    ("other (contexts, graph pools, scratch, runtime, staging)", "#afb8c1", ("other",)),
]


def figure_memory():
    rec = json.load(open(MEMORY))
    GiB = 2 ** 30
    bars = [("VRAM, 2 × RTX 5090", rec["gpu_categories_bytes"], sum(rec["gpu_total_bytes"].values())),
            ("host DRAM", rec["host_categories_bytes"], rec["host_total_bytes"])]
    fig, ax = plt.subplots(figsize=(10.4, 2.9))
    drawn = {}
    for y, (label, cats, cap) in enumerate(bars):
        left = 0.0
        for name, color, prefixes in MEMORY_GROUPS:
            v = sum(b for k, b in cats.items() if k.startswith(prefixes)) / GiB
            if v <= 0:
                continue
            ax.barh(y, v, left=left, color=color, height=0.55, edgecolor="white", linewidth=1.5)
            if v >= 2.5:
                ax.text(left + v / 2, y, f"{v:.1f}", ha="center", va="center", fontsize=8.5, color="white")
            drawn[name] = color
            left += v
        ax.plot([cap / GiB] * 2, [y - 0.36, y + 0.36], color="#24292f", linewidth=1.4)  # the device's capacity
        ax.text(max(left, cap / GiB) + 0.8, y, f"{left:.1f} of {cap / GiB:.1f} GiB", ha="left", va="center",
                fontsize=8.5, color="#24292f")
        print("memory:", label, {n: round(sum(b for k, b in cats.items() if k.startswith(p)) / GiB, 2)
                                 for n, _, p in MEMORY_GROUPS}, "capacity", round(cap / GiB, 2))
    ax.set_yticks(range(len(bars)), [b[0] for b in bars])
    ax.invert_yaxis()
    ax.set_xlabel("GiB")
    ax.set_xlim(0, max(b[2] for b in bars) / GiB * 1.18)
    ax.grid(axis="x", color="#eaeef2")
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=n) for n, c in drawn.items()], fontsize=8, ncol=3, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.set_title("Memory of the served configuration by category, GiB (R900; black tick = capacity)")
    save(fig, "memory.svg", "VRAM and host DRAM of the served configuration by category")


if __name__ == "__main__":
    kinds, conc, prefill = served()
    figure_decode_concurrency(conc)
    figure_c1_by_kind(kinds)
    figure_prefill(prefill)
    figure_history()
    figure_memory()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.svg"))))
