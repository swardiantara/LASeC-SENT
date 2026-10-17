"""Step 11: paper figures (PDF + 300-dpi PNG) in results/paper/figures/.

Palette: validated categorical slots (light surface), assigned in fixed order; identity is never colour-only
(legends / direct labels; every figure has a matching table from make_tables.py).
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasec.cluster import ALGORITHM_LABEL, ALGORITHMS  # noqa: E402
from lasec.experiment import EVENT_TYPES, slug  # noqa: E402
from lasec.report import BASE_ORDER, FAMILY_LABEL, FAMILY_ORDER, SUBSET_LABEL, llm_display, load_all, method_table  # noqa: E402
from lasec.utils import load_config, path  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAMILY_COLOR = {f: SLOTS[i] for i, f in enumerate(["lasec", "pretrained", "string", "parser", "llm-zero", "llm-few"])}
INK, INK2, GRID, OTHER = "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"
ONE, TWO = 3.5, 7.2  # journal column widths (in)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.frameon": False, "lines.linewidth": 1.5, "lines.markersize": 5, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "figure.dpi": 150,
})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  figure {name}")


def fig_algorithm_selection(R, out, cfg):
    g = R["alg_grid"]
    if g is None:
        return print("  figure selection_algorithms: SKIPPED")
    fig, ax = plt.subplots(figsize=(ONE, 2.4))
    for k, alg in enumerate([a for a in ALGORITHMS if a in set(g.algorithm)]):
        x = g[g.algorithm == alg]
        ax.plot(x.threshold, x.ga, color=SLOTS[k], label=ALGORITHM_LABEL[alg], marker="o", markersize=3)
    ax.set_xlabel("Distance threshold (cosine; BIRCH: radius)")
    ax.set_ylabel("Group accuracy (catalogue)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.set_title(f"Clustering algorithms, pre-trained {slug(cfg['selection']['embedding'])}", fontsize=8, loc="left", color=INK)
    save(fig, out, "selection_algorithms")


def fig_embedding_selection(R, out, cfg):
    g, s = R["emb_grid"], R["emb_summary"]
    if g is None:
        return print("  figure selection_embeddings: SKIPPED")
    embs = [b for b in BASE_ORDER if b in set(g.embedding)] + [b for b in g.embedding.unique() if b not in BASE_ORDER]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TWO, 2.4), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.6})
    for k, e in enumerate(embs):
        x = g[g.embedding == e]
        ax.plot(x.threshold, x.ga, color=SLOTS[k], label=e, marker="o", markersize=3)
    ax.set_xlabel("Distance threshold (cosine)")
    ax.set_ylabel("Group accuracy (catalogue)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=6.5)
    ax.set_title(f"Pre-trained encoders, {ALGORITHM_LABEL[cfg['clustering']['algorithm']]}", fontsize=8, loc="left", color=INK)
    s = s.set_index("embedding").loc[embs]
    y = np.arange(len(embs))[::-1]
    for k, (yy, e) in enumerate(zip(y, embs)):
        bx.barh(yy, s.loc[e, "sentences_per_s"], color=SLOTS[k], height=0.4)
        bx.annotate(f"{s.loc[e, 'sentences_per_s']:,.0f}", (s.loc[e, "sentences_per_s"], yy), xytext=(3, -2.5),
                    textcoords="offset points", fontsize=6.5, color=INK2)
    bx.set_yticks(y, embs)
    bx.set_axisbelow(True)
    bx.set_xlabel("Encoding throughput (sentences/s)")
    bx.grid(axis="y", visible=False)
    bx.set_xlim(0, s.sentences_per_s.max() * 1.25)
    save(fig, out, "selection_embeddings")


def fig_cv_threshold(R, out, cfg):
    b = slug(cfg["embedding"]["base_model"])
    s = R["cv"].get(b)
    if not s:
        return print("  figure cv_threshold: SKIPPED (no CV results)")
    fc = s["fold_curves"]
    if "algorithm" in fc:
        fc = fc[fc.algorithm == s.get("algorithm", cfg["clustering"]["algorithm"])]
    fig, ax = plt.subplots(figsize=(ONE, 2.3))
    for enc, fam, lab, off in (("lasec", "lasec", "LASeC (fine-tuned)", (-30, 8)),
                               ("pretrained", "pretrained", "pre-trained", (6, -12))):
        g = fc[fc.encoder == enc].groupby("threshold").ga
        mu, sd = g.mean(), g.std().fillna(0)
        ax.fill_between(mu.index, mu - sd, mu + sd, color=FAMILY_COLOR[fam], alpha=0.15, linewidth=0)
        t = s[enc]["selected_threshold"]
        ax.plot(mu.index, mu.values, color=FAMILY_COLOR[fam], label=f"{lab} (selected t = {t:.2f})")
        ax.plot([t], [mu.loc[t]], "o", color=FAMILY_COLOR[fam], markeredgecolor="white", markeredgewidth=1)
    ax.set_xlabel("Distance threshold (cosine)")
    ax.set_ylabel("Group accuracy (CV mean ± sd)")
    ax.set_ylim(0, 1)
    ax.legend(loc="center right")
    ax.set_title(f"Threshold selection on the catalogue ({b})", fontsize=8, color=INK, loc="left")
    save(fig, out, "cv_threshold")


def fig_fpv_main(R, out, cfg):
    m = R["metrics"]
    if m is None:
        return print("  figure fpv_main: SKIPPED")
    subs = [s for s in ("all", "events") if s in set(m.subset)]
    tabs = {s: method_table(m, s, cfg) for s in subs}
    order = list(tabs[subs[0]].display)[::-1]
    fig, axes = plt.subplots(1, len(subs), figsize=(TWO, 0.22 * len(order) + 1.0), sharey=True, squeeze=False)
    for ax, sub in zip(axes[0], subs):
        t = tabs[sub].set_index("display").loc[order]
        y = np.arange(len(order))
        lo = np.where(t.ga_ci_low.isna(), t.ga - t.ga_sd.fillna(0), t.ga_ci_low)
        hi = np.where(t.ga_ci_high.isna(), t.ga + t.ga_sd.fillna(0), t.ga_ci_high)
        for i, (fam, g, l, h) in enumerate(zip(t.family, t.ga, lo, hi)):
            c = FAMILY_COLOR.get(fam, OTHER)
            ax.plot([l, h], [i, i], color=c, linewidth=1.2, solid_capstyle="round")
            ax.plot([g], [i], "o", color=c, markeredgecolor="white", markeredgewidth=0.8, markersize=6)
            ax.annotate(f"{g:.2f}", (h, i), xytext=(3, -2.5), textcoords="offset points", fontsize=6.5, color=INK2)
        ax.set_yticks(y, order)
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("Group accuracy")
        ax.set_title(SUBSET_LABEL[sub].capitalize(), fontsize=8, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
    fams = [f for f in FAMILY_ORDER if f in set(tabs[subs[0]].family)]
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=FAMILY_COLOR[f], label=FAMILY_LABEL[f]) for f in fams]
    fig.legend(handles=handles, loc="upper center", ncol=min(3, len(fams)), bbox_to_anchor=(0.55, -0.06))
    # error bars: 95% bootstrap CI; LLM rows: +/- sd over runs (stated in the caption / table)
    save(fig, out, "fpv_main")


def fig_case_study(R, out, cfg):
    c, m = R["case_flight"], R["metrics"]
    if c is None or m is None:
        return print("  figure case_study: SKIPPED")
    c = c[c.subset == "all"]
    picks = ["Exact dedup", "Regex-masked dedup", "Drain3", R["primary"]]
    llm = m[(m.subset == "all") & m.family.str.startswith("llm")]
    fam = dict(zip(m.method, m.family))
    series = [(p, p) for p in picks if p in set(c.method)]
    if len(llm):
        best = llm.groupby("llm_config").ga.mean().idxmax()
        series.append((f"LLM {best} run0", llm_display(best, cfg) + ", run 0"))
    markers, seen = {}, {}
    for meth, _ in series:
        k = seen.get(fam[meth], 0)
        markers[meth] = ["o", "s", "D", "^"][k]
        seen[fam[meth]] = k + 1
    fig, axes = plt.subplots(1, 2, figsize=(TWO, 2.4))
    for ax, (col, lab) in zip(axes, (("key_recall", "Key-event recall"), ("workload_reduction", "Workload reduction"))):
        for k, (meth, name) in enumerate(series):
            x = c[c.method == meth].sort_values("flight")
            off = (k - (len(series) - 1) / 2) * 0.12
            ax.plot(x.flight + off, x[col], markers[meth], color=FAMILY_COLOR.get(fam[meth], OTHER),
                    markeredgecolor="white", markeredgewidth=0.8, markersize=5.5, label=name)
        ax.set_xticks(sorted(c.flight.unique()))
        ax.set_xlabel("Flight")
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3)
    save(fig, out, "case_study")


def fig_base_models(R, out, cfg):
    m = R["metrics"]
    if m is None:
        return
    main = cfg["clustering"]["algorithm"]
    x = m[(m.subset == "all") & m.family.isin(["pretrained", "lasec"]) & (m.algorithm.isna() | (m.algorithm == main))]
    if x.family.eq("lasec").sum() == 0:
        return print("  figure base_models: SKIPPED (no LASeC models)")
    fig, ax = plt.subplots(figsize=(ONE, 2.4))
    for b in [b for b in BASE_ORDER if b in set(x.base_model)]:
        p, l = x[(x.base_model == b) & (x.family == "pretrained")], x[(x.base_model == b) & (x.family == "lasec")]
        if len(p) and len(l):
            ax.plot([p.runtime_s.iloc[0], l.runtime_s.iloc[0]], [p.ga.iloc[0], l.ga.iloc[0]], color=OTHER, linewidth=1, zorder=1)
        if len(l):
            ax.annotate(b.replace("all-", ""), (l.runtime_s.iloc[0], l.ga.iloc[0]), xytext=(4, 2), textcoords="offset points",
                        fontsize=6.5, color=INK2)
    for fam, lab in (("pretrained", "pre-trained"), ("lasec", "LASeC (fine-tuned)")):
        y = x[x.family == fam]
        ax.plot(y.runtime_s, y.ga, "o", color=FAMILY_COLOR[fam], markeredgecolor="white", markeredgewidth=0.8, label=lab, zorder=2)
    ax.set_xlabel("Encoding + clustering time on FPV (s)")
    ax.set_ylabel("Group accuracy (FPV)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    save(fig, out, "base_models")


def fig_sensitivity(R, out, cfg):
    s, m = R["sensitivity"], R["metrics"]
    if s is None or len(s) == 0:
        return print("  figure fpv_sensitivity: SKIPPED")
    b = slug(cfg["embedding"]["base_model"])
    fig, ax = plt.subplots(figsize=(ONE, 2.3))
    alg = ALGORITHM_LABEL[cfg["clustering"]["algorithm"]]
    for fam, name, lab in (("pretrained", f"Pre-trained {b} + {alg}", "pre-trained"), ("lasec", R["primary"], "LASeC (fine-tuned)")):
        y = s[(s.method == name) & (s.subset == "all")]
        if not len(y):
            continue
        ax.plot(y.threshold, y.ga, color=FAMILY_COLOR[fam], label=lab)
        row = m[(m.method == name) & (m.subset == "all")]
        if len(row):
            ax.plot([row.threshold.iloc[0]], [row.ga.iloc[0]], "o", color=FAMILY_COLOR[fam], markeredgecolor="white",
                    markeredgewidth=1)
    ax.set_xlabel("Distance threshold (cosine)")
    ax.set_ylabel("Group accuracy (FPV, all sentences)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    ax.set_title("Post-hoc sensitivity (markers: thresholds chosen on the catalogue)", fontsize=7.5, loc="left", color=INK)
    save(fig, out, "fpv_sensitivity")


def fig_repeatability(R, out, cfg):
    rr, m = R["repeat_runs"], R["metrics"]
    if rr is None:
        return print("  figure repeatability: SKIPPED")
    names = sorted(set(rr.method), key=lambda n: ([i for i, b in enumerate(BASE_ORDER) if f"({b})" in n] + [99])[0] * 10 + ("+" in n))
    fig, ax = plt.subplots(figsize=(ONE, 0.35 * len(names) + 0.9))
    rng = np.random.default_rng(0)
    for i, n in enumerate(names):
        g = rr[rr.method == n].ga
        ax.plot(g, i + rng.uniform(-0.12, 0.12, len(g)), "o", color=FAMILY_COLOR["lasec"], alpha=0.6, markersize=4,
                markeredgewidth=0)
        ref = m[(m.method == n) & (m.subset == "all")]
        if len(ref):
            ax.plot([ref.ga.iloc[0]], [i], "|", color=INK, markersize=12, markeredgewidth=1.5)
    ax.set_yticks(range(len(names)), [n.replace("LASeC (", "").replace(")", "") for n in names])
    ax.set_ylim(len(names) - 0.5, -0.5)  # main algorithm on top
    ax.set_xlabel("Group accuracy (FPV) under random input orders")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_title("Dots: random input orders; bar: reported (alphabetical order)", fontsize=7, loc="left", color=INK)
    save(fig, out, "repeatability")


def fig_tsne(R, out, cfg, n_highlight=8):
    """t-SNE of the held-out FPV sentences (never used for training), pre-trained vs fine-tuned primary encoder.

    The FPV gold groups with the most member sentences are coloured; all other sentences are grey.
    Needs the trained model in models/lasec-<base model>/ (run train_final.py first)."""
    base = slug(cfg["embedding"]["base_model"])
    model_dir = path(cfg, "models") / f"lasec-{base}"
    if not model_dir.exists():
        return print(f"  figure tsne: SKIPPED ({model_dir} not found; run train_final.py first)")
    from sklearn.manifold import TSNE

    from lasec.cluster import embed, load_encoder
    from lasec.experiment import load_fpv
    fpv, _ = load_fpv(cfg)
    fpv = fpv.sort_values("sentence").reset_index(drop=True)
    size = fpv.event.value_counts()
    top = list(size[size > 1].index[:n_highlight])
    fig, axes = plt.subplots(1, 2, figsize=(TWO, 3.1))
    for ax, (model, title) in zip(axes, ((cfg["embedding"]["base_model"], f"Pre-trained {base}"),
                                         (str(model_dir), f"Fine-tuned (LASeC, {base})"))):
        X = embed(load_encoder(model), fpv.sentence.tolist(), cfg["embedding"]["batch_size"], True)
        Z = TSNE(n_components=2, metric="cosine", perplexity=15, init="pca", random_state=cfg["seed"]).fit_transform(X)
        rest = ~fpv.event.isin(top).to_numpy()
        ax.scatter(Z[rest, 0], Z[rest, 1], s=8, color=OTHER, linewidths=0, label="other events")
        for k, e in enumerate(top):
            sel = (fpv.event == e).to_numpy()
            ax.scatter(Z[sel, 0], Z[sel, 1], s=18, color=SLOTS[k], edgecolors="white", linewidths=0.4,
                       label=f"{e} ({int(sel.sum())})")
        ax.set_title(title, fontsize=8, loc="left", color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.14))
    save(fig, out, "tsne")


FIGURES = {"selection_algorithms": fig_algorithm_selection, "selection_embeddings": fig_embedding_selection,
           "cv_threshold": fig_cv_threshold, "fpv_main": fig_fpv_main, "case_study": fig_case_study,
           "base_models": fig_base_models, "fpv_sensitivity": fig_sensitivity, "repeatability": fig_repeatability,
           "tsne": fig_tsne}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-tsne", action="store_true")
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES), help="only these figures, e.g. --only tsne")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    R = load_all(cfg)
    out = path(cfg, "results") / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    names = a.only or [n for n in FIGURES if not (n == "tsne" and a.skip_tsne)]
    for n in names:
        FIGURES[n](R, out, cfg)


if __name__ == "__main__":
    main()
