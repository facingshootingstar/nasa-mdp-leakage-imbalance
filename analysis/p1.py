"""Paper 1 analysis: data quality (leakage, raw vs clean) and imbalance treatments.

Revision 1 (deviation 28): the dataset (n = 9, learners averaged) is the unit of every confirmatory test, the effect
size is the matched-pairs rank-biserial r, and the revision experiments of experiments/run_p1_extra.py are added.
"""
from common import *
from scipy.stats import spearmanr

LAB = {"none": "None", "cw": "Class weight", "rus": "RUS", "ros": "ROS", "smote": "SMOTE",
       "bsmote": "B-SMOTE", "smotetomek": "SMOTE-Tomek", "thr_prior": "Thr. (prior)", "thr_cv": "Thr. (inner CV)"}
TREAT = ["cw", "rus", "ros", "smote", "bsmote", "smotetomek"]
CLF3 = ("LR", "RF", "LGBM")
df = load_p1()
out = {}
tex = []


def npos(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return int((d > 0).sum())


def vs_ref(m, metrics, treatments, ref="none", key="resampler", unit="dataset"):
    """Treatment vs reference on unit-level means (Holm over treatments per metric)."""
    rows = []
    for met in metrics:
        base = m[m[key].eq(ref)].set_index(unit)[met]
        for t in treatments:
            a = m[m[key].eq(t)].set_index(unit)[met].reindex(base.index)
            rows.append({"metric": met, key: t, **pair_test(a, base), "n_pos": npos(a, base)})
    return add_holm(pd.DataFrame(rows), by="metric")


# ---- RQ1.1a leakage
raw = df[df.version.eq("raw")]
leak = raw.drop_duplicates(["dataset", "repeat", "fold"]).groupby("dataset").leak_share.agg(["mean", "min", "max"])
out["leak"] = leak.reset_index()
out["leak_overall"] = {"mean": leak["mean"].mean(), "max": leak["mean"].max(), "min": leak["mean"].min()}

# ---- RQ1.1b unseen vs full on raw data (resampler none)
r0 = raw[raw.resampler.eq("none")]
dm = r0.groupby(["dataset", "clf"])[["mcc", "mcc_unseen", "gmean", "gmean_unseen"]].mean().reset_index()
rows = []
for met in ("mcc", "gmean"):
    for c in CLF3:
        s = dm[dm.clf.eq(c)]
        rows.append({"metric": met, "clf": c, **pair_test(s[met], s[f"{met}_unseen"]), "n_pos": npos(s[met], s[f"{met}_unseen"])})
unseen = add_holm(pd.DataFrame(rows), by="metric")
out["unseen_vs_full"] = unseen
out["unseen_by_dataset"] = dm
# exploratory: gap vs leakage share (Spearman over 9 datasets), its sensitivity to mc1, and leave-one-dataset-out
gap = dm.assign(gap=dm.mcc - dm.mcc_unseen).pivot(index="dataset", columns="clf", values="gap")
lk = leak["mean"].reindex(gap.index)
out["gap_vs_leak"] = {}
for c in CLF3:
    rho, p = spearmanr(lk, gap[c])
    keep = gap.index != "mc1"
    rho2, p2 = spearmanr(lk[keep], gap[c][keep])
    loo = [spearmanr(lk.drop(d), gap[c].drop(d))[0] for d in gap.index]
    lo, hi = boot_ci(gap[c][keep].values)
    out["gap_vs_leak"][c] = {"rho": rho, "p": p, "rho_no_mc1": rho2, "p_no_mc1": p2,
                             "rho_loo_min": min(loo), "rho_loo_max": max(loo),
                             "gap_mean": gap[c].mean(), "gap_median": gap[c].median(),
                             "gap_mean_no_mc1": gap[c][keep].mean(), "gap_no_mc1_ci": [lo, hi]}

# ---- RQ1.1c raw vs clean (resampler none)
v = df[df.resampler.eq("none")].groupby(["dataset", "version", "clf"])[["mcc", "gmean", "pd", "pf", "auc"]].mean().reset_index()
rows = []
for met in ("mcc", "gmean"):
    for c in CLF3:
        a = v[(v.clf == c) & (v.version == "raw")].set_index("dataset")[met]
        b = v[(v.clf == c) & (v.version == "clean")].set_index("dataset")[met].reindex(a.index)
        rows.append({"metric": met, "clf": c, **pair_test(a, b), "n_pos": npos(a, b)})
rvc = add_holm(pd.DataFrame(rows), by="metric")
out["raw_vs_clean"] = rvc
ds_tab = pd.read_csv(SUM / "datasets.csv").set_index("dataset")
out["prevalence"] = {d: {"raw": ds_tab.loc[d, "pos_raw"] / ds_tab.loc[d, "n_raw"],
                         "clean": ds_tab.loc[d, "pos_final"] / ds_tab.loc[d, "n_final"]} for d in ds_tab.index}

# ---- RQ1.2 treatments on clean data
cl = df[df.version.eq("clean")]
cm = cl.groupby(["dataset", "clf", "resampler"])[["mcc", "gmean", "pd", "pf", "auc", "prauc"]].mean().reset_index()
cm["block"] = cm.dataset + "|" + cm.clf
dmean = cm.groupby(["dataset", "resampler"])[["mcc", "gmean", "pd", "pf", "auc", "prauc"]].mean().reset_index()

fr = {}
fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.6))
for ax, met in zip(axes, ("mcc", "gmean")):
    chi2, p, ranks, cd, nem, N = friedman_nemenyi(dmean, "dataset", "resampler", met)  # primary unit: dataset (N=9)
    fr[met] = {"chi2": chi2, "p": p, "ranks": ranks, "cd": cd, "N": N, "nemenyi": nem.round(4)}
    cd_diagram(ranks, cd, f"{met.upper() if met == 'mcc' else 'G-Mean'} (Friedman $\\chi^2$={chi2:.1f}, p{peq(p)}, N={N})",
               ax=ax, labels=LAB)
fig.tight_layout(); savefig(fig, 1, "cd_resamplers")
out["friedman"] = fr
# sensitivity: Friedman/Nemenyi on the 27 dataset x learner blocks (the original, pseudo-replicated unit)
fr27 = {}
for met in ("mcc", "gmean"):
    chi2, p, ranks, cd, nem, N = friedman_nemenyi(cm, "block", "resampler", met)
    fr27[met] = {"chi2": chi2, "p": p, "ranks": ranks, "cd": cd, "N": N}
out["friedman_n27"] = fr27

# primary (revised): dataset level, n = 9
vsnone = vs_ref(dmean, ("mcc", "gmean", "pd", "pf"), TREAT)
out["vs_none"] = vsnone
# sensitivity: 27 dataset x learner blocks (the original analysis), and each learner separately at n = 9
out["vs_none_n27"] = vs_ref(cm, ("mcc", "gmean"), TREAT, unit="block")
out["vs_none_per_clf"] = pd.concat([vs_ref(cm[cm.clf.eq(c)], ("mcc", "gmean"), TREAT).assign(clf=c) for c in CLF3])
# threshold-free metrics (exploratory)
out["vs_none_rank"] = vs_ref(dmean, ("auc", "prauc"), TREAT)

pc = cm.groupby(["clf", "resampler"])[["mcc", "gmean", "pd", "pf"]].mean().reset_index()
out["per_clf"] = pc

# ---- clean-vs-raw x treatment (exploratory): is the treatment gain the same with and without cleaning?
rm = raw.groupby(["dataset", "resampler"])[["mcc", "gmean"]].mean().reset_index()
rows, ranks_raw = [], {}
for met in ("mcc", "gmean"):
    gr = rm.pivot(index="dataset", columns="resampler", values=met)
    gc = dmean.pivot(index="dataset", columns="resampler", values=met)
    for t in TREAT:
        a = gc[t] - gc["none"]; b = (gr[t] - gr["none"]).reindex(a.index)
        rows.append({"metric": met, "resampler": t, "gain_raw": b.mean(), "gain_clean": a.mean(), **pair_test(a, b),
                     "n_pos": npos(a, b)})
    rmb = raw.groupby(["dataset", "clf", "resampler"])[met].mean().reset_index()
    rmb["block"] = rmb.dataset + "|" + rmb.clf
    ranks_raw[met] = friedman_nemenyi(rm, "dataset", "resampler", met)[2]  # dataset level, N = 9
out["interaction"] = add_holm(pd.DataFrame(rows), by="metric")
out["ranks_raw"] = ranks_raw

# ---- revision experiments (deviation 28)
X1 = RAW / "p1x"


def part(name):
    """All nine datasets of one revision experiment, or None while the run is incomplete."""
    fs = sorted(X1.glob(f"{name}_*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs]) if len(fs) == 9 else None


g = part("group")
if g is not None:
    # raw data: random CV vs group CV (no test row has a training twin), clean data as a third column
    gm = g.groupby(["dataset", "clf", "resampler"])[["mcc", "gmean", "auc", "prev_test"]].mean().reset_index()
    rnd = raw.groupby(["dataset", "clf", "resampler"])[["mcc", "gmean", "auc"]].mean().reset_index()
    rows = []
    for met in ("mcc", "gmean", "auc"):
        for c in CLF3:
            a = rnd[(rnd.clf == c) & rnd.resampler.eq("none")].set_index("dataset")[met]
            b = gm[(gm.clf == c) & gm.resampler.eq("none")].set_index("dataset")[met].reindex(a.index)
            rows.append({"metric": met, "clf": c, **pair_test(a, b), "n_pos": npos(a, b),
                         "gap_mc1": a["mc1"] - b["mc1"], "gap_no_mc1": (a - b).drop("mc1").mean()})
    out["random_vs_group"] = add_holm(pd.DataFrame(rows), by="metric")
    out["random_vs_group_by_dataset"] = (rnd[rnd.resampler.eq("none")].merge(
        gm[gm.resampler.eq("none")], on=["dataset", "clf"], suffixes=("_random", "_group")))
    # treatment gains under group CV on raw data
    gd = gm.groupby(["dataset", "resampler"])[["mcc", "gmean"]].mean().reset_index()
    out["vs_none_groupcv"] = vs_ref(gd, ("mcc", "gmean"), TREAT)
    # fold-composition diagnostics: largest identical-row group, spread of fold size and prevalence (group vs random)
    from core.data import load
    fd = {}
    for ds in sorted(g.dataset.unique()):
        Xr, yr = load(ds, clean=False)
        gs = Xr.astype(str).agg("|".join, axis=1).value_counts()
        gg = g[(g.dataset == ds) & g.clf.eq("LR") & g.resampler.eq("none")]
        rr = raw[(raw.dataset == ds) & raw.clf.eq("LR") & raw.resampler.eq("none")]
        fd[ds] = {"n": int(len(yr)), "largest_group": int(gs.iloc[0]), "largest_group_share": float(gs.iloc[0] / len(yr)),
                  "group_n_min": int(gg.n_test.min()), "group_n_max": int(gg.n_test.max()),
                  "random_n_min": int(rr.n_test.min()), "random_n_max": int(rr.n_test.max()),
                  "group_prev_min": float(gg.prev_test.min()), "group_prev_max": float(gg.prev_test.max()),
                  "prev_full": float(yr.mean())}
    out["groupcv_folds"] = fd
    gb = gm.assign(block=gm.dataset + "|" + gm.clf)
    out["ranks_groupcv"] = {met: friedman_nemenyi(gd, "dataset", "resampler", met)[2] for met in ("mcc", "gmean")}

s = part("seen")
if s is not None:
    sd = s.groupby("dataset")[["share_seen", "prev_full", "prev_seen", "prev_unseen", "consistent_of_seen"]].mean()
    out["seen_profile"] = sd.reset_index()
    pdiff = (sd.prev_unseen - sd.prev_seen); lk2 = leak["mean"].reindex(pdiff.index)
    r_all = spearmanr(lk2, pdiff); r_nm = spearmanr(lk2.drop("mc1"), pdiff.drop("mc1"))
    out["leak_vs_prevdiff"] = {"rho": float(r_all[0]), "p": float(r_all[1]),
                               "rho_no_mc1": float(r_nm[0]), "p_no_mc1": float(r_nm[1])}
    out["seen_min_pos_unseen"] = s.groupby("dataset").pos_unseen.min().to_dict()
    sm = s.groupby(["dataset", "clf"]).mean(numeric_only=True).reset_index()
    rows = []
    for met in ("mcc", "bacc", "auc"):
        for c in CLF3:
            x = sm[sm.clf.eq(c)]
            rows.append({"metric": met, "clf": c, **pair_test(x[f"{met}_full"], x[f"{met}_unseen"]),
                         "n_pos": npos(x[f"{met}_full"], x[f"{met}_unseen"])})
    out["full_vs_unseen_prevfree"] = add_holm(pd.DataFrame(rows), by="metric")
    out["seen_metrics"] = sm[["dataset", "clf", "mcc_full", "mcc_unseen", "mcc_seen", "auc_full", "auc_unseen",
                              "bacc_full", "bacc_unseen"]]

mj = part("major")
if mj is not None:
    mm = mj.groupby(["dataset", "resampler"])[["mcc", "gmean"]].mean().reset_index()
    out["vs_none_major"] = vs_ref(mm, ("mcc", "gmean"), TREAT)
    out["major_size"] = mj.drop_duplicates("dataset").set_index("dataset")[["n", "pos"]].to_dict(orient="index")
    mb = mj.groupby(["dataset", "clf", "resampler"])[["mcc", "gmean"]].mean().reset_index()
    mb["block"] = mb.dataset + "|" + mb.clf
    out["ranks_major"] = {met: friedman_nemenyi(mm, "dataset", "resampler", met)[2] for met in ("mcc", "gmean")}

th = part("thr")
if th is not None:
    tm = pd.concat([cl[cl.resampler.isin(["none", "smote", "smotetomek", "rus"])],
                    th]).groupby(["dataset", "resampler"])[["mcc", "gmean", "pd", "pf"]].mean().reset_index()
    out["vs_none_thr"] = vs_ref(tm, ("mcc", "gmean", "pd", "pf"), ["thr_prior", "thr_cv"])
    out["thr_vs_smote"] = vs_ref(tm, ("mcc", "gmean"), ["thr_prior", "thr_cv"], ref="smote")
    out["thr_means"] = tm.groupby("resampler")[["mcc", "gmean", "pd", "pf"]].mean()
    out["thr_value"] = th.groupby(["clf", "resampler"]).threshold.median()
    # non-inferiority to SMOTE (exploratory, deviation 28k, revised in round 3): margin = half the SMOTE gain over
    # no treatment; 90% dataset-bootstrap and t intervals; smallest margins at which each rule passes
    from scipy.stats import t as tdist
    rg = np.random.default_rng(20261018)
    w = tm.pivot(index="dataset", columns="resampler", values="mcc")
    smote_gain = float((w["smote"] - w["none"]).mean()); NI = smote_gain / 2
    eqr = {}
    for rule in ("thr_prior", "thr_cv"):
        d = (w[rule] - w["smote"]).values
        bs = [d[rg.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        lo, hi = np.quantile(bs, [0.05, 0.95])
        se = d.std(ddof=1) / np.sqrt(len(d)); q = tdist.ppf(0.95, len(d) - 1)
        tlo, thi = d.mean() - q * se, d.mean() + q * se
        eqr[rule] = {"diff": float(d.mean()), "lo90": float(lo), "hi90": float(hi), "t_lo90": float(tlo),
                     "t_hi90": float(thi), "smote_gain": smote_gain, "ni_margin": NI,
                     "noninferior": bool(lo > -NI), "noninferior_t": bool(tlo > -NI),
                     "min_margin_ni": float(max(0.0, -lo)), "min_margin_ni_t": float(max(0.0, -tlo)),
                     "min_margin_equiv": float(max(-lo, hi)), "min_margin_equiv_t": float(max(-tlo, thi))}
    out["thr_equiv_smote"] = eqr
    # LGBM sensitivity to its inner-CV threshold: MCC of LGBM with the inner-CV rule vs untreated / SMOTE
    lg = pd.concat([cl[cl.clf.eq("LGBM") & cl.resampler.isin(["none", "smote"])], th[th.clf.eq("LGBM")]])
    out["thr_lgbm"] = lg.groupby("resampler")[["mcc", "gmean", "pd", "pf"]].mean()

# ---- figure: pd vs pf (dataset-averaged per classifier x treatment)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
mk = {"none": "o", "cw": "s", "rus": "v", "ros": "^", "smote": "D", "bsmote": "P", "smotetomek": "X"}
for ax, c in zip(axes, CLF3):
    sc = pc[pc.clf.eq(c)].set_index("resampler")
    for i, rsn in enumerate(mk):
        r = sc.loc[rsn]
        ax.scatter(r.pf, r.pd, marker=mk[rsn], color=OI[i], s=40, label=LAB[rsn], edgecolor="k", lw=0.4,
                   alpha=0.85, zorder=3)
    ax.set_title(c); ax.set_xlabel("pf (false alarm rate)"); ax.grid(alpha=0.3)
    ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=0.7)
    ax.set_xlim(0, 0.45); ax.set_ylim(0, 0.9)
axes[0].set_ylabel("pd (recall)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=7, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1)); savefig(fig, 1, "pd_pf")

# ---- figure: leakage per dataset and MCC inflation
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
lks = leak.sort_values("mean")
ax[0].barh(lks.index, 100 * lks["mean"], color=OI[0]); ax[0].set_xlabel("% raw test rows also in training fold")
if "random_vs_group_by_dataset" in out:  # primary RQ1 counterfactual: standard minus group CV, untreated models
    rg_ = out["random_vs_group_by_dataset"]; infl = rg_.assign(gap=rg_.mcc_random - rg_.mcc_group)
    xl = "MCC(standard CV) - MCC(group CV)"
else:
    infl = dm.assign(gap=dm.mcc - dm.mcc_unseen); xl = "MCC(full fold) - MCC(unseen rows)"
for i, c in enumerate(CLF3):
    sc = infl[infl.clf.eq(c)].set_index("dataset").reindex(lks.index)
    ax[1].scatter(sc.gap, sc.index, color=OI[i + 1], marker="osD"[i], label=c, s=22)
ax[1].axvline(0, color="grey", lw=0.7); ax[1].set_xlabel(xl)
ax[1].legend(fontsize=7, frameon=False)
fig.tight_layout(); savefig(fig, 1, "leakage")

dump(out, "p1.json")

# ---- tables
t = ds_tab.reset_index()[["dataset", "n_raw", "pos_raw", "missing_removed", "negative_removed", "conflicting_removed",
                          "duplicate_removed", "n_final", "pos_final"]].copy()
t["prev_raw"] = 100 * t.pos_raw / t.n_raw; t["prev_clean"] = 100 * t.pos_final / t.n_final
t["leak \\%"] = [100 * leak.loc[d, "mean"] for d in t.dataset]
t.columns = ["Dataset", "$n$ raw", "Def. raw", "Missing", "Negative", "Conflicting", "Duplicate", "$n$ clean",
             "Def. clean", "Def.\\% raw", "Def.\\% clean", "Leak \\%"]
f1 = lambda x: f"{x:.1f}"
tex.append(latex_table(t, None, "Datasets, rows removed by each cleaning rule (applied in order), defect prevalence before "
                       "and after cleaning, and the mean share of raw test rows whose metric vector also occurs in the "
                       "training fold (25 folds).", "tab:data",
                       fmt={"Leak \\%": f1, "Def.\\% raw": f1, "Def.\\% clean": f1}))

R = lambda x: f"{x:.2f}"


def ptab(d, keycol):
    t = d[["metric", keycol, "mean_a", "mean_b", "diff", "ci_lo", "ci_hi", "p_holm", "rrb", "n_pos", "n"]].copy()
    t["ci"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(t.ci_lo, t.ci_hi)]
    t["pos"] = [f"{a}/{b}" for a, b in zip(t.n_pos, t.n)]
    t = t[["metric", keycol, "mean_a", "mean_b", "diff", "ci", "p_holm", "rrb", "pos"]]
    t["metric"] = t.metric.map({"mcc": "MCC", "gmean": "G-Mean", "pd": "pd", "pf": "pf", "auc": "AUC",
                                "prauc": "PR-AUC", "bacc": "Bal. acc."})
    return t


FMT = {"$p_{Holm}$": pfmt, "$r_{rb}$": R}
t = ptab(rvc, "clf"); t.columns = ["Metric", "Classifier", "Raw", "Clean", "$\\Delta$", "95\\% CI", "$p_{Holm}$", "$r_{rb}$", "$+$"]
tex.append(latex_table(t, None, "Raw versus cleaned data (no treatment): dataset-level means ($n=9$), mean paired difference "
                       "with bootstrap 95\\% CI, Holm-adjusted Wilcoxon $p$ (family: 3 classifiers per metric), matched-pairs "
                       "rank-biserial $r_{rb}$, and the number of datasets with a positive difference.",
                       "tab:rawclean", fmt=FMT))
t = ptab(unseen, "clf"); t.columns = ["Metric", "Classifier", "Full fold", "Unseen", "$\\Delta$", "95\\% CI", "$p_{Holm}$", "$r_{rb}$", "$+$"]
tex.append(latex_table(t, None, "Raw data, random folds: performance on the full test fold versus its unseen rows ($n=9$ "
                       "datasets; columns as in Table~\\ref{tab:rawclean}).", "tab:unseen", fmt=FMT))
if "random_vs_group" in out:
    t = ptab(out["random_vs_group"], "clf")
    t.columns = ["Metric", "Classifier", "Random CV", "Group CV", "$\\Delta$", "95\\% CI", "$p_{Holm}$", "$r_{rb}$", "$+$"]
    tex.append(latex_table(t, None, "Raw data, no treatment: standard stratified CV versus stratified group CV in which "
                           "identical metric vectors share a fold, so that no test row has a copy in training ($n=9$; "
                           "Holm family: 3 classifiers per metric; post-hoc primary RQ1 analysis, added after the SAP was frozen).", "tab:groupcv", fmt=FMT))
    fdt = pd.DataFrame(out["groupcv_folds"]).T.reset_index().rename(columns={"index": "ds"})
    t = pd.DataFrame({"Dataset": fdt.ds, "Rows": fdt.n.astype(int),
                      "Largest group": [f"{int(a)} ({100 * b:.1f}\\%)" for a, b in zip(fdt.largest_group, fdt.largest_group_share)],
                      "Fold size, random": [f"{int(a)}--{int(b)}" for a, b in zip(fdt.random_n_min, fdt.random_n_max)],
                      "Fold size, group": [f"{int(a)}--{int(b)}" for a, b in zip(fdt.group_n_min, fdt.group_n_max)],
                      "Prevalence": [f"{a:.3f}" for a in fdt.prev_full],
                      "Fold prev., group": [f"{a:.3f}--{b:.3f}" for a, b in zip(fdt.group_prev_min, fdt.group_prev_max)]})
    tex.append(latex_table(t, None, "Fold composition under group CV (raw data): the largest group of identical metric "
                           "vectors, the range of test-fold sizes under random and group CV, and the range of test-fold "
                           "prevalence under group CV over the 25 folds.", "tab:groupfolds"))
t = ptab(vsnone, "resampler"); t["resampler"] = t.resampler.map(LAB)
t.columns = ["Metric", "Treatment", "Treated", "None", "$\\Delta$", "95\\% CI", "$p_{Holm}$", "$r_{rb}$", "$+$"]
tex.append(latex_table(t, None, "Each imbalance treatment versus no treatment on cleaned data. Unit: dataset, learners "
                       "averaged ($n=9$); Holm family: 6 treatments per metric. pd and pf are secondary and use their "
                       "own Holm family.", "tab:vsnone", fmt=FMT))
pcw = pc.pivot_table(index="resampler", columns="clf", values=["mcc", "gmean"]).reindex(list(mk))
t = pd.DataFrame({"Treatment": [LAB[i] for i in pcw.index]})
for met in ("mcc", "gmean"):
    for c in CLF3:
        t[f"{'MCC' if met == 'mcc' else 'G-Mean'} {c}"] = pcw[(met, c)].values
tex.append(latex_table(t, None, "Mean MCC and G-Mean over the nine cleaned datasets per classifier and treatment.", "tab:perclf"))

# robustness table: MCC gain vs none under each analysis variant
var = [("Primary: clean, $n=9$", out["vs_none"]), ("Blocks, $n=27$", out["vs_none_n27"])]
for c in CLF3:
    var.append((f"{c} only, $n=9$", out["vs_none_per_clf"][out["vs_none_per_clf"].clf.eq(c)]))
if "vs_none_major" in out:
    var.append(("Majority-vote cleaning, $n=9$", out["vs_none_major"]))
if "vs_none_groupcv" in out:
    var.append(("Raw data, group CV, $n=9$", out["vs_none_groupcv"]))
rows = []
for name, d in var:
    d = d[d.metric.eq("mcc")].set_index("resampler")
    rows.append({"Analysis": name, **{LAB[t_]: f"{d.loc[t_, 'diff']:+.3f}{'*' if d.loc[t_, 'p_holm'] < 0.05 else ''}"
                                      for t_ in TREAT}})
tex.append(latex_table(pd.DataFrame(rows), None, "Robustness of the MCC result: mean MCC gain of each treatment over no "
                       "treatment under alternative units and data versions (* Holm-adjusted $p<0.05$ within the row).",
                       "tab:robust"))
# seen-row profile (exploratory, deviation 28)
if "seen_profile" in out:
    sp = out["seen_profile"].set_index("dataset")
    sm = out["seen_metrics"].pivot_table(index="dataset", columns="clf", values="mcc_seen")
    t = pd.DataFrame({"Dataset": sp.index, "Seen \\%": 100 * sp.share_seen.values,
                      "Prev. seen": sp.prev_seen.values, "Prev. unseen": sp.prev_unseen.values,
                      "Label agree": sp.consistent_of_seen.values})
    for c in CLF3:
        t[f"MCC seen {c}"] = [sm.loc[d, c] for d in sp.index]
    tex.append(latex_table(t, None, "Raw data, random folds: profile of the \\emph{seen} test rows (rows whose metric vector "
                           "also occurs in the training fold). Prev.: defect prevalence; Label agree: share of seen rows whose "
                           "label equals the majority label of their training copies; MCC seen: MCC of the untreated model on "
                           "the seen rows only (0 when no positive is predicted or present). Exploratory.", "tab:seen",
                           fmt={"Seen \\%": lambda x: f"{x:.1f}"}))
t = ptab(out["vs_none_rank"], "resampler"); t["resampler"] = t.resampler.map(LAB)
t.columns = ["Metric", "Treatment", "Treated", "None", "$\\Delta$", "95\\% CI", "$p_{Holm}$", "$r_{rb}$", "$+$"]
tex.append(latex_table(t, None, "Threshold-free metrics: each treatment versus no treatment on cleaned data ($n=9$; Holm "
                       "family: 6 treatments per metric; exploratory).", "tab:rank", fmt=FMT))
it = out["interaction"].copy()
t = it[["metric", "resampler", "gain_raw", "gain_clean", "diff", "ci_lo", "ci_hi", "p_holm", "rrb", "n_pos", "n"]].copy()
t["ci"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(t.ci_lo, t.ci_hi)]
t["pos"] = [f"{a}/{b}" for a, b in zip(t.n_pos, t.n)]
t = t[["metric", "resampler", "gain_raw", "gain_clean", "diff", "ci", "p_holm", "rrb", "pos"]]
t["metric"] = t.metric.map({"mcc": "MCC", "gmean": "G-Mean"}); t["resampler"] = t.resampler.map(LAB)
t.columns = ["Metric", "Treatment", "Gain raw", "Gain clean", "$\\Delta$", "95\\% CI", "$p_{Holm}$", "$r_{rb}$", "$+$"]
tex.append(latex_table(t, None, "Data version $\\times$ treatment interaction: gain over no treatment on cleaned data minus "
                       "gain on raw data ($n=9$; Holm family: 6 treatments per metric; exploratory).", "tab:interact",
                       fmt=FMT))
if "vs_none_thr" in out:
    THL = {"thr_prior": "Threshold = prior", "thr_cv": "Threshold = inner CV"}
    a = ptab(out["vs_none_thr"], "resampler"); a.insert(0, "Ref.", "None")
    b = ptab(out["thr_vs_smote"], "resampler"); b.insert(0, "Ref.", "SMOTE")
    t = pd.concat([a, b]); t["resampler"] = t.resampler.map(THL)
    t.columns = ["Ref.", "Metric", "Rule", "Thresholded", "Ref. value", "$\\Delta$", "95\\% CI", "$p_{Holm}$", "$r_{rb}$", "$+$"]
    tex.append(latex_table(t, None, "Threshold moving without resampling (cleaned data, $n=9$, learners averaged) against no "
                           "treatment and against SMOTE. Holm family: the two rules within each metric and reference. "
                           "Exploratory (deviation 28).", "tab:thr", fmt=FMT))
(ROOT / "paper1" / "tables.tex").write_text("\n".join(tex), encoding="utf-8")

pd.set_option("display.width", 220)
print("leak", out["leak_overall"]); print(json.dumps(out["gap_vs_leak"], indent=0, default=float))
for k in ("thr_equiv_smote", "leak_vs_prevdiff", "groupcv_folds"):
    if k in out:
        print("==", k); print(json.dumps(out[k], indent=0, default=float))
for k in ("raw_vs_clean", "unseen_vs_full", "vs_none", "vs_none_n27", "vs_none_per_clf", "vs_none_rank", "interaction",
          "random_vs_group", "vs_none_groupcv", "full_vs_unseen_prevfree", "vs_none_major", "vs_none_thr", "thr_vs_smote"):
    if k in out:
        print("==", k); print(out[k].drop(columns=["W", "delta", "mag"], errors="ignore").round(3).to_string())
for k in ("seen_profile", "thr_means", "thr_value"):
    if k in out:
        print("==", k); print(out[k].round(3).to_string())
print({m: (round(f["chi2"], 2), f["p"], round(f["cd"], 3), f["ranks"].round(2).to_dict()) for m, f in fr.items()})
print("n27", {m: (round(f["chi2"], 2), round(f["p"], 4), round(f["cd"], 3), f["ranks"].round(2).to_dict()) for m, f in fr27.items()})
print("ranks_raw", {m: r.round(2).to_dict() for m, r in ranks_raw.items()})
for k in ("ranks_groupcv", "ranks_major"):
    if k in out:
        print(k, {m: r.round(2).to_dict() for m, r in out[k].items()})
