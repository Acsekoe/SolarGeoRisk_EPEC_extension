"""Quick analysis script for sensitivity results."""
import pandas as pd
import os, glob

base = r"d:\Alexander\Studium\EEG\Complementarity Modelling\SolarGeoRisk_EPEC_extension\outputs\sensitivity_20260216_161237"
files = sorted(glob.glob(os.path.join(base, "run_*.xlsx")))
regions = ["ch", "eu", "us", "apac", "roa", "row"]

print(f"Found {len(files)} files\n")

all_flows = []
all_regions = []

for f in files:
    name = os.path.basename(f).replace(".xlsx", "")
    fl = pd.read_excel(f, sheet_name="flows")
    rg = pd.read_excel(f, sheet_name="regions")
    fl["run"] = name
    rg["run"] = name
    all_flows.append(fl)
    all_regions.append(rg)

flows_df = pd.concat(all_flows, ignore_index=True)
regs_df = pd.concat(all_regions, ignore_index=True)

# Filter only significant flows (x > 0.1 GW)
sig = flows_df[flows_df["x"] > 0.1].copy()
sig["route"] = sig["exp"] + "->" + sig["imp"]


def classify(name):
    if "_default" in name:
        return "EqA_CH_first"
    return "EqB_CH_later"


print("=" * 90)
print("SIGNIFICANT TRADE FLOWS (x > 0.1 GW) - Grouped by Equilibrium")
print("=" * 90)

for eq_label in ["EqA_CH_first", "EqB_CH_later"]:
    runs = [
        os.path.basename(f).replace(".xlsx", "")
        for f in files
        if classify(os.path.basename(f).replace(".xlsx", "")) == eq_label
    ]
    eq_flows = sig[sig["run"].isin(runs)]

    print(f"\n--- {eq_label} ({len(runs)} runs) ---")

    routes = (
        eq_flows.groupby("route")
        .agg(
            mean_x=("x", "mean"),
            min_x=("x", "min"),
            max_x=("x", "max"),
            mean_te=("tau_exp", "mean"),
            count=("x", "count"),
        )
        .sort_values("mean_x", ascending=False)
    )

    header = f"{'Route':<15} {'Mean_x':>10} {'Min_x':>10} {'Max_x':>10} {'Tau_exp':>10} {'N_runs':>8}"
    print(header)
    for route, row in routes.iterrows():
        n = int(row["count"])
        print(
            f"{route:<15} {row.mean_x:>10.2f} {row.min_x:>10.2f} {row.max_x:>10.2f} {row.mean_te:>10.2f} {n:>8}"
        )


print("\n")
print("=" * 90)
print("REGIONAL SUMMARY - Q_offer, Imports, Exports, Lambda, Obj")
print("=" * 90)

for eq_label in ["EqA_CH_first", "EqB_CH_later"]:
    runs = [
        os.path.basename(f).replace(".xlsx", "")
        for f in files
        if classify(os.path.basename(f).replace(".xlsx", "")) == eq_label
    ]
    eq_regs = regs_df[regs_df["run"].isin(runs)]

    print(f"\n--- {eq_label} ({len(runs)} runs) ---")
    summary = eq_regs.groupby("r").agg(
        Q_mean=("Q_offer", "mean"),
        Q_std=("Q_offer", "std"),
        lam_mean=("lam", "mean"),
        imp_mean=("imports", "mean"),
        exp_mean=("exports", "mean"),
        obj_mean=("obj", "mean"),
    )
    summary = summary.reindex(regions)
    header = f"{'Rgn':<6} {'Q_offer':>10} {'Q_std':>8} {'Lambda':>10} {'Imports':>10} {'Exports':>10} {'Obj':>12}"
    print(header)
    for r, row in summary.iterrows():
        print(
            f"{r:<6} {row.Q_mean:>10.1f} {row.Q_std:>8.1f} {row.lam_mean:>10.1f} "
            f"{row.imp_mean:>10.1f} {row.exp_mean:>10.1f} {row.obj_mean:>12.1f}"
        )


print("\n")
print("=" * 90)
print("FULL TRADE FLOW MATRIX (mean x) per Equilibrium")
print("=" * 90)

for eq_label in ["EqA_CH_first", "EqB_CH_later"]:
    runs = [
        os.path.basename(f).replace(".xlsx", "")
        for f in files
        if classify(os.path.basename(f).replace(".xlsx", "")) == eq_label
    ]
    eq_flows = flows_df[flows_df["run"].isin(runs)]

    print(f"\n--- {eq_label}: Mean trade flow x(exp->imp) ---")
    pivot = eq_flows.pivot_table(
        values="x", index="exp", columns="imp", aggfunc="mean"
    )
    pivot = pivot.reindex(index=regions, columns=regions)
    # Format as table
    header = f"{'exp\\imp':<8}" + "".join(f"{c:>10}" for c in regions)
    print(header)
    for r in regions:
        vals = "".join(f"{pivot.loc[r, c]:>10.2f}" for c in regions)
        print(f"{r:<8}{vals}")

    # Also tau_exp matrix
    print(f"\n--- {eq_label}: Mean tau_exp(exp->imp) ---")
    pivot_te = eq_flows.pivot_table(
        values="tau_exp", index="exp", columns="imp", aggfunc="mean"
    )
    pivot_te = pivot_te.reindex(index=regions, columns=regions)
    header = f"{'exp\\imp':<8}" + "".join(f"{c:>10}" for c in regions)
    print(header)
    for r in regions:
        vals = "".join(f"{pivot_te.loc[r, c]:>10.4f}" for c in regions)
        print(f"{r:<8}{vals}")


print("\n")
print("=" * 90)
print("CH DOMESTIC vs EXPORT SPLIT")
print("=" * 90)
ch_flows = flows_df[flows_df["exp"] == "ch"].copy()
ch_flows["eq"] = ch_flows["run"].apply(classify)
ch_pivot = ch_flows.pivot_table(values="x", index="run", columns="imp", aggfunc="sum")
ch_pivot = ch_pivot.reindex(columns=regions)
ch_pivot["eq"] = [classify(r) for r in ch_pivot.index]
ch_pivot["total_export"] = ch_pivot[["eu", "us", "apac", "roa", "row"]].sum(axis=1)

for eq_label in ["EqA_CH_first", "EqB_CH_later"]:
    subset = ch_pivot[ch_pivot["eq"] == eq_label]
    print(f"\n--- {eq_label} ---")
    header = f"{'Run':<45} {'Domestic':>10} {'Exports':>10} {'->EU':>8} {'->US':>8} {'->APAC':>8} {'->ROA':>8} {'->ROW':>8}"
    print(header)
    for run, row in subset.iterrows():
        rn = run.replace("run_", "")
        ex = row["total_export"]
        print(
            f"{rn:<45} {row['ch']:>10.1f} {ex:>10.1f} "
            f"{row['eu']:>8.1f} {row['us']:>8.1f} {row['apac']:>8.1f} "
            f"{row['roa']:>8.1f} {row['row']:>8.1f}"
        )
