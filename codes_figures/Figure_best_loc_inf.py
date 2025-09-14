import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------- loader (standalone) ----------
def load_best_params_table(base_path, Freq, Den, ObsErrs, suffix):
    """
    Returns a DataFrame with columns:
      inflation, loc_scale, nens, alpha, ntemp, method, obserr
    One row per (nens, alpha, ntemp, obserr) with finite best params.
    """
    if not isinstance(ObsErrs, (list, tuple, np.ndarray)):
        ObsErrs = [ObsErrs]

    def _fmt_obs(x):
        try:
            return f"{float(x):g}"
        except Exception:
            return str(x)

    rows = []
    for obs in ObsErrs:
        obs_str = _fmt_obs(obs)
        fname = f"Best_params_Freq{Freq}_Den{Den}_ObsErr{obs_str}_{suffix}.npz"
        path = os.path.join(base_path, fname)
        if not os.path.exists(path):
            print(f"[warn] missing file: {path}")
            continue

        d = np.load(path, allow_pickle=True)
        infl = d["loc_inflation"]   # (n_nens, n_alpha, n_methods)
        locs = d["loc_scale"]
        nens_labels  = np.array(d["nens_labels"]).astype(int)
        alpha_values = np.array(d["alpha_range"]).astype(int)

        meta = d.get("meta", None)
        if meta is not None and isinstance(meta.item(), dict):
            methods = list(meta.item().get("methods", range(1, infl.shape[2] + 1)))
        else:
            methods = list(range(1, infl.shape[2] + 1))

        def mlabel(m):
            return {1: "LETKF", 2: "LETKF-T2", 3: "LETKF-T3"}.get(int(m), f"NTemp{m}")

        for i, nens in enumerate(nens_labels):
            for j, alpha in enumerate(alpha_values):
                for k, m in enumerate(methods):
                    x = float(infl[i, j, k])
                    y = float(locs[i, j, k])
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    rows.append({
                        "inflation": x,
                        "loc_scale": y,
                        "nens": int(nens),
                        "alpha": int(alpha),
                        "ntemp": int(m),
                        "method": mlabel(m),
                        "obserr": obs_str,
                    })

    return pd.DataFrame(rows)

def plot_method_contours_3panels(base_path, output_dir, Freq, Den, ObsErrs, suffix,
                                 tag="", fill=True, levels=5):
    """
    Three side-by-side KDE panels (LETKF, LETKF-T2, LETKF-T3), same axes.
    Each panel shows a faint scatter, KDE contours (optionally filled), and a star at the median.
    """
    df = load_best_params_table(base_path, Freq, Den, ObsErrs, suffix)
    if df.empty:
        raise ValueError("No data loaded.")

    os.makedirs(output_dir, exist_ok=True)
    methods = [(1, "LETKF", "tab:gray"),
               (2, "LETKF-T2", "tab:blue"),
               (3, "LETKF-T3", "tab:red")]

    # common axis limits for fair comparison
    x = df["inflation"].to_numpy()
    y = df["loc_scale"].to_numpy()
    xr = x.max() - x.min()
    yr = y.max() - y.min()
    xpad = 0.07 * (xr if xr > 0 else 1.0)
    ypad = 0.07 * (yr if yr > 0 else 1.0)
    xlim = (x.min() - xpad, x.max() + xpad)
    ylim = (y.min() - ypad, y.max() + ypad)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300, sharex=True, sharey=True)

    for ax, (m, label, color) in zip(axes, methods):
        sub = df[df["ntemp"] == m]
        ax.scatter(sub["inflation"], sub["loc_scale"], s=12, color=color, alpha=0.35, edgecolor="none", zorder=1)

        if len(sub) >= 5:
            sns.kdeplot(
                x=sub["inflation"], y=sub["loc_scale"],
                levels=levels, fill=fill, alpha=0.25 if fill else 1.0,
                color=color, lw=1.7, thresh=0.05, ax=ax
            )

        # median marker
        if not sub.empty:
            ax.scatter(np.median(sub["inflation"]), np.median(sub["loc_scale"]),
                       marker="*", s=180, color=color, edgecolor="k", linewidth=0.6, zorder=3)

        ax.set_title(f"{label}  (N={len(sub)})", fontsize=13, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    axes[0].set_ylabel("Best localization scale", fontsize=12)
    for ax in axes:
        ax.set_xlabel("Best multiplicative inflation", fontsize=12)

    fig.suptitle("Where do best combos concentrate?  (KDE by method)", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    tag = f"{tag}_" if tag else ""
    f = os.path.join(output_dir, f"MethodContours_3panels_{tag}.png")
    plt.savefig(f, bbox_inches="tight", dpi=300)
    plt.savefig(f.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved:", f, "and .pdf")

def plot_method_hist_3panels(base_path, output_dir, Freq, Den, ObsErrs, suffix, tag="", levels=5):
    """
    Three side-by-side hist panels (LETKF, LETKF-T2, LETKF-T3), same axes.
    Each panel shows a faint scatter, KDE contours (optionally filled), and a star at the median.
    """
    df = load_best_params_table(base_path, Freq, Den, ObsErrs, suffix)
    if df.empty:
        raise ValueError("No data loaded.")

    os.makedirs(output_dir, exist_ok=True)
    methods = [(1, "LETKF", "tab:gray"),
               (2, "LETKF-T2", "tab:blue"),
               (3, "LETKF-T3", "tab:red")]

    # common axis limits for fair comparison
    x = df["inflation"].to_numpy()
    y = df["loc_scale"].to_numpy()
    xr = x.max() - x.min()
    yr = y.max() - y.min()
    xpad = 0.07 * (xr if xr > 0 else 1.0)
    ypad = 0.07 * (yr if yr > 0 else 1.0)
    xlim = (x.min() - xpad, x.max() + xpad)
    ylim = (y.min() - ypad, y.max() + ypad)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300, sharex=True, sharey=True)

    for ax, (m, label, color) in zip(axes, methods):
        sub = df[df["ntemp"] == m]
        ax.scatter(sub["inflation"], sub["loc_scale"], s=12, color=color, alpha=0.35, edgecolor="none", zorder=1)

        if len(sub) >= 5:
            sns.histplot(
                x=sub["inflation"], y=sub["loc_scale"],
                color=color, lw=1.7, thresh=0.05, ax=ax
            )

        # median marker
        if not sub.empty:
            ax.scatter(np.median(sub["inflation"]), np.median(sub["loc_scale"]),
                       marker="*", s=180, color=color, edgecolor="k", linewidth=0.6, zorder=3)

        ax.set_title(f"{label}  (N={len(sub)})", fontsize=13, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    axes[0].set_ylabel("Best localization scale", fontsize=12)
    for ax in axes:
        ax.set_xlabel("Best multiplicative inflation", fontsize=12)

    fig.suptitle("Where do best combos concentrate?  (KDE by method)", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    tag = f"{tag}_" if tag else ""
    f = os.path.join(output_dir, f"Methodhist_3panels_{tag}.png")
    plt.savefig(f, bbox_inches="tight", dpi=300)
    plt.savefig(f.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved:", f, "and .pdf")

if __name__ == "__main__":
    work_place = "hydra" # 'hydra', 'ubuntu' or 'win'
    if work_place == 'hydra':
        base_path = r'/home/jorge.gacitua/salidas/L96_multiple_experiments/data/summary'
        output_dir = r'/home/jorge.gacitua/salidas/L96_multiple_experiments/figures_Prespinup/best_params'
    elif work_place == 'ubuntu':
        base_path = r'/media/jgacitua/storage/L96_multiple_experiments/data/summary'
        output_dir = r'/media/jgacitua/storage/L96_multiple_experiments/data/figures_Prespinup/best_params'
    elif work_place == 'win':
        base_path  = r'C:\Users\jyndd\OneDrive\Doctorado\Experimentos\L96_multiple_experiments\data\summary'
        output_dir = r'C:\Users\jyndd\OneDrive\Doctorado\Experimentos\L96_multiple_experiments\figures_Prespinup\best_params'


    ObsErrs = [0.3, 1, 5, 25]
    suffix = "NOGEC_500_Prespinup200_inf1.2"

    plot_method_contours_3panels(base_path, output_dir, 4, 1.0, ObsErrs, suffix, tag="Freq4_Den1.0")
    
    plot_method_contours_3panels(base_path, output_dir, 4, 0.5, ObsErrs, suffix, tag="Freq4_Den0.5")

    plot_method_hist_3panels(base_path, output_dir, 4, 1.0, ObsErrs, suffix, tag="Freq4_Den1.0")

    plot_method_hist_3panels(base_path, output_dir, 4, 0.5, ObsErrs, suffix, tag="Freq4_Den1.0")
    