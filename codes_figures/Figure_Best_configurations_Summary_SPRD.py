import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_best_params(base_path, output_dir, Freq, Den, ObsErr, suffix, percentual=True):
    """
    Loads Best_params_*.npz and produces:
      - Two heatmaps: (LETKF-T2 − LETKF) and (LETKF-T3 − LETKF), % by default
      - Two bar charts with SPRD means (over alpha, over Nens)

    """

    def _clean_ticklabels(arr):
        # Pretty ints where possible, otherwise compact float text
        labels = []
        for v in np.asarray(arr):
            try:
                f = float(v)
                labels.append(int(f) if f.is_integer() else float(f))
            except Exception:
                labels.append(str(v))
        return labels

    # --- locate file ---------------------------------------------------------
 
    filename = f"Best_params_Freq{Freq}_Den{Den}_ObsErr{ObsErr}_{suffix}.npz"
    npz_path = os.path.join(base_path, filename)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    # Expecting keys present in your NPZ
    sprd = data["sprd_values"]            # shape (n_nens, n_alpha, 3)
    nens_labels = data["nens_labels"]     # will be used on x-axis of heatmaps/bars
    alpha_vals  = data["alpha_range"]

    # basic shape checks (fail fast if file content mismatches)
    if sprd.ndim != 3 or sprd.shape[2] < 3:
        raise ValueError("sprd_values must have shape (n_nens, n_alpha, 3).")

    # Extract methods
    letkf    = sprd[:, :, 0]  # (nens, alpha)
    letkf_t2 = sprd[:, :, 1]
    letkf_t3 = sprd[:, :, 2]

    # Differences (percent or absolute), NaN-safe
    if percentual:
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_t2 = (letkf_t2 - letkf) / letkf * 100.0
            pct_t3 = (letkf_t3 - letkf) / letkf * 100.0
            pct_t2 = np.where(letkf == 0, np.nan, pct_t2)
            pct_t3 = np.where(letkf == 0, np.nan, pct_t3)
        diff_t2, diff_t3 = pct_t2, pct_t3
        cbar_label = "Difference [%]"
    else:
        diff_t2 = letkf_t2 - letkf
        diff_t3 = letkf_t3 - letkf
        cbar_label = "Difference (sprd)"

    # Symmetric color limits around zero, data-driven
    max_abs = np.nanmax(np.abs(np.concatenate([diff_t2.ravel(), diff_t3.ravel()])))
    if not np.isfinite(max_abs) or max_abs == 0:
        vmax = 1.0
    else:
        # round up to a “nice” bound
        step = 5.0 if percentual else 0.1
        vmax = step * np.ceil(max_abs / step)
    vmin = -vmax

    # --- Figure --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=300)
    axes = axes.flatten()

    # Clean ticks for aesthetics
    nens_ticklabels  = _clean_ticklabels(nens_labels)
    alpha_ticklabels = _clean_ticklabels(alpha_vals)

    # Heatmaps use transpose so axes are (alpha, nens)
    hm_t2 = sns.heatmap(
        diff_t2.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=vmin, vmax=vmax, xticklabels=nens_ticklabels, yticklabels=alpha_ticklabels,
        ax=axes[0], cbar=False, linewidths=.5, annot_kws={"fontsize": 10, "fontweight": "bold"}
    )
    axes[0].set_title("LETKF-T2 vs LETKF", fontsize=16, fontweight='bold')
    axes[0].set_xlabel("Ensemble Size", fontsize=14)
    axes[0].set_ylabel(r"Tempering slope [$\alpha_s$]", fontsize=14)
    axes[0].invert_yaxis()

    hm_t3 = sns.heatmap(
        diff_t3.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=vmin, vmax=vmax, xticklabels=nens_ticklabels, yticklabels=alpha_ticklabels,
        ax=axes[1], cbar=False, linewidths=.5, annot_kws={"fontsize": 10, "fontweight": "bold"}
    )
    axes[1].set_title("LETKF-T3 vs LETKF", fontsize=16, fontweight='bold')
    axes[1].set_xlabel("Ensemble Size", fontsize=14)
    axes[1].set_ylabel(r"Tempering slope [$\alpha_s$]", fontsize=14)
    axes[1].invert_yaxis()

    # Single colorbar shared with right heatmap
    
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(hm_t3.collections[0], cax=cax)
    cbar.set_label(cbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Bar chart: mean over alpha (axis=1)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    mean_sprd_letkf    = np.nanmean(letkf, axis=1)     # mean over alpha
    mean_sprd_letkf_t2 = np.nanmean(letkf_t2, axis=1)  # mean over alpha
    mean_sprd_letkf_t3 = np.nanmean(letkf_t3, axis=1)  # mean over alpha
    x = np.arange(len(nens_labels)) * 4
    axes[2].bar(x,     mean_sprd_letkf,    color='gray',  label='LETKF')
    axes[2].bar(x + 1, mean_sprd_letkf_t2, color='Teal',  label='LETKF-T2')
    axes[2].bar(x + 2, mean_sprd_letkf_t3, color='Maroon',label='LETKF-T3')
    axes[2].set_xticks(x + 1)
    axes[2].set_xticklabels(nens_ticklabels)
    axes[2].set_ylabel("Spread", fontsize=14)
    axes[2].set_title(r"Mean over Tempering slope [$\alpha_s$]", fontsize=14, fontweight='bold')
    axes[2].set_xlabel(r"Ensemble Size [$N_{ens}$]", fontsize=14)
    axes[2].set_ylim(0, 3.6)
    axes[2].legend(loc='upper right', fontsize=10, frameon=False)

    # Bar chart: mean over ensemble size (axis=0)
    axes[3].grid(True, linestyle='--', alpha=0.5)
    mean_sprd_letkf    = np.nanmean(letkf, axis=0)     # mean over Nens
    mean_sprd_letkf_t2 = np.nanmean(letkf_t2, axis=0)  # mean over Nens
    mean_sprd_letkf_t3 = np.nanmean(letkf_t3, axis=0)  # mean over Nens
    x = np.arange(len(alpha_vals)) * 4
    axes[3].bar(x,     mean_sprd_letkf,    color='gray',  label='LETKF')
    axes[3].bar(x + 1, mean_sprd_letkf_t2, color='Teal',  label='LETKF-T2')
    axes[3].bar(x + 2, mean_sprd_letkf_t3, color='Maroon',label='LETKF-T3')
    axes[3].set_xticks(x + 1)
    axes[3].set_xticklabels(alpha_ticklabels)
    axes[3].set_ylabel("Spread", fontsize=14)
    axes[3].set_title(r"Mean over Ensemble Size [$N_{ens}$]", fontsize=14, fontweight='bold')
    axes[3].set_xlabel(r"Tempering slope [$\alpha_s$]", fontsize=14)
    axes[3].set_ylim(0, 3.6)
    axes[3].legend(loc='upper right', fontsize=10, frameon=False)

    # Save
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    figname = f"Best_params_Freq{Freq}_Den{Den}_ObsErr{ObsErr}_{suffix}"
    output_pdf = os.path.join(output_dir, f"{figname}.pdf")
    output_png = os.path.join(output_dir, f"{figname}.png")
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_png, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"Figure saved as:\n  PDF: {output_pdf}\n  PNG: {output_png}")

def plot_best_params_heatmaps(base_path, output_dir, Freq, Den, ObsErr, suffix, percentual=True):
    """
    Create ONLY the two heatmaps (LETKF-T2 vs LETKF, LETKF-T3 vs LETKF).
    Saves as ..._heatmaps.(pdf|png)
    """
    # --- small helpers (local to keep this function standalone) -------------

    def _clean_ticklabels(arr):
        labels = []
        for v in np.asarray(arr):
            try:
                f = float(v)
                labels.append(int(f) if f.is_integer() else float(f))
            except Exception:
                labels.append(str(v))
        return labels

    # --- load ---------------------------------------------------------------
    filename = f"Best_params_Freq{Freq}_Den{Den}_ObsErr{ObsErr}_{suffix}.npz"
    npz_path = os.path.join(base_path, filename)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    sprd         = data["sprd_values"]      # (n_nens, n_alpha, 3)
    nens_labels  = data["nens_labels"]
    alpha_vals   = data["alpha_range"]

    letkf    = sprd[:, :, 0]
    letkf_t2 = sprd[:, :, 1]
    letkf_t3 = sprd[:, :, 2]

    # Differences
    if percentual:
        with np.errstate(divide="ignore", invalid="ignore"):
            diff_t2 = (letkf_t2 - letkf) / letkf * 100.0
            diff_t3 = (letkf_t3 - letkf) / letkf * 100.0
            diff_t2 = np.where(letkf == 0, np.nan, diff_t2)
            diff_t3 = np.where(letkf == 0, np.nan, diff_t3)
        cbar_label = "Difference [%]"
        step = 5.0
    else:
        diff_t2 = letkf_t2 - letkf
        diff_t3 = letkf_t3 - letkf
        cbar_label = "Difference (sprd)"
        step = 0.1

    # symmetric color scale
    max_abs = np.nanmax(np.abs(np.concatenate([diff_t2.ravel(), diff_t3.ravel()])))
    vmax = step * np.ceil((max_abs if np.isfinite(max_abs) and max_abs > 0 else 1.0) / step)
    vmin = -vmax

    # ticks
    xticklabels = _clean_ticklabels(nens_labels)
    yticklabels = _clean_ticklabels(alpha_vals)

    # --- figure (only heatmaps) --------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
    ax0, ax1 = axes

    hm_t2 = sns.heatmap(
        diff_t2.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=vmin, vmax=vmax, xticklabels=xticklabels, yticklabels=yticklabels,
        ax=ax0, cbar=False, linewidths=.5, annot_kws={"fontsize": 10, "fontweight": "bold"}
    )
    ax0.set_title("LETKF-T2 vs LETKF", fontsize=16, fontweight="bold")
    ax0.set_xlabel("Ensemble Size", fontsize=14)
    ax0.set_ylabel(r"Tempering slope [$\alpha_s$]", fontsize=14)
    ax0.invert_yaxis()

    hm_t3 = sns.heatmap(
        diff_t3.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=vmin, vmax=vmax, xticklabels=xticklabels, yticklabels=yticklabels,
        ax=ax1, cbar=True, cbar_kws={"label": cbar_label}, linewidths=.5,
        annot_kws={"fontsize": 10, "fontweight": "bold"}
    )
    ax1.set_title("LETKF-T3 vs LETKF", fontsize=16, fontweight="bold")
    ax1.set_xlabel("Ensemble Size", fontsize=14)
    ax1.set_ylabel(r"Tempering slope [$\alpha_s$]", fontsize=14)
    ax1.invert_yaxis()

    # adjust
    cbar = hm_t3.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    figname = f"Best_params_Freq{Freq}_Den{Den}_ObsErr{ObsErr}_{suffix}_heatmaps"
    pdf = os.path.join(output_dir, f"{figname}.pdf")
    png = os.path.join(output_dir, f"{figname}.png")
    plt.savefig(pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    print(f"Heatmaps saved:\n  PDF: {pdf}\n  PNG: {png}")


def plot_best_params_bars(base_path, output_dir, Freq, Den, ObsErr, suffix, ylim=(0, 3.6)):
    """
    Create ONLY the two bar charts:
      - Mean sprd over alpha (per Nens)
      - Mean sprd over Nens (per alpha)
    Saves as ..._bars.(pdf|png)
    """
    def _clean_ticklabels(arr):
        labels = []
        for v in np.asarray(arr):
            try:
                f = float(v)
                labels.append(int(f) if f.is_integer() else float(f))
            except Exception:
                labels.append(str(v))
        return labels


    filename = f"Best_params_Freq{Freq}_Den{Den}_ObsErr{ObsErr}_{suffix}.npz"
    npz_path = os.path.join(base_path, filename)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    sprd         = data["sprd_values"]      # (n_nens, n_alpha, 3)
    nens_labels  = data["nens_labels"]
    alpha_vals   = data["alpha_range"]

    letkf    = sprd[:, :, 0]
    letkf_t2 = sprd[:, :, 1]
    letkf_t3 = sprd[:, :, 2]

    # means
    mean_over_alpha_letkf    = np.nanmean(letkf,    axis=1)
    mean_over_alpha_t2       = np.nanmean(letkf_t2, axis=1)
    mean_over_alpha_t3       = np.nanmean(letkf_t3, axis=1)

    mean_over_nens_letkf     = np.nanmean(letkf,    axis=0)
    mean_over_nens_t2        = np.nanmean(letkf_t2, axis=0)
    mean_over_nens_t3        = np.nanmean(letkf_t3, axis=0)

    # ticks
    nens_ticklabels  = _clean_ticklabels(nens_labels)
    alpha_ticklabels = _clean_ticklabels(alpha_vals)

    # --- figure (only bars) -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    # Left: mean over alpha (x = Nens)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    x0 = np.arange(len(nens_labels)) * 4
    axes[0].bar(x0,     mean_over_alpha_letkf, color="gray",   label="LETKF")
    axes[0].bar(x0 + 1, mean_over_alpha_t2,    color="Teal",   label="LETKF-T2")
    axes[0].bar(x0 + 2, mean_over_alpha_t3,    color="Maroon", label="LETKF-T3")
    axes[0].set_xticks(x0 + 1)
    axes[0].set_xticklabels(nens_ticklabels)
    axes[0].set_ylabel("Spread", fontsize=14)
    axes[0].set_title(r"Mean over Tempering slope [$\alpha_s$]", fontsize=14, fontweight="bold")
    axes[0].set_xlabel(r"Ensemble Size [$N_{ens}$]", fontsize=14)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].legend(loc="upper right", fontsize=10, frameon=False)

    # Right: mean over Nens (x = alpha)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    x1 = np.arange(len(alpha_vals)) * 4
    axes[1].bar(x1,     mean_over_nens_letkf, color="gray",   label="LETKF")
    axes[1].bar(x1 + 1, mean_over_nens_t2,    color="Teal",   label="LETKF-T2")
    axes[1].bar(x1 + 2, mean_over_nens_t3,    color="Maroon", label="LETKF-T3")
    axes[1].set_xticks(x1 + 1)
    axes[1].set_xticklabels(alpha_ticklabels)
    axes[1].set_ylabel("Spread", fontsize=14)
    axes[1].set_title(r"Mean over Ensemble Size [$N_{ens}$]", fontsize=14, fontweight="bold")
    axes[1].set_xlabel(r"Tempering slope [$\alpha_s$]", fontsize=14)
    if ylim is not None:
        axes[1].set_ylim(*ylim)
    axes[1].legend(loc="upper right", fontsize=10, frameon=False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    figname = f"Best_params_Freq{Freq}_Den{Den}_ObsErr{ObsErr}_{suffix}_bars"
    pdf = os.path.join(output_dir, f"{figname}.pdf")
    png = os.path.join(output_dir, f"{figname}.png")
    plt.savefig(pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    print(f"Bar charts saved:\n  PDF: {pdf}\n  PNG: {png}")


if __name__ == "__main__":
    work_place = "hydra" # 'hydra', 'ubuntu' or 'win'
    if work_place == 'hydra':
        base_path = r'/home/jorge.gacitua/salidas/L96_multiple_experiments/data/summary'
        output_dir = r'/home/jorge.gacitua/salidas/L96_multiple_experiments/figures_Prespinup/Best_params_sprd'
    elif work_place == 'ubuntu':
        base_path = r'/media/jgacitua/storage/L96_multiple_experiments/data/summary'
        output_dir = r'/media/jgacitua/storage/L96_multiple_experiments/data/figures_Prespinup/Best_params_sprd'
    elif work_place == 'win':
        base_path  = r'C:\Users\jyndd\OneDrive\Doctorado\Experimentos\L96_multiple_experiments\data\summary'
        output_dir = r'C:\Users\jyndd\OneDrive\Doctorado\Experimentos\L96_multiple_experiments\figures_Prespinup\Best_params_sprd'

    Freq = 4
    Den = 0.5
    ObseErrs = np.array(['0.3','1','5','25'])
 
    suffix = 'NOGEC_500_Prespinup200_inf1.2'

    for ObsErr in ObseErrs:
        plot_best_params(base_path, output_dir,Freq,Den,ObsErr,suffix)
        plot_best_params_heatmaps(base_path, output_dir,Freq,Den,ObsErr,suffix)
        plot_best_params_bars(base_path, output_dir,Freq,Den,ObsErr,suffix)


