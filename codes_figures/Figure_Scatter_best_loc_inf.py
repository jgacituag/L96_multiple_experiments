import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- load & flatten all best combos from one or many ObsErr files ----------
def gather_best_param_points(base_path, Freq, Den, ObsErrs, suffix):
    """
    Returns a dict of 1D arrays with one row per (nens, alpha, method, obs_err) tuple
    where best_inflation/loc_scale are finite.
    Expects NPZ files written by your 'best_params(...)' saver (keys: loc_inflation, loc_scale, ...).
    """
    if not isinstance(ObsErrs, (list, tuple, np.ndarray)):
        ObsErrs = [ObsErrs]

    records = {
        "inflation": [], "loc_scale": [],
        "nens": [], "alpha": [], "ntemp": [], "method_label": [],
        "obserr": []
    }

    def _fmt_obserr(x):
        if isinstance(x, str):
            return x
        try:
            return f"{float(x):g}"
        except Exception:
            return str(x)

    for obs in ObsErrs:
        obs_str = _fmt_obserr(obs)
        filename = f"Best_params_Freq{Freq}_Den{Den}_ObsErr{obs_str}_{suffix}.npz"
        path = os.path.join(base_path, filename)
        if not os.path.exists(path):
            print(f"[warn] missing file: {path}")
            continue

        data = np.load(path, allow_pickle=True)
        infl = data["loc_inflation"]    # (n_nens, n_alpha, n_methods)
        locs = data["loc_scale"]        # same shape
        nens_labels = data["nens_labels"]
        alpha_range = data["alpha_range"]

        # method indices as saved (likely [1,2,3] for NTemp)
        meta = data.get("meta", None)
        if meta is not None and isinstance(meta.item(), dict):
            methods = meta.item().get("methods", [1,2,3])
        else:
            # fallback to 1..K
            methods = list(range(1, infl.shape[2] + 1))

        # labeling map
        def method_label(m):
            return {1: "LETKF", 2: "LETKF-T2", 3: "LETKF-T3"}.get(int(m), f"NTemp{m}")

        for i_nens, nens in enumerate(nens_labels):
            for i_alpha, alpha in enumerate(alpha_range):
                for k, m in enumerate(methods):
                    x = infl[i_nens, i_alpha, k]
                    y = locs[i_nens, i_alpha, k]
                    if not np.isfinite(x) or not np.isfinite(y):
                        continue
                    records["inflation"].append(float(x))
                    records["loc_scale"].append(float(y))
                    records["nens"].append(int(nens))
                    records["alpha"].append(int(alpha))
                    records["ntemp"].append(int(m))
                    records["method_label"].append(method_label(m))
                    records["obserr"].append(_fmt_obserr(obs))

    # convert lists -> numpy arrays for easy seaborn feeding
    for k in list(records.keys()):
        records[k] = np.array(records[k], dtype=object)
    return records


# ---------- scatter plotter (color_by in {'nens','alpha','obserr'}) ----------
def plot_scatter_best_params(records, output_dir, tag, color_by="nens"):
    """
    Scatter of (inflation, loc_scale):
      color = color_by ('nens' | 'alpha' | 'obserr')
      marker shape = tempering steps (ntemp 1/2/3)
    Saves PDF+PNG with 'tag' in filename.
    """
    if len(records["inflation"]) == 0:
        raise ValueError("No points to plot. Did the NPZ files exist and contain finite best params?")

    os.makedirs(output_dir, exist_ok=True)

    x = records["inflation"].astype(float)
    y = records["loc_scale"].astype(float)
    ntemp = records["ntemp"].astype(int)
    color_vals = records[color_by]

    # choose palette: categorical for 'obserr' or small unique counts; otherwise continuous-like
    unique_vals = np.unique(color_vals)
    if color_by == "obserr" or len(unique_vals) <= 10:
        palette = sns.color_palette("tab10", n_colors=len(unique_vals))
    else:
        palette = "viridis"

    # markers per tempering steps
    marker_map = {1: "o", 2: "s", 3: "^"}
    markers = [marker_map.get(int(m), "o") for m in ntemp]

    # build style arrays for seaborn
    # seaborn scatterplot can take arrays for hue and style directly
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)
    sns.scatterplot(
        x=x, y=y,
        hue=color_vals,
        style=ntemp,
        markers=marker_map,
        palette=palette,
        edgecolor="black", linewidth=0.5, s=70, ax=ax
    )

    ax.set_xlabel("Best multiplicative inflation", fontsize=13)
    ax.set_ylabel("Best localization scale", fontsize=13)
    title_map = {
        "nens": "colored by Ensemble Size ($N_{ens}$)",
        "alpha": r"colored by Tempering slope ($\alpha_s$)",
        "obserr": "colored by Observation Error"
    }
    ax.set_title(f"Best (inflation, localization) combos — {title_map.get(color_by, color_by)}",
                 fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    # improve legends: split hue and style legends
    handles, labels = ax.get_legend_handles_labels()
    # seaborn mixes them; let’s keep the default but tighten layout
    plt.tight_layout()

    fname = f"BestCombos_scatter_{tag}_colorby-{color_by}"
    pdf = os.path.join(output_dir, f"{fname}.pdf")
    png = os.path.join(output_dir, f"{fname}.png")
    plt.savefig(pdf, bbox_inches="tight", format="pdf")
    plt.savefig(png, bbox_inches="tight", format="png")
    plt.close(fig)
    print(f"Scatter saved:\n  PDF: {pdf}\n  PNG: {png}")


# ---------- convenience driver to make 3 figures with different color encodings ----------
def scatter_best_params_three_figs(base_path, output_dir, Freq, Den, ObsErrs, suffix, tag="all"):
    """
    Builds three figures from all provided ObsErrs combined:
      1) color by Nens
      2) color by alpha (tempering slope)
      3) color by ObsErr
    """
    rec = gather_best_param_points(base_path, Freq, Den, ObsErrs, suffix)
    plot_scatter_best_params(rec, output_dir, tag=tag, color_by="nens")
    plot_scatter_best_params(rec, output_dir, tag=tag, color_by="alpha")
    plot_scatter_best_params(rec, output_dir, tag=tag, color_by="obserr")


if __name__ == "__main__":
    work_place = "hydra" # 'hydra', 'ubuntu' or 'win'
    if work_place == 'hydra':
        base_path = r'/home/jorge.gacitua/salidas/L96_multiple_experiments/data/summary'
        output_dir = r'/home/jorge.gacitua/salidas/L96_multiple_experiments/figures_Prespinup/scatter_best_params'
    elif work_place == 'ubuntu':
        base_path = r'/media/jgacitua/storage/L96_multiple_experiments/data/summary'
        output_dir = r'/media/jgacitua/storage/L96_multiple_experiments/data/figures_Prespinup/scatter_best_params'
    elif work_place == 'win':
        base_path  = r'C:\Users\jyndd\OneDrive\Doctorado\Experimentos\L96_multiple_experiments\data\summary'
        output_dir = r'C:\Users\jyndd\OneDrive\Doctorado\Experimentos\L96_multiple_experiments\figures_Prespinup\scatter_best_params'

    #Freq = 4
    #Den = 1.0
    #ObseErrs = np.array(['0.3','1','5','25'])
 
    #suffix = 'NOGEC_500_Prespinup200_inf1.2'

    ObsErrs = [0.3, 1, 5, 25]
    scatter_best_params_three_figs(
        base_path, output_dir,
        Freq=4, Den=1.0, ObsErrs=ObsErrs,
        suffix="NOGEC_500_Prespinup200_inf1.2",
        tag="Freq4_Den1.0"
)


