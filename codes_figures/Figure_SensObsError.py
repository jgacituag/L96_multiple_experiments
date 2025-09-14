import matplotlib.pyplot as plt
import numpy as np
import os
import common_function as cf
import matplotlib.colors as mcolors

def plot_RMSE_sensitivity_to_observations_errors(base_path,output_dir,nens,freq,den,alpha,gec,vmin=0,vmax=2):

    obs_errs = ['0.3', '1', '5', '25']
    ntemps = [1, 2, 3]
    labels = ['LETKF', 'LETKF-T2', 'LETKF-T3']

    fig, axes = plt.subplots(len(obs_errs), len(ntemps), figsize=(14, 13), 
                            sharex=False, sharey=False, 
                            gridspec_kw={'hspace': 0.3, 'wspace': 0.1})

    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])  # x, y, width, height

    panel_label_counter = 0
    output_dir = f"{output_dir}/RMSE_OBS_ERR_Sensitivity"
    os.makedirs(output_dir, exist_ok=True)

    for i, obs in enumerate(obs_errs):
        for j, ntemp in enumerate(ntemps):
            ax = axes[i, j]
            filename = f'LETKF_Paper_Nature_Freq{freq}_Den{den}_Type3_ObsErr{obs}_Nens{nens}_NTemp{ntemp}_alpha{alpha}_{gec}_500_Prespinup200_inf1.2.npz'
            filepath = os.path.join(base_path, filename)
            if not os.path.exists(filepath):
                ax.text(0.5, 0.5, 'file not found', ha='center', va='center', fontsize=10, alpha=0.6)
                ax.set_axis_off()
                panel_label_counter += 1
                continue

            data = np.load(filepath, allow_pickle=True)

            mult_inf_range = data.get('mult_inf_range', np.arange(1.0, 1.6, 0.05))
            loc_scale_range = data.get('loc_scale_range', np.arange(0.5, 5.0, 0.5))

            total_analysis_rmse = data['total_analysis_rmse']

            NormalEnd = (1 - data['NormalEnd']).astype(bool)
            total_analysis_rmse[NormalEnd] = np.nan
            total_analysis_rmse = cf.outlier_rmse_filter( total_analysis_rmse )

            min_rmse = np.nanmin(total_analysis_rmse)
            idx_min = np.where(total_analysis_rmse == min_rmse)
            best_mult_inf = mult_inf_range[idx_min[0][0]]
            best_loc_scale = loc_scale_range[idx_min[1][0]]

            total_analysis_rmse_smooth = cf.smooth_filter(total_analysis_rmse, size=5)
            
            min_rmse_smooth = np.nanmin(total_analysis_rmse_smooth)
            idx_min_smooth = np.where(total_analysis_rmse_smooth == min_rmse_smooth)
            best_mult_inf_smooth = mult_inf_range[idx_min_smooth[0][0]]
            best_loc_scale_smooth = loc_scale_range[idx_min_smooth[1][0]]

            levels = np.arange(vmin, vmax + 0.1, 0.1)  # From 0 to 2, step 0.25
            norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256)

            im = ax.pcolormesh(mult_inf_range, loc_scale_range, total_analysis_rmse_smooth.T,
                            cmap='YlGn', norm=norm, edgecolors='lightgray', linewidth=0.001)            
            ax.plot(best_mult_inf_smooth, best_loc_scale_smooth, 'w.', markersize=12, markeredgecolor='black',markeredgewidth=0.5)

            nan_mask = np.isnan(total_analysis_rmse_smooth.T)
            for (y, x), is_nan in np.ndenumerate(nan_mask):
                if is_nan:
                    xi = mult_inf_range[x]
                    yi = loc_scale_range[y]
                    ax.text(xi, yi, '×', ha='center', va='center', color='gray', fontsize=12, fontweight='bold')

            panel_label = rf"$\bf{{{chr(97 + panel_label_counter)}}}$"
            ax.set_title(f"({panel_label}) - Min. RMSE={min_rmse_smooth:.2f}", fontsize=9)

            panel_label_counter += 1

            if j == 0:
                ax.set_ylabel('Localization Scale', fontsize=10)
            if i == len(obs_errs) - 1:
                ax.set_xlabel('Multiplicative Inflation', fontsize=10)

    for i, obs in enumerate(obs_errs):
        axes[i, 0].annotate(rf"$\bf{{\sigma^2_0 = {obs}}}$", xy=(-0.15, 0.5), xycoords='axes fraction',
                            rotation=90, va='center', ha='right', fontsize=14, weight='bold')
    for j, t in enumerate(ntemps):
        axes[0, j].annotate(labels[j], xy=(0.5, 1.15), xycoords='axes fraction',
                            va='center', ha='center', fontsize=14, weight='bold')
    cbar = fig.colorbar(im, cax=cbar_ax,ticks=np.arange(vmin, vmax + 0.2, 0.2))
    cbar.set_label('Total Analysis RMSE', fontsize=14)

    
    figname = f"RMSE_ObsError_Sensitivity_Nens{nens}_Freq{freq}_Den{den}_Type3_PTemp{alpha}_{gec}"
    output_pdf = os.path.join(output_dir, f"{figname}.pdf")
    output_png = os.path.join(output_dir, f"{figname}.png")

    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_png, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"Figure saved as: \n PDF: {output_pdf} \n PNG: {output_png}")

def plot_RMSE_sensitivity_to_observations_errors_wide(base_path,output_dir,nens,freq,den,alpha,gec,vmin=0,vmax=2):

    obs_errs = ['0.3', '1', '5', '25']
    ntemps = [1, 2, 3]
    labels = ['LETKF', 'LETKF-T2', 'LETKF-T3']

    fig, axes = plt.subplots(len(ntemps), len(obs_errs), figsize=(18, 8), 
                            sharex=False, sharey=False, 
                            gridspec_kw={'hspace': 0.3, 'wspace': 0.15})

    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])  # x, y, width, height
    output_dir = f"{output_dir}/RMSE_OBS_ERR_Sensitivity_WIDE"
    panel_label_counter = 0
    os.makedirs(output_dir, exist_ok=True)

    for i, ntemp in enumerate(ntemps):
        for j, obs in enumerate(obs_errs):
            ax = axes[i, j]
            filename = f'LETKF_Paper_Nature_Freq{freq}_Den{den}_Type3_ObsErr{obs}_Nens{nens}_NTemp{ntemp}_alpha{alpha}_{gec}_500_Prespinup200_inf1.2.npz'
            filepath = os.path.join(base_path, filename)
            if not os.path.exists(filepath):
                ax.text(0.5, 0.5, 'file not found', ha='center', va='center', fontsize=10, alpha=0.6)
                ax.set_axis_off()
                panel_label_counter += 1
                continue

            data = np.load(filepath, allow_pickle=True)

            mult_inf_range = data.get('mult_inf_range', np.arange(1.0, 1.6, 0.05))
            loc_scale_range = data.get('loc_scale_range', np.arange(0.5, 5.0, 0.5))

            total_analysis_rmse = data['total_analysis_rmse']

            NormalEnd = (1 - data['NormalEnd']).astype(bool)
            total_analysis_rmse[NormalEnd] = np.nan
            total_analysis_rmse = cf.outlier_rmse_filter( total_analysis_rmse )

            min_rmse = np.nanmin(total_analysis_rmse)
            idx_min = np.where(total_analysis_rmse == min_rmse)
            best_mult_inf = mult_inf_range[idx_min[0][0]]
            best_loc_scale = loc_scale_range[idx_min[1][0]]

            total_analysis_rmse_smooth = cf.smooth_filter(total_analysis_rmse, size=5)
            
            min_rmse_smooth = np.nanmin(total_analysis_rmse_smooth)
            idx_min_smooth = np.where(total_analysis_rmse_smooth == min_rmse_smooth)
            best_mult_inf_smooth = mult_inf_range[idx_min_smooth[0][0]]
            best_loc_scale_smooth = loc_scale_range[idx_min_smooth[1][0]]

            levels = np.arange(vmin, vmax + 0.1, 0.1)  # From 0 to 2, step 0.25
            norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=256)

            im = ax.pcolormesh(mult_inf_range, loc_scale_range, total_analysis_rmse_smooth.T,
                            cmap='YlGn', norm=norm, edgecolors='lightgray', linewidth=0.001)            
            ax.plot(best_mult_inf_smooth, best_loc_scale_smooth, 'w.', markersize=12, markeredgecolor='black',markeredgewidth=0.5)

            nan_mask = np.isnan(total_analysis_rmse_smooth.T)
            for (y, x), is_nan in np.ndenumerate(nan_mask):
                if is_nan:
                    xi = mult_inf_range[x]
                    yi = loc_scale_range[y]
                    ax.text(xi, yi, '×', ha='center', va='center', color='gray', fontsize=12, fontweight='bold')

            panel_label = rf"$\bf{{{chr(97 + panel_label_counter)}}}$"
            ax.set_title(f"({panel_label}) - Min. RMSE={min_rmse_smooth:.2f}", fontsize=9)

            panel_label_counter += 1

            if j == 0:
                ax.set_ylabel('Localization Scale', fontsize=10)
            if i == len(obs_errs) - 1:
                ax.set_xlabel('Multiplicative Inflation', fontsize=10)

    for j, obs in enumerate(obs_errs):
        axes[0, j].annotate(rf"$\bf{{\sigma^2_0 = {obs}}}$", xy=(0.5, 1.12), xycoords='axes fraction',
                            va='bottom', ha='center', fontsize=14, weight='bold')
    for i, t in enumerate(ntemps):
        axes[i, 0].annotate(labels[i], xy=(-0.2, 0.5), xycoords='axes fraction',rotation=90, 
                            va='center', ha='center', fontsize=14, weight='bold')
    cbar = fig.colorbar(im, cax=cbar_ax,ticks=np.arange(vmin, vmax + 0.2, 0.2))
    cbar.set_label('Total Analysis RMSE', fontsize=14)

    
    figname = f"RMSE_ObsError_Sensitivity_Nens{nens}_Freq{freq}_Den{den}_Type3_PTemp{alpha}_{gec}_wide"
    output_pdf = os.path.join(output_dir, f"{figname}.pdf")
    output_png = os.path.join(output_dir, f"{figname}.png")

    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_png, dpi=300, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"Figure saved as: \n PDF: {output_pdf} \n PNG: {output_png}")


work_place = "hydra" # 'hydra', 'ubuntu' or 'win'
if work_place == 'hydra':
    base_path = r'/home/jorge.gacitua/salidas/L96_multiple_experiments/data/LETKF'
    output_dir = r'/home/jorge.gacitua/salidas/L96_multiple_experiments/figures_Prespinup'
elif work_place == 'ubuntu':
    base_path = r'/media/jgacitua/storage/L96_multiple_experiments/data/LETKF'
    output_dir = r'/media/jgacitua/storage/L96_multiple_experiments/figures_Prespinup'
elif work_place == 'win':
    base_path  = r'C:\Users\jyndd\OneDrive\Doctorado\Experimentos\L96_multiple_experiments\data\LETKF'
    output_dir = r'C:\Users\jyndd\OneDrive\Doctorado\Experimentos\L96_multiple_experiments\figures_Prespinup'
# Configuración


freq = 4
den = 0.5
gec = 'NOGEC'
alphas = [0,1,2,3]
nens = [10,20,40,60,80]
for nen in nens:
    for alpha in alphas:
        plot_RMSE_sensitivity_to_observations_errors(base_path,output_dir,nen,freq,den,alpha,gec,vmin=0,vmax=2)
        plot_RMSE_sensitivity_to_observations_errors_wide(base_path,output_dir,nen,freq,den,alpha,gec,vmin=0,vmax=2)

