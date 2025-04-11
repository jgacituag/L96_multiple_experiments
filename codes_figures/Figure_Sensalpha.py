import matplotlib.pyplot as plt
import numpy as np
import os
import common_function as cf
# Configuración
ntemps = [1, 2, 3]
labels = ['LETKF', 'LETKF-T2', 'LETKF-T3']
base_path  = '/home/jorge.gacitua/datosmunin/L96_multiple_experiments/data'
output_dir = '/home/jorge.gacitua/experimentos/L96_multiple_experiments/figures'
freq = 4
den = 1.0
alpha = 2.0
ObseErr = '5'
gec = '_NO_GEC'
fig, axes = plt.subplots(len(nens), len(ntemps), figsize=(12, 12), 
                         sharex=False, sharey=False, 
                         gridspec_kw={'hspace': 0.25, 'wspace': 0.1})


cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])  # x, y, width, height

vmin, vmax = 0, 4
panel_label_counter = 0
alphas = [0,1,2,3]
nen = 20



for i, alpha in enumerate(alphas):
    for j, ntemp in enumerate(ntemps):
        # Load the data
        filename = f'LETKF_Paper_Nature_Freq{freq}_Den{den}_Type3_{ObseErr}_Nens{nen}_NTemp{ntemp}_alpha{alpha}{gec}.npz'
        filepath = os.path.join(base_path, filename)
        print(filepath)
        # Check if the file exists
        if os.path.exists(filepath):
            data = np.load(filepath, allow_pickle=True)
            #results = data['results']
            mult_inf_range = data['MultInfRange']#data['mult_inf_range']
            loc_scale_range = data['LocScaleRange']#data['loc_scale_range']

            ax = axes[i, j]
            total_analysis_rmse = data['total_analysis_rmse'][:,:,j,0,i]
            NormalEnd = np.zeros((len(mult_inf_range), len(loc_scale_range)))
            for ii in range(len(mult_inf_range)):
                for jj in range(len(loc_scale_range)):
                    NormalEnd[ii,jj] = np.any(np.isnan(data['XAMean'][ii,jj,j,0,2,:,:]))
                    #NormalEnd[ii,jj,kk] = np.any(Output['XAMean'][ii,jj,kk,iens,ialpha,:,:]==0.)
                    NormalEnd[ii,jj] = np.any(data['total_analysis_rmse'][ii,jj,j,0,2]>=7.)
            NormalEnd=NormalEnd.astype(bool)
            total_analysis_rmse[NormalEnd] = np.nan
            total_analysis_rmse = cf.outlier_rmse_filter( total_analysis_rmse )
            min_rmse = np.nanmin(total_analysis_rmse)
            idx_min = np.where(total_analysis_rmse == min_rmse)
            best_mult_inf = mult_inf_range[idx_min[0][0]]
            best_loc_scale = loc_scale_range[idx_min[1][0]]

            im = ax.pcolormesh(mult_inf_range, loc_scale_range, total_analysis_rmse.T, 
                                vmin=vmin, vmax=vmax, cmap='PuBu',edgecolors='lightgray',linewidth=0.001)
            ax.plot(best_mult_inf, best_loc_scale, 'k.', markersize=6)

            panel_label = rf"$\bf{{{chr(97 + panel_label_counter)}}}$"
            ax.set_title(f"({panel_label}) - Min. RMSE={min_rmse:.2f}", fontsize=10)
            panel_label_counter += 1

            if j == 0:
                ax.set_ylabel('Localization Scale', fontsize=10)
            if i == len(alphas) - 1:
                ax.set_xlabel('Multiplicative Inflation', fontsize=10)

for i, alpha in enumerate(alphas):
    axes[i, 0].annotate(rf"$\bf{{\alpha_s = {alpha}}}$", xy=(-0.15, 0.5), xycoords='axes fraction',
                        rotation=90, va='center', ha='right', fontsize=14, weight='bold')
for j, t in enumerate(ntemps):
    axes[0, j].annotate(labels[j], xy=(0.5, 1.15), xycoords='axes fraction',
                        va='center', ha='center', fontsize=14, weight='bold')
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Total Analysis RMSE', fontsize=14)

os.makedirs(output_dir, exist_ok=True)
figname = f"Figure_Alpha_AnalysisRMSE_Freq{freq}_Den{den}_Type3"
output_pdf = os.path.join(output_dir, f"{figname}.pdf")
output_png = os.path.join(output_dir, f"{figname}.png")

plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(output_png, dpi=300, bbox_inches='tight', format='png')
#plt.show()

print(f"Figure saved as: \n PDF: {output_pdf} \n PNG: {output_png}")