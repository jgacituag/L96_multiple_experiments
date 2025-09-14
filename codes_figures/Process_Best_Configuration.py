import numpy as np
import os
import common_function as cf

def best_params(base_path, output_dir,Freq,Den,ObsErr,nen_range,alpha_range,
                suffix,method_indices = (1, 2, 3),smooth_size = 5):

    n_alpha = len(alpha_range)
    n_nens = len(nen_range)

    best_inflation = np.full((n_nens, n_alpha, len(method_indices)), np.nan, dtype=float)
    best_localization = np.full((n_nens, n_alpha, len(method_indices)), np.nan, dtype=float)
    rmse_values = np.full((n_nens, n_alpha, len(method_indices)), np.nan, dtype=float)
    sprd_values = np.full((n_nens, n_alpha, len(method_indices)), np.nan, dtype=float)
    ratio_values = np.full((n_nens, n_alpha, len(method_indices)), np.nan, dtype=float)

    for k, method_idx in enumerate(method_indices):
        for iens, nens in enumerate(nen_range):
            for ialpha, alpha in enumerate(alpha_range):
                filename = (
                    f"LETKF_Paper_Nature_Freq{Freq}_Den{Den}_Type3_ObsErr{ObsErr}"
                    f"_Nens{nens}_NTemp{method_idx}_alpha{alpha}_{suffix}.npz"
                )
                filepath = os.path.join(base_path, filename)

                if not os.path.exists(filepath):
                    print(f"file not found {filename}")
                    continue

                data = np.load(filepath, allow_pickle=True)
                rmse = data["total_analysis_rmse"]
                sprd = data['total_analysis_sprd']

                # drop incomplete runs marked by NormalEnd==0
                if "NormalEnd" in data:
                    normal_end_mask = (1 - data["NormalEnd"]).astype(bool)
                    rmse[normal_end_mask] = np.nan
                    sprd[normal_end_mask] = np.nan

                mult_inf_range = data.get('mult_inf_range', np.arange(1.0, 1.6, 0.05))
                loc_scale_range = data.get('loc_scale_range', np.arange(0.5, 5.0, 0.5))


                # outlier filter and smoothing
                rmse = cf.outlier_rmse_filter(rmse)
                rmse_smooth = cf.smooth_filter(rmse, size=smooth_size)
                sprd_smooth = cf.smooth_filter(sprd, size=smooth_size)
                ratio_smooth = sprd_smooth/rmse_smooth
                # select minimum
                if np.all(np.isnan(rmse_smooth)):
                    print(f'All nan in {filename}')
                    continue
                min_idx = np.nanargmin(rmse_smooth)
                inf_idx, loc_idx = np.unravel_index(min_idx, rmse_smooth.shape)

                best_inflation[iens, ialpha, k] = mult_inf_range[inf_idx]
                best_localization[iens, ialpha, k] = loc_scale_range[loc_idx]
                rmse_values[iens, ialpha, k] = float(np.nanmin(rmse_smooth))
                sprd_values[iens, ialpha, k] = float(sprd_smooth[inf_idx,loc_idx])
                ratio_values[iens, ialpha, k] = float(ratio_smooth[inf_idx,loc_idx])
    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_name = f"Best_params_Freq{Freq}_Den{Den}_ObsErr{ObsErr}_{suffix}.npz"
    out_path = os.path.join(output_dir, out_name)
    np.savez(
        out_path,
        rmse_values=rmse_values,
        sprd_values=sprd_values,
        ratio_values=ratio_values,
        loc_inflation=best_inflation,
        loc_scale=best_localization,
        nens_labels=nen_range,
        alpha_range=alpha_range,
        meta=dict(Freq=Freq, Den=Den, ObsErr=ObsErr, methods=list(method_indices), suffix=suffix),
    )
if __name__ == "__main__":
    work_place = "hydra" # 'hydra', 'ubuntu' or 'win'
    if work_place == 'hydra':
        base_path = r'/home/jorge.gacitua/salidas/L96_multiple_experiments/data/LETKF'
        output_dir = r'/home/jorge.gacitua/salidas/L96_multiple_experiments/data/summary'
    elif work_place == 'ubuntu':
        base_path = r'/media/jgacitua/storage/L96_multiple_experiments/data/LETKF'
        output_dir = r'/media/jgacitua/storage/L96_multiple_experiments/data/summary'
    elif work_place == 'win':
        base_path  = r'C:\Users\jyndd\OneDrive\Doctorado\Experimentos\L96_multiple_experiments\data\LETKF'
        output_dir = r'C:\Users\jyndd\OneDrive\Doctorado\Experimentos\L96_multiple_experiments\data\summary'

    Freq = 4
    Den = 0.5
    ObseErrs = np.array(['0.3','1','5','25'])
    nen_range = np.array([10,20,40,60,80,100])
    suffix = 'NOGEC_500_Prespinup200_inf1.2'
    alpha_range = np.array([0, 1, 2, 3])
    for ObsErr in ObseErrs:
        best_params(base_path, output_dir,Freq,Den,ObsErr,nen_range,alpha_range,suffix)

