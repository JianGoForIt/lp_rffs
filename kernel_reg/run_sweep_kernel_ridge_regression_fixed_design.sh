seed=${1}
lambda=${2}
output=${3}
sigma=30.0


for n_fp_rff in 1024 8192
  do 
    # run with pca rff
    for noise in 1e-2 1e-1 1e0 1e1 1e2 1e3 1e4 1e5
      do
        if [ ! -f "${output}_n_fp_feat_${n_fp_rff}_fp_rff_noise_sigma_${noise}/results.pkl" ]; then
          echo "${output}_n_fp_feat_${n_fp_rff}_fp_rff_noise_sigma_${noise}/results.pkl"
          python rff_kernel_census.py --fixed_design --fixed_design_data_sample_int=1 --fixed_design_noise_level=${noise} --n_fp_rff=${n_fp_rff} --sigma=${sigma} --random_seed=${seed} --do_fp --reg_lambda=${lambda} --output_folder="${output}_n_fp_feat_${n_fp_rff}_fp_rff_noise_sigma_${noise}"
        fi
      done
  done