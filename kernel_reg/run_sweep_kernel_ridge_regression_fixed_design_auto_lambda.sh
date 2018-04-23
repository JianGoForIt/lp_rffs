seed=${1}
output=${2}
sigma=30.0


for n_fp_rff in 32 64 128 256 512 1024 2048 4096 8192 #1024 #8192  
  do 
    # run with pca rff
    for noise in 1e-2 1e-1 #1e0 1e1 #1e2 1e3 1e4 1e5
      do
        if [ ! -f "${output}_exact_noise_sigma_${noise}/results.pkl" ]; then
          echo "${output}_exact_noise_sigma_${noise}/results.pkl"
          python rff_kernel_census.py --fixed_design --fixed_design_opt_reg --fixed_design_data_sample_int=1 --fixed_design_noise_level=${noise} \
            --sigma=${sigma} --random_seed=${seed} --exact_kernel --output_folder="${output}_exact_noise_sigma_${noise}"
        fi

        if [ ! -f "${output}_n_fp_feat_${n_fp_rff}_fp_rff_noise_sigma_${noise}/results.pkl" ]; then
          echo "${output}_n_fp_feat_${n_fp_rff}_fp_rff_noise_sigma_${noise}/results.pkl"
          python rff_kernel_census.py --fixed_design --fixed_design_opt_reg --fixed_design_data_sample_int=1 --fixed_design_noise_level=${noise} \
            --n_fp_rff=${n_fp_rff} --sigma=${sigma} --random_seed=${seed} --do_fp --output_folder="${output}_n_fp_feat_${n_fp_rff}_fp_rff_noise_sigma_${noise}"
        fi

        for nbit in 32 16 8 4 2 1
          do
            if [ ! -f "${output}_n_fp_feat_${n_fp_rff}_lp_rff_nbit_${nbit}_noise_sigma_${noise}/results.pkl" ]; then
              echo "${output}_n_fp_feat_${n_fp_rff}_lp_rff_nbit_${nbit}_noise_sigma_${noise}/results.pkl"
              python rff_kernel_census.py --fixed_design --fixed_design_opt_reg --fixed_design_data_sample_int=1 --fixed_design_noise_level=${noise} \
                --n_fp_rff=${n_fp_rff} --sigma=${sigma} --random_seed=${seed} --n_bit=${nbit} --output_folder="${output}_n_fp_feat_${n_fp_rff}_lp_rff_nbit_${nbit}_noise_sigma_${noise}"
            fi
          done
      done
  done
