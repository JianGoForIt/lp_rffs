seed=${1}
lambda=${2}
percentile=${3}
output=${4}
sigma=30.0
# run exact kernel
# if [ ! -f "${output}_exact/results.pkl" ]; then
#   echo "${output}_exact/results.pkl"
#   python rff_kernel_census.py --exact_kernel --sigma=${sigma} --reg_lambda=${lambda} --output_folder="${output}_exact"
# fi

for n_fp_rff in 32 64 128 256 512 1024
  do 
    # run with pca rff
    for perc in 0.0 0.1 1.0 10.0
      do
        if [ ! -f "${output}_n_fp_feat_${n_fp_rff}_lp_pca_sqr_rff_n_base_feat_1024_auto_scale_perc_${perc}/results.pkl" ]; then
          echo "${output}_n_fp_feat_${n_fp_rff}_lp_pca_sqr_rff_n_base_feat_1024_auto_scale_perc_${perc}/results.pkl"
          python rff_kernel_census.py --pca_rff --pca_rff_auto_scale --pca_rff_perc=${perc} --n_fp_rff=${n_fp_rff} --pca_rff_n_base_fp_feat=1024 --pca_rff_mu=${mu} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda} --output_folder="${output}_n_fp_feat_${n_fp_rff}_lp_pca_sqr_rff_n_base_feat_1024_auto_scale_perc_${perc}"
        fi
      done
  done


for n_fp_rff in 32 64 128 256 512 1024 2048 4096 8192 
  do 
    # run with pca rff
    for perc in 0.0 0.1 1.0 10.0
      do
        if [ ! -f "${output}_n_fp_feat_${n_fp_rff}_lp_pca_sqr_rff_n_base_feat_8192_auto_scale_perc_${perc}/results.pkl" ]; then
          echo "${output}_n_fp_feat_${n_fp_rff}_lp_pca_sqr_rff_n_base_feat_8192_auto_scale_perc_${perc}/results.pkl"
          python rff_kernel_census.py --pca_rff --pca_rff_auto_scale --pca_rff_perc=${perc} --n_fp_rff=${n_fp_rff} --pca_rff_n_base_fp_feat=8192 --pca_rff_mu=${mu} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda} --output_folder="${output}_n_fp_feat_${n_fp_rff}_lp_pca_sqr_rff_n_base_feat_8192_auto_scale_perc_${perc}"
        fi
      done
  done
