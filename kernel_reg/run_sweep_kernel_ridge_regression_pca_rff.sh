seed=${1}
lambda=${2}
output=${3}
# n_fp_rff=${4}
sigma=30.0
# run exact kernel
if [ ! -f "${output}_exact/results.pkl" ]; then
  echo "${output}_exact/results.pkl"
  python rff_kernel_census.py --exact_kernel --sigma=${sigma} --reg_lambda=${lambda} --output_folder="${output}_exact"
fi

for n_fp_rff in 32 64 128 256 512 1024
  do 
    # run with pca rff
    for mu in 2.0 1.0 5.0 10.0 20.0 50.0
      do
        if [ ! -f "${output}_n_fp_feat_${n_fp_rff}_lp_pca_sqr_rff_mu_${mu}_n_base_feat_1024/results.pkl/results.pkl" ]; then
          echo "${output}_n_fp_feat_${n_fp_rff}_lp_pca_sqr_rff_mu_${mu}_n_base_feat_1024/results.pkl/results.pkl"
          python rff_kernel_census.py --pca_rff --n_fp_rff=${n_fp_rff} --pca_rff_n_base_fp_feat=1024 --pca_rff_mu=${mu} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda} --output_folder="${output}_n_fp_feat_${n_fp_rff}_lp_pca_sqr_rff_mu_${mu}_n_base_feat_1024/results.pkl"
        fi
      done
    # run fp rff kernel
    if [ ! -f "${output}_n_fp_feat_${n_fp_rff}_fp_rff/results.pkl" ]; then
      echo "${output}_n_fp_feat_${n_fp_rff}_fp_rff"
      python rff_kernel_census.py --do_fp --n_fp_rff=${n_fp_rff} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda} --output_folder="${output}_n_fp_feat_${n_fp_rff}_fp_rff"
    fi
    # run lp rff kernel
    # 1 2 4 bits already in the 64_bit experiment folder
    for nbit in 8 16 32 1 2 4
      do
        if [ ! -f "${output}_nbit_${nbit}_n_fp_feat_${n_fp_rff}_lp_rff/results.pkl" ]; then
          echo "${output}_nbit_${nbit}_n_fp_feat_${n_fp_rff}_lp_rff"
          python rff_kernel_census.py --n_bit=${nbit} --n_fp_rff=${n_fp_rff} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda} --output_folder="${output}_nbit_${nbit}_n_fp_feat_${n_fp_rff}_lp_rff"
        fi
      done
  done


for n_fp_rff in 32 64 128 256 512 1024 2048 4096 8192 
  do 
    # run with pca rff
    for mu in 2.0 1.0 5.0 10.0 20.0 50.0
      do
        if [ ! -f "${output}_n_fp_feat_${n_fp_rff}_lp_pca_sqr_rff_mu_${mu}_n_base_feat_8192/results.pkl/results.pkl" ]; then
          echo "${output}_n_fp_feat_${n_fp_rff}_lp_pca_sqr_rff_mu_${mu}_n_base_feat_8192/results.pkl/results.pkl"
          python rff_kernel_census.py --pca_rff --n_fp_rff=${n_fp_rff} --pca_rff_n_base_fp_feat=8192 --pca_rff_mu=${mu} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda} --output_folder="${output}_n_fp_feat_${n_fp_rff}_lp_pca_sqr_rff_mu_${mu}_n_base_feat_8192/results.pkl"
        fi
      done
    # run fp rff kernel
    if [ ! -f "${output}_n_fp_feat_${n_fp_rff}_fp_rff/results.pkl" ]; then
      echo "${output}_n_fp_feat_${n_fp_rff}_fp_rff"
      python rff_kernel_census.py --do_fp --n_fp_rff=${n_fp_rff} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda} --output_folder="${output}_n_fp_feat_${n_fp_rff}_fp_rff"
    fi
    # run lp rff kernel
    # 1 2 4 bits already in the 64_bit experiment folder
    for nbit in 8 16 32 #1 2 4
      do
        if [ ! -f "${output}_nbit_${nbit}_n_fp_feat_${n_fp_rff}_lp_rff/results.pkl" ]; then
          echo "${output}_nbit_${nbit}_n_fp_feat_${n_fp_rff}_lp_rff"
          python rff_kernel_census.py --n_bit=${nbit} --n_fp_rff=${n_fp_rff} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda} --output_folder="${output}_nbit_${nbit}_n_fp_feat_${n_fp_rff}_lp_rff"
        fi
      done
  done
