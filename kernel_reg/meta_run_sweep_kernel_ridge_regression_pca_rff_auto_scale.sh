# seed=${1}
# output=${2}
# lambda=${3}
# n_fp_rff=${4}
for seed in 1 2 3 4 5 #6 7 8 9 10
  do 
    for lambda in 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2 
      do
        bash run_sweep_kernel_ridge_regression_pca_rff_auto_scale.sh ${seed} ${lambda} /dfs/scratch0/zjian/lp_kernel/census_results_64_bit_pca_rff_auto_scale/lambda_${lambda}_seed_${seed} 
        # bash run_sweep_kernel_ridge_regression_pca_rff.sh ${seed} ${lambda} ./test/census_results_64_bit_pca_rff/lambda_${lambda}_seed_${seed}
      done
  done
