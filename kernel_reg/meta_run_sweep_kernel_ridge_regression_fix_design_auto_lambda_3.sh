# seed=${1}
# output=${2}
# lambda=${3}
# n_fp_rff=${4}
for seed in 1 #2 3 #6 7 8 9 10
  do 
    bash run_sweep_kernel_ridge_regression_fixed_design_auto_lambda_3.sh ${seed} /dfs/scratch0/zjian/lp_kernel/census_results_64_bit_fixed_design_opt_lambda/lambda_${lambda}_seed_${seed} 
  done
