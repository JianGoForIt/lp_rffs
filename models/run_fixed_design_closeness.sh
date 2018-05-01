for sigma in 1e4 #1e3 1e2 1e5
  do
    python run_model.py --model=ridge_regression  \
      --kernel_sigma=28.8675134595 --random_seed=1  \
      --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=/dfs/scratch0/zjian/lp_kernel/closeness/fixed_design/exact_auto_l2_reg_noise_sigma_${sigma} --approx_type=exact \
      --do_fp_feat --collect_sample_metrics --fixed_design --fixed_design_auto_l2_reg --fixed_design_noise_sigma=${sigma}
  done
