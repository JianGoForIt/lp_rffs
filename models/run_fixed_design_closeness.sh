# exact kernel optimal lambda
# label noise sigma     lambda
# 1e3                   1.76055378e-06
# 1e4                   0.00431012
# 1e2                   0.0
# 1e5                   1.04123522
for sigma in 1e2 #1e3 1e2 1e5
  do
#    python run_model.py --model=ridge_regression  \
#      --kernel_sigma=28.8675134595 --random_seed=1  \
#      --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=/dfs/scratch0/zjian/lp_kernel/closeness/fixed_design/exact_auto_l2_reg_noise_sigma_${sigma} --approx_type=exact \
#      --collect_sample_metrics --fixed_design --fixed_design_auto_l2_reg --fixed_design_noise_sigma=${sigma}

    for seed in 1 2 3
      do
        for n_feat in 1250 2500 5000 10000 20000
          do
            python run_model.py --model=ridge_regression  \
              --kernel_sigma=28.8675134595 --random_seed=${seed}  \
              --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=/dfs/scratch0/zjian/lp_kernel/closeness/fixed_design/nystrom_noise_sigma_${sigma}_n_feat_${n_feat}_seed_${seed} --approx_type=nystrom --do_fp_feat --n_fp_rff=${n_feat} \
              --collect_sample_metrics --fixed_design --l2_reg=0.0 --fixed_design_noise_sigma=${sigma}
          done
        for n_feat in 1250 2500 5000 10000 20000 50000 100000 200000 400000
          do
            python run_model.py --model=ridge_regression  \
              --kernel_sigma=28.8675134595 --random_seed=${seed}  \
              --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=/dfs/scratch0/zjian/lp_kernel/closeness/fixed_design/rff_noise_sigma_${sigma}_n_feat_${n_feat}_seed_${seed} --approx_type=rff --do_fp_feat --n_fp_rff=${n_feat} \
              --collect_sample_metrics --fixed_design --l2_reg=0.0 --fixed_design_noise_sigma=${sigma}
          done
      done
  done
