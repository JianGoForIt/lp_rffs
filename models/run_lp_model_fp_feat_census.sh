for seed in 1 2 3
  do
    for lr in 0.5 0.1
      do
        # run plain sgd
        python run_model.py --model=ridge_regression --minibatch=250 --l2_reg=0.0 \
          --kernel_sigma=30.0 --n_fp_rff=10000 --random_seed=${seed} --do_fp_feat --learning_rate=${lr} \
          --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=sgd --epoch=1 --do_fp_model --cuda \
          --save_path=/dfs/scratch0/zjian/lp_kernel/census_results_sgd_lp_model_fp_feat/sgd_lr_${lr}
      done
    for model_bit in 8 16 4
      do
        for lr in 0.5 0.1
          # run lp sgd
          for scale in 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1
            do
              python run_model.py --model=ridge_regression --minibatch=250 --l2_reg=0.0 \
                --kernel_sigma=30.0 --n_fp_rff=10000 --random_seed=${seed} --do_fp_feat --learning_rate=${lr} \
                --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=lp_sgd --epoch=1 \
                --n_bit_model=${model_bit} --scale_model=${scale} --cuda \
                --save_path=/dfs/scratch0/zjian/lp_kernel/census_results_sgd_lp_model_fp_feat/lpsgd_lr_${lr}_model_bits_${model_bit}_scale_${scale}
            done 
          # run halp 
          for mu in 1e-1 1e0 1e1 1e2 1e3 1e4 1e5
            do
              python run_model.py --model=ridge_regression --minibatch=250 --l2_reg=0.0 \
                --kernel_sigma=30.0 --n_fp_rff=10000 --random_seed=2 --do_fp_feat --learning_rate=${lr} \
                --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=halp --epoch=1 \
                --halp_mu=${mu} --halp_epoch_T=1.0 --cuda \
                --save_path=/dfs/scratch0/zjian/lp_kernel/census_results_sgd_lp_model_fp_feat/halp_lr_${lr}_model_bits_${model_bit}_mu_${mu}_epoch_T_1.0
            done
      done
  done