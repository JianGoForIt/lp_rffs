lr_in=${1}
model_bit_in=${2}
feat_bit_in=${3}
budget_rff_number=$((10000 * ${feat_bit_in} / 32))
echo ${budget_rff_number}
for seed in 1 2 3
  do
#    for lr in ${lr_in} #0.5 0.1
#      do
#        # run plain sgd
#        python run_model.py --model=ridge_regression --minibatch=250 --l2_reg=0.0 \
#          --kernel_sigma=30.0 --n_fp_rff=10000 --random_seed=${seed} --do_fp_feat --learning_rate=${lr} \
#          --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=sgd --epoch=100 --do_fp_model --cuda \
#          --save_path=/dfs/scratch0/zjian/lp_kernel/census_results_sgd_lp_model_fp_feat/fpsgd_lr_${lr}_seed_${seed}
#      done
    for model_bit in ${model_bit_in} #8 #16 4
      do
        for lr in ${lr_in} #0.5 0.1
          do
          # run lp sgd
            for scale in 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1
              do
                python run_model.py --model=ridge_regression --minibatch=250 --l2_reg=0.0 \
                  --kernel_sigma=30.0 --n_fp_rff=${budget_rff_number} --random_seed=${seed} --learning_rate=${lr} \
                  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=lpsgd --epoch=250 \
                  --n_bit_model=${model_bit} --scale_model=${scale} --cuda --n_bit_feat=${feat_bit_in} \
                  --save_path=/dfs/scratch0/zjian/lp_kernel/census_results_sgd_lp_model_lp_feat/lpsgd_lr_${lr}_model_bits_${model_bit}_scale_${scale}_seed_${seed}
              done 
            # run halp 
            for mu in 1e-3 1e-2 1e-1 1e0 1e1 1e2 1e3 1e4
              do
                python run_model.py --model=ridge_regression --minibatch=250 --l2_reg=0.0 \
                  --kernel_sigma=30.0 --n_fp_rff=${budget_rff_number} --random_seed=${seed} --learning_rate=${lr} \
                  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=halp --epoch=250 \
                  --halp_mu=${mu} --halp_epoch_T=2.0 --cuda --n_bit_model=${model_bit} --n_bit_feat=${feat_bit_in} \
                  --save_path=/dfs/scratch0/zjian/lp_kernel/census_results_sgd_lp_model_lp_feat/halp_lr_${lr}_model_bits_${model_bit}_mu_${mu}_epoch_T_2.0_seed_${seed}
              done

            # run bit centering
            for mu in 1e-3 1e-2 1e-1 1e0 1e1 1e2 1e3 1e4
              do
                python run_model_bit_center.py --model=ridge_regression --minibatch=250 --l2_reg=0.0 \
                  --kernel_sigma=30.0 --n_fp_rff=${budget_rff_number} --random_seed=${seed} --learning_rate=${lr} \
                  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=halp --epoch=100 \
                  --halp_mu=${mu} --halp_epoch_T=2.0 --cuda --n_bit_model=${model_bit} --n_bit_feat=${feat_bit_in} \
                  --save_path=/dfs/scratch0/zjian/lp_kernel/census_results_sgd_lp_model_lp_feat/bit_center_lr_${lr}_model_bits_${model_bit}_mu_${mu}_epoch_T_2.0_seed_${seed}
              done
          done
      done
  done
