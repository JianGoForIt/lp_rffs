lr_in=${1}
#feat_bit_in=${2}
for seed in 1 2 3
  do
    for lr in ${lr_in} #0.5 0.1
      do
        # run plain sgd
        for feat_bit_in in 32 16 8 4 2 1
          do
            budget_rff_number=$((10000 * ${feat_bit_in} / 32))
            python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
              --kernel_sigma=1.12 --n_fp_rff=${budget_rff_number} --random_seed=${seed} --learning_rate=${lr} \
              --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/codrna --opt=sgd --epoch=100 --do_fp_model --n_bit_feat=${feat_bit_in} --cuda \
              --save_path=/dfs/scratch0/zjian/lp_kernel/codrna_results_sgd_fp_model_lp_feat/fpsgd_lr_${lr}_seed_${seed}_feat_nbit_${feat_bit_in}
          done
      done
  done
