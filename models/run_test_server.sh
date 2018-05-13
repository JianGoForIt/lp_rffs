#python run_model.py --model=ridge_regression --minibatch=250 --l2_reg=0.0 \
#  --kernel_sigma=28.87 --n_fp_rff=10000 --random_seed=2 --do_fp --learning_rate=0.5 \
#  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=sgd --epoch=70 --cuda
#
#python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#  --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --do_fp --learning_rate=100.0 \
#  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/adult --opt=sgd --epoch=40  --cuda

# test multiple LP algorithm using adault dataset
#python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#  --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --do_fp_feat --learning_rate=100.0 \
#  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/adult --opt=lpsgd --epoch=1 --n_bit_model=2 --scale_model=0.001 --cuda
#
#python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#  --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --do_fp_feat --learning_rate=100.0 \
#  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/adult --opt=halp --epoch=1 --n_bit_model=2 --halp_mu=10.0 \
#  --halp_epoch_T=1.0 --cuda
#
#python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#  --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --n_bit_feat=4 --learning_rate=100.0 \
#  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/adult --opt=lpsgd --epoch=1 --n_bit_model=2 --scale_model=0.001 --cuda
#
#python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#  --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --n_bit_feat=4 --learning_rate=100.0 \
#  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/adult --opt=halp --epoch=1 --n_bit_model=2 --halp_mu=10.0 \
#  --halp_epoch_T=1.0 --cuda

# python run_model.py --model=ridge_regression --minibatch=250 --l2_reg=0.0 \
#   --kernel_sigma=28.87 --n_fp_rff=10000 --random_seed=2 --do_fp --learning_rate=0.5 \
#   --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=sgd --epoch=70

# python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#   --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --do_fp --learning_rate=100.0 \
#   --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/adult --opt=sgd --epoch=40

# python run_model.py --model=ridge_regression --minibatch=250 --dataset=census --l2_reg=0.0001 \
#   --kernel_sigma=30.0 --n_fp_rff=256 --random_seed=1 --do_fp --learning_rate=1.0 \
#   --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --cuda

# python run_model.py --model=logistic_regression --minibatch=250 --dataset=adult --l2_reg=0.0001 \
#   --kernel_sigma=30.0 --n_fp_rff=256 --random_seed=1 --do_fp --learning_rate=1.0 \
#   --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/adult --cuda

# python run_model.py --model=ridge_regression --minibatch=250 --dataset=census --l2_reg=0.0001 \
#   --kernel_sigma=30.0 --n_fp_rff=256 --n_bit_feat=4 --random_seed=1 --learning_rate=1.0 \
#   --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --cuda

# python run_model.py --model=logistic_regression --minibatch=250 --dataset=adult --l2_reg=0.0001 \
#   --kernel_sigma=30.0 --n_fp_rff=256 --n_bit_feat=4 --random_seed=1 --learning_rate=1.0 \
#   --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/adult --cuda

#python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#  --kernel_sigma=2.24 --n_fp_rff=2500 --random_seed=2 --learning_rate=2.5 \
#  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/adult --opt=sgd --epoch=1 \
#  --save_path=./test/123/ --approx_type=nystrom

# Nystrom test

#python run_model.py --model=ridge_regression --minibatch=250 --l2_reg=0.0 \
#  --kernel_sigma=30.0 --n_fp_rff=2500 --random_seed=2 --learning_rate=0.5 \
#  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=sgd --epoch=100 \
#  --save_path=./test/123/ --approx_type=nystrom --cuda

#python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#  --kernel_sigma=1.12 --n_fp_rff=2500 --random_seed=2 --learning_rate=12.5 \
#  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/codrna --opt=sgd --epoch=100 \
#  --save_path=./test/123/ --approx_type=nystrom --cuda

# # test lr decay approac
# python run_model.py --model=ridge_regression --minibatch=250 --l2_reg=0.0 \
#   --kernel_sigma=30.0 --n_fp_rff=10000 --random_seed=2 --learning_rate=0.5 \
#   --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=sgd --epoch=300 \
#   --save_path=./test/123/ --do_fp_feat --approx_type=rff --cuda --do_fp_feat 

# python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#   --kernel_sigma=1.12 --n_fp_rff=10000 --random_seed=2 --learning_rate=12.5 \
#   --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/codrna --opt=sgd --epoch=200 \
#   --save_path=./test/123/ --approx_type=rff --cuda --do_fp_feat

# test fixed design functionality
#python run_model.py --model=ridge_regression --minibatch=250 \
#  --l2_reg=0.0  --kernel_sigma=30.0 --n_fp_rff=2500 --random_seed=1 --learning_rate=0.5  \
#  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=sgd --epoch=300  --save_path=./test/123 --approx_type=exact \
#  --do_fp_feat --collect_sample_metrics --fixed_design --fixed_design_noise_sigma=1e4

#python run_model.py --model=ridge_regression --minibatch=250 \
#  --l2_reg=0.0  --kernel_sigma=30.0 --n_fp_rff=256 --random_seed=1 --learning_rate=0.5  \
#  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=sgd --epoch=300  --save_path=./test/rff --approx_type=rff \
#  --do_fp_feat --collect_sample_metrics --fixed_design --fixed_design_noise_sigma=1e4
#
# python run_model.py --model=ridge_regression --minibatch=250 \
#   --l2_reg=0.0  --kernel_sigma=28.8675134595 --n_fp_rff=200 --random_seed=1 --learning_rate=0.5  \
#   --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=sgd --epoch=300  --save_path=./test/nystrom --approx_type=nystrom \
#   --do_fp_feat --collect_sample_metrics --fixed_design --fixed_design_noise_sigma=1e4 --fixed_design_auto_l2_reg

#python run_model.py --model=ridge_regression --minibatch=250 \
#  --l2_reg=0.0  --kernel_sigma=30.0 --n_fp_rff=2500 --random_seed=1 --learning_rate=0.5  \
#  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=sgd --epoch=300  --save_path=./test/exact --approx_type=exact \
#  --do_fp_feat --collect_sample_metrics --fixed_design --fixed_design_noise_sigma=1e4


# test closed form real setting lambda star search closeness experiments
#python run_model.py --model=ridge_regression --minibatch=250 \
#  --l2_reg=1e-3  --kernel_sigma=28.8675134595 --random_seed=1 --learning_rate=0.5  \
#  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=./test/exact --approx_type=exact \
#  --collect_sample_metrics --closed_form_sol --do_fp_feat


# test for fixed design exp script
#sigma=1e4
#seed=1
#n_feat=8192
#
#python run_model.py --model=ridge_regression  \
#     --kernel_sigma=28.8675134595 --random_seed=1  \
#     --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=/dfs/scratch0/zjian/lp_kernel/delta/fixed_design/exact_auto_l2_reg_noise_sigma_${sigma} --approx_type=exact \
#     --collect_sample_metrics --fixed_design --fixed_design_auto_l2_reg --fixed_design_noise_sigma=${sigma} --exit_after_collect_metric
#
#python run_model.py --model=ridge_regression  \
#              --kernel_sigma=28.8675134595 --random_seed=${seed}  \
#              --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=/dfs/scratch0/zjian/lp_kernel/delta/fixed_design/nystrom_noise_sigma_${sigma}_n_feat_${n_feat}_seed_${seed} \
#              --approx_type=nystrom --do_fp_feat --n_fp_rff=${n_feat} \
#              --collect_sample_metrics --fixed_design --fixed_design_auto_l2_reg --fixed_design_noise_sigma=${sigma} --exit_after_collect_metric
#            
#python run_model.py --model=ridge_regression  \
#              --kernel_sigma=28.8675134595 --random_seed=${seed}  \
#              --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=/dfs/scratch0/zjian/lp_kernel/delta/fixed_design/rff_noise_sigma_${sigma}_n_feat_${n_feat}_seed_${seed} --approx_type=rff --do_fp_feat --n_fp_rff=${n_feat} \
#              --collect_sample_metrics --fixed_design --fixed_design_auto_l2_reg --fixed_design_noise_sigma=${sigma} --exit_after_collect_metric



python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
                  --kernel_sigma=1.12 --n_fp_rff=10000 --random_seed=5 --learning_rate=2.5 \
                  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/codrna --opt=lm_halp --epoch=100 \
                  --halp_mu=0.01 --halp_epoch_T=2.0 --cuda --n_bit_model=4 --n_bit_feat=4 \
                  --save_path=./test/lm_halp_lr_2.5_model_bits_4_mu_0.01_epoch_T_2.0_seed_11

#python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#                  --kernel_sigma=1.12 --n_fp_rff=10000 --random_seed=1 --learning_rate=2.5 \
#                  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/codrna --opt=halp --epoch=100 \
#                  --halp_mu=0.01 --halp_epoch_T=2.0 --cuda --n_bit_model=4 --n_bit_feat=4 \
#                  --save_path=./test/halp_lr_2.5_model_bits_4_mu_0.01_epoch_T_2.0_seed_1



