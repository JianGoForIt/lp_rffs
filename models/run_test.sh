# python run_model.py --model=ridge_regression --minibatch=250 --l2_reg=0.0 \
#   --kernel_sigma=28.87 --n_fp_rff=10000 --random_seed=2 --do_fp --learning_rate=0.5 \
#   --data_path=../../data/census/census --opt=sgd --epoch=70

# python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#   --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --do_fp_feat --learning_rate=100.0 \
#   --data_path=../../data/adult/adult --opt=sgd --epoch=5

# python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#   --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --do_fp_feat --learning_rate=100.0 \
#   --data_path=../../data/adult/adult --opt=lpsgd --epoch=1 --n_bit_model=2 --scale_model=0.001 \
#   --save_path=./test/123/


# python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#   --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --do_fp_feat --learning_rate=100.0 \
#   --data_path=../../data/adult/adult --opt=lpsgd --epoch=1 --n_bit_model=2 --scale_model=0.001

# python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#   --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --do_fp_feat --learning_rate=100.0 \
#   --data_path=../../data/adult/adult --opt=halp --epoch=1 --n_bit_model=2 --halp_mu=10.0 \
#   --halp_epoch_T=1.0

# python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#   --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --n_bit_feat=4 --learning_rate=100.0 \
#   --data_path=../../data/adult/adult --opt=lpsgd --epoch=1 --n_bit_model=2 --scale_model=0.001

# python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#   --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --n_bit_feat=4 --learning_rate=100.0 \
#   --data_path=../../data/adult/adult --opt=halp --epoch=1 --n_bit_model=2 --halp_mu=10.0 \
#   --halp_epoch_T=1.0

# python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
#   --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --n_bit_feat=2 --learning_rate=100.0 \
#   --data_path=../../data/adult/adult --opt=halp --epoch=5 --n_bit_model=2 --halp_mu=10.0 \
#   --halp_epoch_T=1.0

# python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=100.0 \
#   --kernel_sigma=1.1 --n_fp_rff=256 --random_seed=2 --do_fp --learning_rate=0.001 \
#   --data_path=../../data/codrna/codrna --opt=sgd

# python run_model.py --model=ridge_regression --minibatch=250 --l2_reg=0.0001 \
#   --kernel_sigma=30.0 --n_fp_rff=256 --random_seed=1 --do_fp --learning_rate=0.1 \
#   --data_path=../../data/census/census --opt=halp

# python run_model.py --model=logistic_regression --minibatch=250  --l2_reg=0.0001 \
#   --kernel_sigma=30.0 --n_fp_rff=256 --random_seed=1 --do_fp --learning_rate=0.01 \
#   --data_path=../../data/adult/adult --opt=halp

# python run_model.py --model=ridge_regression --minibatch=250 --dataset=census --l2_reg=0.0001 \
#   --kernel_sigma=30.0 --n_fp_rff=256 --n_bit_feat=4 --random_seed=1 --learning_rate=0.1 \
#   --data_path=../../data/census/census

# python run_model.py --model=logistic_regression --minibatch=250 --dataset=adult --l2_reg=0.0001 \
#   --kernel_sigma=30.0 --n_fp_rff=256 --n_bit_feat=4 --random_seed=1 --learning_rate=0.1 \
#   --data_path=../../data/adult/adult

# # test n feat > n sample issue in nystrom
# python run_model.py --model=ridge_regression --minibatch=250 --l2_reg=0.0 \
#   --kernel_sigma=30.0 --n_fp_rff=100000 --random_seed=2 --learning_rate=0.5 \
#   --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --opt=sgd --epoch=300 \
#   --save_path=./test/123/ --do_fp_feat --approx_type=rff --cuda --do_fp_feat 


# # test sampled data spectrum visualization
# python run_model.py --model=ridge_regression --minibatch=250 --l2_reg=0.0 \
#   --kernel_sigma=30.0 --n_fp_rff=20000 --random_seed=2 --learning_rate=0.5 \
#   --data_path=../../data/census/census --opt=sgd --epoch=300 \
#   --save_path=./test/123/ --do_fp_feat --approx_type=nystrom --cuda --do_fp_feat --n_sample=100000


# # test fixed design functionality
# python run_model.py --model=ridge_regression --minibatch=250 \
#   --l2_reg=0.0  --kernel_sigma=30.0 --n_fp_rff=2500 --random_seed=3 --learning_rate=0.5  \
#   --data_path=../../data/census/census --opt=sgd --epoch=300  --save_path=./test/123 --approx_type=exact \
#   --do_fp_feat --collect_sample_metrics --fixed_design --fixed_design_noise_sigma=1e2

# python run_model.py --model=ridge_regression --minibatch=250 \
#   --l2_reg=0.0  --kernel_sigma=28.8675134595 --n_fp_rff=2500 --random_seed=3  \
#   --data_path=../../data/census/census --opt=sgd --epoch=300  --save_path=./test/rff --approx_type=rff \
#   --do_fp_feat --collect_sample_metrics --closed_form_sol

# python /lfs/1/zjian/lp_kernel/lp_kernel/models/run_model.py --model=ridge_regression --minibatch=250 \
#   --l2_reg=0.0  --kernel_sigma=28.8675134595 --n_fp_rff=2500 --random_seed=3 --learning_rate=0.5  \
#   --data_path=../../data/census/census --opt=sgd --epoch=300  --save_path=./test/123 --approx_type=nystrom \
#   --do_fp_feat --collect_sample_metrics

# # test closed form real setting lambda star search closeness experiments
python run_model.py --model=ridge_regression --minibatch=250 \
  --l2_reg=1e-3  --kernel_sigma=28.8675134595 --random_seed=1 --learning_rate=0.5  \
  --data_path=../../data/census/census --save_path=./test/rff_256 --approx_type=rff --n_fp_rff=256 \
  --collect_sample_metrics --closed_form_sol

python run_model.py --model=ridge_regression --minibatch=250 \
  --l2_reg=1e-3  --kernel_sigma=28.8675134595 --random_seed=1 --learning_rate=0.5  \
  --data_path=../../data/census/census --save_path=./test/nystrom_256 --approx_type=nystrom --n_fp_rff=256 \
  --collect_sample_metrics --closed_form_sol

python run_model.py --model=ridge_regression --minibatch=250 \
  --l2_reg=1e-3  --kernel_sigma=28.8675134595 --random_seed=1 --learning_rate=0.5  \
  --data_path=../../data/census/census --save_path=./test/nystrom_20000 --approx_type=nystrom --n_fp_rff=20000 \
  --collect_sample_metrics --closed_form_sol
