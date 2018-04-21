python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
  --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --do_fp_feat --learning_rate=100.0 \
  --data_path=../../data/adult/adult --opt=lpsgd --epoch=1 --n_bit_model=2 --scale_model=0.001 --cuda

python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
  --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --do_fp_feat --learning_rate=100.0 \
  --data_path=../../data/adult/adult --opt=halp --epoch=1 --n_bit_model=2 --halp_mu=10.0 \
  --halp_epoch_T=1.0 --cuda

python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
  --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --n_bit_feat=4 --learning_rate=100.0 \
  --data_path=../../data/adult/adult --opt=lpsgd --epoch=1 --n_bit_model=2 --scale_model=0.001 --cuda

python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
  --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --n_bit_feat=4 --learning_rate=100.0 \
  --data_path=../../data/adult/adult --opt=halp --epoch=1 --n_bit_model=2 --halp_mu=10.0 \
  --halp_epoch_T=1.0 --cuda

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
