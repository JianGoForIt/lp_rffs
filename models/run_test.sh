# python run_model.py --model=ridge_regression --minibatch=250 --l2_reg=0.0 \
#   --kernel_sigma=28.87 --n_fp_rff=10000 --random_seed=2 --do_fp --learning_rate=0.5 \
#   --data_path=../../data/census/census --opt=sgd --epoch=70

python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
  --kernel_sigma=2.24 --n_fp_rff=10000 --random_seed=2 --do_fp --learning_rate=100.0 \
  --data_path=../../data/adult/adult --opt=sgd --epoch=40

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