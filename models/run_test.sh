python run_model.py --model=ridge_regression --minibatch=250 --dataset=census --l2_reg=0.0001 \
  --kernel_sigma=30.0 --n_fp_rff=256 --random_seed=1 --do_fp --learning_rate=1.0 \
  --data_path=../../data/census/census

python run_model.py --model=logistic_regression --minibatch=250 --dataset=adult --l2_reg=0.0001 \
  --kernel_sigma=30.0 --n_fp_rff=256 --random_seed=1 --do_fp --learning_rate=1.0 \
  --data_path=../../data/adult/adult

python run_model.py --model=ridge_regression --minibatch=250 --dataset=census --l2_reg=0.0001 \
  --kernel_sigma=30.0 --n_fp_rff=256 --n_bit_feat=4 --random_seed=1 --learning_rate=1.0 \
  --data_path=../../data/census/census

python run_model.py --model=logistic_regression --minibatch=250 --dataset=adult --l2_reg=0.0001 \
  --kernel_sigma=30.0 --n_fp_rff=256 --n_bit_feat=4 --random_seed=1 --learning_rate=1.0 \
  --data_path=../../data/adult/adult