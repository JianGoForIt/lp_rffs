# unit test for models and approximation methods
cd ./models
python ridge_regression.py
python logistic_regression.py
python kernel_regressor.py
cd ../kernels
python rff.py
python nystrom.py
python circulant_rff.py
python ensemble_nystrom.py
cd ..
# make sure the following runs
# test closed form solution for kernel ridge regression
census_path='/dfs/scratch0/zjian/data/lp_kernel_data/census'
python run_model.py --model=ridge_regression --l2_reg=0.0005 --kernel_sigma=28.867513459481287 \
  --random_seed=2 --do_fp_feat --data_path=${census_path} --save_path=./test/tmp --approx_type=rff \
  --n_feat=1250 --closed_form_sol --collect_sample_metrics

python run_model.py --model=ridge_regression --l2_reg=0.0005 --kernel_sigma=28.867513459481287 \
  --random_seed=2 --n_bit_feat=4 --data_path=${census_path} --save_path=./test/tmp --approx_type=rff \
  --n_feat=1250 --closed_form_sol --collect_sample_metrics

python run_model.py --model=ridge_regression --l2_reg=0.0005 --kernel_sigma=28.867513459481287 \
  --random_seed=2 --n_bit_feat=4 --data_path=${census_path} --save_path=./test/tmp --approx_type=cir_rff \
  --n_feat=1250 --closed_form_sol --collect_sample_metrics

python run_model.py --model=ridge_regression --l2_reg=0.0005 --kernel_sigma=28.867513459481287 \
  --random_seed=2 --do_fp_feat --data_path=${census_path} --save_path=./test/tmp --approx_type=nystrom \
  --n_feat=1250 --closed_form_sol --collect_sample_metrics

# test sgd based training
adult_path='/dfs/scratch0/zjian/data/lp_kernel_data/adult'
python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
  --kernel_sigma=2.24 --n_feat=10000 --random_seed=2 --do_fp_feat --learning_rate=100.0 \
  --data_path=${adult_path} --save_path=./test/tmp --opt=sgd --epoch=3 --approx_type=rff --cuda 

python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
  --kernel_sigma=2.24 --n_feat=10000 --random_seed=2 --n_bit_feat=2 --learning_rate=100.0 \
  --data_path=${adult_path} --save_path=./test/tmp --opt=sgd --epoch=3 --approx_type=cir_rff --cuda

python run_model.py --model=logistic_regression --minibatch=250 --l2_reg=0.0 \
  --kernel_sigma=2.24 --n_feat=10000 --random_seed=2 --do_fp_feat --learning_rate=100.0 \
  --data_path=${adult_path} --save_path=./test/tmp --opt=sgd --epoch=3 --approx_type=nystrom --cuda





