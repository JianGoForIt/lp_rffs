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


## tests using census datasets for closed form solutions
echo rff fp closed form
# cir rff fp closed form new
cd /dfs/scratch0/zjian/lp_kernel_code_release/lp_kernel && python /dfs/scratch0/zjian/lp_kernel_code_release/lp_kernel/run_model.py \
 --model=ridge_regression --l2_reg=0.0005  --kernel_sigma=28.867513459481287 --random_seed=1 \
 --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=./tmp/new --approx_type=rff --n_feat=1000  --do_fp_feat  --closed_form_sol --collect_sample_metric \
  | grep "spectral_norm_error\|test error"
# cir rff fp closed form old
cd /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models && python /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models/run_model.py \
 --model=ridge_regression --l2_reg=0.0005  --kernel_sigma=28.867513459481287 --random_seed=1 \
 --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=./tmp/old --approx_type=rff --n_fp_rff=1000  --do_fp_feat  --closed_form_sol --collect_sample_metric \
  | grep "spectral_norm_error\|test error"
cat /dfs/scratch0/zjian/lp_kernel/lp_rff/regression_real_setting_independent_quant_seed/census_type_rff_l2_reg_0.0005_n_feat_1000_n_bit_64_seed_1/eval_metric.txt
echo

# nystrom fp closed form new
echo nystrom fp closed form
cd /dfs/scratch0/zjian/lp_kernel_code_release/lp_kernel && python /dfs/scratch0/zjian/lp_kernel_code_release/lp_kernel/run_model.py \
 --model=ridge_regression --l2_reg=0.0005  --kernel_sigma=28.867513459481287 --random_seed=1 \
 --do_fp_feat --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=./tmp/new --approx_type=nystrom --n_feat=2500 --closed_form_sol --collect_sample_metrics \
  | grep "spectral_norm_error\|test error"
# nystrom fp closed form old
cd /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models && python /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models/run_model.py \
 --model=ridge_regression --l2_reg=0.0005  --kernel_sigma=28.867513459481287 --random_seed=1 \
 --do_fp_feat --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=./tmp/old --approx_type=nystrom --n_fp_rff=2500 --closed_form_sol --collect_sample_metrics \
  | grep "spectral_norm_error\|test error"
cat /dfs/scratch0/zjian/lp_kernel/closeness/regression_real_setting/census_type_nystrom_l2_reg_0.0005_n_fp_feat_2500_seed_1/eval_metric.txt
echo

echo cir rff fp closed form
# cir rff fp closed form new
cd /dfs/scratch0/zjian/lp_kernel_code_release/lp_kernel && python /dfs/scratch0/zjian/lp_kernel_code_release/lp_kernel/run_model.py \
 --model=ridge_regression --l2_reg=0.0005  --kernel_sigma=28.867513459481287 --random_seed=1 \
 --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=./tmp/new --approx_type=cir_rff --n_feat=1000  --do_fp_feat  --closed_form_sol --collect_sample_metric \
  | grep "spectral_norm_error\|test error"
# cir rff fp closed form old
cd /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models && python /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models/run_model.py \
 --model=ridge_regression --l2_reg=0.0005  --kernel_sigma=28.867513459481287 --random_seed=1 \
 --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=./tmp/old --approx_type=cir_rff --n_fp_rff=1000  --do_fp_feat  --closed_form_sol --collect_sample_metric \
  | grep "spectral_norm_error\|test error"
cat /dfs/scratch0/zjian/lp_kernel/lp_rff/regression_real_setting_independent_quant_seed/census_type_cir_rff_l2_reg_0.0005_n_feat_1000_n_bit_64_seed_1/eval_metric.txt
echo

echo cir rff lp 4 bit closed form
# cir rff lp 4 bit closed form new
cd /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models/ && python /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models/run_model.py \
  --model=ridge_regression --l2_reg=0.0005  --kernel_sigma=28.867513459481287 --random_seed=1 \
  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=./tmp/new --approx_type=cir_rff --n_fp_rff=1000  --n_bit_feat=4 --closed_form_sol --collect_sample_metrics \
  | grep "spectral_norm_error\|test error"
# cir rff lp 4 bit closed form old
cd /dfs/scratch0/zjian/lp_kernel_code_release/lp_kernel/ && python /dfs/scratch0/zjian/lp_kernel_code_release/lp_kernel/run_model.py \
  --model=ridge_regression --l2_reg=0.0005  --kernel_sigma=28.867513459481287 --random_seed=1 \
  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=./tmp/old --approx_type=cir_rff --n_feat=1000  --n_bit_feat=4 --closed_form_sol --collect_sample_metrics \
  | grep "spectral_norm_error\|test error"
cat /dfs/scratch0/zjian/lp_kernel/lp_rff/regression_real_setting_independent_quant_seed/census_type_cir_rff_l2_reg_0.0005_n_feat_1000_n_bit_4_seed_1/eval_metric.txt
echo 

# cir rff lp 8 bit closed form fixed design new
echo cir rff lp 8 bit closed form fixed design
cd /dfs/scratch0/zjian/lp_kernel_code_release/lp_kernel && python /dfs/scratch0/zjian/lp_kernel_code_release/lp_kernel/run_model.py \
  --model=ridge_regression --l2_reg=0.0005  --kernel_sigma=28.867513459481287 --random_seed=1 \
  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=./tmp/old --approx_type=cir_rff --n_feat=2000 \
  --n_bit_feat=4 --closed_form_sol --fixed_design --fixed_design_auto_l2_reg --fixed_design_noise_sigma=1e4  --collect_sample_metrics \
  | grep "spectral_norm_error\|test error"
# cir rff lp 8 bit closed form fixed design old
cd /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models && python /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models/run_model.py \
  --model=ridge_regression --l2_reg=0.0005  --kernel_sigma=28.867513459481287 --random_seed=1 \
  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=./tmp/old --approx_type=cir_rff --n_fp_rff=2000 \
  --n_bit_feat=4 --closed_form_sol --fixed_design --fixed_design_auto_l2_reg --fixed_design_noise_sigma=1e4  --collect_sample_metrics \
  | grep "spectral_norm_error\|test error"
cat /dfs/scratch0/zjian/lp_kernel/lp_rff/fixed_design/census_type_cir_rff_l2_reg_0.0005_n_feat_2000_n_bit_4_seed_1/eval_metric.txt
echo

# single nystrom lp closed form new
echo single nystrom lp closed form
cd /dfs/scratch0/zjian/lp_kernel_code_release/lp_kernel && python /dfs/scratch0/zjian/lp_kernel_code_release/lp_kernel/run_model.py \
  --model=ridge_regression --l2_reg=0.0005  --kernel_sigma=28.867513459481287 --random_seed=1 \
  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=./tmp/new --approx_type=ensemble_nystrom --n_feat=2500 \
  --closed_form_sol --n_ensemble_nystrom=1 --collect_sample_metrics --n_bit_feat=8 --closed_form_sol \
  | grep "spectral_norm_error\|test error"
# single nystrom lp closed form old
cd /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models && python /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models/run_model.py \
  --model=ridge_regression --l2_reg=0.0005  --kernel_sigma=28.867513459481287 --random_seed=1 \
  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=./tmp/old --approx_type=ensemble_nystrom --n_fp_rff=2500 \
  --closed_form_sol --n_ensemble_nystrom=1 --collect_sample_metrics --n_bit_feat=8 --closed_form_sol \
  | grep "spectral_norm_error\|test error"
cat /dfs/scratch0/zjian/lp_kernel/lp_ensemble_nystrom/regression_real_setting/census_type_ensemble_nystrom_l2_reg_0.0005_n_feat_2500_n_bit_8_n_learner_1_seed_1/eval_metric.txt
echo

# ensembled nystrom lp closed form new
echo ensembled nystrom lp closed form
cd /dfs/scratch0/zjian/lp_kernel_code_release/lp_kernel && python /dfs/scratch0/zjian/lp_kernel_code_release/lp_kernel/run_model.py \
  --model=ridge_regression --l2_reg=0.0005  --kernel_sigma=28.867513459481287 --random_seed=1 \
  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=./tmp/new --approx_type=ensemble_nystrom --n_feat=2500 \
  --closed_form_sol --n_ensemble_nystrom=10 --collect_sample_metrics --n_bit_feat=8 --closed_form_sol \
  | grep "spectral_norm_error\|test error"
# ensembled nystrom lp closed form old
cd /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models && python /dfs/scratch0/zjian/lp_kernel_code/lp_kernel/models/run_model.py \
  --model=ridge_regression --l2_reg=0.0005  --kernel_sigma=28.867513459481287 --random_seed=1 \
  --data_path=/dfs/scratch0/zjian/data/lp_kernel_data/census --save_path=./tmp/old --approx_type=ensemble_nystrom --n_fp_rff=2500 \
  --closed_form_sol --n_ensemble_nystrom=10 --collect_sample_metrics --n_bit_feat=8 --closed_form_sol \
  | grep "spectral_norm_error\|test error"
cat /dfs/scratch0/zjian/lp_kernel/lp_ensemble_nystrom/regression_real_setting/census_type_ensemble_nystrom_l2_reg_0.0005_n_feat_2500_n_bit_8_n_learner_10_seed_1/eval_metric.txt
echo

## tests using subsample covtype for sgd based solutions
# rff fp new

# rff fp old


# nystrom fp new

# nystrom fp old


# cir rff fp new

# cir rff fp old


# cir rff lp 4 bit new

# cir rff lp 4 bit old


# lm halp 8 bit new

# lm halp 8 bit old


# lm bit center sgd 8 bit new

# lm bit center sgd 8 bit old


