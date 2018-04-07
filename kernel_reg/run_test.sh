# python rff_kernel_census.py --exact_kernel --sigma=30 --reg_lambda=0.0005
# python rff_kernel_census.py --do_fp --n_fp_rff=32 --sigma=30 --reg_lambda=0.005 --random_seed=1
# python rff_kernel_census.py --n_bit=2  --n_fp_rff=1024 --sigma=30 --reg_lambda=0.005 --random_seed=2
# python rff_kernel_census.py --n_bit=2  --n_fp_rff=1024 --sigma=30 --reg_lambda=0.005 --random_seed=1
# python rff_kernel_census.py --n_bit=2  --n_fp_rff=1024 --sigma=30 --reg_lambda=0.005 --test_var_reduce --random_seed=1
# python rff_kernel_census.py --n_bit=2  --n_fp_rff=1024 --sigma=30 --reg_lambda=0.005 --test_var_reduce --random_seed=2
# python rff_kernel_census.py --do_fp --n_fp_rff=16384 --sigma=28.87 --reg_lambda=0.005
# python rff_kernel_census.py --n_bit=8 --n_fp_rff=512 --sigma=10.0 --reg_lambda=10.0

python rff_kernel_census.py --pca_rff --pca_rff_mu=10.0 \
  --n_fp_rff=128 --pca_rff_n_base_fp_feat=1024 --sigma=30.0 --random_seed=1 --reg_lambda=1e-6

python rff_kernel_census_bak.py --pca_rff --pca_rff_mu=10.0 \
  --n_fp_rff=128 --pca_rff_n_base_fp_feat=1024 --sigma=30.0 --random_seed=1 --reg_lambda=1e-6

## test auto pca quantizer
#python rff_kernel_census.py --pca_rff --pca_rff_auto_scale --pca_rff_perc=0.0\
#  --n_fp_rff=128 --pca_rff_n_base_fp_feat=1024 --sigma=30.0 --random_seed=1 --reg_lambda=1e-6
#
#python rff_kernel_census.py --pca_rff --pca_rff_auto_scale --pca_rff_perc=10.0\
#  --n_fp_rff=128 --pca_rff_n_base_fp_feat=1024 --sigma=30.0 --random_seed=1 --reg_lambda=1e-6
#
#python rff_kernel_census.py --pca_rff --pca_rff_auto_scale --pca_rff_perc=10.0\
#  --n_fp_rff=128 --pca_rff_n_base_fp_feat=1024 --sigma=30.0 --random_seed=1 --reg_lambda=1e-6
