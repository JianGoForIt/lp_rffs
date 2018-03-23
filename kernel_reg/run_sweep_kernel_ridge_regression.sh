seed=${1}
lambda=${2}
output=${3}
# n_fp_rff=${4}
sigma=30.0
# run exact kernel
if [ ! -f "${output}_exact/results.pkl" ]; then
  echo "${output}_exact/results.pkl"
  python rff_kernel_census.py --exact_kernel --sigma=${sigma} --reg_lambda=${lambda} --output_folder="${output}_exact"
fi

<<<<<<< b011e943a8087ffaa5fba116878c11970942141b
for n_fp_rff in 32 
=======
for n_fp_rff in 32 64 128 256 512 1024 2048 4096 8192
>>>>>>> add 32 feature test
  do 
    # run fp rff kernel
    if [ ! -f "${output}_n_fp_feat_${n_fp_rff}_fp_rff/results.pkl" ]; then
      echo "${output}_n_fp_feat_${n_fp_rff}_fp_rff"
      python rff_kernel_census.py --do_fp --n_fp_rff=${n_fp_rff} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda} --output_folder="${output}_n_fp_feat_${n_fp_rff}_fp_rff"
    fi    

    # run lp rff kernel
    for nbit in 1 2 4 8 16 32
      do
	if [ ! -f "${output}_nbit_${nbit}_n_fp_feat_${n_fp_rff}_lp_rff/results.pkl" ]; then
          echo "${output}_nbit_${nbit}_n_fp_feat_${n_fp_rff}_lp_rff"
          python rff_kernel_census.py --n_bit=${nbit} --n_fp_rff=${n_fp_rff} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda} --output_folder="${output}_nbit_${nbit}_n_fp_feat_${n_fp_rff}_lp_rff"
        fi 
      done
   done

# for lambda in 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1
#   do
#     python rff_kernel_census.py --exact_kernel --sigma=${sigma} --reg_lambda=${lambda}
#   done

# for n_fp_rff in 64 128 256 512 1024 2048 4096 8192 16384 
#   do
#     for lambda in 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1
#       do
#         python rff_kernel_census.py --do_fp --n_fp_rff=${n_fp_rff} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda}
#         for nbit in 32 16 8 4 2 1
#           do
#             python rff_kernel_census.py --n_bit=${nbit} --n_fp_rff=${n_fp_rff} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda}
#           done
#       done
#   done
