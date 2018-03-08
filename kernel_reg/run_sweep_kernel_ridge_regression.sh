seed=${1}
output=${2}
lambda=${3}
n_fp_rff=${4}
sigma=30.0
# run exact kernel
python rff_kernel_census.py --exact_kernel --sigma=${sigma} --reg_lambda=${lambda} --output_folder="${output}_exact"

# run fp rff kernel
python rff_kernel_census.py --do_fp --n_fp_rff=${n_fp_rff} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda} --output_folder="${output}_fp_rff"

# run lp rff kernel
for nbit in 32 16 8 4 2 1
  do
    python rff_kernel_census.py --n_bit=${nbit} --n_fp_rff=${n_fp_rff} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda} --output_folder="${output}_lp_rff_nbit_${nbit}"
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
