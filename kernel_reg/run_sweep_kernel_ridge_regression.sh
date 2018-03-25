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

for n_fp_rff in 32 4096 64 128 256 512 1024 2048  #8192
  do 
    # run fp rff kernel
    if [ ! -f "${output}_n_fp_feat_${n_fp_rff}_fp_rff/results.pkl" ]; then
      echo "${output}_n_fp_feat_${n_fp_rff}_fp_rff"
      python rff_kernel_census.py --do_fp --n_fp_rff=${n_fp_rff} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda} --output_folder="${output}_n_fp_feat_${n_fp_rff}_fp_rff"
    fi
    # run lp rff kernel
    for nbit in 1 2 4 
      do
        if [ ! -f "${output}_nbit_${nbit}_n_fp_feat_${n_fp_rff}_lp_rff/results.pkl" ]; then
          echo "${output}_nbit_${nbit}_n_fp_feat_${n_fp_rff}_lp_rff"
          python rff_kernel_census.py --n_bit=${nbit} --n_fp_rff=${n_fp_rff} --sigma=${sigma} --random_seed=${seed} --reg_lambda=${lambda} --output_folder="${output}_nbit_${nbit}_n_fp_feat_${n_fp_rff}_lp_rff"
        fi
      done
  done
