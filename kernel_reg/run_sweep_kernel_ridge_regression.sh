sigma=30.0
for lambda in 1e-4 5e-4 1e-3 #5e-3 1e-2 5e-2 1e-1 5e-1 1e0 5e0
  do
    python rff_kernel_census.py --exact_kernel --sigma=${sigma} --reg_lambda=${lambda}
  done

for n_fp_rff in 1024 4096 16384 65536 262144 1048576
  do
    for lambda in 1e-4 5e-4 1e-3 #5e-3 1e-2 5e-2 1e-1 5e-1 1e0 5e0
      do
        python rff_kernel_census.py --do_fp --n_fp_rff=${n_fp_rff} --sigma=${sigma} --reg_lambda=${lambda}
        for nbit in 32 16 8 4 2 1
          do
            python rff_kernel_census.py --n_bit=${nbit} --n_fp_rff=${n_fp_rff} --sigma=${sigma} --reg_lambda=${lambda}
          done
      done
  done
