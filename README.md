# Low Precision Random Fourier Features (LP-RFFs)

**LP-RFFs is a library to train classification and regression models using fixed-point low-precision random Fourier features.** Random Fourier feature (RFFs) is one of the major kernel approximation approach for kernel methods on large scale datasets. The generalization performance of kernel methods using RFFs is highly correlated with the number of RFFs; larger number of RFFs typically indicates better generalizaton, yet requires significantly larger memory footprint in minibatch-based training. **LP-RFFs use low-precision fixed-point representation for the kernel approximation features. It can achieve similar performance as with full-precision RFFs using 5-10X less memory during training with theoretical guarantees.** LP-RFFs currently supports closed-form kernel ridge regression, SGD-based training for kernel ridge regression and kernel logistic regression in float-point-based simulation. LP-RFFs also support low-precision training with LM-HALP and LM-Bit-Center-SGD using the implementation from the [HALP repo](https://github.com/mleszczy/halp). For more technical details, please refer to our paper [Low-Precision Random Fourier Features]().

## Content
* [Setup instructions](#setup-instructions)
* [Command guidelines](#command-guidelines)
* [Citation](#citation)
* [Acknowledgement](#acknowledgement)

## Setup instructions
* Install PyTorch. Our implementation is tested under PyTorch 0.3.1.
* Clone the LP-RFFs repo, along with the HALP repo (lp_kernel branch) in the same folder.
```
git clone https://github.com/JianGoForIt/lp_kernel.git
git clone https://github.com/mleszczy/halp.git
git checkout lp_kernel
```
* Download data from dropbox in the same folder with the cloned LP-RFFs and HALP repo. We provide preprocessed training and heldout dataset in our paper, including the Census, CovType and YearPred datasets. For the TIMIT dataset, we do not provide it here due to licensing restriction. We refer to the our [paper]() for details in preprocessing the raw TIMIT dataset.
```
wget https://www.dropbox.com/s/l1jy7ilifrknd82/LP-RFFs-Data.zip?dl=0
```

## Command guidelines

* Key arguments

  * specify kernel approximation method
  ```
  --approx_type: specifies the kernel approximation method.
  --n_feat: the number of kernel approximation features.
  --do_fp_feat: use full precision kernel approximation features.
  --n_bit_feat: the number of bits for low precision fixed-point representation of kernel approximation features.

  LP-RFFs currently support:
  * FP-RFFs (--approx_type=rff --do_fp_feat)
  * circulant FP-RFFs (--approx_type=cir_rff --do_fp_feat)
  * FP-Nystrom (--approx_type=nystrom --do_fp_feat)
  * ensemble FP-Nystrom (--approx_type=ensemble_nystrom --do_fp_feat)
  * LP-RFFs (--approx_type=cir_rff --n_bit_feat=# of bits)
  * LP-Nystrom (--approx_type=ensemble_nystrom --n_bit_feat=# of bits --n_ensemble_nystrom=1)
  * ensemble LP-Nystrom (--approx_type=ensemble_nystrom --n_bit_feat=# of bits --n_ensemble_nystrom=# of learners of ensemble Nystrom).
  ```
  
  * specify training approach
  ```
  LP-RFFs currently support training the following models:
  * closed-form kernel ridge regression: 
    --closed_form_sol --model=ridge_regression
  * mini-batch based iterative training for kernel ridge regression: 
    --model=ridge_regression --opt=type of the optimizer
  * mini-batch based iterative training for logistic regression: 
    --model=logistic regression --opt=type of the optimizer
    
  LP-RFFs can use the following optimizers for min-batch based iterative training:
  * plain SGD (full precision training):
    --opt=sgd
  * LM-HALP (low precision training):
    --opt=lm_halp --n_bit_model=# of bit for model parameter during training --halp_mu=the value do determine the scale factor in LM-HALP --halp_epoch_T=#of epochs as interval to compute the scale factor in LM-HALP
  * LM-Bit-Center SGD (low precision training):
    --opt=lm_bit_center_sgd --n_bit_model=# of bit for model parameter during training --halp_mu=the value do determine the scale factor in LM-Bit-Center SGD --halp_epoch_T=# of epochs as interval to compute the scale factor in LM-Bit-Center SGD
    
  The learning rate and minibatch size can be specified using --learning_rate and --minibatch.
  For GPU based iterative training, please use --cuda.
  ```

  * --collect_sample_metrics indicates to calculate relative spectral distance, Frobenius norm error, spectral norm error on the heldout set kernel matrix. For large datasest, these metrics can be computed on a subsampled heldout set, the size of the subsampled heldout set can be specified by --n_sample=size of subsampled heldout set.
  
  * The dataset path and the output saving path can be specified with --data_path and --save_path

## Citation
If you use LP-RFFs in your project, please cite our paper
```
ArXiv entry
```

## Acknowledgement
We thank Fred Sala, Virginia Smith, Will Hamilton, Paroma Varma, Sen Wu and Megan Leszczynski for the helpful discussion and feedbacks.
