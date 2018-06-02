# Low Precision Random Fourier Features (LP-RFFs)

**LP-RFFs is a library to train classification and regression models using fixed-point low-precision random Fourier features.** Random Fourier feature (RFFs) is one of the major kernel approximation approach for kernel methods on large scale datasets. The generalization performance of kernel methods using RFFs is highly correlated with the number of RFFs; larger number of RFFs typically indicates better generalizaton, yet requires significantly larger memory footprint in minibatch-based training. **LP-RFFs use low-precision fixed-point representation for the kernel approximation features, which can achieve similar performance as with full-precision RFFs using 5-10X less memory during training.** LP-RFFs currently supports closed-form kernel ridge regression, SGD-based training for kernel ridge regression and kernel logistic regression in float-point-based simulation. 

## Content
* [Setup instructions](#setup-instructions)
* [Command guidelines](#command-guidelines)
* [Citation](#citation)
* [Acknowledgement](#acknowledgement)

## Setup instructions

## Command guidelines

## Citation

## Acknowledgement
