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

## Command guidelines

## Citation
If you use LP-RFFs in your project, please cite our paper
```
ArXiv entry
```

## Acknowledgement
We thank Fred Sala, Virginia Smith, Will Hamilton, Paroma Varma, Sen Wu and Megan Leszczynski for the helpful discussion and feedbacks.
