# DNN Backdoor Trigger Recovery

This repository contains a reimplementation of the underlying Sequential Monte Carlo method, dubbed Importance Splitting, used in my paper named "REStore: Exploring a Black-Box Defense against DNN Backdoors using Rare Event Simulation" [1,2].

### What this method is for

**Importance Splitting is a Sequential Monte Carlo method developped by Cérou et al.** [3] to efficiently samples observations of a rare event in a given search space, by iteratively partitioning this search space into nested, progressively rarer regions. 
The rare event thus occupies the bottom-most region in this search space.

In [1], empirical evidence shows that a **black-box** Deep Neural Network can be interrogated with Importance Splitting to find high-scoring input perturbations.
The paper demonstrates such perturbations can be used to reverse-engineer hidden backdoor triggers or universal adversarial patterns.

### Implementation in this repository

This code is a reimplementation, not the original code used to produce the paper, of Importance Splitting.
It contains:

1. a quick-and-dirty training pipeline for a CIFAR10 classifier model
2. a example backdoor injection method (poison-label, all-to-one) as a python class
3. a trained model containing 2 backdoor triggers
4. a corresponding example recovery for each trigger

### How to reproduce the content of this repository

Here is how to setup the environment for pytorch GPU 11.8 [4] using conda only (have it installed first [5]).

```console
$ conda create -n pytorch python=3.10
$ conda activate pytorch
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then you can train and run the importance splitting recovery method as such:

```console
$ python train.py
$ # TO COMPLETE
```


### How to cite the research paper

```
@INPROCEEDINGS{10516624,
  author={Le Roux, Quentin and Kallas, Kassem and Furon, Teddy},
  booktitle={2024 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)}, 
  title={REStore: Exploring a Black-Box Defense against DNN Backdoors using Rare Event Simulation}, 
  year={2024},
  volume={},
  number={},
  pages={286-308},
  keywords={Monte Carlo methods;Purification;Pipelines;Closed box;Artificial neural networks;Machine learning;Robustness;deep neural networks;backdoor defense;black-box;trigger reconstruction;input purification},
  doi={10.1109/SaTML59370.2024.00021}}
```

### links

[1] https://ieeexplore.ieee.org/document/10516624/ (citation source)

[2] https://hal.univ-lille.fr/IRISA_SET/hal-04485197v1 (open access .pdf)

[3] https://hal.science/inria-00584352/

[4] https://pytorch.org/get-started/locally/

[5] https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html