# QuantGenMdl

This repository contains the official Python implementation of [*Generative Quantum Machine Learning via Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2310.05866), an article by [Bingzhi Zhang](https://sites.google.com/view/bingzhi-zhang/home), [Peng Xu](https://francis-hsu.github.io/), [Xiaohui Chen](https://the-xiaohuichen.github.io/), and [Quntao Zhuang](https://sites.usc.edu/zhuang).

## Citation
```
@misc{zhang2023generative,
      title={Generative quantum machine learning via denoising diffusion probabilistic models}, 
      author={Bingzhi Zhang and Peng Xu and Xiaohui Chen and Quntao Zhuang},
      year={2023},
      eprint={2310.05866},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

## Prerequisite
The simulation of quantum circuit is performed via the [TensorCircuit](https://tensorcircuit.readthedocs.io/en/latest/#) package. We explored with the [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), and [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) backends during development. As a result, all three are required in order to run all the notebooks presented in this repository. Use of GPU is not required, but highly recommended.

Additionally, the packages [POT](https://pythonot.github.io/) and [OTT](https://ott-jax.readthedocs.io/en/latest/) are required for the computation of Wasserstein distance, [`opt_einsum`](https://optimized-einsum.readthedocs.io/en/stable/) is used for speeding up certain evaluation, and [Optax](https://github.com/google-deepmind/optax) is needed for optmization with JAX backend.

## File Structure
Presented in this repository are notebooks.