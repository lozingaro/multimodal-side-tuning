# Multimodal `Side-tuning`

<div>
  <img src="./assets/img/method_model_overview.png" alt="method model overview" width="1462" />
</div>

In this repository we provide implementation of side-tuning framework using multimodal input features. The side-tuning framework was originally proposed in the paper [Side-Tuning: Network Adaptation via Additive Side Networks](https://arxiv.org/abs/1912.13503).

## Multimodal Side-Tuning for Document Classification

 method proposed in the paper `citation`.

## Contents 
- [Quickstart](#quickstart-)
- [Running experiments](#running-experiments-)
- [Citation](#citation-)
- [References](#references-)

## Quickstart [\[^\]](#Contents)

The proposed implementation depends on `Python3.6+` and uses the `PyTorch` deep-learning framework, a list of used packages along with their version can be found in the [requirements.txt](requirements.txt). Simpy install it via:

```sh
pip install -r requirements.txt
```

However, some specification should be made:

1. the `FastText` package has been compiled [from the source repository](https://github.com/facebookresearch/fastText) and not installed from `PyPI`.

Download the Tobacco3482 dataset to test the demo or try with the RVL-CDIP full dataset.

## Running experiments [\[^\]](#Contents)

Describe the process to perform demo at least with Tobacco3482 dataset.

## Citation [\[^\]](#Contents)

If you find this repository useful, then please cite the work:

```bibtex
@inproceedings{Zingaro2020,
    title={Multimodal Side-Tuning for Document Classification},
    author={Stefano P. Zingaro and Giuseppe Lisanti and Maurizio Gabbrielli},
    year={2020},
}
```

## References [\[^\]](#Contents)
 
- J. O. Zhang, A. Sax, A. Zamir, L. Guibas, and J. Malik, “Side-Tuning: Network Adaptation via Additive Side Networks,” 2019.
- N. Audebert, C. Herold, K. Slimani, and C. Vidal, “Multimodal deep networks for text and image-based document classification,” Jul. 2019.
- M. Z. Afzal, A. Kolsch, S. Ahmed, and M. Liwicki, “Cutting the Error by Half: Investigation of Very Deep CNN and Advanced Training Strategies for Document Image Classification,” Proc. Int. Conf. Doc. Anal. Recognition, ICDAR, vol. 1, pp. 883–888, 2018.
- A. W. Harley, A. Ufkes, and K. G. Derpanis, “Evaluation of deep convolutional nets for document image classification and retrieval,” Proc. Int. Conf. Doc. Anal. Recognition, ICDAR, vol. 2015-November, pp. 991–995, 2015.
