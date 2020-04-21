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

## Running experiments [\[^\]](#Contents)



## Citation [\[^\]](#Contents)

If you find this repository useful, then please cite the work:

```bibtex
@inproceedings{Zingaro2020,
    title={Multimodal Side-Tuning for Document Classification},
    author={Stefano P. Zingaro and Giuseppe Lisanti and Maurizio Gabbrielli},
    year={2020},
}
```

and the original paper proposing the Side-tuning framework:

```bibtex
@inproceedings{sidetuning2019,
    title={Side-tuning: Network Adaptation via Additive Side Networks},
    author={Jeffrey O. Zhang and Alexander Sax and Amir Zamir and Leonidas J. Guibas and Jitendra Malik},
    year={2019},
}
```

## References [\[^\]](#Contents)
 

