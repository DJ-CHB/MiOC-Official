# MiOC Official Implementation


The official implementattion of the paper "One-Class SVM-guided Negative Mixing for
Enhanced Contrastive Learning". The paper is availabe at [Arxiv]().

* This repository is based on [MoCo Official Implementation](https://github.com/facebookresearch/moco) , with modifications.
## Requirements

```
pip install numpy torch blobfile tqdm pyYaml pillow jaxtyping beartype pytorch-lightning omegaconf hydra-core wandb
```

## Mixing One-Class SVM Negative Samples - MiOC

While for MiOC,
- We use the One-Class SVM to find the negative samples that are most similar to the query.

- we generate $s_1'$, synthetic negative samples by mixing a random query $z^q$ with random negative samples$z^-$.
- we generate the $s_2'$, synthetic negative samples by mixing a random query $z^q$ with inner One-Class SVM negative samples$z^-$.

<img src="Images/MiOC.png" alt="MiOC">

Please refer to the paper for more details.

The code is present in the [Code](Code) folder.
Please go through the [Instructions](Code/Instructions.md) for the implementation details. 

Here are the results for the Imagenet-100 dataset. 

<img src="Images/Imagenet-100.png" alt="Imagenet-100" width="400">

## See Also
We would recommend to read the 
- [MoCo Article](https://arxiv.org/abs/1911.05722).
- [MoCHI Article](https://europe.naverlabs.com/research/publications/hard-negative-mixing-for-contrastive-learning/).

## License

This repository is released under the MIT license. 

## Contributing to MiOC

Please see this link -  [How to CONTRIBUTE](.github/CONTRIBUTING.md) for more details.

## Code of Conduct

Please see the [CODE OF CONDUCT](.github/CODE_OF_CONDUCT.md) for more details.
