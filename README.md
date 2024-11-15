# Pre-training Q-Former
This repository contains code for pre-training Q-Former using the transformers library. The code supports both training from scratch and converting pre-trained LAVIS BLIP-2 models to the PyTorch transformers format.

### Features

- Pre-train Q-Former from scratch using transformers library
- Convert LAVIS BLIP-2 Q-Former weights to transformers format

## Usage

### From Scratch
To run the script for pre-training Q-Former from scratch, use the following command:
```
sh run.sh
```


### From LAVIS BLIP-2
To run the script for pre-training Q-Former from lavis, use the following command:
```
```

## Citation
```bibtex
@article{blip2,
    title={BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models},
    author={Junnan Li and Dongxu Li and Silvio Savarese and Steven Hoi},
    journal={arXiv:2301.12597},
    year={2023}
}
```