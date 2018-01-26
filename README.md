# CapsNet-pytorch
This repo aims to implement a "Dynamic Routing Between Capsules"

## TODO

- [x] Network building
- [x] Training on MNIST
- [ ] Training on MultiMNIST
- [ ] Weight load and inference
- [ ] Reconstruction at test time
- [ ] Reconstructed sample visualization
- [ ] Results reproduce and report on README.md

## Usage

For training, use following command. GPU Selection is not available on this repo.

Use CUDA_VISIBLE_DEVICES=$GPU, instead of GPU selection

See details in trainer.py to modify the hyper-params

```
python trainer.py --cuda
```

or try this. if you don't want to use GPU

```
python trainer.py
```
