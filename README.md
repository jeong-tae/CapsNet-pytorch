# CapsNet-pytorch
This repo aims to implement a "Dynamic Routing Between Capsules"[paper](https://arxiv.org/abs/1710.09829)

## Requirements
- python 3
- [Pytorch 0.4](https://github.com/pytorch/pytorch#from-source)
- [torchvision](https://github.com/pytorch/vision)
- numpy
- [tensorflow](https://www.tensorflow.org/install/)
    - Not using right now. This for tensorboard

## TODO

- [x] Network building
- [x] Training on MNIST
- [ ] Training on MultiMNIST
- [ ] Weight load and inference
- [ ] Reconstruction at test time
- [ ] Reconstructed sample visualization
- [ ] Results reproduce and report on README.md

## Usage

For training, use following command.

See details in trainer.py to modify the hyper-params or --help will let you know

```bash
$ python trainer.py --cuda
```

Use CUDA_VISIBLE_DEVICES=$GPU, if you want to select GPU devices

```bash
$ CUDA_VISIBLE_DEVICES=0 python trainer.py --cuda
```

or try this. if you don't want to use GPU

```bash
$ python trainer.py
```

## References

- [Another pytorch implementation](https://github.com/cedrickchee/capsule-net-pytorch)
- [Tensorflow implementation](https://github.com/naturomics/CapsNet-Tensorflow)
- [Blog post: Understanding capsnet](https://becominghuman.ai/understanding-capsnet-part-1-e274943a018d)
- [Video: CapsNet tutorial](https://youtu.be/pPN8d0E3900)
    - This is best of best explanation in my mind
