# LBAM

PyTorch implementation of "Image Inpainting with Learnable Bidirectional Attention Maps (ICCV 2019)" [[Paper]](https://arxiv.org/abs/1909.00968)

**Authors**: _Chaohao Xie, Shaohui Liu, Chao Li, Ming-Ming Cheng, Wangmeng Zuo, Xiao Liu, Shilei Wen, Errui Ding_

## Prerequisites

* Python 3
* PyTorch 1.0
* NVIDIA GPU + CUDA cuDNN

## Installation

* Clone this repo:

```
git clone https://github.com/Xiefan-Guo/LBAM.git
cd LBAM
```

## Usage

### Training

To train the LBAM model:

```
python train.py \ 
    --image_root [path to input image directory] \ 
    --mask_root [path to masks directory]
```

### Evaluating

To evaluate the model:

```
python eval.py \
    --pre_trained [path to checkpoints] \
    --image_root [path to input image directory] \ 
    --mask_root [path to masks directory]
```
