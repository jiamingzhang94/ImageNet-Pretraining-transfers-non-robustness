# ImageNet_Pretraining_transfers_non-robustness
Code relative to "[ImageNet Pre-training also Transfers Non-Robustness](https://arxiv.org/abs/2106.10989)", AAAI2023

## Requirements
- torch
- foolbox
- torchvision
- Pillow

## How to use?

Regarding the so many experiments conducted in the original paper, we provide some key codes here. 
If you have any request about the rest implementations, plz contact me.

### Fine-tune
Fine-tune a model starting from ImageNet pre-trained model.
```
python train.py
```

### Discrepancy Mitigating
Robust fine-tuning via Discrepancy Mitigating (DM).
```
python train_dm.py
```


## Citation
If you find this code to be useful for your research, please consider citing.
```
@inproceedings{zhang2023imageNet,
  title={ImageNet Pre-training also Transfers Non-Robustness},
  author={Zhang, Jiaming and Sang, Jitao and Yi, Qi and Yang, Yunfan and Dong, Huiwen and Yu, Jian},
  booktitle="Proceedings of the AAAI Conference on Artificial Intelligence",
  year={2023}
}
```