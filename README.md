# LLA: Loss-aware Label Assignment for Dense Pedestrian Detection

![GitHub](https://img.shields.io/github/license/Megvii-BaseDetection/DeFCN)

This project provides an implementation for "[LLA: Loss-aware Label Assignment for Dense Pedestrian Detection](https://arxiv.org/abs/2101.04307)" on PyTorch. 

**LLA is the first one-stage detector that surpasses two-stage detectors (e.g., Faster R-CNN) on CrowdHuman dataset**. Experiments in the paper were conducted on the internal framework, thus we reimplement them on [cvpods](https://github.com/Megvii-BaseDetection/cvpods) and report details as below.

<img src="./result.png" width="800" height="400">

## Requirements
* [cvpods](https://github.com/Megvii-BaseDetection/cvpods)
* scipy >= 1.5.4

## Get Started

* install cvpods locally (requires cuda to compile)
```shell

python3 -m pip install 'git+https://github.com/Megvii-BaseDetection/cvpods.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/Megvii-BaseDetection/cvpods.git
python3 -m pip install -e cvpods

# Or,
pip install -r requirements.txt
python3 setup.py build develop
```

* prepare datasets
```shell
cd /path/to/cvpods/datasets
ln -s /path/to/your/crowdhuman/dataset crowdhuman
```

* Train & Test
```shell
git clone https://github.com/Megvii-BaseDetection/LLA.git
cd LLA/playground/detection/crowdhuman/lla.res50.fpn.crowdhuman.800size.30k  # for example

# Train
pods_train --num-gpus 8

# Test
pods_test --num-gpus 8 \
    MODEL.WEIGHTS /path/to/your/save_dir/ckpt.pth # optional
    OUTPUT_DIR /path/to/your/save_dir # optional

# Multi node training
## sudo apt install net-tools ifconfig
pods_train --num-gpus 8 --num-machines N --machine-rank 0/1/.../N-1 --dist-url "tcp://MASTER_IP:port"

```

## Results on CrowdHuman val set

| Model | Backbone | LR Sched. | Aux. Branch | NMS Thr. | MR | AP50 |  Recall | Download |
|:------| :----:   | :----: |:---:| :---:| :---:|:---:| :---: | :--------: |
|  [FCOS](https://github.com/Joker316701882/LLA/tree/main/playground/detection/crowdhuman/fcos.res50.fpn.crowdhuman.800size.30k) | Res50   | 30k       | CenterNess | 0.6 | 54.4     |  86.0       | 94.1    | [weights](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EY6b1GdqgjROlY2G9Joe45YBmesD804Agf7mRG6lJBHiOQ) |
|  [ATSS](https://github.com/Joker316701882/LLA/tree/main/playground/detection/crowdhuman/atss.res50.fpn.crowdhuman.800size.30k) | Res50   | 30k       | CenterNess | 0.6 | 49.4     |  87.3       | 94.1    | [weights](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EYLuG9lfetJKqdXXu5vc0yMB82pzTdN6xYy-wmypnpIKGg?e=wsBvhk) |
| [Faster R-CNN](https://github.com/Megvii-BaseDetection/cvpods/tree/master/playground/detection/crowdhuman/rcnn/faster_rcnn.res50.fpn.crowdhuman.800size.1x) | Res50  | 30k | -       | 0.5 | 48.5         |   84.3    | 87.1       |  [weights](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EV-hTTWEZSJCnw08Eg0Dr18BxKh-jiIaMVW_DBQUZe0cKw?e=YyeNa8)    |
| [LLA.FCOS](https://github.com/Joker316701882/LLA/tree/main/playground/detection/crowdhuman/lla.res50.fpn.crowdhuman.800size.30k) | Res50 | 30k       | IoU        | 0.6 | **47.5**  | **88.2**    | **94.4** | [weights](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EdVJdAm0RINGnS5LoroQ2eUBg-Gwcaf7sbSl7eu7QX35rw) |

## Acknowledgement
This repo is developed based on cvpods. Please check [cvpods](https://github.com/Megvii-BaseDetection/cvpods) for more details and features.

## License
This repo is released under the Apache 2.0 license. Please see the LICENSE file for more information.

## Citing
If you use this work in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:
```
@article{ge2021lla,
  title={LLA: Loss-aware Label Assignment for Dense Pedestrian Detection},
  author={Ge, Zheng and Wang, Jianfeng and Huang, Xin and Liu, Songtao and Yoshie, Osamu},
  journal={arXiv preprint arXiv:2101.04307},
  year={2021}
}
```
