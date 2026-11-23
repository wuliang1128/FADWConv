1. Repository Structure

├── datasets/               # Dataset directory (Place your OBCD data here)
│   └── OBCD_YOLO/ 
│       └── dataset.yaml
├── my_modules/             # Core custom modules (FA_WTConv2dv3, SAB, etc.)
│   ├── fa_wtconv_v3.py
│   ├── lightweight_sab.py
│   ├── wtconv2d.py
│   └── ...
├── ultralytics/            # Modified Ultralytics YOLO framework
│   ├── cfg/models/         # Custom YAML files (e.g., yolov8-fa-v3.yaml)
│   └── nn/modules/block.py # Modified block.py containing C2f_WTConvV3
└── README.md

2. Environment Setup
We recommend using Anaconda to create a virtual environment. Our experiments are conducted on Python 3.9, PyTorch 2.0.1, and CUDA 11.8.

bash
conda create -n xxx python=3.9 -y
conda activate xxx

# Alternatively, manually install the required packages:
# pip install torch torchvision ultralytics opencv-python pyyaml tqdm pandas matplotlib

3.Dataset Preparation
Download the OBCD dataset from https://jgw.aynu.edu.cn/home/down/index.html.

Organize the dataset into the YOLO format and place it in the datasets/ directory:

text
datasets/
└── OBCD_YOLO/
    ├── dataset.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
Update the dataset paths in your dataset.yaml file:

yaml:
path: ./datasets/OBCD_YOLO
train: train/images
val: val/images
nc: 1
#class_0 denotes oracle bone inscription (OBI) region
names: ['class_0']