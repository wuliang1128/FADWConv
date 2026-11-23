1. Repository Structure

├── datasets/               # Dataset directory (Place your OBIMD data here)
│   └── OBIMD_tapian_YOLO/ 
│       └── dataset.yaml
├── my_modules/             # Core custom modules (FA_WTConv2dv3, SAB, etc.)
│   ├── fa_wtconv_v3.py
│   ├── lightweight_sab.py
│   ├── wtconv2d.py
│   └── ...
├── ultralytics/            # Modified Ultralytics YOLO framework
│   ├── cfg/models/         # Custom YAML files (e.g., yolo26n-fadg-wtconv.yaml)
│   └── nn/modules/block.py # Modified block.py containing C3k2_FADG_WTConv
└── README.md

2. Environment Setup
We recommend using Anaconda to create a virtual environment. Our experiments are conducted on Python 3.9, PyTorch 2.0.1, and CUDA 11.8.

bash
conda create -n xxx python=3.9 -y
conda activate xxx

# Alternatively, manually install the required packages:
# pip install torch torchvision ultralytics opencv-python pyyaml tqdm pandas matplotlib

3.Dataset Preparation
Download the OBIMD_tapian_YOLO(OBIM_YOLO) dataset via the download link provided in the GitHub repository README (https://github.com/ZCDMW/OracleDet), extract it, and place it in the datasets/ directory：

text
datasets/
└── OBIMD_tapian_YOLO/
    ├── dataset.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
Update the dataset paths in your dataset.yaml file:

yaml:
path: ./datasets/OBIMD_tapian_YOLO
train: train/images
val: val/images

nc: 1
names: ['Oracle']