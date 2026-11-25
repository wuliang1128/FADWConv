# FADWConv: A Frequency-Aware Dual-Gated Wavelet Convolution for Oracle Bone Inscription Detection
Automated detection of oracle bone inscriptions (OBI) in scanned rubbings is hindered by crack textures, whose high-frequency components overlap with character strokes. Most spatial-domain models struggle to separate these overlapping frequency signals. Although Wavelet Transform Convolution (WTConv) introduces frequency analysis, its uniform scaling and fixed fusion fail to distinguish structure-bearing low frequencies from crack-induced high frequencies. We propose FADWConv, a frequency-aware dual-gated wavelet convolution comprising a spatial branch and a wavelet branch. The wavelet branch incorporates a subband attention (SBA) module and independent per-channel scaling for low- and high-frequency groups. A spatial gate and a channel gate then fuse these two branches at pixel and channel levels. Integrated into YOLOv8m, FADWConv improves precision from 91\% to 94.9\% and mAP@0.5 from 84.7\% to 86.1\% on OBCD. On OBIMD, it raises precision from 87.7\% to 89.0\% and mAP@0.5 from 85.2\% to 86.6\%. As a plug-and-play module, it also consistently improves accuracy in detectors including YOLOv11m, YOLO26m/n, and OracleDet.

In this study, we propose **FADWConv**, a frequency-aware dual-gated wavelet convolution module, and incorporate it into YOLOv8m and YOLO26 for Oracle Bone Inscription (OBI) detection. In this repository, we provide the complete evolution of our module design (v1 to v4) and the corresponding training/validation scripts.

***Note: The original name was FADG-WTConv, which was subsequently changed to FADWConv.*
## Repository Structure

We provide two independent integration pipelines to ensure reproducibility. The core module versions are organized separately.

```text
.
├── fa_wtconv_v1/             # Version 1: Differentiated scaling + global fusion
├── fa_wtconv_v2/             # Version 2: Dual-branch + spatial-channel gating
├── fa_wtconv_v3/             # Version 3: Dual-branch + SAB + dual gating [BEST]
├── fa_wtconv_v4/             # Version 4: Dual-path learnable wavelet decomposition (Discarded)
│
├── 调用方案/                  # Integration Pipelines (Two runnable projects)
│   ├── yolov8m(插入式改进C2f)/ 
│   │   ├── datasets/         # Place your OBCD data here
│   │   ├── my_modules/       # Core custom modules (v3, SAB, WTConv, util)
│   │   ├── ultralytics/      # Modified Ultralytics framework (Insertion in C2f)
│   │   └── README.md
│   │
│   └── yolo26(替换bottleneck中卷积改进C3k2)/
│       ├── datasets/         # Place your OBIMD data here
│       ├── my_modules/       # Core custom modules (v3, SAB, WTConv, util)
│       ├── ultralytics/      # Modified Ultralytics framework (Replacement in C3k2)
│       └── README.md
│
└── README.md
```

## Version Evolution

| Version | Core Improvement (compared to Original WTConv)                                                                                                            | Design Logic                                                                                                                                                |
| :------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **v1**  | Introduced learnable differentiated scaling (LL & H independent) and fusion coefficient.                                                                  | Original WTConv directly adds spatial and wavelet branches. v1 lets the network learn weights for LL (structure) and H (noise) separately.                  |
| **v2**  | Upgraded global scalar to a **spatial-channel dual gating** mechanism and added residual connections.                                                     | Global scalars cannot adapt to different image regions. v2 uses spatial (pixel-wise) and channel (channel-wise) gating to fuse the two branches adaptively. |
| **v3**  | Introduced **Subband Attention (SAB)** within the wavelet branch, forming an "intra-frequency purification + inter-domain selection" two-level attention. | Convolution acts as a coarse filter, while SAB performs fine screening. BatchNorm is replaced with GroupNorm to prevent FP16 numerical corruption.          |
| **v4**  | Replaced fixed Haar wavelet with a dual-path learnable wavelet decomposition.                                                                             | Attempted to adapt wavelet bases to OBI stroke directionality.                                                                                              |

## Why Version 3 is the Final Choice (and Why Version 4 was Discarded)

**1. Theoretical Risk of v4: Breaking Perfect Reconstruction**
v3 uses a fixed Haar basis, ensuring lossless reconstruction (IWT). The learnable filters in v4 easily deviate from mathematical constraints on small datasets, causing spatial artifacts and contaminating clean high-frequency stroke information.

**2. Experimental Results of v4: Severe Overfitting**
The dual-path parallel structure in v4 significantly increases the parameter count. However, the OBCD dataset only has 8,895 training images, leading to severe overfitting.

**3. v3 is Already Sufficient**
v3 perfectly resolves the trade-off between noise removal and stroke edge preservation via SAB and dual gating. Accordingly, it delivers the best experimental performance.

## Quick Start: Training & Validation

We provide two integration strategies for different baselines. Both use the **v3** module.

### YOLOv8m (Insertion-style in C2f)

* **Config file**: `ultralytics/cfg/models/v8/yolov8m-fa-v3.yaml`

* **Module**: `C2f_WTConvV3` (FA-WTConv inserted before the Bottleneck sequence)

**CLI:**

```
yolo detect train model=ultralytics/cfg/models/v8/yolov8m-fa-v3.yaml data=datasets/OBCD_YOLO/dataset.yaml epochs=250 batch=16 imgsz=640 device=0 
```

**Python API:**

```
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v8/yolov8m-fa-v3.yaml')
model.train(
    data='datasets/OBCD_YOLO/dataset.yaml',
    epochs=250,
    batch=16,
    imgsz=640,
    device=0,
)
```

**Note**: The training commands and hyperparameters provided above are merely examples.

### YOLOv26 (Replacement-style in C3k2)

* **Config file**: `ultralytics/cfg/models/v26/yolo26n-fadg-wtconv.yaml`

* **Module**: `C3k2_FADG_WTConv` (3x3 spatial conv in Bottleneck replaced by FA-WTConv)

**CLI:**

```
yolo detect train model=ultralytics/cfg/models/v26/yolo26n-fadg-wtconv.yaml data=datasets/OBIMD_tapian_YOLO/dataset.yaml epochs=150 batch=16 imgsz=640 device=0
```

**Python API:**

```
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v26/yolo26n-fadg-wtconv.yaml')
model.train(
    data='datasets/OBIMD_tapian_YOLO/dataset.yaml',
    epochs=150,
    batch=16,
    imgsz=640,
    device=0,
)
```

**Note**: The training commands and hyperparameters provided above are merely examples.
