## UEOF: A Benchmark Dataset for Underwater Event-Based Optical Flow

### Overview

Underwater imaging is fundamentally challenging due to wavelength-dependent
light attenuation, strong scattering from suspended particles,
turbidity-induced blur, and nonuniform illumination. These effects impair
standard cameras and make ground-truth motion nearly impossible to obtain. To
address this problem, we introduce the first synthetic underwater benchmark
dataset for event-based optical flow derived from physically-based ray-traced
RGBD sequences.

<p align="center">
  <img src="images/ueof_overview.png" alt="ueof overview" width="400"/>
</p>

This repository provides source code for our 2026 WACV Workshop paper titled
"[UEOF: A Benchmark Dataset for Underwater Event-Based Optical
Flow](https://openaccess.thecvf.com/content/WACV2026W/EVGEN-2026/papers/Truong_UEOF_A_Benchmark_Dataset_for_Underwater_Event-Based_Optical_Flow_WACVW_2026_paper.pdf)."
Using a modern video-to-event pipeline applied to rendered underwater videos,
we produce realistic event data streams with dense ground-truth flow, depth,
and camera motion. Moreover, we benchmark state-of-the-art learning-based and
model-based optical flow prediction methods to understand how underwater light
transport affects event formation and motion estimation accuracy. Our dataset
establishes a new baseline for future development and evaluation of underwater
event-based perception algorithms.

More information on the project can be found on the [UEOF
website](https://robotic-vision-lab.github.io/ueof).

### Citation

If you find this project useful, then please consider citing both our paper and
dataset.

```bibtex
@inproceedings{truong2026ueof,
  title={{UEOF}: A benchmark dataset for underwater event-based optical flow},
  author={Truong, Nick and Karmokar, Pritam P and Beksi, William J},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
  pages={645--655},
  year={2026}
}

@data{mavmatrix/dataset.2026.02.045,
  title={{UEOF}},
  author={Truong, Nick and Karmokar, Pritam P and Beksi, William J},
  publisher={MavMatrix},
  version={V1},
  url={https://doi.org/10.32855/dataset.2026.02.045},
  doi={10.32855/dataset.2026.02.045},
  year={2026}
}
```

### UEOF Pipeline 

<p align="center">
  <img src="images/ueof_pipeline.png" alt="ueof pipeline" width="800"/>
</p>

### Installation

### Dataset

The dataset can be downloaded from
[MavMatrix](https://mavmatrix.uta.edu/rvl_ebv_datasets/1/). It consists of 12
minutes and 51 seconds of data across 13,714 RGB frames. This results in a
total of 4.94 billion events across all scenes.

### UEOF Source Code License

[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/robotic-vision-lab/UEOF-A-Benchmark-Dataset-For-Underwater-Event-Based-Optical-Flow/blob/main/LICENSE)

### UEOF Dataset License

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
