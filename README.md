# [BMVC2024] HFGS: 4D Gaussian Splatting with Emphasis on Spatial and Temporal High-Frequency Components for Endoscopic Scene Reconstruction

This is the official code for [HFGS](https://arxiv.org/abs/2405.17872).

## Introduction

Robot-assisted minimally invasive surgery benefits from enhancing dynamic scene
reconstruction, as it improves surgical outcomes. While Neural Radiance Fields (NeRF)
have been effective in scene reconstruction, their slow inference speeds and lengthy training durations limit their applicability. To overcome these limitations, 3D Gaussian Splatting (3D-GS) based methods have emerged as a recent trend, offering rapid inference
capabilities and superior 3D quality. However, these methods still struggle with underreconstruction in both static and dynamic scenes. In this paper, we propose HFGS,
a novel approach for deformable endoscopic reconstruction that addresses these challenges from spatial and temporal frequency perspectives. Our approach incorporates deformation fields to better handle dynamic scenes and introduces Spatial High-Frequency
Emphasis Reconstruction (SHF) to minimize discrepancies in spatial frequency spectra
between the rendered image and its ground truth. Additionally, we introduce Temporal
High-Frequency Emphasis Reconstruction (THF) to enhance dynamic awareness in neural rendering by leveraging flow priors, focusing optimization on motion-intensive parts.
Extensive experiments on two widely used benchmarks demonstrate that HFGS achieves
superior rendering quality.

## Installation

Clone this repository and install packages:
```
git clone https://github.com/zhaohaoyu376/HFGS.git
conda env create --file environment.yml
conda activate gs
pip install git+https://github.com/ingra14m/depth-diff-gaussian-rasterization.git@depth
pip install git+https://github.com/facebookresearch/pytorch3d.git
```
Note: for the submodule diff-gaussian-rasterization of the [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting), we use the depth branch of https://github.com/ingra14m/depth-diff-gaussian-rasterization.

## Dataset

You can download the [EndoNeRF](https://github.com/med-air/EndoNeRF) from their website.

You can download the SCARED [here](https://endovissub2019-scared.grand-challenge.org/).

Use [COLMAP](https://demuc.de/colmap/) to estimate the initial point clouds. Store the files (`cameras.bin, images.bin, points3D.bin`) in the data path (e.g., `./data/cutting_tissues_twice/sparse/`).

## Training
```
python train.py {data path} --workspace {workspace}
## e.g.,
python train.py data/cutting_tissues_twice/ --workspace output/cutting/
```

## Inference
```
python inference.py {data path} --model_path {model path}
## e.g.,
python inference.py data/cutting_tissues_twice/ --model_path output/cutting/point_cloud/iteration_60000
```

## Evaluation
```
python eval_rgb.py --gt_dir {gt_dir path} --mask_dir {mask_dir path} --img_dir {rendered image path}
## e.g.,
python eval_rgb.py --gt_dir data/cutting_tissues_twice/images --mask_dir data/cutting_tissues_twice/gt_masks --img_dir output/cutting/point_cloud/iteration_60000/render
```
Note: we should use the same masks in training and evaluation. If the name 'gt_masks' exist, we use 'gt_masks'; if not, use 'masks'. And we exclude the unseen pixels in gt and rendered images for PSNR.

## Citation

If you find our work useful, please kindly cite as:
```
@article{zhao2024hfgs,
  title={HFGS: 4D Gaussian Splatting with Emphasis on Spatial and Temporal High-Frequency Components for Endoscopic Scene Reconstruction},
  author={Zhao, Haoyu and Zhao, Xingyue and Zhu, Lingting and Zheng, Weixi and Xu, Yongchao},
  journal={arXiv preprint arXiv:2405.17872},
  year={2024}
}
```

## Acknowledgement
* The codebase is developed based on [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) (Kerbl et al.), [4D-GS](https://github.com/hustvl/4DGaussians) (Wu et al.), [SuGaR](https://github.com/Anttwo/SuGaR) (Guédon et al.), [EndoNeRF](https://github.com/med-air/EndoNeRF) (Wang et al.), [unimatch](https://github.com/autonomousvision/unimatch/tree/master), and [EndoGS](https://github.com/HKU-MedAI/EndoGS) (Zhu et al.).
