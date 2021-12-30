# ORB_SLAM3_Monodepth

## Introduction
This repository was forked from [ORB-SLAM3] (https://github.com/UZ-SLAMLab/ORB_SLAM3).
ORB-SLAM3-Monodepth is an extended version of ORB-SLAM3 that utilizes a deep monocular depth estimation network.
For this pre-trained models of [Monodepth2] (https://github.com/nianticlabs/monodepth2) are used.
The monocular depth network is deployed using LibTorch and executed in an asynchronous thread in parallel with the ORB feature detection to optimize runtime.
The estimated metric depth is used to initialize map points and in the cost function similar to the stereo/RGBD case, and can significantly reduce the scale drift in the monocular case.
This approach is based on DVSO and CNN-SVO, which have extended DSO and SVO, respectively, with a monocular depth network.

 ## Example
Comparison between the monocular case and monocular case with depth estimation network (KITTI Sequence 01).

Monocular:

![](mono.gif)

Monocular with depth estimation network:

![](mono_depth.gif)

## Related Publications
[ORB-SLAM3] Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel and Juan D. Tardós, **ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM**, *IEEE Transactions on Robotics 37(6):1874-1890, Dec. 2021*. **[PDF](https://arxiv.org/abs/2007.11898)**.

[Monodepth2] Clément Godard, Oisin Mac Aodha, Michael Firman and Gabriel J. Brostow, **Digging Into Self-Supervised Monocular Depth Estimation**, *ICCV 2019*. **[PDF](https://arxiv.org/abs/1806.01260)**.

[DVSO] Nan Yang, Rui Wang, Jörg Stückler and Daniel Cremers, **Deep Virtual Stereo Odometry: Leveraging Deep Depth Prediction for Monocular Direct Sparse Odometry**, *ECCV 2018*. **[PDF](https://arxiv.org/abs/1807.02570)**.

[CNN-SVO] Shing Yan Loo, Ali Jahani Amiri, Syamsiah Mashohor, Sai Hong Tang and Hong Zhang, **CNN-SVO: Improving the Mapping in Semi-Direct Visual Odometry Using Single-Image Depth Prediction**, *ICRA 2019*. **[PDF](https://arxiv.org/abs/1810.01011)**.

# 1. License (from ORB-SLAM3)
See LICENSE file.

# 2. Prerequisites
The library is tested in **Ubuntu 16.04** and **18.04**, but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.

## C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

## Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at leat 3.0. Tested with OpenCV 3.2.0 and 4.4.0**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## DBoW2 and g2o (Included in Thirdparty folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

## Python
Required to calculate the alignment of the trajectory with the ground truth. **Required Numpy module**.

* (win) http://www.python.org/downloads/windows
* (deb) `sudo apt install libpython2.7-dev`
* (mac) preinstalled with osx

## ROS (optional)
We provide some examples to process input of a monocular, monocular-inertial, stereo, stereo-inertial or RGB-D camera using ROS. Building these examples is optional. These have been tested with ROS Melodic under Ubuntu 18.04.

## Pytorch/LibTorch
The Pytorch C++ API (LibTorch) is used for deployment. Download the pre-built version here https://pytorch.org/ (important select the cxx11 ABI).

# 3. Building ORB-SLAM3 library and examples

Clone the repository:
```
git clone https://github.com/jan9419/ORB_SLAM3_Monodepth.git ORB_SLAM3_Monodepth
```

We provide a script `build.sh` to build the *Thirdparty* libraries and *ORB-SLAM3-Monodepth*. Please make sure you have installed all required dependencies (see section 2) and the correct `LIBTORCH_PATH` is set in `build.sh`. Execute:
```
cd ORB_SLAM3_Monodepth
chmod +x build.sh
./build.sh
```

This will create **libORB_SLAM3_Monodepth.so**  at *lib* folder and the executables in *Examples* folder.

# 4. KITTI RGB-MonoDepth Examples

1. Download the dataset (color images) from http://www.cvlibs.net/datasets/kitti/eval_odometry.php 

2. Export pre-trained Monodepth2 models (trained on the KITTI dataset) to TorchScript models. For this please add the Monodepth2 repository to the `PYTHONPATH` environment variable. Furthermore, the depth decoder in the Monodepth2 repository (`networks/depth_decoder.py`) needs to be modified to return the last dictionary element (`self.outputs[("disp", 0)]`). Note that when exporting the TorchScript models, the same device (cpu or cuda) must be selected as for deployment (`DepthEstimator.device` (cpu or gpu)).  
```
python tools/export_models.py --input_encoder_path PATH_TO_MONODEPTH_PRETRAINED_MODEL/encoder.pth --input_decoder_path PATH_TO_MONODEPTH_PRETRAINED_MODEL/decoder.pth --output_encoder_path tools/encoder.pt --output_decoder_path tools/decoder.pt --device cuda
```

3. Set the correct path to the exported TorchScript models in `KITTIX.yaml` (`DepthEstimator.encoderPath` and `DepthEstimator.decoderPath`).

4. Execute the following command. Change `KITTIX.yaml` by KITTI00-02.yaml, KITTI03.yaml or KITTI04-12.yaml for sequence 0 to 2, 3, and 4 to 12 respectively. Change `PATH_TO_DATASET_FOLDER` to the uncompressed dataset folder. Change `SEQUENCE_NUMBER` to 00, 01, 02,.., 11. 
```
./Examples/RGBMonoDepth/rgb_monodepth Vocabulary/ORBvoc.txt Examples/RGBMonoDepth/KITTIX.yaml PATH_TO_DATASET_FOLDER/data_odometry_color/dataset/sequences/SEQUENCE_NUMBER
```
