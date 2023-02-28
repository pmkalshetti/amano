# Overview
This directory contains code for our SCA 2022 paper "Local-scale Adaptation to Hand Shape Model for Accurate and Robust Hand Tracking".

# Setup
The below steps have been tested on Ubuntu 18.04

1. Install required libraries by following the instructions in install_packages.sh

2. Execute setup_dev_env.sh to set up the development environment in the current terminal
    
3. Install gptoolbox from https://github.com/alecjacobson/gptoolbox

# Download data
1. MANO [1]
    Download the MANO model from https://mano.is.tue.mpg.de/ to `data/mano/` directory.

2. NYU hand pose dataset [2]
    Download the NYU hand pose dataset from https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm to `data/nyu/` directory.

3. AWR [3] predictions
    Download the AWR predicted results on NYU test set from https://github.com/Elody-07/AWR-Adaptive-Weighting-Regression/blob/master/results/resnet_18.txt and place it under `data/awr/nyu_predictions/` directory.

# Usage
1. Create aMANO
    - Create .obj file from MANO .pkl file (for easy reading)
        ```
        python src/hand_model/create_obj_from_pkl.py
        ```

    - Define vertex ids surrounding keypoints
        ```
        python src/hand_model/define_verts_around_keypoints.py
        ```
    
    - Define rotation axes
        ```
        python src/hand_model/define_axis_per_dof.py
        ```
    
    - Compute bone and endpoint weights
        - Create .tgf files required as per specification in gptoolbox.
            ```
            python src/hand_model/lbs_weights/create_skeleton_tgf.py
            ```

        - Bounded biharmonic weight [4] computation requires meshes to not have boundaries. Specifically, tetgen requires this condition. So we close the holes in `output/hand_model/mesh.obj` (load in MeshLab, Filters->Remeshing, Simplification and Reconstruction->Close Holes) and export the mesh as `output/hand_model/mesh_hole_closed.obj`.
        
        - Compute bone and endpoint weights using `src/hand_model/lbs_weights/compute_weights.m` which writes the weights at `./output/hand_model/lbs_weights/W.mat`

    - Generate pose prior from synthetic data
        ```
        python src/hand_model/create_syn_data.py
        python src/hand_model/compute_theta_prior.py
        ```

2. Register aMANO on NYU using AWR as fingertip reinitializer
    ```
    python src/nyu/register_amano.py
    ```

# References
1. Javier Romero, Dimitrios Tzionas, and Michael J. Black. 930 Embodied hands: Modeling and capturing hands and bodies together. *ACM TOG*, 36(6):245:1–245:17, 2017.
2. Jonathan Tompson, Murphy Stein, Yann Lecun, and Ken Perlin. Real-time continuous pose recovery of human hands using convolutional networks. *ACM TOG*, 33, 2014.
3. Weiting Huang, Pengfei Ren, Jingyu Wang, Qi Qi, and Haifeng Sun. Awr: Adaptive weighting regression for 3d hand pose estimation. In *AAAI*, 2020.
4. Alec Jacobson, Ilya Baran, Jovan Popović, and Olga Sorkine. Bounded biharmonic weights for real-time deformation. *ACM TOG*, 30(4):78:1–78:8, 2011.