conda create --name env_amano python=3.8
conda activate env_amano

# install libraries only available with conda before installing any packages with pip (https://www.anaconda.com/blog/using-pip-in-a-conda-environment)
conda install -c conda-forge igl

# upgrade pip
pip install -U pip

# numerical libraries
pip install scipy scikit-learn
pip install --upgrade "jax[cpu]"

# hyperparameter tuning
pip install optuna

# image, video, graphics io
pip install scikit-image opencv-python open3d
pip install kaleido plotly
pip install ipympl
pip install colorcet
pip install moderngl moderngl-window

# utility
pip install tqdm
pip install numba

# chumpy required for loading MANO model
pip install chumpy