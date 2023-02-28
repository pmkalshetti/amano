# path
from pathlib import Path
import shutil
import logging
import os

# numerical
import numpy as np
import scipy
import scipy.io as sio
from scipy.spatial.transform import Rotation
from scipy import ndimage
from scipy.spatial import KDTree
from scipy.sparse import load_npz
from scipy.spatial import distance
import jax.numpy as jnp

from numba import njit
from jax import jit

# image
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from skimage.transform import rescale, rotate
import cv2 as cv
import open3d as o3d

# graphics
import igl

# plot
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import colorcet as cc

# i/o
import pickle

# utility
from tqdm.auto import tqdm
import cProfile
import pstats
import subprocess
import time
from functools import partial
import argparse