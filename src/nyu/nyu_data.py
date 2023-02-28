from utils.freq_imports import *
from utils.perspective_projection import uvd2xyz, xyz2uv

def read_nyu_depth(img_path):
    """Depth is encoded in png image as: green channel contains top 8 bits, blue channel: contains latter 8 bits."""
    img = cv.imread(str(img_path))[:, :, ::-1].astype(np.float32)
    depth = 255*img[:, :, 1] + img[:, :, 2]
    return depth

def get_nyu_camera_params():
    fx = 588.03
    fy = 587.07
    cx = 320.
    cy = 240.
    return fx, fy, cx, cy
    

def crop_depth_using_keypoints(D, k_xyz, fx, fy, cx, cy):
    # crop around hand region in 3D
    k_uv = xyz2uv(k_xyz, fx, fy, cx, cy)
    
    # get skeleton bounds from keypoints
    uv_min = np.amin(k_uv, axis=0)
    uv_max = np.amax(k_uv, axis=0)

    # pad skeleton to cover hand
    pad = 30    # pixels on each side
    u_min = int(max(uv_min[0] - pad, 0))
    v_min = int(max(uv_min[1] - pad, 0))
    u_max = int(min(uv_max[0] + pad, D.shape[1]))
    v_max = int(min(uv_max[1] + pad, D.shape[0]))

    # crop
    D_crop = np.zeros_like(D)
    D_crop[v_min:v_max, u_min:u_max] = D[v_min:v_max, u_min:u_max]

    # z bounds on skeleton
    z_min = np.amin(k_xyz[:, 2])
    z_max = np.amax(k_xyz[:, 2])

    # pad skeleton to cover hand
    pad_z = 0.02    # m on each side
    z_min = z_min - (pad_z + 0.03)  # additional min_z handles offset from keypoints to hand points 
    z_max = z_max + pad_z
    d_min = z_min*1000; d_max = z_max*1000

    D_clip = np.copy(D_crop)
    D_clip[D_clip < d_min] = 0
    D_clip[D_clip > d_max] = 0

    return D_clip