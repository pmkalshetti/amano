from utils.freq_imports import *
from utils.helper import create_dir

"""
Write control vertices and edges in .tgf file

terminology:
- an affine transformation is associated at each control handle (Hj in paper)
- edges can be one of "bone", "pseudo" or "cage";
  - bone: useful for defining skeleton
  - pseudo: assists providing rotations for point handles (used only in UI and not for computing bbw)
  - cage: useful for precise area/volume controls; for a cage edge the weights vary linearly along the bone end points and is zero for all other handles.
- control handles not attached to any bone serve as point handles (if no point handles are given, all control handles are used a point handles)

.tgf format is used to describe the control handles and edges (required for using the 'bounded biharmonic weights (bbw)' matlab code)
there are two sets of lines separated by a line with '#'
each line in the first set of lines, contain the index of the control handle and its position (e.g. "4   0.02439  -0.01741   0.07192"); Note: control handles are 1-indexed
each line in the second set of lines, describe each edge by a set of 5 numbers
 - index of control handle corresponding to the endpoint of the edge 
 - index of control handle corresponding to the startpoint of the edge
 - 1 if bone edge else 0
 - 1 if pseudo edge else 0
 - 1 if cage edge else 0

In our case, we have a control handle corresponding to each keypoint
all our control handles are point handles
For computing bone weights: all our edges are bone edges
For computing end point weights: all our edges are cage edges, because for a cage edge the weights vary linearly along the bone end points and is zero for all other handles.
"""

def main():
    v, _, _, F, _, _ = igl.read_obj("./output/hand_model/mesh.obj")
    K = load_npz("./output/hand_model/K.npz")
    k = K @ v

    out_dir = "./output/hand_model/lbs_weights"; create_dir(out_dir, False)

    with open(f'{out_dir}/skeleton.tgf', 'w') as file:
        # write skeleton joints (indexing starts at 1)
        for i in range(len(k)):
            file.write(f'{i+1:>2d}{k[i, 0]:>10.5f}{k[i, 1]:>10.5f}{k[i, 2]:>10.5f}\n')

        file.write('#\n')

        # write skeleton edges (indexing starts at 1)
        for id_finger in range(5):
            mcp_id = id_finger * 4 + 2
            pip_id = mcp_id + 1
            dip_id = pip_id + 1
            tip_id = dip_id + 1

            file.write(f'{1:>3d}{mcp_id:>3d}{1:>3d}{0:>3d}{0:>3d}\n')
            file.write(f'{mcp_id:>3d}{pip_id:>3d}{1:>3d}{0:>3d}{0:>3d}\n')
            file.write(f'{pip_id:>3d}{dip_id:>3d}{1:>3d}{0:>3d}{0:>3d}\n')
            file.write(f'{dip_id:>3d}{tip_id:>3d}{1:>3d}{0:>3d}{0:>3d}\n')

        file.write('#\n')
    

if __name__ == "__main__":
    main()
