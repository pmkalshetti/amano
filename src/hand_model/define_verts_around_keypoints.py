from utils.freq_imports import *
from utils.helper import create_dir
from utils.plotly_wrapper import scatter3d, wireframe3d, remove_fig_background, update_fig_size, mesh3d, update_fig_camera
from scipy.sparse import lil_array, save_npz


def main():
    v, _, _, F, _, _ = igl.read_obj('./output/hand_model/mesh.obj')

    log_dir = "./log/hand_model"; create_dir(log_dir, False)
    
    # plot vertex ids
    scat_v = scatter3d(v, text=len(v), size=5, color='peru')
    wireframe = wireframe3d(v, F)
    fig = go.Figure([scat_v, wireframe])
    remove_fig_background(fig)
    update_fig_size(fig)
    fig.write_html(f"{log_dir}/verts_with_id.html")

    # # manually defined 4 vertex ids per keypoint that surround it
    # Note: if the keypoint positions are made to resemble the Bighand dataset (https://sites.google.com/view/hands2019/challenge#h.p_Xy44LIGf8uwJ), then there are skinning creates artifacts at mcp
    I_v_surr_k = np.array([
        [35, 110, 209, 192],  # root

        # thumb
        [113, 229, 114, 232],   # mcp
        [123, 126, 105, 286],   # pip
        [753, 711, 712, 707],   # dip 
        [766, 729, 745, 744],   # tip


        # index
        [168, 274, 62, 144],    # mcp
        [48, 87, 156, 225],     # pip
        [342, 295, 300, 297],   # dip
        [353, 319, 333, 320],   # tip

        # middle
        [288, 271, 270, 220],     # mcp
        [365, 358, 394, 373],   # pip
        [413, 452, 406, 408],   # dip
        [465, 445, 444, 443],   # tip

        # ring
        [141, 290, 77, 183],     # mcp
        [470, 477, 504, 479],   # pip
        [565, 518, 523, 520],   # dip
        [576, 556, 555, 554],   # tip

        # small
        [770, 604, 202, 83],     # mcp
        [582, 588, 603, 586],   # pip
        [682, 658, 685, 637],   # dip
        [693, 673, 672, 671],   # tip
    ])

    # let's construct the sparse matrix K which will be used to obtain keypoints from vertices
    # k = K @ v
    n_k = I_v_surr_k.shape[0]; n_v = v.shape[0]
    K = lil_array((n_k, n_v))   # (|k|, |v|)
    for k_id in range(n_k):
        K[k_id, I_v_surr_k[k_id]] = 1/4 # each keypoint is surrounded by 4 vertices
    K = K.tocsr()
    out_dir = "./output/hand_model/"; create_dir(out_dir, False)
    np.save(f'{out_dir}/vertex_ids_surr_keypoints.npy', I_v_surr_k)
    save_npz(f"{out_dir}/K.npz", K)

    # plot
    k = K @ v
    v_surr_k = np.take(v, I_v_surr_k, axis=0) # (21, 4, 3)
    scat_v_surr_k = scatter3d(v_surr_k.reshape(-1, 3), size=5, color='blue')
    mesh = mesh3d(v, F, color='silver', opacity=0.5)
    scat_k = scatter3d(k, size=10, color='brown')
    fig = go.Figure([mesh, scat_k, scat_v_surr_k])
    remove_fig_background(fig)
    update_fig_size(fig)
    update_fig_camera(fig)
    fig.write_html(f"{log_dir}/verts_surr_k.html")
    

if __name__ == "__main__":
    main()
