import numpy as np

def compute_face_ids_per_part(F, W_bone):
    # part label to vertices
    p_v = np.argmax(W_bone, axis=1)    # (|v|,)
    # part label to faces
    p_F = p_v[F[:, 0]]    # (|F|,)
    
    n_p = W_bone.shape[1]
    
    # cm_glasbey = cc.cm.glasbey
    # colorscale = []
    # for s in np.linspace(0, 1, n_p):
    #     rgba = cm_glasbey(s)
    #     rgba_str = f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})'
    #     colorscale.append([s, rgba_str])
    # mesh = plotly_utils.mesh3d(v, F, intensitymode='cell', intensity=p_F, colorscale=colorscale)
    # fig = go.Figure(mesh)
    # plotly_utils.remove_fig_background(fig)
    # plotly_utils.update_fig_size(fig)
    # fig.show()

    # list of face ids for each part (inverse map of above)
    i_F_per_part = []; n_f_per_part = []
    for i_p in range(n_p):
        i_F_p = np.nonzero(p_F == i_p)[0]   # [0] since it's 1D
        i_F_per_part.append(i_F_p)
        n_f_per_part.append(len(i_F_p))

    # len(i_F_per_part) = 20; i_F_per_part[i] contains face ids belonging to part i
    # n_f_per_part[i] denote the number of faces belonging to part i
    return i_F_per_part, n_f_per_part