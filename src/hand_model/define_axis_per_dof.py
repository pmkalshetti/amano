from utils.freq_imports import *
from utils.helper import create_dir
from utils import plotly_wrapper
from utils.array import normalize


def define_axes(v, k):
    I_v = [
        # thumb
        
        [100],          # mcp1
        [114],          # mcp2
        [123],          # pip
        [711],          # dip

        # index
        [132, 168],     # mcp1
        [127, 128, 172],# mcp2
        [156],          # pip
        [297, 331],     # dip

        # middle
        [271, 288],     # mcp1
        [150, 144],     # mcp2
        [373, 362],     # pip
        [409],          # dip

        # ring
        [142],          # mcp1
        [74, 206],      # mcp2
        [483, 495, 474, 479], # pip
        [521, 520],          # dip

        # pinky
        [595, 770],     # mcp1
        [776, 289],     # mcp2
        [599, 598],     # pip
        [669]           # dip
    ]
    ends = [np.mean(v[i_v], axis=0) for i_v in I_v]

    axes = []
    for i_f in range(5):
        wrist_id = 4*i_f; mcp_id = wrist_id+1; pip_id = mcp_id+1; dip_id = pip_id+1
        i_t_mcp1 = 4*i_f; i_t_mcp2 = i_t_mcp1+1; i_t_pip = i_t_mcp2+1; i_t_dip = i_t_pip+1

        axes.append(normalize(ends[i_t_mcp1] - k[mcp_id]))
        axes.append(normalize(ends[i_t_mcp2] - k[mcp_id]))
        axes.append(normalize(ends[i_t_pip] - k[pip_id]))
        axes.append(normalize(ends[i_t_dip] - k[dip_id]))
    axes = np.array(axes)

    return axes

def main():
    # read mesh
    v, _, _, F, _, _ = igl.read_obj("./output/hand_model/mesh.obj")
    K = load_npz("./output/hand_model/K.npz")
    k = K @ v

    # define axes and write to file
    axes = define_axes(v, k)
    out_dir = './output/hand_model'; create_dir(out_dir, False)
    np.save(f'{out_dir}/axis_per_dof.npy', axes)

    # plot axes
    log_dir = './log/hand_model'; create_dir(log_dir, False)
    lines = []
    for i_f in range(5):
        wrist_id = 4*i_f; mcp_id = wrist_id+1; pip_id = mcp_id+1; dip_id = pip_id+1
        i_t_mcp1 = 4*i_f; i_t_mcp2 = i_t_mcp1+1; i_t_pip = i_t_mcp2+1; i_t_dip = i_t_pip+1
        
        lines.append(plotly_wrapper.line3d(k[mcp_id], k[mcp_id]+0.01*axes[i_t_mcp1], color='blue', width=4))
        lines.append(plotly_wrapper.line3d(k[mcp_id], k[mcp_id]+0.01*axes[i_t_mcp2], color='red', width=4))
        lines.append(plotly_wrapper.line3d(k[pip_id], k[pip_id]+0.01*axes[i_t_pip], color='red', width=4))
        lines.append(plotly_wrapper.line3d(k[dip_id], k[dip_id]+0.01*axes[i_t_dip], color='red', width=4))

    mesh = plotly_wrapper.mesh3d(v, F, color='silver', opacity=0.5)
    scat_k = plotly_wrapper.scatter3d(k, size=10, color='brown')
    fig = go.Figure([mesh, scat_k, *lines])
    plotly_wrapper.remove_fig_background(fig)
    plotly_wrapper.update_fig_size(fig)
    fig.write_html(f"{log_dir}/axis_per_dof.html")
    
if __name__ == "__main__":
    main()
