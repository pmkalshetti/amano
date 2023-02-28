from utils.freq_imports import *
from utils.mesh import compute_vertex_normals
from utils.helper import create_dir
import pandas as pd
from hand_model.amano import Amano
from utils.moderngl_render import Pointcloud_and_HandRenderer
from utils.image import colormap_depth

def download_data(download_dir):
    urls = [
        'https://ls7-gv.cs.tu-dortmund.de/downloads/publications/2014/icra14/HandData/JointAngles/GeneralMovements/20101220_152546_Trial01.mat',
        'https://ls7-gv.cs.tu-dortmund.de/downloads/publications/2014/icra14/HandData/JointAngles/GeneralMovements/20101220_152738_Trial02.mat',
        'https://ls7-gv.cs.tu-dortmund.de/downloads/publications/2014/icra14/HandData/JointAngles/GeneralMovements/20101220_153004_Trial03.mat',
        'https://ls7-gv.cs.tu-dortmund.de/downloads/publications/2014/icra14/HandData/JointAngles/GeneralMovements/20101220_153120_Trial04.mat',
        'https://ls7-gv.cs.tu-dortmund.de/downloads/publications/2014/icra14/HandData/JointAngles/GeneralMovements/20101220_153305_Trial05.mat',
        'https://ls7-gv.cs.tu-dortmund.de/downloads/publications/2014/icra14/HandData/JointAngles/GeneralMovements/20101220_154117_Trial07.mat',
        'https://ls7-gv.cs.tu-dortmund.de/downloads/publications/2014/icra14/HandData/JointAngles/GeneralMovements/20101220_154202_Trial08.mat'
    ]
    for url in urls:
        df = pd.read_csv(url, header=None, sep=' ')
        filename = url[-11:-4]  # Trial0i
        df.to_csv(f'{download_dir}/{filename}.txt', header=False, index=False, sep=' ')

    return

def main():
    amano = Amano()

    # download data
    download_dir = './data/mario_botsch'; create_dir(download_dir, False)
    download_data(download_dir)

    # For now let's fix global transformation
    t_glob = np.array([0, 0, 0.4])  # 400mm away from camera
    theta_glob = np.deg2rad(np.array([-63.35355476,  76.25372491, 160.19133785]))   # these angles are estimated from NYU's frame 1
    R_glob_ref_x = Rotation.from_euler('X', theta_glob[0]).as_matrix()
    R_glob_ref_y = Rotation.from_euler('Y', theta_glob[1]).as_matrix()
    R_glob_ref_z = Rotation.from_euler('Z', theta_glob[2]).as_matrix()
    R_glob = R_glob_ref_x @ R_glob_ref_y @ R_glob_ref_z

    fx = 613.6845; fy = 613.7108; cx = 323.2300; cy = 236.7877  # use realsense camera parameters as example
    z_far = t_glob[2] + 0.05
    z_near = z_far - 0.2
    point_cloud_and_hand_renderer = Pointcloud_and_HandRenderer(640, 480, fx, fy, cx, cy, z_near, z_far, np.array(amano.F), len(amano.v))


    # each trial contains a sequence of pose
    trial_ids = [1, 2, 3, 4, 5, 7, 8]
    for trial_id in trial_ids:
        path_to_trial_data = f'{download_dir}/Trial{trial_id:02d}.txt'
        trial_data = np.loadtxt(path_to_trial_data)

        out_dir = Path(f'./output/syn_data/trial_{trial_id:02d}')
        out_color_dir = out_dir / 'color'; create_dir(out_color_dir, True)
        out_depth_dir = out_dir / 'depth_png'; create_dir(out_depth_dir, True)
        out_glob_dir = out_dir / 'glob'; create_dir(out_glob_dir, True)
        out_theta_dir = out_dir / 'theta'; create_dir(out_theta_dir, True)
        out_k_dir = out_dir / 'k'; create_dir(out_k_dir, True)

        # convention:
        # 3: global translation x,y,z
        # 3: global orientation x,y,z
        # 4: thumb mcp1, mcp2, pip, dip
        # 4: index mcp1, mcp2, pip, dip
        # 4: middle mcp1, mcp2, pip, dip
        # 4: ring mcp1, mcp2, pip, dip
        # 4: pinky mcp1, mcp2, pip, dip

        
        thetas = trial_data[:, 6:]

        # as per our convention
        # mcp2, pip, dip axis are reversed
        for i_f in range(5):
            i_t_mcp1 = 4*i_f; i_t_mcp2 = i_t_mcp1+1; i_t_pip = i_t_mcp2+1; i_t_dip = i_t_pip+1
            
            thetas[:, i_t_mcp2] *= -1
            thetas[:, i_t_pip] *= -1
            thetas[:, i_t_dip] *= -1

        for sample_id in tqdm(range(len(thetas)), dynamic_ncols=True, desc=f'Generating synthetic data for trial {trial_id}'):
            theta = thetas[sample_id]
            v_art = amano.deform(phi=np.ones(20), beta=np.zeros(10), theta=theta)
            v_p = v_art @ R_glob.T + t_glob
            n_p = compute_vertex_normals(v_p, amano.F)
            k_p = amano.K @ v_p

            point_cloud_and_hand_renderer.write_vbo_model(v_p, n_p)
            point_cloud_and_hand_renderer.render_model()
            color = point_cloud_and_hand_renderer.extract_fbo_color()
            cv.imwrite(f"{out_color_dir}/{sample_id:05d}.png", color[:, :, ::-1])

            depth = point_cloud_and_hand_renderer.extract_fbo_depth()
            depth_cm = colormap_depth(depth, z_near, z_far, cv.COLORMAP_HOT)
            cv.imwrite(f"{out_depth_dir}/{sample_id:05d}.png", depth_cm[:, :, ::-1])

            np.save(f'{out_glob_dir}/{sample_id:05d}.npy', np.concatenate([t_glob, theta_glob]))
            np.save(f'{out_theta_dir}/{sample_id:05d}.npy', theta)
            np.save(f'{out_k_dir}/{sample_id:05d}.npy', k_p)
            
            cv.imshow(f"render", np.hstack([color[:, :, ::-1], depth_cm[:, :, ::-1]]))
            key = cv.waitKey(1)
            if key == ord('q'):
                exit()

        subprocess.run(["./src/utils/create_video_from_frames.sh", "-f", "30", "-s", "0", "-w", "%05d.png", f"{out_color_dir}"])
        subprocess.run(["./src/utils/create_video_from_frames.sh", "-f", "30", "-s", "0", "-w", "%05d.png", f"{out_depth_dir}"])

if __name__ == "__main__":
    main()
