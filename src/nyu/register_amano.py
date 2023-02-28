from utils.freq_imports import *
from utils.helper import create_dir
from utils import plotly_wrapper
from utils.perspective_projection import uvd2xyz, xyz2uv
from utils.moderngl_render import Pointcloud_and_HandRenderer
from utils.image import colormap_depth, calculate_depth_diff_img
from utils.mesh import compute_vertex_normals
import metric
from registration import preprocess, global_transform
from registration.pose_registration import PoseRegistration
from registration.shape_registration import ShapeRegistration
from nyu import nyu_data


def main():
    shape_reg = ShapeRegistration()
    pose_reg = PoseRegistration()

    shape_reg.w_beta = 1e-4

    i_k_nyu_palm = np.array([
        29, 28, 23, 17, 11, 5
    ])
    i_k_nyu_phi = np.array([
        29,  # root
        
        # mcp, pip, dip, tip
        28, 27, 25, 24, # thumb
        23, 21, 19, 18, # index
        17, 15, 13, 12, # middle
        11, 9, 7, 6,    # ring
        5, 3, 1, 0,     # pinky
    ])

    i_k_nyu_eval = np.array([
        # tip, pip
        24, 27,  # thumb
        18, 21,   # index
        12, 15,   # middle
        6, 9,   # ring
        0, 3,   # pinky
    ])
    i_k_amano_eval = np.array([
        # tip, pip
        4,  2,    # thumb    
        8,  6,    # index
        12, 10,  # middle    
        16, 14,  # ring    
        20, 18,  # pinky
    ])
    i_k_awr_eval = np.array([
        # tip, pip
        8, 10,  # thumb
        6, 7,   # index
        4, 5,   # middle
        2, 3,   # ring
        0, 1,   # pinky
    ])

    i_k_awr_reinit = np.array([
        # tip, pip
        8, 10,  # thumb
        6, 7,   # index
        4, 5,   # middle
        2, 3,   # ring
        0, 1,   # pinky
    ])
    i_k_amano_reinit = np.array([
        # tip, pip
        4,  2,    # thumb    
        8,  6,    # index
        12, 10,  # middle    
        16, 14,  # ring    
        20, 18,  # pinky
    ])
    shape_reg.set_i_k_amano_reinit(i_k_amano_reinit)
    pose_reg.set_i_k_amano_reinit(i_k_amano_reinit)

    i_k_awr_reg_k = i_k_awr_eval
    i_k_amano_reg_k = i_k_amano_eval
    pose_reg.set_i_k_amano_reg_k(i_k_amano_reg_k)

    H, W = 480, 640
    z_near, z_far = 0.3, 1.2    # m
    d_near = z_near * 1000; d_far = z_far * 1000    # depth uses mm as units
    fx, fy, cx, cy = nyu_data.get_nyu_camera_params()
    shape_reg.set_camera_params(fx, fy, cx, cy)
    pose_reg.set_camera_params(fx, fy, cx, cy)
    point_cloud_and_hand_renderer = Pointcloud_and_HandRenderer(W, H, fx, fy, cx, cy, z_near, z_far, np.array(pose_reg.amano.F), len(pose_reg.amano.v))

    path_to_nyu = Path('./data/nyu')
    split = 'test'
    path_to_data_dir = path_to_nyu / split
    anno_file = path_to_data_dir / 'joint_data.mat'
    img_paths = sorted(path_to_data_dir.glob('depth_1*.png'))
    anno_dict = sio.loadmat(anno_file)
    
    path_to_awr_nyu_pred = Path('./data/awr/nyu_predictions/resnet_18.txt')
    Kvec_uvd_awr = np.loadtxt(path_to_awr_nyu_pred)   # (8252, 42) 
    K_uvd_awr = np.reshape(Kvec_uvd_awr, (-1, 14, 3))    # (8252, 14, 3)
    
    out_dir = f"./output/nyu/amano"; create_dir(out_dir, True)
    for subject in range(1, 3):    
        out_sub_dir = f"{out_dir}/{subject:02d}"; create_dir(out_sub_dir, True)
        out_depth_proc_dir = f"{out_sub_dir}/depth_proc"; create_dir(out_depth_proc_dir, True)
        out_depth_ren_dir = f"{out_sub_dir}/depth_ren"; create_dir(out_depth_ren_dir, True)
        out_depth_diff_dir = f"{out_sub_dir}/depth_diff"; create_dir(out_depth_diff_dir, True)
        out_points_mesh_dir = f"{out_sub_dir}/points_mesh"; create_dir(out_points_mesh_dir, True)
        out_metric_dir = f"{out_sub_dir}/metric"; create_dir(out_metric_dir, True)
        d2m_mean_file = f"{out_metric_dir}/d2m_mean.txt"; d2m_max_file = f"{out_metric_dir}/d2m_max.txt"
        m2d_mean_file = f"{out_metric_dir}/m2d_mean.txt"; m2d_max_file = f"{out_metric_dir}/m2d_max.txt"
        k_error_mean_file = f"{out_metric_dir}/k_error_mean.txt"; k_error_max_file = f"{out_metric_dir}/k_error_max.txt"
        k_error_mean_awr_file = f"{out_metric_dir}/k_error_mean_awr.txt"; k_error_max_awr_file = f"{out_metric_dir}/k_error_max_awr.txt"
        metric_cum_avg_file = f"{out_metric_dir}/cum_avg.txt"
        with open(metric_cum_avg_file, "w") as file:
            file.write((
                    "Frame id"
                    " | d2m_mean | d2m_max"
                    " | m2d_mean | m2d_max"
                    " | k_error_mean | k_error_max"
                    " | k_error_mean_awr | k_error_max_awr"
                    "\n"
                ))
        
        cum_d2m_mean = cum_d2m_max = 0.0
        cum_m2d_mean = cum_m2d_max = 0.0
        cum_k_error_mean = cum_k_error_max = 0.0
        cum_k_error_mean_awr = cum_k_error_max_awr = 0.0
        cnt_frame = 0

        # the first 2440 frames correspond to person 1 as per Sec 4.4 of https://lgg.epfl.ch/publications/2017/HOnline/paper.pdf
        if subject == 1:
            i_frame_start = 0; i_frame_end = 2439
        else:
            i_frame_start = 2440; i_frame_end = len(img_paths)-1

        calibration = True
        for i_frame in tqdm(range(i_frame_start, i_frame_end+1), dynamic_ncols=True):
            # read and process data
            k_nyu_mm = anno_dict['joint_xyz'][0][i_frame]
            k_nyu = k_nyu_mm / 1000
            # flip y axis when doing xyz <-> uvd transformation
            # Ref: https://github.com/Elody-07/AWR-Adaptive-Weighting-Regression/blob/master/dataloader/nyu_loader.py#L34
            k_nyu[:, 1] *= -1

            k_uvd_awr = K_uvd_awr[i_frame]
            k_awr_mm = uvd2xyz(k_uvd_awr, fx, fy, cx, cy)   # (14, 3)
            # k_awr_mm[:, 1] *= -1
            k_awr = k_awr_mm / 1000

            depth = nyu_data.read_nyu_depth(img_paths[i_frame])
            depth_proc = nyu_data.crop_depth_using_keypoints(depth, k_nyu, fx, fy, cx, cy)
            I_D_vu = preprocess.compute_sil_idx_at_each_pixel(depth_proc)
            x_dense, x, xn = preprocess.depth_to_point_cloud(depth_proc, fx, fy, cx, cy, n_x=pose_reg.n_x)
            if x is None:
                print(f"Frame {i_frame}, invalid point cloud")
                continue
            
            if i_frame == i_frame_start:
                ## register to keypoints to initialize pose
                # init params
                beta = np.zeros(10)
                phi, k_s = pose_reg.calculate_phi(k_nyu[i_k_nyu_phi], beta, return_k_s=True)
                R_glob_init, t_glob = global_transform.compute_global_trans_from_palm_keypoints(k_nyu[i_k_nyu_palm], k_s[pose_reg.i_k_amano_palm])
                theta_glob = np.zeros(3)
                theta = np.zeros(20)
                k_p_prev = None

                # register pose to keypoints
                v_p, n_p, k_p, axis_per_dof, pivot_per_dof = pose_reg.deform_and_compute_linearized_info(phi, beta, R_glob_init, theta_glob, t_glob, theta)
                theta_glob, t_glob, theta, v_p, n_p, k_p, axis_per_dof, pivot_per_dof = pose_reg.register_to_keypoints(
                    k_awr[i_k_awr_reg_k], 
                    phi, beta, R_glob_init, theta_glob, t_glob, theta,
                    v_p, n_p, k_p, axis_per_dof, pivot_per_dof,
                    k_p_prev=None,
                )

                # add residual global rotation into initial global rotation
                R_glob_ref_x = Rotation.from_euler('X', theta_glob[0]).as_matrix()
                R_glob_ref_y = Rotation.from_euler('Y', theta_glob[1]).as_matrix()
                R_glob_ref_z = Rotation.from_euler('Z', theta_glob[2]).as_matrix()
                R_glob = R_glob_init @ R_glob_ref_x @ R_glob_ref_y @ R_glob_ref_z
                R_glob_init = R_glob.copy() # use this as initial global transformation for next frame
                theta_glob = np.zeros(3)

                # ready for first iteration of shape registration
                v_p, n_p, k_p, axis_per_dof, pivot_per_dof, J_beta = shape_reg.deform_and_compute_linearized_info(phi, beta, R_glob_init, theta_glob, t_glob, theta)

                k_data = k_awr[i_k_awr_reinit]

            else:
                # use previous frame's global transformation to initialize current frame's global transformation
                R_glob_ref_x = Rotation.from_euler('X', theta_glob[0]).as_matrix()
                R_glob_ref_y = Rotation.from_euler('Y', theta_glob[1]).as_matrix()
                R_glob_ref_z = Rotation.from_euler('Z', theta_glob[2]).as_matrix()
                R_glob = R_glob_init @ R_glob_ref_x @ R_glob_ref_y @ R_glob_ref_z
                R_glob_init = R_glob.copy() # use this as initial global transformation for next frame

                # since theta_glob captures offset from initial global transform, init to zero
                theta_glob = np.zeros(3)
                # t_glob, theta, beta are initialized using previous frame's estimates

                k_p_prev = k_p
                # use keypoints for current frame
                k_data = k_awr[i_k_awr_reinit]

            # register to point cloud
            if calibration:
                beta, theta_glob, t_glob, theta, v_p, n_p, k_p, axis_per_dof, pivot_per_dof, J_beta, y, yn, p, q = shape_reg.register_to_pointcloud(
                    k_data, x, xn, I_D_vu,
                    phi, beta, R_glob_init, theta_glob, t_glob, theta,
                    v_p, n_p, k_p, axis_per_dof, pivot_per_dof, J_beta,
                    k_p_prev
                )

                if (i_frame - i_frame_start) == 50:
                    print("calibration complete")
                    calibration = False
                    np.save(f"{out_sub_dir}/beta.npy", beta)

            else:
                theta_glob, t_glob, theta, v_p, n_p, k_p, axis_per_dof, pivot_per_dof, y, yn, p, q = pose_reg.register_to_pointcloud(
                    k_data, x, xn, I_D_vu,
                    phi, beta, R_glob_init, theta_glob, t_glob, theta,
                    v_p, n_p, k_p, axis_per_dof, pivot_per_dof,
                    k_p_prev,
                )
            
            ## plot

            # mesh and point cloud; downsample point cloud so that it's not too dense
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x_dense))
            pcd = pcd.voxel_down_sample(0.003)
            x_plot = np.asarray(pcd.points)
            point_cloud_and_hand_renderer.write_vbo(v_p, n_p, x_plot, x, xn, y, yn)
            point_cloud_and_hand_renderer.render_x_dense_and_mesh()
            color = point_cloud_and_hand_renderer.extract_fbo_color()
            cv.imwrite(f"{out_points_mesh_dir}/{i_frame:08d}.png", color[:, :, ::-1])

            # input processed depth
            depth_proc_cm = colormap_depth(depth_proc, d_near, d_far, cv.COLORMAP_INFERNO)
            cv.imwrite(f"{out_depth_proc_dir}/{i_frame:08d}.png", depth_proc_cm[:, :, ::-1])

            # rendered depth
            point_cloud_and_hand_renderer.render_model()
            depth_ren = point_cloud_and_hand_renderer.extract_fbo_depth()
            depth_ren_cm = colormap_depth(depth_ren, d_near, d_far, cv.COLORMAP_INFERNO)
            cv.imwrite(f"{out_depth_ren_dir}/{i_frame:08d}.png", depth_ren_cm[:, :, ::-1])
            
            # depth difference
            depth_diff_cm = calculate_depth_diff_img(depth_proc, depth_ren, diff_threshold=10)
            cv.imwrite(f"{out_depth_diff_dir}/{i_frame:08d}.png", depth_diff_cm[:, :, ::-1])


            ## write metric
            d2m_mean, d2m_max = metric.compute_d2m_mean_max(depth_proc, depth_ren, fx, fy, cx, cy)
            with open(d2m_mean_file, "a") as file:
                file.write(f"{d2m_mean:.3f}\n")
            with open(d2m_max_file, "a") as file:
                file.write(f"{d2m_max:.3f}\n")
            
            
            m2d_mean, m2d_max = metric.compute_m2d_mean_max(depth_proc, depth_ren)
            with open(m2d_mean_file, "a") as file:
                file.write(f"{m2d_mean:.3f}\n")
            with open(m2d_max_file, "a") as file:
                file.write(f"{m2d_max:.3f}\n")

            k_error_mean = metric.compute_k_error_mean(k_nyu[i_k_nyu_eval], k_p[i_k_amano_eval])
            with open(k_error_mean_file, "a") as file:
                file.write(f"{k_error_mean:.3f}\n")
            k_error_max = metric.compute_k_error_max(k_nyu[i_k_nyu_eval], k_p[i_k_amano_eval])
            with open(k_error_max_file, "a") as file:
                file.write(f"{k_error_max:.3f}\n")

            k_error_mean_awr = metric.compute_k_error_mean(k_nyu[i_k_nyu_eval], k_awr[i_k_awr_eval])
            with open(k_error_mean_awr_file, "a") as file:
                file.write(f"{k_error_mean_awr:.3f}\n")
            k_error_max_awr = metric.compute_k_error_max(k_nyu[i_k_nyu_eval], k_awr[i_k_awr_eval])
            with open(k_error_max_awr_file, "a") as file:
                file.write(f"{k_error_max_awr:.3f}\n")

            cum_d2m_mean += d2m_mean; cum_d2m_max += d2m_max
            cum_m2d_mean += m2d_mean; cum_m2d_max += m2d_max
            cum_k_error_mean += k_error_mean; cum_k_error_max += k_error_max
            cum_k_error_mean_awr += k_error_mean_awr; cum_k_error_max_awr += k_error_max_awr
            cnt_frame += 1
            metric_cum_avg_str = (
                f"{i_frame:08d}"
                f"| {cum_d2m_mean/cnt_frame:.3f} | {cum_d2m_max/cnt_frame:.3f}"
                f"| {cum_m2d_mean/cnt_frame:.3f} | {cum_m2d_max/cnt_frame:.3f}"
                f"| {cum_k_error_mean/cnt_frame:.3f} | {cum_k_error_max/cnt_frame:.3f}"
                f"| {cum_k_error_mean_awr/cnt_frame:.3f} | {cum_k_error_max_awr/cnt_frame:.3f}"
                "\n"
            )
            with open(metric_cum_avg_file, 'a') as file:
                file.write(metric_cum_avg_str)

        subprocess.run(["./src/utils/create_video_from_frames.sh", "-f", "30", "-s", f"{i_frame_start}", "-w", "%08d.png", f"{out_depth_proc_dir}"])
        subprocess.run(["./src/utils/create_video_from_frames.sh", "-f", "30", "-s", f"{i_frame_start}", "-w", "%08d.png", f"{out_depth_ren_dir}"])
        subprocess.run(["./src/utils/create_video_from_frames.sh", "-f", "30", "-s", f"{i_frame_start}", "-w", "%08d.png", f"{out_depth_diff_dir}"])
        subprocess.run(["./src/utils/create_video_from_frames.sh", "-f", "30", "-s", f"{i_frame_start}", "-w", "%08d.png", f"{out_points_mesh_dir}"])


if __name__ == "__main__":
    main()