from utils.freq_imports import *
from registration.base.registration_base import RegistrationBase
from registration.base.pca_prior import compute_I_minus_Pi_M_PiT
from registration import utils, jacobian, correspondences
from utils.mesh import compute_vertex_normals
from utils.array import normalize

class PoseRegistration(RegistrationBase):
    def __init__(self,
                 w_nor=1e-5,
                 w_data_2d=1e-7,
                 w_theta_bound=1e-4,
                 w_pca=1e-2,    # w_4 in paper https://lgg.epfl.ch/publications/2015/Htrack_ICP/paper.pdf
                 w_pca_mean=1e-3,
                 w_int=1e-1,
                 w_k_reinit=1e1,
                 w_vel=1e-1,
                 w_damp_init=1e-4,
                 ):
        super().__init__()

        # number of points in sampled point cloud
        self.n_x = 200

        # hyperparameters
        self.w_pos = 1
        self.w_nor = w_nor
        self.n_s_approx = 212    # 212 results in 200 points
        self.w_data_3d = 1
        self.w_data_2d = w_data_2d
        self.w_theta_bound = w_theta_bound
        self.w_pca = w_pca  # w_4 in paper https://lgg.epfl.ch/publications/2015/Htrack_ICP/paper.pdf
        self.w_pca_mean = w_pca_mean    # w_5 in paper https://lgg.epfl.ch/publications/2015/Htrack_ICP/paper.pdf
        self.I_minus_Pi_M_PiT = compute_I_minus_Pi_M_PiT(self.Pi, self.Sigma, self.w_pca, self.w_pca_mean)
        self.w_int = w_int
        self.w_k_reinit = w_k_reinit
        self.w_vel = w_vel
        self.w_damp_init = w_damp_init

        self.i_dof_update = np.arange(6+20)

    def deform_and_compute_linearized_info(self, phi, beta, R_glob_init, theta_glob, t_glob, theta):
        # shape and articualate
        v_art, Rs_rel_art, Rs, a_prime = self.amano.deform(phi, beta, theta, return_intermediate=True)
        # global transform
        R_glob_ref_x = Rotation.from_euler('X', theta_glob[0]).as_matrix()
        R_glob_ref_y = Rotation.from_euler('Y', theta_glob[1]).as_matrix()
        R_glob_ref_z = Rotation.from_euler('Z', theta_glob[2]).as_matrix()
        R_glob = R_glob_init @ R_glob_ref_x @ R_glob_ref_y @ R_glob_ref_z
        v_p = v_art @ R_glob.T + t_glob
        n_p = compute_vertex_normals(v_p, self.amano.F)

        k_p = self.amano.K @ v_p

        # linearized info required for computing derivatives of energy
        axis_per_dof = jacobian.compute_axis_per_dof_in_world_space(R_glob_init, R_glob_ref_x, R_glob_ref_y, R_glob_ref_z, Rs_rel_art, self.amano.axes)
        pivot_per_dof = jacobian.compute_pivot_per_dof_in_world_space(R_glob, t_glob, a_prime)   # (26, 3)

        return v_p, n_p, k_p, axis_per_dof, pivot_per_dof

    def compute_theta_bound_terms(self, theta):
        # E = mask_max (dtheta + theta - theta_max) + mask_min (dtheta + theta - theta_min)
        # J = mask_max + min_mask
        # e = mask_max (theta_max - theta) + mask_min (theta_min - theta)

        mask_max = theta > self.theta_max    # (20,)
        mask_min = theta < self.theta_min    # (20,)
        # e = mask_max * (theta_max - theta) + mask_min * (theta_min - theta) # (20,)
        e = mask_max * (theta - self.theta_max) + mask_min * (theta - self.theta_min) # (20,)
        J = np.zeros((20, 26))
        J[:, 6:] = -np.diag(mask_max*1 + mask_min*1)  # (20, 20)

        JtJ = J.T @ J   # (26, 26)
        Jte = J.T @ e   # (26,)

        return e, JtJ, Jte

    def compute_pca_prior_terms(self, theta):
        # E = (I - Pi @ M @ Pi.T)(dtheta + theta - mu)
        # J = (I - Pi @ M @ Pi.T)
        # e = (I - Pi @ M @ Pi.T) @ (mu - theta)
        e = self.I_minus_Pi_M_PiT @ (self.mu - theta)  # (20,)
        J_theta = self.I_minus_Pi_M_PiT    # (20, 20)
        J = np.zeros((len(e), 26))  # (20, 26)
        J[:, 6:] = J_theta

        JtJ = J.T @ J   # (26, 26)
        Jte = J.T @ e   # (26,)

        return e, JtJ, Jte

    def compute_intersection_penalty_term(self, v_p, axis_per_dof, pivot_per_dof):
        def compute_sphere_center(v_p):
            return np.mean(v_p[self.i_v_per_sphere], axis=1)   # (28, 3)

        # E = n1.T ( (J_skel(x1) - J_skel(x2)) dpose + (x1 - x2)  )
        # J = n1.T (J_skel(x1) - J_skel(x2)
        # e = n1.T (x2 - x1)

        c_per_sphere = compute_sphere_center(v_p) # (28, 3)
        m_dof_per_sphere = np.any(self.m_dof_per_vert[self.i_v_per_sphere], axis=1)   # (28, 26)

        # center for each pair
        c1 = c_per_sphere[self.i_s_per_pair[:, 0]]   # (257, 3)
        c2 = c_per_sphere[self.i_s_per_pair[:, 1]]   # (257, 3)

        # radii for each pair
        r1 = self.r_per_sphere[self.i_s_per_pair[:, 0]]   # (257,)
        r2 = self.r_per_sphere[self.i_s_per_pair[:, 1]]   # (257,)

        # dof influenced by each pair
        m_dof_per_c1 = m_dof_per_sphere[self.i_s_per_pair[:, 0]] # (257, 26)
        m_dof_per_c2 = m_dof_per_sphere[self.i_s_per_pair[:, 1]] # (257, 26)

        # normalized direction vectors from c1 to c2 and vice-versa (shortest ray)
        v_1_2 = c2 - c1; n1 = normalize(v_1_2)    # (257, 3)
        v_2_1 = -v_1_2.copy(); n2 = normalize(v_2_1)   # (257, 3)

        # endpoint on sphere along shortest ray
        x1 = c1 + r1[:, np.newaxis] * n1     # (257, 3)
        x2 = c2 + r2[:, np.newaxis] * n2     # (257, 3)

        J_skel_x1 = jacobian.compute_J_skel(x1, m_dof_per_c1, axis_per_dof, pivot_per_dof)    # (257, 3, 26)
        J_skel_x2 = jacobian.compute_J_skel(x2, m_dof_per_c2, axis_per_dof, pivot_per_dof)    # (257, 3, 26)
        J = np.sum(n1[:, :, np.newaxis] * (J_skel_x1 - J_skel_x2), axis=1)    # (257, 26)
        e = np.sum(n1 * (x2 - x1), axis=1)
        
        # intersection indicator
        intersection_amount = (r1 + r2)**2 - np.sum((c1 - c2)**2, axis=1)   # (257,)
        m_int = intersection_amount > 0 # (257,)
        J[~m_int] = 0; e[~m_int] = 0


        JtJ = J.T @ J   # (26, 26)
        # the below line is surprisingly slow, so use jax jit
        # Jte = J.T @ e   # (26,)
        Jte = np.array(jacobian.compute_Jte(J, e))

        return e, JtJ, Jte

    def compute_keypoint_terms(self, k_data, k_model, m_dof_per_k, axis_per_dof, pivot_per_dof):
        # E = J_skel(k_model) @ dpose + (k_model - k_data)
        # J = J_skel(k_model)
        # e = k_data - k_model
        J_skel = jacobian.compute_J_skel(k_model, m_dof_per_k, axis_per_dof, pivot_per_dof)  # (10, 3, 26)
        J = np.reshape(J_skel, (-1, J_skel.shape[2]))   # (2*10, 26)
        e = np.reshape(k_data - k_model, (-1))  # (2*10,)

        JtJ = J.T @ J   # (26, 26)
        Jte = J.T @ e   # (26,)

        return e, JtJ, Jte

    def compute_velocity_terms(self, k_p_prev, k_p, axis_per_dof, pivot_per_dof):
        J_skel = jacobian.compute_J_skel(k_p, self.m_dof_per_k, axis_per_dof, pivot_per_dof)  # (21, 3, 26)
        J = np.reshape(J_skel, (-1, J_skel.shape[2]))   # (3*21, 26)
        e = np.reshape(k_p_prev - k_p, (-1))  # (3*21,)

        JtJ = J.T @ J   # (26, 26)
        Jte = J.T @ e   # (26,)

        return e, JtJ, Jte

    def solve(self, JtJ, Jte, w_damp):
        I_J = np.ix_(self.i_dof_update, self.i_dof_update)
        JtJ_dof = JtJ[I_J]
        Jte_dof = Jte[self.i_dof_update]
        dpose_dof = scipy.linalg.solve(JtJ_dof + w_damp * np.identity(JtJ_dof.shape[0]), Jte_dof, assume_a='pos')  # (|i_dof|,)
        dpose = np.zeros(len(Jte))
        dpose[self.i_dof_update] = dpose_dof

        return dpose

    def update_pose(self, dpose, t_glob, theta_glob, theta):
        dt_glob = dpose[:3]; t_glob_new = t_glob + dt_glob
        dtheta_glob = dpose[3:6]; theta_glob_new = theta_glob + dtheta_glob
        dtheta = dpose[6:]; theta_new = theta + dtheta; theta_new = np.clip(theta_new, self.theta_min, self.theta_max)

        return t_glob_new, theta_glob_new, theta_new


    def register_to_keypoints(self, k_data, 
        phi, beta, R_glob_init, theta_glob, t_glob, theta,
        v_p, n_p, k_p, axis_per_dof, pivot_per_dof,
        k_p_prev,
        n_iter=10,
        ):
        
        def compute_energy(k_data, theta, v_p, k_p, k_p_prev, axis_per_dof, pivot_per_dof):
            e_theta_bound, JtJ_theta_bound, Jte_theta_bound = self.compute_theta_bound_terms(theta)
            e_pca, JtJ_pca, Jte_pca = self.compute_pca_prior_terms(theta)
            e_int, JtJ_int, Jte_int = self.compute_intersection_penalty_term(v_p, axis_per_dof, pivot_per_dof)
            e_k_reinit, JtJ_k_reinit, Jte_k_reinit = self.compute_keypoint_terms(k_data, k_p[self.i_k_amano_reg_k], self.m_dof_per_k[self.i_k_amano_reg_k], axis_per_dof, pivot_per_dof)
            
            JtJ = self.w_theta_bound*JtJ_theta_bound + self.w_pca*JtJ_pca + self.w_int*JtJ_int + self.w_k_reinit*JtJ_k_reinit
            Jte = self.w_theta_bound*Jte_theta_bound + self.w_pca*Jte_pca + self.w_int*Jte_int + self.w_k_reinit*Jte_k_reinit
            E = self.w_theta_bound*np.sum(e_theta_bound**2) + self.w_pca*np.sum(e_pca**2) + self.w_int*np.sum(e_int**2) + self.w_k_reinit*np.sum(e_k_reinit**2)
            
            if k_p_prev is not None:
                e_vel, JtJ_vel, Jte_vel = self.compute_velocity_terms(k_p_prev, k_p, axis_per_dof, pivot_per_dof)
                JtJ += self.w_vel*JtJ_vel
                Jte += self.w_vel*Jte_vel
                E += self.w_vel*np.sum(e_vel**2)
            
            return JtJ, Jte, E

        # compute energy terms for current pose
        JtJ, Jte, E = compute_energy(k_data, theta, v_p, k_p, k_p_prev, axis_per_dof, pivot_per_dof)

        # LM
        w_damp = self.w_damp_init
        for iter in range(n_iter):
            # solve and update parameters
            dpose = self.solve(JtJ, Jte, w_damp)
            t_glob_new, theta_glob_new, theta_new = self.update_pose(dpose, t_glob, theta_glob, theta)
            
            # pose mesh using new parameters
            v_p_new, n_p_new, k_p_new, axis_per_dof_new, pivot_per_dof_new = self.deform_and_compute_linearized_info(phi, beta, R_glob_init, theta_glob_new, t_glob_new, theta_new)
            # compute energy terms for new pose
            JtJ_new, Jte_new, E_new = compute_energy(k_data, theta_new, v_p_new, k_p_new, k_p_prev, axis_per_dof_new, pivot_per_dof_new)
            
            if E_new < E:   # iteration successful   
                w_damp *= 0.8
                JtJ, Jte, E = JtJ_new, Jte_new, E_new
                t_glob, theta_glob, theta = t_glob_new, theta_glob_new, theta_new
                v_p, n_p, k_p, axis_per_dof, pivot_per_dof = v_p_new, n_p_new, k_p_new, axis_per_dof_new, pivot_per_dof_new
            else:   # iteration unsuccessful
                w_damp *= 10
        
        return (
            theta_glob, t_glob, theta,
            v_p, n_p, k_p, axis_per_dof, pivot_per_dof
        )

    def compute_3d_data_terms(self, x, y, yn, m_dof_per_y, axis_per_dof, pivot_per_dof):
        # E = n.T * (J_skel @ dp + (y-x))
        # J = n.T * J_skel
        # e = n.T * (x-y)
        J_skel = jacobian.compute_J_skel(y, m_dof_per_y, axis_per_dof, pivot_per_dof)    # (|y|, 3, 26)
        J = np.sum(yn[:, :, np.newaxis] * J_skel, axis=1)   # (|y|, 26)
        e = np.sum(yn * (x - y), axis=1)    # (|y|,)

        JtJ = J.T @ J   # (26, 26)
        Jte = J.T @ e   # (26,)

        return e, JtJ, Jte

    def compute_2d_data_terms(self, x_bg, p, q, axis_per_dof, pivot_per_dof):
        # E = n.T * (J_persp @ J_skel @ dp + (p-q))
        # J = n.T * J_persp @ J_skel
        # e = n.T * (q-p)
        J_skel = jacobian.compute_J_skel(x_bg, self.m_dof_per_bg, axis_per_dof, pivot_per_dof)    # (|x|, 3, 26)
        J_persp = jacobian.compute_J_persp(x_bg, self.fx, self.fy)             # (|x|, 2, 3)
        J = np.reshape(J_persp @ J_skel, (-1, J_skel.shape[2]))    # (2*|x|, 26)
        e = np.reshape(q - p, (-1)).astype(np.float64)    # (2*|x|,)

        JtJ = J.T @ J   # (26, 26)
        
        Jte = np.array(jacobian.compute_Jte(J, e))
        # the below line is surprisingly slow, so use jax jit
        # Jte = J.T @ e   # (26,)
        # np.ones((1982, 26)).T @ np.ones((1982))

        return e, JtJ, Jte

    def register_to_pointcloud(self, k_data, x, xn, I_D_vu,
        phi, beta, R_glob_init, theta_glob, t_glob, theta,
        v_p, n_p, k_p, axis_per_dof, pivot_per_dof,
        k_p_prev,
        n_iter=10,
        ):

        def compute_energy(k_data, x, xn, I_D_vu,
                y, yn, m_dof_per_y,
                x_bg, p, q,
                theta, v_p, k_p, axis_per_dof, pivot_per_dof,
                k_p_prev
            ):
            e_data_3d, JtJ_data_3d, Jte_data_3d = self.compute_3d_data_terms(x, y, yn, m_dof_per_y, axis_per_dof, pivot_per_dof)
            e_data_2d, JtJ_data_2d, Jte_data_2d = self.compute_2d_data_terms(x_bg, p, q, axis_per_dof, pivot_per_dof)
            e_theta_bound, JtJ_theta_bound, Jte_theta_bound = self.compute_theta_bound_terms(theta)
            e_pca, JtJ_pca, Jte_pca = self.compute_pca_prior_terms(theta)
            e_int, JtJ_int, Jte_int = self.compute_intersection_penalty_term(v_p, axis_per_dof, pivot_per_dof)
            
            JtJ = self.w_data_3d*JtJ_data_3d + self.w_data_2d*JtJ_data_2d + self.w_theta_bound*JtJ_theta_bound + self.w_pca*JtJ_pca + self.w_int*JtJ_int
            Jte = self.w_data_3d*Jte_data_3d + self.w_data_2d*Jte_data_2d + self.w_theta_bound*Jte_theta_bound + self.w_pca*Jte_pca + self.w_int*Jte_int
            E = self.w_data_3d*np.sum(e_data_3d**2) + self.w_data_2d*np.sum(e_data_2d**2) + self.w_theta_bound*np.sum(e_theta_bound**2) + self.w_pca*np.sum(e_pca**2) + self.w_int*np.sum(e_int**2)
            
            if k_data is not None:
                e_k_reinit, JtJ_k_reinit, Jte_k_reinit = self.compute_keypoint_terms(k_data, k_p[self.i_k_amano_reinit], self.m_dof_per_k[self.i_k_amano_reinit], axis_per_dof, pivot_per_dof)
                JtJ += self.w_k_reinit*JtJ_k_reinit
                Jte += self.w_k_reinit*Jte_k_reinit
                E += self.w_k_reinit*np.sum(e_k_reinit**2)

            if k_p_prev is not None:
                e_vel, JtJ_vel, Jte_vel = self.compute_velocity_terms(k_p_prev, k_p, axis_per_dof, pivot_per_dof)
                JtJ += self.w_vel*JtJ_vel
                Jte += self.w_vel*Jte_vel
                E += self.w_vel*np.sum(e_vel**2)
            
            return JtJ, Jte, E

        # compute correspondences for current pose
        y, yn, i_F_y, b_y, m_dof_per_y = correspondences.compute_3d_correspondences(v_p, n_p, x, xn, self.rng, self.amano.F, self.i_F_per_part, self.n_f_per_part, self.m_dof_per_face, self.w_pos, self.w_nor, self.n_s_approx)
        x_bg, p, q = correspondences.compute_2d_correspondences(v_p, I_D_vu, self.amano.F, self.i_F_bg, self.b_bg, self.fx, self.fy, self.cx, self.cy)
        # compute energy terms for current pose
        JtJ, Jte, E = compute_energy(k_data, x, xn, I_D_vu, 
            y, yn, m_dof_per_y, 
            x_bg, p, q, 
            theta, v_p, k_p, axis_per_dof, pivot_per_dof,
            k_p_prev
        )

        # LM
        w_damp = self.w_damp_init
        for iter in range(n_iter):
            # solve and update parameters
            dpose = self.solve(JtJ, Jte, w_damp)
            t_glob_new, theta_glob_new, theta_new = self.update_pose(dpose, t_glob, theta_glob, theta)
            
            # pose mesh using new parameters
            v_p_new, n_p_new, k_p_new, axis_per_dof_new, pivot_per_dof_new = self.deform_and_compute_linearized_info(phi, beta, R_glob_init, theta_glob_new, t_glob_new, theta_new)
            # update correspondences for new pose
            y_new, yn_new, i_F_y_new, b_y_new, m_dof_per_y_new = correspondences.update_3d_correspondences(v_p_new, n_p_new, i_F_y, b_y, m_dof_per_y, x, xn, self.rng, self.amano.F, self.i_F_per_part, self.n_f_per_part, self.m_dof_per_face, self.w_pos, self.w_nor, self.n_s_approx)
            x_bg_new, p_new, q_new = correspondences.compute_2d_correspondences(v_p_new, I_D_vu, self.amano.F, self.i_F_bg, self.b_bg, self.fx, self.fy, self.cx, self.cy)
            # compute energy terms for new pose
            JtJ_new, Jte_new, E_new = compute_energy(k_data, x, xn, I_D_vu, 
                y_new, yn_new, m_dof_per_y_new, 
                x_bg_new, p_new, q_new, 
                theta_new, v_p_new, k_p_new, axis_per_dof_new, pivot_per_dof_new,
                k_p_prev,
            )

            if E_new < E:   # iteration successful   
                w_damp *= 0.8
                JtJ, Jte, E = JtJ_new, Jte_new, E_new
                t_glob, theta_glob, theta = t_glob_new, theta_glob_new, theta_new
                v_p, n_p, k_p, axis_per_dof, pivot_per_dof = v_p_new, n_p_new, k_p_new, axis_per_dof_new, pivot_per_dof_new
                y, yn, i_F_y, b_y, m_dof_per_y = y_new, yn_new, i_F_y_new, b_y_new, m_dof_per_y_new
                x_bg, p, q = x_bg_new, p_new, q_new
            else:   # iteration unsuccessful
                w_damp *= 10
        
        return (
            theta_glob, t_glob, theta,
            v_p, n_p, k_p, axis_per_dof, pivot_per_dof,
            y, yn,
            p, q
        )

