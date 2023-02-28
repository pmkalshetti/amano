from utils.freq_imports import *
from utils.mesh import compute_vertex_normals


class Amano:
    def __init__(self):
        # read hand model
        self.v, _, _, self.F, _, _ = igl.read_obj('./output/hand_model/mesh.obj')
        self.n = compute_vertex_normals(self.v, self.F)
        self.axes = np.load("./output/hand_model/axis_per_dof.npy")  # (20, 3)
        W = sio.loadmat('./output/hand_model/lbs_weights/W.mat')
        self.W_bone = W['W_bone']            # (778, 20)
        self.W_endpoint = W['W_endpoint']    # (778, 20)
        self.K = load_npz("./output/hand_model/K.npz")  # (21, 778)
        self.k = self.K @ self.v    # (21, 3)

        path_to_mano = './data/mano/mano_v1_2/models/MANO_RIGHT.pkl'
        with open(path_to_mano, "rb") as file:
            mano_data = pickle.load(file, encoding="latin1")
            self.mano_S = np.array(mano_data['shapedirs']) # (778, 3, 10)
            self.mano_P = mano_data["posedirs"]   # (778, 3, 135)

    def compute_relative_rotation_matrix_at_each_dof(self, theta):
        Rs_rel = Rotation.from_rotvec(theta[:, np.newaxis] * self.axes).as_matrix()  # (20, 3, 3)
        return Rs_rel

    def deform(self, phi, beta, theta, return_intermediate=False):
        Rs_rel_art = self.compute_relative_rotation_matrix_at_each_dof(theta)
        
        # shape blends
        v_s = self.v + self.mano_S @ beta  # (|v|, 3)
        # obtain keypoint positions in shaped mesh
        k_s = self.K @ v_s  # (21, 3)
        # pose blends; requires reordering Rs_rel_art to resemble MANO
        i_Rs_rel_art_mano = np.array([
            4, 5, 6,    # index: mcp, pip, dip
            7, 8, 9,    # middle: mcp, pip, dip
            13, 14, 15, # pinky: mcp, pip, dip
            10, 11, 12, # ring: mcp, pip, dip
            1, 2, 3,    # thumb: mcp, pip, dip
        ]) - 1
        v_s += self.mano_P @ (Rs_rel_art[i_Rs_rel_art_mano] - np.eye(3)).flatten()  # (|v|, 3)

        ## stretchable LBS: vectorized eq 4 from https://igl.ethz.ch/projects/skinning/stretchable-twistable-bones/stretchable-twistable-bones-siggraph-asia-2011-jacobson-sorkine.pdf
        
        # recursively accumulate rotations matrices
        Rs = []
        for i_f in range(5):
            i_t_mcp1 = 4*i_f; i_t_mcp2 = i_t_mcp1+1; i_t_pip = i_t_mcp2+1; i_t_dip = i_t_pip+1

            # add rotation for wrist, so that the matrix dimensions are compatible with the bone weights
            R_wrist = np.eye(3); Rs.append(R_wrist)  # no rotation around wrist
            R_mcp = Rs_rel_art[i_t_mcp1] @ Rs_rel_art[i_t_mcp2] ; Rs.append(R_mcp)
            R_pip = R_mcp @ Rs_rel_art[i_t_pip]; Rs.append(R_pip)
            R_dip = R_pip @ Rs_rel_art[i_t_dip]; Rs.append(R_dip)
        Rs = np.array(Rs)   # (20, 3, 3)

        # calculate start and end points of bones in rest pose
        a, b = [], []
        for i_f in range(5):
            i_k_mcp = 4*i_f + 1; i_k_pip = i_k_mcp+1; i_k_dip = i_k_pip+1; i_k_tip = i_k_dip+1
            a.append(k_s[0]);         b.append(k_s[i_k_mcp])  # wrist -> mcp
            a.append(k_s[i_k_mcp]);   b.append(k_s[i_k_pip])  # mcp -> pip
            a.append(k_s[i_k_pip]);   b.append(k_s[i_k_dip])  # pip -> dip
            a.append(k_s[i_k_dip]);   b.append(k_s[i_k_tip])  # dip -> tip
        a = np.array(a); b = np.array(b)    # (20, 3)

        # calculate start points of bones in deformed pose
        a_prime = []
        for i_f in range(5):
            i_wrist = 4*i_f; i_mcp = i_wrist+1; i_pip = i_mcp+1; i_dip = i_pip+1
            a_prime_wrist = a[0]; a_prime.append(a_prime_wrist)
            a_prime_mcp = Rs[i_wrist] @ (phi[i_wrist] * (a[i_mcp] - a[i_wrist])) + a_prime_wrist; a_prime.append(a_prime_mcp)
            a_prime_pip = Rs[i_mcp] @ (phi[i_mcp] * (a[i_pip] - a[i_mcp])) + a_prime_mcp; a_prime.append(a_prime_pip)
            a_prime_dip = Rs[i_pip] @ (phi[i_pip] * (a[i_dip] - a[i_pip])) + a_prime_pip; a_prime.append(a_prime_dip)
        a_prime = np.array(a_prime)

        # lbs: vectorized eq 4 from https://igl.ethz.ch/projects/skinning/stretchable-twistable-bones/stretchable-twistable-bones-siggraph-asia-2011-jacobson-sorkine.pdf
        # Tp = R(p - a + es) + a' = Rp - Ra + a'
        n_v = len(self.v); n_b = len(Rs)
        # vectorized R
        R = Rs.reshape(-1, 3)  # (16*3, 3)
        # vectorized R*p (p is v in this casce)
        Rp = (R @ v_s.T).T.reshape(n_v, n_b, 3).transpose(0, 2, 1)    # (n_v, 3, n_b)
        # vectorized R*a
        Ra = (Rs @ a[:, :, np.newaxis]).squeeze() # (n_b, 3)
        # vectorized Rs
        s = (phi - 1)[:, np.newaxis] * (b-a) # (n_b, 3)
        Rs_ = (Rs @ s[:, :, np.newaxis]).squeeze()   # (n_b, 3) _ is used to distinguish between earlier used Rs
        # vectorized eRs
        eRs = self.W_endpoint[:, np.newaxis, :] * Rs_.T[np.newaxis, :, :]  # (n_v, 3, n_b)
        # vectorized Tp (eqn 1 and 2)
        Tp = Rp - Ra.T[np.newaxis, :, :] + a_prime.T[np.newaxis, :, :] + eRs  # (n_v, 3, n_b)
        # vectorized lbs
        v_art = np.sum(self.W_bone[:, np.newaxis, :] * Tp, axis=2)    # (n_v, 3)
        
        if return_intermediate:
            return v_art, Rs_rel_art, Rs, a_prime
        return v_art

    