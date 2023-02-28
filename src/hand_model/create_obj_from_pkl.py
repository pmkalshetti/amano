from utils.freq_imports import *
from utils.helper import create_dir


def main():
    path_to_mano_pkl = "./data/mano/mano_v1_2/models/MANO_RIGHT.pkl"
    
    # read vertices and faces from pkl file
    with open(path_to_mano_pkl, "rb") as file:
        mano_data = pickle.load(file, encoding="latin1")   
    v_pkl = mano_data["v_template"] # (778, 3)
    F_pkl = mano_data["f"].astype(np.int32) # (1538, 3)
    

    # save as .obj file
    out_dir = "./output/hand_model/"; create_dir(out_dir, False)
    igl.write_obj(f'{out_dir}/mesh.obj', v_pkl, F_pkl)

    # verify
    v_obj, _, _, F_obj, _, _ = igl.read_obj(f'{out_dir}/mesh.obj')
    print(np.allclose(v_obj, v_pkl, atol=1e-5))
    print(np.allclose(F_obj, F_pkl))
    
if __name__ == "__main__":
    main()
