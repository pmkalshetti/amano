import numpy as np

def get_proxy_details():
    # Note: Sphere are defined from tip to MCP
    n_spheres_per_finger = [4, 6, 7, 6, 5]
    n_spheres_cumulative = np.cumsum(n_spheres_per_finger)
    sphere_ids_per_fingers = [
        list(range(0, n_spheres_per_finger[0])),
        list(range(n_spheres_cumulative[0], n_spheres_cumulative[1])),
        list(range(n_spheres_cumulative[1], n_spheres_cumulative[2])),
        list(range(n_spheres_cumulative[2], n_spheres_cumulative[3])),
        list(range(n_spheres_cumulative[3], n_spheres_cumulative[4])),
    ]

    # sphere pairs
    sphere_id_pairs = []
    for finger_id_1 in range(5):
        for sphere_id_1 in sphere_ids_per_fingers[finger_id_1]:
            for finger_id_2 in range(finger_id_1+1, 5):
                for id_sphere_id_2, sphere_id_2 in enumerate(sphere_ids_per_fingers[finger_id_2]):

                    # dont include neighbor finger's base(last) sphere, prevents neigbor finger vertices from influencing current finger
                    n_spheres_in_finger_2 = len(sphere_ids_per_fingers[finger_id_2])
                    if id_sphere_id_2 == n_spheres_in_finger_2 - 1:
                        continue
                    
                    sphere_id_pairs.append([sphere_id_1, sphere_id_2])
    sphere_id_pairs = np.array(sphere_id_pairs) # (257, 2)

    # radii for each sphere
    radii = [
        0.008, 0.009, 0.0095, 0.0105,           
        0.0055, 0.007, 0.0075, 0.0082, 0.009, 0.0095,          
        0.006, 0.007, 0.0075, 0.0082, 0.009, 0.0095, 0.011,
        0.0055, 0.007, 0.0072, 0.008, 0.0085, 0.009, 
        0.005, 0.006, 0.0065, 0.007, 0.0085  
    ]
    radii = np.array(radii) # (28,)

    vert_ids_per_sphere = [
        [727, 763, 748, 734],
        [731, 756, 749, 733],
        [708, 754, 710, 713],
        [250, 267, 249, 28],

        [350, 314, 337, 323],
        [343, 316, 322, 336],
        [342, 295, 299, 297],
        [280, 56, 222, 155],
        [165, 133, 174, 189],
        [136, 139, 176, 170],

        [462, 426, 460, 433],
        [423, 455, 448, 432],
        [430, 454, 457, 431],
        [397, 405, 390, 398],
        [357, 364, 391, 372],
        [375, 366, 381, 367],
        [379, 399, 384, 380],

        [573, 537, 560, 544],
        [566, 534, 559, 543],
        [565, 541, 542, 523],
        [507, 476, 501, 508],
        [496, 498, 491, 495],
        [489, 509, 494, 490],

        [690, 654, 677, 664],
        [682, 658, 642, 669],
        [581, 633, 619, 629],
        [614, 616, 609, 613],
        [607, 627, 612, 608],
    ]
    vert_ids_per_sphere = np.array(vert_ids_per_sphere) # (28, 4)


    return sphere_id_pairs, radii, vert_ids_per_sphere

def compute_sphere_center(v, i_v_per_sphere):
    return np.mean(v[i_v_per_sphere], axis=1)   # (28, 3)
