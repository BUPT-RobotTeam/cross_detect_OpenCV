import cv2
import numpy as np

img_size = (640, 480)

def get_params(camera_matrix, dist_coefs):
    global img_size
    return {
        'camera_matrix': camera_matrix,
        'dist_coefs': dist_coefs,
        'new_cammtx': cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coefs, img_size, 0
        )[0]
    }


def undistort(params, frame):
    return cv2.undistort(frame, params['camera_matrix'],
                         params['dist_coefs'], None, params['new_cammtx']
                         )


g200_params = get_params(camera_matrix=np.array([
    [391.459091, 0.000000, 329.719318],
    [0.000000, 391.714735, 229.722416],
    [0.000000, 0.000000, 1.000000]
]),
    dist_coefs=np.array([
        -0.34456177150589806, 0.08938559391911026,
        0.0026686183140887153, -0.0035206005954522388, 0.0
    ])
)

s908_params = get_params(camera_matrix=np.array([
    [452.2764826544423, 0.0, 359.89953113602235],
    [0.0, 451.6034994124971, 254.37194889475958],
    [0.000000, 0.000000, 1.000000]
]),
    dist_coefs=np.array([
        -0.35302786455375235, 0.08773253097338021,
        0.00402433714716149, -0.00787628412411706, 0.0
    ])
)