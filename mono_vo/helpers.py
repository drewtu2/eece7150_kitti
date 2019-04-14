from parameters import *
import numpy as np
from typing import Tuple
import cv2

def match_features(matcher, kp1, desc1, kp2, desc2, ratio = LOWE_RATIO):
    """
    Matches a set of features and points using the given features matcher and 
    filtering the points using the LOWE RATIO Test.
    """
    matches = matcher.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    good_matches = []
    candidate_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
        else:
            candidate_matches.append(m)
    # Filter out the kps and desc for the good matches
    kp1_good = [kp1[mat.queryIdx] for mat in good_matches]
    kp2_good = [kp2[mat.trainIdx] for mat in good_matches]
    desc1_good = [desc1[mat.queryIdx] for mat in good_matches]
    desc2_good = [desc2[mat.queryIdx] for mat in good_matches]
    
    kp1_candidate = [kp1[mat.queryIdx] for mat in candidate_matches]
    kp2_candidate = [kp2[mat.trainIdx] for mat in candidate_matches]
    desc1_candidate = [desc1[mat.queryIdx] for mat in candidate_matches]
    desc2_candidate = [desc2[mat.queryIdx] for mat in candidate_matches]


    print("Matches Pre Ratio Test: " + str(len(matches)))
    print("Matches Post Ratio Test: " + str(len(good_matches)))

    if len(good_matches) < 5:
        print("\n\nONLY {} MATCHES DETECTED\n\n".format(len(good_matches)))

    return kp1_good, desc1_good, kp2_good, desc2_good, \
           kp1_candidate, desc1_candidate, kp2_candidate, desc2_candidate

def keypoints_to_umat(keypoints):
    return np.float32([point.pt for point in keypoints])


####################################################
# Extraction Helper Functions
####################################################
def extract_parameters() -> Tuple[float, float, float]:
    """
    Extracts the necessary intrinisc parameters from the INTRINISC_MATRIX. 
    """
    f = INTRINSIC_MATRIX[0, 0]
    cx = INTRINSIC_MATRIX[0, 2]
    cy = INTRINSIC_MATRIX[1, 2]

    return f, cx, cy

def extract_pose(transition_matrix) -> Pose6Dof:
    r, p, y = extract_rpy(transition_matrix)
    x, y, z = extract_xyz(transition_matrix)

    return (x, y, z, r, p, y)

def extract_xyz(transition_matrix) -> Tuple[float, float, float]:
    x = transition_matrix[0, 3]
    y = transition_matrix[1, 3]
    z = transition_matrix[2, 3]

    return x, y, z

def extract_rpy(transition_matrix) -> Tuple[float, float, float]:
    """
    Extracts the roll pitch and yaw from the 3x4 transition matrix. 
    """
    r = transition_matrix[0:3, 0:3]

    yaw = np.arctan2(r[1, 0], r[0, 0])
    roll = np.arctan2(r[2, 1], r[2, 2])
    
    denom = np.sqrt(r[2,1]**2 + r[2, 2]**2)
    pitch = np.arctan2(-r[2, 0], denom)

    return roll, pitch, yaw

def scale_landmarks(landmarks):
    scaled_landmarks = landmarks/landmarks[3, :]
    return scaled_landmarks[0:3, :]

def extract_landmarks(landmarks) -> List[Point3D]:
    """
    Takes in an 4xN ndarray with each column representing 
    the (X, Y, Z, S) of a landmark and returns it as a
    list of tuples.  
    """
    scaled = scale_landmarks(landmarks)
    scaled = scaled.T
    # scaled_landmarks is now Nx3
    # return scaled_landmarks.tolist()
    return [tuple(l) for l in scaled]


#############
# From Vikrant
#############
def T_from_PNP(coord_3d, img_pts, K, D):
    '''
    This function accepts a Nx3 vector of 3D landmark coordinates and Nx2 vector of 
    corresponding image coordinates and returns the homogenous transform T to the 
    origin of the coordinates.
    Returns success, T, mask
    '''
    success, rvec_to_obj, tvecs_to_obj, inliers = cv2.solvePnPRansac(coord_3d, img_pts, 
                                                   K, D, iterationsCount=250, reprojectionError=4.0,
                                                   confidence=0.9999)

    if success:    
        R_to_obj, _ = cv2.Rodrigues(rvec_to_obj)
        mask = np.zeros(len(img_pts)).astype('bool')
        mask[inliers[:,0]]=True

        return success, compose_T(*pose_inv(R_to_obj, tvecs_to_obj)), mask
    else: 
        return success, None, None

def compose_T(R,t):
    #return np.vstack((np.hstack((R,t)),np.array([0, 0, 0, 1])))
    return np.hstack((R,t))

def decompose_T(T_in):
    return T_in[:3,:3], T_in[:3,[-1]].T

def pose_inv(R_in, t_in):
    t_out = -np.matmul((R_in).T,t_in)
    R_out = R_in.T
    return R_out,t_out

def T_inv(T_in):
    R_in = T_in[:3,:3]
    t_in = T_in[:3,[-1]]
    R_out = R_in.T
    t_out = -np.matmul(R_out,t_in)
    return np.vstack((np.hstack((R_out,t_out)),np.array([0, 0, 0, 1])))
