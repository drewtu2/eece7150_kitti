from parameters import *
import numpy as np
from typing import Tuple
import cv2

def match_features(matcher, kp1, desc1, kp2, desc2, ratio = LOWE_RATIO):
    """
    Matches a set of features and points using the given features matcher and 
    filtering the points using the LOWE RATIO Test.
    """
    matches = matcher.knnMatch(desc2, desc1, k=2)

    # Apply ratio test
    good_matches = []
    non_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
        else:
            non_matches.append(m)

    # Filter out the kps and desc for the good matches
    kp1_good = [kp1[mat.trainIdx] for mat in good_matches]
    kp2_good = [kp2[mat.queryIdx] for mat in good_matches]
    desc1_good = [desc1[mat.trainIdx] for mat in good_matches]
    desc2_good = [desc2[mat.queryIdx] for mat in good_matches]
    
    kp1_non_match = [kp1[mat.trainIdx] for mat in non_matches]
    kp2_non_match = [kp2[mat.queryIdx] for mat in non_matches]
    desc1_non_match = [desc1[mat.trainIdx] for mat in non_matches]
    desc2_non_match = [desc2[mat.queryIdx] for mat in non_matches]


    print("Matches Pre Ratio Test: " + str(len(matches)))
    print("Matches Post Ratio Test: " + str(len(good_matches)))

    if len(good_matches) < 5:
        print("\n\nONLY {} MATCHES DETECTED\n\n".format(len(good_matches)))

    return kp1_good, desc1_good, kp2_good, desc2_good, \
           kp1_non_match, desc1_non_match, kp2_non_match, desc2_non_match

def match_features_indices(matcher, kp1, desc1, kp2, desc2, ratio = LOWE_RATIO):
    """
    Matches a set of features and points using the given features matcher and 
    filtering the points using the LOWE RATIO Test.
    """
    matches = matcher.knnMatch(desc2, desc1, k=2)

    # Apply ratio test
    good_matches = []
    non_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
        else:
            non_matches.append(m)

    # Filter out the kps and desc for the good matches
    kp1_good = [mat.trainIdx for mat in good_matches]
    kp2_good = [mat.queryIdx for mat in good_matches]
    
    kp1_non_match = [mat.trainIdx for mat in non_matches]
    kp2_non_match = [mat.queryIdx for mat in non_matches]


    print("Matches Pre Ratio Test: " + str(len(matches)))
    print("Matches Post Ratio Test: " + str(len(good_matches)))

    if len(good_matches) < 5:
        print("\n\nONLY {} MATCHES DETECTED\n\n".format(len(good_matches)))

    return kp1_good, kp2_good, kp1_non_match, kp2_non_match

def keypoints_to_umat(keypoints):
    return np.float32([point.pt for point in keypoints])

def check_non_matched_promotion(feature_matcher, non_matched_kps, non_matched_desc, new_kps, new_desc):
    """
    """

    # new landmarks
    new_landmark_kp = []
    new_landmark_desc = []
    
    # new keypoints
    updated_candidates_kp = []
    updated_candidates_desc = []
    updated_candidates_frames = []

    # get the INDICES of the matches between images (instead of the actual kps/desc)
    # we can use the indices to access the kps/desc/frames accordingly. 
    candidate_match_index, new_match_index, candidate_non_matched_index, new_non_match_index \
        = match_features_indices(feature_matcher, candidate_kp, candidate_desc, new_kps, new_desc)

    for ii in len(candidate_match_index):
        index = candidate_match_index[ii]
        new_index = new_match_index[ii]
        if euclidean_distance_from_kp(candidate_kp[index], new_kps[new_index] > CANDIDATE_THRESHOLD):
            # If the distance between the two keypoints is large enough, lets promote
            # to a landmark. Do not mark to add back as keypoint
            new_landmark_kp.append(candidate_kp[index])
            new_landmark_desc.append(candidate_desc[index])
        else:
            # This candidate did not move far enough in the match. It needs to stay
            # as a candidate. 
            updated_candidates_kp.append(candidates.get_kp(index))
            updated_candidates_desc.append(candidates.get_desc(index))
            updated_candidates_frames.append(candidates.get_frame(index))
    
    # Update the values. Only the features that were seen this round were kept. 
    candidates.set_kps_desc_frame(updated_candidates_kp, updated_candidates_desc, updated_candidates_frames)

    new_non_matched = [new_kps[ii] for ii in new_non_match_index]
    desc_non_matched = [new_desc[ii] for ii in new_non_match_index]
    return new_landmark_kp, new_landmark_desc, candidates, new_non_matched, desc_non_matched

def check_candidate_promotion(feature_matcher, candidates, new_kps, new_desc):
    """
    This function takes in a set of candidates and a set of new keypoints and descriptors. It
    looks for any matches between the new keypoints and the existing candidates. If a match 
    is found between the two, the euclidean distance in pixel space is calcualted. If that
    distance is above a mininum threshold, enough parallax is assumed and the candidate is 
    removed from the candidate set and promoted to a newly registered keypoint. 

    The candidate set is pruned to only keep the kps/descriptors that were in found in this frame. 

    Returns newly registered keypoints/descriptors, updated candidates, and non matched candidates/keypoints. 
    """

    # get the relevant keypoints
    candidate_kp = candidates.get_kp()
    candidate_desc = candidates.get_descs()

    # Short circuit the null candidate case 
    if len(candidate_kp) == 0:
        return [], [], candidates, new_kps, new_desc

    # new landmarks
    new_landmark_kp = []
    new_landmark_desc = []
    
    # new keypoints
    updated_candidates_kp = []
    updated_candidates_desc = []
    updated_candidates_frames = []

    # get the INDICES of the matches between images (instead of the actual kps/desc)
    # we can use the indices to access the kps/desc/frames accordingly. 
    candidate_match_index, new_match_index, candidate_non_matched_index, new_non_match_index \
        = match_features_indices(feature_matcher, candidate_kp, candidate_desc, new_kps, new_desc)

    for ii in len(candidate_match_index):
        index = candidate_match_index[ii]
        new_index = new_match_index[ii]
        if euclidean_distance_from_kp(candidate_kp[index], new_kps[new_index] > CANDIDATE_THRESHOLD):
            # If the distance between the two keypoints is large enough, lets promote
            # to a landmark. Do not mark to add back as keypoint
            new_landmark_kp.append(candidate_kp[index])
            new_landmark_desc.append(candidate_desc[index])
        else:
            # This candidate did not move far enough in the match. It needs to stay
            # as a candidate. 
            updated_candidates_kp.append(candidates.get_kp(index))
            updated_candidates_desc.append(candidates.get_desc(index))
            updated_candidates_frames.append(candidates.get_frame(index))
    
    # Update the values. Only the features that were seen this round were kept. 
    candidates.set_kps_desc_frame(updated_candidates_kp, updated_candidates_desc, updated_candidates_frames)

    new_non_matched = [new_kps[ii] for ii in new_non_match_index]
    desc_non_matched = [new_desc[ii] for ii in new_non_match_index]
    return new_landmark_kp, new_landmark_desc, candidates, new_non_matched, desc_non_matched

def euclidean_distance_from_kp(kp1, kp2):
    """
    Calculates the euclidean distance between two OpenCV KeyPoints. 
    """
    x1, y1 = kp1.pt
    x2, y2 = kp2.pt

    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

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
