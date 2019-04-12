from parameters import *

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

def extract_parameters():
    """
    Extracts the necessary intrinisc parameters from the INTRINISC_MATRIX. 
    """
    f = INTRINSIC_MATRIX[0][0]
    cx = INTRINSIC_MATRIX[0][2]
    cy = INTRINSIC_MATRIX[1][2]

    return f, cx, cy