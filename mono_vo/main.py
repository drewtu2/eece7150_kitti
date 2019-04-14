import numpy as np
import cv2
import glob
from TiledDetector import TiledDetector
from typing import *
from VoClasses import *
from helpers import *
from parameters import *

class VO_Pipeline:
    def __init__(self, root_folder):
        '''
        Creates a vo pipeline 
        '''
        
        self.dataset = self.load_images(root_folder)
        self.current_state = State()
        self.prev_state = State()
        self.current_pose = np.zeros((3,1))

        # Feature Detector
        self.feature_detector = TiledDetector(cv2.ORB_create(), 8, 17)
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    def load_images(self, image_folder):
        """
        Loads all the images
        """
        img_array = []
        
        for filename in sorted(glob.glob(image_folder)):
            img = cv2.imread(filename)
            img_array.append(img)
        
        print("Loaded {} images".format(len(img_array)))
        return img_array


    def initialize_pipeline(self, f1: int, f2: int):
        '''
        Initialization step

        input: f1 and f2 are the first two frames selected to initialize the pipeline
        output: update self.current_state
        '''

        # Get Keypoints and descriptors
        # kps_xx, desc_xx are np arrays of OpenCV Keypoints and OpenCV Descriptors
        kps1, desc1 = self.feature_detector.detectAndCompute(self.dataset[f1])

        kps2, desc2 = self.feature_detector.detectAndCompute(self.dataset[f2])

        # Match features, throw away candidates
        kps1, desc1, kps2, desc2, _, _, _, _= match_features(self.feature_matcher, kps1, desc1, kps2, desc2)

        kps1_umat = keypoints_to_umat(kps1)
        kps2_umat = keypoints_to_umat(kps2)

        f, cx, cy = extract_parameters()
        # Find the esential matrix, throw away mask
        essential_matrix, _ = cv2.findEssentialMat(kps1_umat, kps2_umat, focal=f, pp= (cx, cy))

        # Find the pose 
        _, rotation, translation, _ = cv2.recoverPose(essential_matrix, kps1_umat, kps2_umat, INTRINSIC_MATRIX, 50.0)

        #   R    T
        #  3x3  3x1
        pose_cam1 = np.eye(4)[0:3, :]
        pose_cam2 = np.hstack([rotation, translation])
    
        print("Cam1: ", pose_cam1)
        print("Cam2: ", pose_cam2)

        # Find the Landmarks in Camera 1 Coordinate Frame
        landmarks = cv2.triangulatePoints(INTRINSIC_MATRIX @ pose_cam1, INTRINSIC_MATRIX @ pose_cam2, 
        kps1_umat.reshape(-1, 1, 2), kps2_umat.reshape(-1, 1, 2))

        #TODO: Need to save pose of the current images origin wrt to the world frame!


    def associate_keypoints(self, current_frame, prev_frame):
        '''
        4.1 Assosciate keypoints
            detect features in current_frame
            match features of current_frame to prev_frame
            update landmark informations

            input: current frame, previous frame, self.prev_state
            output: update self.current_state (keypoints, landmarks)
        '''
        pass

    def estimate_current_pose(self):
        '''
        4.2 Estimate current pose
            estimate current pose based on 2d-3d correspondences

            input: self.current_state(keypoints, landmarks)
            output: self.current_pose
        '''
        pass

    def triangulate_new_landmarks(self, current_frame, prev_frame):
        '''
        4.3 Triangulate new landmarks
            triangulate new landmarks from unregistered features and landmarks

            input: self.prev_state(candidates, pose_history)
            output: self.current_state(keypoints, landmarks, candidates, pose_history)
        '''
        pass

    def _update_correspondences(self):
        '''
        helper function to update 2d-3d correspondences (3.2-3.5)

        '''
        pass

    
if __name__ == "__main__":
    vp = VO_Pipeline("sample/*.png")
    vp.initialize_pipeline(0, 1)
