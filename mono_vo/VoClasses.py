from collections import namedtuple
from typing import Tuple, List
from parameters import *

class State:
    def __init__(self):
        '''
        the actual data type for each member variable depends on implementation
        '''
        self.registered_kp = []                 # List of Tuples(X, Y) - Index corresponds to matching landmark keypoint
        self.landmarks = []                     # List of Tuples(X, Y, Z) - Index corresponds to matching registered keypoint
        self.candidate_kp = Candidates()        # Candidate Keypoints
        self.pose_history = [(0, 0, 0, 0, 0, 0)]                  # List of Tuples(X, Y, Z, R, P, Y) 

    def add_pose(self, new_pose: Pose6Dof):
        self.pose_history.append(new_pose)

    def add_landmarks(self, new_landmarks: List[Point3D]):
        self.landmarks.extend(new_landmarks)

    def get_pose_history(self) -> List[Pose6Dof]:
        return self.pose_history.copy()
    
    def get_landmarks(self) -> List[Point3D]:
        return self.landmarks.copy()
    
    def get_registered_kp(self) -> List[Point2D]:
        return self.registered_kp.copy()
    
    #def get_candidate_kp(self) -> List[Point2D]:
    #    return self.candidate_kp.copy()
    
    

class Candidates:
    def __init__(self):
        self.descriptors = []   # List of OpenCV descriptors
        self.keypoints = []     # List of FirstAndLast((X, Y, Frame Number), (X, Y, Frame Number))
    
    def add_descriptor(self, descriptor):
        self.descriptors.append(descriptor)
    
    def add_keypoints(self, keypoints):
        self.keypoints.extend(keypoints)

    #def copy(self) -> Candidates:
    #    return Candidates()