from collections import namedtuple
from typing import Tuple, List
from parameters import *

class State:
    def __init__(self):
        '''
        the actual data type for each member variable depends on implementation
        '''
        # The registered set
        self.registered_kp = []                 # List of CV Keypoints - Index corresponds to matching landmark keypoint
        self.registered_desc = []               # List of CV Descriptors - Index corresponds to matching landmark keypoint
        self.landmarks = []                     # List of Tuples(X, Y, Z) - Index corresponds to matching registered keypoint
        
        # Candidates
        self.candidates = Candidates()       
        
        # These are the current frames non matched features which need to be carried forward
        self.non_matched_kp = []
        self.non_matched_desc = []
        
        # History of all poses
        self.pose_history = [(0, 0, 0, 0, 0, 0)]                  # List of Tuples(X, Y, Z, R, P, Y) 

    def add_pose(self, new_pose: Pose6Dof):
        self.pose_history.append(new_pose)

    def set_non_matched(self, new_non_matched_kp, new_non_matched_desc):
        """
        Update the non matched features/states
        """
        self.non_matched_kp = new_non_matched_kp.copy()
        self.non_matched_desc = new_non_matched_desc.copy()

    def set_lm_kp(self, new_landmarks, new_keypoints, new_desc):
        """
        Updates the existing landmarks and corresponding keypoints with the new values.
        """
        self._set_landmarks(new_landmarks)
        self._set_registered_keypoints(new_keypoints)
        self._set_registered_descriptors(new_desc)

    def set_candidates(self, candidates):
        #TODO: Fix this
        self.candidates = candidates

    def get_pose_history(self) -> List[Pose6Dof]:
        return self.pose_history.copy()
    
    def get_landmarks(self) -> List[Point3D]:
        return self.landmarks.copy()
    
    def get_registered_kp(self) -> List[Point2D]:
        return self.registered_kp.copy()
    
    def get_registered_desc(self):
        return self.registered_desc.copy()     
       
    #def get_candidate_kp(self) -> List[Point2D]:
    #    return self.candidate_kp.copy()
    
    def _set_landmarks(self, new_landmarks: List[Point3D]):
        """
        Overwites the existing set of landmarks with a new set of keypoints
        """
        self.landmarks = new_landmarks.copy()
    
    def _set_registered_keypoints(self, new_keypoints):
        """
        Overwites the existing set of keypoints with a new set of keypoints
        """
        self.landmarks = new_keypoints.copy()
    
    def _set_registered_descriptors(self, new_desc):
        """
        Overwites the existing set of keypoints with a new set of keypoints
        """
        self.registered_desc = new_desc.copy()

    
    

class Candidates:
    def __init__(self):
        self.descriptors = []   # List of OpenCV descriptors
        self.keypoints = []     # List of KeyPoint
        self.frames = []        # List of ints corresponding to the first observed frame number 

    def set_kps_desc_frame(self, l_kps, l_desc, l_frames):
        self.keypoints = l_kps.copy()
        self.descriptors = l_desc.copy()
        self.frames = l_frames.copy()

    def get_kps(self):
        return self.keypoints.copy()
    
    def get_descs(self):
        return self.descriptors.copy()
    
    def get_frames(self):
        return self.frames().copy()
    
    def get_frame(self, index):
        return self.frames[index]
    
    def get_kp(self, index):
        return self.keypoints[index]
    
    def get_desc(self, index):
        return self.descriptors[index]
    
    def extend(self, new_candidates):
        self.keypoints.extend(new_candidates.get_kps())
        self.descriptors.extend(new_candidates.get_descs())
        self.frames.extend(new_candidates.get_frames())
    #def copy(self) -> Candidates:
    #    return Candidates()