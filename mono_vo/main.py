import numpy as np

class State:
    def __init__(self):
        '''
        the actual data type for each member variable depends on implementation
        '''
        self.keypoints = []
        self.landmarks = []
        self.candidates = []
        self.pose_history = []

class VO_Pipeline:
    def __init__(self, dataset):
        '''
        dataset: image dataset (list of images)

        '''
        self.dataset = dataset
        self.current_state = State()
        self.prev_state = State()
        self.current_pose = np.zeros((3,1))
        pass
    
    def initialize_pipeline(self, f1, f2):
        '''
        Initialization step

        input: f1 and f2 are the first two frames selected to initialize the pipeline
        output: update self.current_state
        '''
        pass

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

    
