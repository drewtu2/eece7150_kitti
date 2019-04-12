from collections import namedtuple

FirstAndLast = namedtuple('FirstAndLast', ['first', 'last'])

class State:
    def __init__(self):
        '''
        the actual data type for each member variable depends on implementation
        '''
        self.registered_kp = []                 # List of Tuples(X, Y) - Index corresponds to matching landmark keypoint
        self.landmarks = []                     # List of Tuples(X, Y, Z) - Index corresponds to matching registered keypoint
        self.candidate_kp = Candidates()        # Candidate Keypoints
        self.pose_history = [(0., 0., 0., 0., 0., 0.)]                  # List of Tuples(X, Y, Z, R, P, Y) 

class Candidates:
    def __init__(self):
        self.descriptors = []   # List of OpenCV descriptors
        self.keypoints = []     # List of FirstAndLast((X, Y, Frame Number), (X, Y, Frame Number))
    
    def add_descriptor(self, descriptor):
        self.descriptors.append(descriptor)
