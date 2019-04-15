import numpy as np

from collections import namedtuple
from typing import Tuple, List

FirstAndLast = namedtuple('FirstAndLast', ['first', 'last'])
Pose6Dof = Tuple[float, float, float, float, float, float]
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]

LOWE_RATIO = .7                         
CANDIDATE_THRESHOLD = 5                 # in pixels

INTRINSIC_MATRIX = np.float32([ 
 [718.856,   0.0,   607.1928],
[0.0,     718.856, 185.2157],
[0.0,       0.0,     1.0   ]])

D = [0.0, 0.0, 0.0, 0.0, 0.0]