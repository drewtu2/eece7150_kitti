import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from typing import List, Dict, Tuple

Odom = Tuple[float, float, float]
LinkDict = Dict[int, Dict[int, Odom]]
def rotation_matrix_fromRPY(R,P,Y):
   R_mat = np.array([[np.cos(Y)*np.cos(P), np.cos(Y)*np.sin(P)*np.sin(R)-np.sin(Y)*np.cos(R), np.cos(Y)*np.sin(P)*np.cos(R)+np.sin(Y)*np.sin(R)],
       [np.sin(Y)*np.cos(P), np.sin(Y)*np.sin(P)*np.sin(R)+np.cos(Y)*np.cos(R), np.sin(Y)*np.sin(P)*np.sin(R)-np.cos(Y)*np.sin(R)],
       [-np.sin(P), np.cos(P)*np.sin(R), np.cos(P)*np.cos(R)]])
   return R_mat

def plot_3d_landmarks(pose,landmarks):
    values=np.asarray(pose)
    a,b=values.shape
    index=range(a)
    print(index)
    ax = plt.gca(projection="3d")
    xFrame=3
    yFrame=1
    zFrame=xFrame/3
    for i in range(np.size(values,0)):
        frameCorners=np.vstack(([-xFrame,yFrame,zFrame],[-xFrame,yFrame,-zFrame],[xFrame,yFrame,-zFrame],[xFrame,yFrame,zFrame],[-xFrame,yFrame,zFrame]))
        rotationMatrix=rotation_matrix_fromRPY(values[i,3],values[i,4],values[i,5])
        for j in range(5):
            frameCorners[j,:]=(rotationMatrix@frameCorners[j,:].T).T
        frameCorners[:,0]+=values[i,0]
        frameCorners[:,1]+=values[i,1]
        frameCorners[:,2]+=values[i,2]
        ax.plot(frameCorners[:,0],frameCorners[:,1],frameCorners[:,2], color='b',alpha=0.5)
        for j in range(5):
            ax.plot((frameCorners[j,0],values[i,0]),(frameCorners[j,1],values[i,1]),(frameCorners[j,2],values[i,2]), color='b',alpha=0.5)
    ax.scatter(values[:,0],values[:,1],values[:,2], c=index, cmap='gist_ncar',s=100)
    ax.plot(values[:,0],values[:,1],values[:,2], color='r')
    landmarkValues=np.asarray(landmarks)
    a,b=landmarkValues.shape
    index=range(a)
    ax.scatter(landmarkValues[:,0],landmarkValues[:,1],landmarkValues[:,2], c=index, cmap='cool',s=10, alpha=1)
    #ax.scatter(landmarkValues[:,0],landmarkValues[:,1],landmarkValues[:,2], c='k', s=10, alpha=0.5)
    plt.show()
def get_image_corners(shape):
    """
    Takes in an image shape and returns the corners as an 3x4 matrix
    :param shape:(HEIGHT, WIDTH)
    :return: 3xN np array where each column represents a point
    """
    tl = [0, 0, 1]
    tr = [shape[1], 0, 1]
    br = [shape[1], shape[0], 1]
    bl = [0, shape[0], 1]

    corners = np.hstack(
        [np.array(tl).reshape(3, 1),
         np.array(tr).reshape(3, 1),
         np.array(br).reshape(3, 1),
         np.array(bl).reshape(3, 1)])

    return corners


def plot_image_outlines(ax: Axes, polygons: List[np.ndarray]):
    """
    Plots the image outlines based on sets of points
    :param ax: the axis to plot onto
    :param polygons: a list of Nx2 np.array representing the vertices
    :return: the updated axis
    """
    patches = []
    x = []
    y = []
    num_polygons = len(polygons)

    for i in range(num_polygons):
        patches.append(Polygon(polygons[i]))
        x.append(polygons[i][0, 0])
        y.append(polygons[i][0, 1])

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

    #colors = 100*np.random.rand(len(patches))  # Set to random colors
    colors = [100 for n in range(len(patches))] # Set to blue apparently?
    p.set_array(np.array(colors))

    ax.add_collection(p)
    ax.plot(x, y, "g*", label="Corners")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    return ax


def plot_links(ax: Axes, corners: Dict[int, Tuple[int, int, int]], links: LinkDict):
    """
    Plots the links between every corner
    :param ax: the axis to plot onto
    :param corners: a list of Nx2 np.arrays representing the corners of each image.
    :param links: a dictionary of dictionaries of tuples. The keys are the image numbers. Tuple is an odom tuple.
    :return:
    """
    # For every corner...
    for src_index in range(len(corners)):
        # For every link from this corner
        try:
            for key in links[src_index]:
                corner = corners[key]
                relative_odom = links[src_index][key]
                x = [corner[0], corner[0] + relative_odom[0]]
                y = [corner[1], corner[1] + relative_odom[1]]

                link_color = "r"
                label = "First Pass Link"

                if abs(src_index - key) != 1:
                    link_color = "c"
                    label="Proposed Link"

                ax.plot(x, y, link_color, label=label)
        except Exception as e:
            print(e)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    return ax



def displayMatches(img_left, kp1, img_right, kp2, matches, mask, display_invalid, in_image=None, color=(0, 255, 0)):
    '''
    This function extracts takes a 2 images, set of keypoints and a mask of valid
    (mask as a ndarray) keypoints and plots the valid ones in green and invalid in red.
    The mask should be the same length as matches
    '''

    bool_mask = mask.astype(bool)

    single_point_color = (0, 255, 255)

    if in_image is None:
        mode_flag = 0
    else:
        mode_flag = 1

    img_valid = cv2.drawMatches(img_left, kp1, img_right, kp2, matches, in_image,
                               matchColor=color,
                               singlePointColor=single_point_color,
                               matchesMask=bool_mask.ravel().tolist(), flags=mode_flag)

    if display_invalid:
        img_valid = cv2.drawMatches(img_left, kp1, img_right, kp2, matches, img_valid,
                                   matchColor=(0, 0, 255),
                                   singlePointColor=single_point_color,
                                   matchesMask=np.invert(bool_mask).ravel().tolist(),
                                   flags=1)
    return img_valid

if __name__ == "__main__":
    pose_history = np.array([[5,0,-4,0,0,0],
       [3.5,3.5,-3,-0.1,0.1,np.pi/4],
       [0,5,-2,-0.1,0.1,np.pi/2],
       [-3.5,3.5,-1,-0.1,0.1,3*np.pi/4],
       [-5,0,0,-0.1,0.1,np.pi],
       [-3.5,-3.5,1,-0.1,0.1,5*np.pi/4],
       [0,-5,2,-0.1,0.1,3*np.pi/2],
       [3.5,-3.5,3,-0.1,0.1,7*np.pi/4]])
    landmarks = np.array([[5,10,-4],
       [4,11,-5],
       [3,12,-3],
       [6,9,-1],
       [7,8,-2]])
    plot_3d_landmarks(pose_history,landmarks)