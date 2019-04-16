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

#########################
# Images
#########################
def setup_img_plot(fig_num):
    """
    Setup the figures and axes for an image
    """
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111)        
    return fig, ax

def displayLandmarkMatches(fig, ax, img_prev, kp_prev, img_current, kp_current, matches, non_matches):
    """
    Draw the Landmark Matches on the given axes
    """
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    kp_current = kp_current.tolist()
    out = cv2.drawMatches(img_current, kp_current, img_prev, kp_prev, matches, None,
                               matchColor=green,
                               singlePointColor=yellow,
                               flags=0)
    
    #out = cv2.drawMatches(img_current, kp_current, img_prev, kp_prev, non_matches, out,
    #                           matchColor=red,
    #                           singlePointColor=yellow,
    #                           flags=1)

    show_img(fig, ax, cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def display_of(fig, ax, image, kp_from, kp_to):
    """
    Display Feature Based Optical Flow
    Draw arrows from every keypiont in kp_from to kp_to
    """
    green = (0, 255, 0)
    width = 2
    
    for index in range(len(kp_from)):
        arrow_from = tuple(np.array(kp_from[index].pt, dtype=int))
        arrow_to = tuple(np.array(kp_to[index].pt, dtype=int))
        cv2.arrowedLine(image, arrow_from, arrow_to, green, width)

    show_img(fig, ax, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def frame_summary(fig, ax, base_image, current_state):
    """
    Plots the current registered landmarks in red and the current
    candidates in green. 
    """
    green = (0, 255, 0)
    red = (0, 0, 255)

    #first plot keypoints
    img_kp = cv2.drawKeypoints(base_image, current_state.get_registered_kp(), \
        outImage=np.array([]), color=green)
    #then plot candidates
    img_kp = cv2.drawKeypoints(img_kp, current_state.candidates.get_kps(), \
        outImage=np.array([]), color=red)
    
    show_img(fig, ax, cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))

def show_img(fig, ax, image):
    """
    Helper to get image onto an axis and queue to be drawn at next pause
    """
    ax.imshow(image)
    fig.canvas.draw_idle() # Will draw on the next pause

#########################
# Modified James/Chris 
#########################
def plot_3d_landmarks(fig, ax, new_pose, landmarks):
    """
    @param pose: 3x4 [R | T]
    @param landmarks: List(Point3D) 
    """
    
    xFrame=3
    yFrame=1
    zFrame=xFrame/3
    
    transl = new_pose[:,-1]
    frameCorners=np.vstack(([-xFrame,yFrame,zFrame],[-xFrame,yFrame,-zFrame],[xFrame,yFrame,-zFrame],[xFrame,yFrame,zFrame],[-xFrame,yFrame,zFrame]))
    #rotationMatrix=rotation_matrix_fromRPY(values[i,3],values[i,4],values[i,5])
    rotationMatrix=new_pose[0:3, 0:3]

    for j in range(5):
        frameCorners[j,:]=(rotationMatrix@frameCorners[j,:].T).T
        
    frameCorners[:,0] += transl[0]
    frameCorners[:,1] += transl[1]
    frameCorners[:,2] += transl[2]
    ax.plot(frameCorners[:,0],frameCorners[:,1],frameCorners[:,2], color='b',alpha=0.5)
    for j in range(5):
        ax.plot((frameCorners[j,0],transl[0]),(frameCorners[j,1],transl[1]),(frameCorners[j,2],transl[2]), color='b',alpha=0.5)
    
    ax.scatter(transl[0], transl[1], transl[2], cmap='gist_ncar', s=100)
    
    landmarkValues = np.asarray(landmarks)
    a,b = landmarkValues.shape
    index = range(a)
    ax.scatter(landmarkValues[:,0],landmarkValues[:,1],landmarkValues[:,2], c=index, cmap='cool',s=10, alpha=1)
    #ax.scatter(landmarkValues[:,0],landmarkValues[:,1],landmarkValues[:,2], c='k', s=10, alpha=0.5)
    plt.show()
    
def setup_plots():
    # Setup figure
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection='3d')
    fig1.subplots_adjust(0,0,1,1)
    #plt.get_current_fig_manager().window.setGeometry(640,430,640,676)
    ax1.set_aspect('equal')       
    fig1.suptitle('Cool plot')

    return fig1, ax1
    
##################
# Old...
##################
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
def plot_3d_landmarks_old(pose, landmarks):
    """
    @param pose: 3x4 [R | T]
    @param landmarks: List(Point3D) 
    """
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
        #rotationMatrix=rotation_matrix_fromRPY(values[i,3],values[i,4],values[i,5])
        rotationMatrix=pose[0:3, 0:3]

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