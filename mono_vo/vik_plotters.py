from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from helpers import decompose_T


def plot_pose3_on_axes(axes, T, axis_length=0.1, center_plot=False, line_obj_list=None):
    """Plot a 3D pose 4x4 homogenous transform  on given axis 'axes' with given 'axis_length'."""
    return plot_pose3RT_on_axes(axes, *decompose_T(T), axis_length, center_plot, line_obj_list)

def plot_pose3RT_on_axes(axes, gRp, origin, axis_length=0.1, center_plot=False, line_obj_list=None):
    """Plot a 3D pose on given axis 'axes' with given 'axis_length'."""
    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    linex = np.append(origin, x_axis, axis=0)
    
    y_axis = origin + gRp[:, 1] * axis_length
    liney = np.append(origin, y_axis, axis=0)

    z_axis = origin + gRp[:, 2] * axis_length
    linez = np.append(origin, z_axis, axis=0)


    if line_obj_list is None:
        xaplt = axes.plot(linex[:, 0], linex[:, 1], linex[:, 2], 'r-')    
        yaplt = axes.plot(liney[:, 0], liney[:, 1], liney[:, 2], 'g-')    
        zaplt = axes.plot(linez[:, 0], linez[:, 1], linez[:, 2], 'b-')
    
        if center_plot:
            #center_3d_plot_around_pt(axes,origin[0])
            pass
        return [xaplt, yaplt, zaplt]
    
    else:
        line_obj_list[0][0].set_data(linex[:, 0], linex[:, 1])
        line_obj_list[0][0].set_3d_properties(linex[:,2])
        
        line_obj_list[1][0].set_data(liney[:, 0], liney[:, 1])
        line_obj_list[1][0].set_3d_properties(liney[:,2])
        
        line_obj_list[2][0].set_data(linez[:, 0], linez[:, 1])
        line_obj_list[2][0].set_3d_properties(linez[:,2])

        if center_plot:
            #center_3d_plot_around_pt(axes,origin[0])
            pass
        return line_obj_list

def plot_3d_points(axes, vals, line_obj=None, *args, **kwargs):
    if line_obj is None:
        graph, = axes.plot(vals[:,0], vals[:,1], vals[:,2], *args, **kwargs)
        return graph

    else:
        line_obj.set_data(vals[:,0], vals[:,1])
        line_obj.set_3d_properties(vals[:,2])
        return line_obj


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def update_plot(fig, ax2, new_pose, new_landmarks, lm_graph=None):
    # new_pose: 3x4
    # landmark: nx3
    
    # will remove previous landmarks 
    #graph.remove() # this is if you want to remove the previous object if it slows things down too much

    # Plot first pose (show origin)
    cam_pose_0 = plot_pose3_on_axes(ax2, np.eye(4), axis_length=0.5)
    
    new_pose = np.vstack([new_pose, [0, 0, 0, 1]])
    new_landmarks  = np.array(new_landmarks)

    # Plot current pose
    cam_pose = plot_pose3_on_axes(ax2, new_pose, axis_length=1.0)

    # Update the points using line_obj
    #lm_graph = plot_3d_points(ax2, new_landmarks, linestyle="", marker=".", markersize=2, color='r', line_obj=lm_graph)
    
    fig.canvas.draw_idle(); plt.pause(0.01) # this updates the plot
    #set_axes_equal(ax2)
    #ax2.view_init(0, -90) # Show top view
    return lm_graph

def setup_plots():
    # Setup figure
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection='3d')
    fig1.subplots_adjust(0,0,1,1)
    #plt.get_current_fig_manager().window.setGeometry(640,430,640,676)
    ax1.set_aspect('equal')       
    fig1.suptitle('Cool plot')
    
    ax1.autoscale(enable=True)

    return fig1, ax1