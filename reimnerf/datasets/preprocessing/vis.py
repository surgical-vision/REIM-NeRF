import numpy as np

def get_camera_wireframe(scale: float = 1):
    """
    Returns a 3D wireframe of a camera in openGL convention.
    """
    a = 0.5 * np.array([2, 1.5, -4])
    up1 = 0.5 * np.array([0, 1.5, -4])
    up2 = 0.5 * np.array([0, 2, -4])
    b = 0.5 * np.array([-2, 1.5, -4])
    c = 0.5 * np.array([2, -1.5, -4])
    d = 0.5 * np.array([-2, -1.5, -4])
    C = np.zeros(3)
    camera_points = [a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C]
    lines = np.stack([x.astype(float) for x in camera_points]) * scale
    return lines

def get_axis_wireframe(scale:float=1):
    return scale*np.array([[0,0,0],
                           [1,0,0],
                           [0,0,0],
                           [0,0,0],
                           [0,1,0],
                           [0,0,0],
                           [0,0,0],
                           [0,0,1]])


def get_cube_wireframe(center=[0,0,0], length=1):  # pragma: no cover
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    c, l = center, length
    cube_lines = [[c[0]-l/2, c[1]-l/2, c[2]-l/2],
                  [c[0]-l/2, c[1]-l/2, c[2]+l/2],
                  [c[0]-l/2, c[1]+l/2, c[2]+l/2],
                  [c[0]-l/2, c[1]+l/2, c[2]-l/2],
                  [c[0]-l/2, c[1]-l/2, c[2]-l/2],
                  [c[0]+l/2, c[1]-l/2, c[2]-l/2],
                  [c[0]+l/2, c[1]-l/2, c[2]+l/2],
                  [c[0]+l/2, c[1]+l/2, c[2]+l/2],
                  [c[0]+l/2, c[1]+l/2, c[2]-l/2],
                  [c[0]+l/2, c[1]-l/2, c[2]-l/2],
                  [c[0]+l/2, c[1]+l/2, c[2]-l/2],
                  [c[0]-l/2, c[1]+l/2, c[2]-l/2],
                  [c[0]-l/2, c[1]+l/2, c[2]+l/2],
                  [c[0]+l/2, c[1]+l/2, c[2]+l/2],
                  [c[0]+l/2, c[1]-l/2, c[2]+l/2],
                  [c[0]-l/2, c[1]-l/2, c[2]+l/2],
                ]
    return np.array(cube_lines)


def plot_cube(fig, color=(255,0,0), center=(0,0,0), length=1, name='cube'):
    cube_wf = get_cube_wireframe(center=center, length=length)
    color= [f'rgb({color[0]}, {color[1]}, {color[2]})']*len(cube_wf)
    fig.add_scatter3d(x=cube_wf[:,0], y=cube_wf[:,1], z=cube_wf[:,2], mode='lines', line_width=1, line_color=color, name = name)

def plot_camera(fig, pose,  name='', color=(50,50,50), axis=True, scale=1):
    # get camera wire-frame
    wf = get_camera_wireframe(scale)
    color= [f'rgb({color[0]}, {color[1]}, {color[2]})']*len(wf)
    line_w =[1]*len(wf)

    # get axis wire-frame and color and concatenate 
    if axis:
        wf = np.vstack((wf, get_axis_wireframe(scale)))
        axis_color = ['rgb(255,0,0)',
                  'rgb(255,0,0)',
                  'rgb(255,0,0)',
                  'rgb(0,255,0)',
                  'rgb(0,255,0)',
                  'rgb(0,255,0)',
                  'rgb(0,0,255)',
                  'rgb(0,0,255)',
                  'rgb(0,0,255)'
                  ]
        color.extend(axis_color)
        line_w.extend([1]*len(axis_color))


    # convert wf to homogeneous coordinates and transform it based on pose
    # transform based on pose paying attention to c2w format??
    wf = np.hstack((wf, np.ones((len(wf),1))))

    wf = (pose@wf.T)[:-1]

    # add trace to fig
    fig.add_scatter3d(x=wf[0], y=wf[1], z=wf[2], mode='lines', line_width=2, line_color=color)

def plot_wireframe(fig, wireframe, color=(10,10,10), line_width=1):
    color = f"rgb({color[0]},{color[1]},{color[2]})"
    fig.add_scatter3d(x=wireframe[:,0], y=wireframe[:,1], z=wireframe[:,2], mode='lines', line_width=1, line_color=color)


def plot_axis(fig, pose:np.ndarray, scale=1):

    zero_frame_axis = scale*np.array([[0,0,0,1],
                                      [1,0,0,1],
                                      [0,0,0,1],
                                      [0,0,0,1],
                                      [0,1,0,1],
                                      [0,0,0,1],
                                      [0,0,0,1],
                                      [0,0,1,1]]).T
    axis_color = ['rgb(255,0,0)',
                  'rgb(255,0,0)',
                  'rgb(255,0,0)',
                  'rgb(0,255,0)',
                  'rgb(0,255,0)',
                  'rgb(0,255,0)',
                  'rgb(0,0,255)',
                  'rgb(0,0,255)',
                  'rgb(0,0,255)']

    frame_axis = pose@zero_frame_axis
    fig.add_scatter3d(x=frame_axis[0], y=frame_axis[1], z=frame_axis[2], mode='lines',line_width=4, line_color=axis_color)



def plot_pt_cloud(fig, ptc, color=(128,64,128), subsample=100):

    if ptc.shape[0]!=3 and ptc.shape[1]==3:
        ptc=ptc.T
    p_color = f"rgb({color[0]},{color[1]},{color[2]})"#*(ptc.shape[0]//subsample)
    fig.add_scatter3d(x=ptc[0,::subsample], y=ptc[1,::subsample], z=ptc[2,::subsample], mode='markers', marker_color=p_color)
    fig.update_traces(marker_size=1)


import cv2
import mediapy

def show_video(dataset):
    images = [cv2.imread(str(i))[...,::-1] for i in dataset.image_paths]
    images = [cv2.resize(i, (i.shape[1]//4, i.shape[0]//4)) for i in images]
    mediapy.show_video(images)

def show_depthmaps(dataset):
    depthmaps = [cv2.resize(i, (i.shape[1]//4, i.shape[0]//4)) for i in dataset.depthmaps]
    mediapy.show_video(depthmaps)

def show_point_projections(ds):
    # usefull to visualize sparse pointclouds
    dms = []
    for dm in ds.depthmaps:
        dm=(dm>0).astype(np.uint8)*255
        dm = cv2.resize(dm, (dm.shape[1]//4, dm.shape[0]//4))
        dm=(dm>0).astype(np.uint8)*255
        dms.append(dm)
    mediapy.show_video(dms)
    return dms