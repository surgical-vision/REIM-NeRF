import numpy as np



def convert_ocv_ogl(pose):
    t_m = np.array([[1,0,0,0],
                    [0,-1,0,0],
                    [0,0,-1,0],
                    [0,0,0,1]])
    return pose@t_m


def convert_ocv_llff(pose):
    t_m = np.array([[0,1,0,0],
                    [1,0,0,0],
                    [0,0,-1,0],
                    [0,0,0,1]])
    return pose@t_m


def convert_ogl_llff(pose):
    t_m = np.array([[0,1,0,0],
                    [-1,0,0,0],
                    [0,0,1,0],
                    [0,0,0,1]])
    return pose@t_m




def transform_left_ptcloud(ptc, transformation):
    ptc = np.hstack((ptc, np.ones((ptc.shape[0],1))))
    ptc = (transformation@ptc.T).T
    ptc = ptc[:,:3]/(ptc[:,3].reshape(-1,1))
    return ptc

def transform_right_ptcloud(ptc, transformation):
    ptc = np.hstack((ptc, np.ones((ptc.shape[0],1))))
    ptc = (ptc@transformation)
    ptc = ptc[:,:3]/(ptc[:,3].reshape(-1,1))
    return ptc

ocv_ogl=np.array([[1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1]])


def cubify(points):
    #"points must be a Nx3 array"
    """computes the transformation by which the pointcloud needs to be manipulated 
    in order to fit inside a cube of length 2, positioned in the origin"""
    p = np.hstack((points.copy(),np.ones((points.shape[0],1))))
    m = np.mean(points,axis=0)
    # center the pointcloud
    t = np.eye(4)
    t[:-1,-1]=-m
    p_t = (t@p.T).T

    # scale the pointcloud to fit inside a cube with dimentions 2x2x2
    scale_factor = 1/np.max(np.abs(p_t[:,:-1]))
    p_t = p_t[:,:-1]* scale_factor

    return p_t, t, scale_factor