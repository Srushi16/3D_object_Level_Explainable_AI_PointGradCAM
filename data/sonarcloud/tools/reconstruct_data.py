#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:53:05 2017
@author: serkan
"""

import ply
import numpy as np
import math
import numpy as np
def reconstruct3d(image,depth_map,x,y,z,yaw,camera_parameters,**kwargs):
    """ Reconstructs scene 
    
    Arguments:
    image -- image of the reconstructed scene
    depth_map -- depth map of the image
    camera_parameters -- Camera intrinsic matrix
                                [fx,  0, cx]
                          K =    [ 0, fy, cy]
                                [ 0,  0,  0]
                          where fx,fy is the focal length of the camera
                          and   cx,cy is the principle point of the camera
    Keyword arguments:
    step -- Controls how many 3d points point cloud will have
            if step == 1 all pixels in the image will be converted
            into a point in the cloud (default 1)
    mesh -- If True, resulting Ply object will contain faces,
            Otherwise it will only contain vertices
    
    transformation -- 4x4 matrix that represents a transformation
            Transforms all points with given transformation
            If None, transformation is not applied (None = Identity)
    """ 
    
    step = kwargs.pop('step', 1)
    mesh = kwargs.pop('mesh', True)
    transformation = kwargs.pop('transformation', None)
    #mesh=True
    #if (transformation is None):
    #    transformation = np.identity(4)
    #else:
    #transformation2 = [[0, 0, -1, 120],
    #                      [0, 1, 0, 0],
    #                      [1, 0, 0, 120],
    #                      [0, 0, 0, 1]]
                          
    #transformation3 =   [[-1, 0, 0, 0],
    #                      [0, 1, 0, 0],
    #                      [0, 0, -1, 240],
    #                      [0, 0, 0, 1]]
    #transformation =     [[0, 0, 1, -120],
    #                      [0, 1, 0, 0],
    #                      [-1, 0, 0, 120],
    #                      [0, 0, 0, 1]]
    ####theta=math.degrees(yaw)
    
    ###########################################       pix2pix/DenseDepth       ################################################
    ''' 
    #theta_x=-0.26
    #theta_x=0.436
    theta_x=0.436
    theta_y=-yaw
    theta_z=0.0
    print('x ', float(x))
    print('y ', float(y))
    print('z ', float(z))
    '''
    #################################################################################################################
    
    ###########################################       2D to 3D       ################################################
    
    ''' theta_x rotating angle '''
    ''' theta_y sonar tilt angle '''
    #theta_x=yaw
    #theta_y=0.436
    #theta_z=0.0
    '''
    theta_x=0.436
    theta_y=yaw
    theta_z=0.0
    '''
    #theta_x=-0.261799
    #theta_x=-0.436332 #PROPER
    theta_x=-0.523599
    #theta_x=-1.0
    theta_y=yaw
    theta_z=0.0
    #print('x ', float(x))
    #print('y ', float(y))
    #print('z ', float(z))
    #x=x*100
    #y=y*100
    #z=700
    #################################################################################################################
    
    
    '''
    Rz=[[(float(np.cos(theta)))*(float(np.cos(theta))), 0, float(np.sin(theta))],
                          [0, 1, 0],
                          [-float((np.sin(theta))), 0, float(np.cos(theta))],
                          [0, 0, 0]]
                          
    Ry=[[(float(np.cos(theta)))*(float(np.cos(theta))), 0, float(np.sin(theta))],
                          [0, 1, 0],
                          [-float((np.sin(theta))), 0, float(np.cos(theta))],
                          [0, 0, 0]]
                          
    Rx=[[(float(np.cos(theta)))*(float(np.cos(theta))), 0, float(np.sin(theta))],
                          [0, 1, 0],
                          [-float((np.sin(theta))), 0, float(np.cos(theta))],
                          [0, 0, 0]]
    '''                                                                  
    #transformation =     [[(float(np.cos(theta)))*(float(np.cos(theta))), 0, float(np.sin(theta)), float(y)],
    #                      [0, 1, 0, float(z)],
    #                      [-float((np.sin(theta))), 0, float(np.cos(theta)), float(x)],
    #                      [0, 0, 0, 1]] 
    #####################################################################################################################
    '''
    transformation =[[(float(np.cos(theta_y)))*(float(np.cos(theta_z))), -((float(np.sin(theta_z)))*(float(np.cos(theta_x))))+(float(np.cos(theta_z)))*(float(np.sin(theta_y)))*(float(np.sin(theta_x))),(float(np.sin(theta_z)))*(float(np.sin(theta_x)))+(float(np.cos(theta_z)))*(float(np.sin(theta_y)))*(float(np.cos(theta_x))), float(y)],
                      [(float(np.cos(theta_y)))*(float(np.sin(theta_z))), (float(np.cos(theta_z)))*(float(np.cos(theta_x)))+(float(np.sin(theta_z)))*(float(np.sin(theta_y)))*(float(np.sin(theta_x))), -((float(np.cos(theta_z)))*(float(np.sin(theta_x))))+(float(np.sin(theta_z)))*(float(np.sin(theta_y)))*(float(np.cos(theta_x))), float(z)],
                      [-(float((np.sin(theta_y)))),(float(np.cos(theta_y)))*(float(np.sin(theta_x))),(float(np.cos(theta_y)))*(float(np.cos(theta_x))), float(x)],
                      [0, 0, 0, 1]]                         
    #print('transformation: ',transformation)                                        
    '''
    ############################################################################################################################
    '''
    theta_x=-0.64
    theta_y=0
    theta_z=0
    '''
    '''
    transformation =[[(float(np.cos(theta_y)))*(float(np.cos(theta_z))), -((float(np.sin(theta_z)))*(float(np.cos(theta_x))))+      (float(np.cos(theta_z)))*(float(np.sin(theta_y)))*(float(np.sin(theta_x))),(float(np.sin(theta_z)))*(float(np.sin(theta_x)))+(float(np.cos(theta_z)))*(float(np.sin(theta_y)))*(float(np.cos(theta_x))), float(y)],
                      [(float(np.cos(theta_y)))*(float(np.sin(theta_z))), (float(np.cos(theta_z)))*(float(np.cos(theta_x)))+(float(np.sin(theta_z)))*(float(np.sin(theta_y)))*(float(np.sin(theta_x))), -((float(np.cos(theta_z)))*(float(np.sin(theta_x))))+(float(np.sin(theta_z)))*(float(np.sin(theta_y)))*(float(np.cos(theta_x))), float(z)],
                      [-(float((np.sin(theta_y)))),(float(np.cos(theta_y)))*(float(np.sin(theta_x))),(float(np.cos(theta_y)))*(float(np.cos(theta_x))), float(x)],
                      [0, 0, 0, 1]]                         
    #print('transformation: ',transformation)
    '''
    #x=0
    #y=0
    transformation =[[(float(np.cos(theta_y)))*(float(np.cos(theta_z))), -((float(np.sin(theta_z)))*(float(np.cos(theta_x))))+      (float(np.cos(theta_z)))*(float(np.sin(theta_y)))*(float(np.sin(theta_x))),(float(np.sin(theta_z)))*(float(np.sin(theta_x)))+(float(np.cos(theta_z)))*(float(np.sin(theta_y)))*(float(np.cos(theta_x))), float(y)],
                      [(float(np.cos(theta_y)))*(float(np.sin(theta_z))), (float(np.cos(theta_z)))*(float(np.cos(theta_x)))+(float(np.sin(theta_z)))*(float(np.sin(theta_y)))*(float(np.sin(theta_x))), -((float(np.cos(theta_z)))*(float(np.sin(theta_x))))+(float(np.sin(theta_z)))*(float(np.sin(theta_y)))*(float(np.cos(theta_x))), float(x)],
                      [-(float((np.sin(theta_y)))),(float(np.cos(theta_y)))*(float(np.sin(theta_x))),(float(np.cos(theta_y)))*(float(np.cos(theta_x))), float(z)],
                      [0, 0, 0, 1]]                         
    #print('transformation: ',transformation)
    
    scene = ply.PLY()
    
    image_width, image_height = image.shape[1], image.shape[0]
    
    inv_intr = np.linalg.inv(camera_parameters)
    
    point_size = (len(range(0,image_width, step)), len(range(0,image_height, step)))
    points = [[None for x in range(point_size[1])] for y in range(point_size[0])]
    z_list=[]
    zlist_min=[]
    transformed_points=[]
    for v in range(0,image_height, step):
        for u in range(0,image_width, step):
            projected_point = np.array([u, v, 1])
            image_point     = np.matmul(inv_intr, projected_point)
            
            x, y = (image_point[0] / image_point[2]), (image_point[1] / image_point[2])
            
            depth = depth_map[v, u]
            
            Z = depth
            #Z = (990)*(depth/255)+10
            #Z = (9.9)*(depth/255)+0.1            
            Y = Z * y
            X = Z * x
            z_list.append(Z)
            transformed_point = np.matmul(transformation, np.array([X, Y, Z, 1]))
            transformed_point_1 = transformed_point[:3]
            transformed_point = transformed_point / transformed_point[3]
            transformed_points.append(transformed_point_1)
            #print('transformed_point: ',transformed_point )
            p = ply.Vertex(transformed_point[0:3].tolist()) #TODO: READ COLOR FROM ORIGINAL IMAGE
            p.r = image[v,u,0] / 255.0
            p.g = image[v,u,1] / 255.0
            p.b = image[v,u,2] / 255.0
            
            p.texture_coordinates = [u / float(image_width) ,v / float(image_height)]
            
            scene.add_vertex(p)
            
            if (mesh):
                points[int(u/step)][int(v/step)] = p
    #print('transformed_points_list: ',len(transformed_points)) 
    #print('transformed_points_list: ',transformed_points[0])            
#            append(p)
    '''
    z_arr = np.array(z_list)
    print('z_arr: ',len(z_arr))
    for l in range(0,len(z_arr)):
        if z_arr[l] != 0:
            zlist_min.append(z_arr[l])
    zlist_min_arr = np.array(zlist_min)
    print('min z float : ',np.min(zlist_min_arr))
    print('max z float : ',np.max(z_arr))
    '''
    if (mesh):
        for u in range(0,point_size[0] - 1):
            for v in range(0,point_size[1] - 1):
                face = ply.Face([points[u    ][v],
                                 points[u + 1][v],
                                 points[u + 1][v + 1],
                                 points[u    ][v + 1],
                                 ])

                scene.add_face(face)
            
    return scene,transformed_points
