import numpy as np
import open3d as o3d
from pandas import DataFrame
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
from libtiff import TIFF 
from PIL import Image
import tifffile as tiff
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
import glob
import argparse

filename = ""
def PCA(data, correlation=False, sort=True):
    data = np.asarray(data.points)
    average_data = np.mean(data,axis=0)       
    decentration_matrix = data - average_data   
    H = np.dot(decentration_matrix.T,decentration_matrix)  
    eigenvectors,eigenvalues,eigenvectors_T = np.linalg.svd(H)    

    if sort:
        sort = eigenvalues.argsort()[::-1]     
        eigenvalues = eigenvalues[sort]       
        eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors



def project2image(pcd,abnormal_index,savepath,direction=1, bigger=4):
    # pcd: 3D Points
    # direction 1/-1
    # bigger: scale


    point_stack = np.asarray(pcd.points)
    x = point_stack[:, 0]
    y = point_stack[:, 1]
    z = -direction * point_stack[:, 2]

    
    print('min max ', np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z))
    width = np.max(x) - np.min(x)
    hight =  np.max(y) - np.min(y)


    voxel_size = max(width,hight)/(224*0.8)
    voxel_size = voxel_size / bigger
    print("voxel_size: ",voxel_size)

    scale = np.max(z)-np.min(z)
    max_z = np.max(z)
    image_grey = np.zeros((224*bigger,224*bigger,1),dtype=np.float32)
    image_xyz = np.zeros((224*bigger,224*bigger,3),dtype=np.float32)


    for i in range(len(x)):
        dx = x[i]//voxel_size + image_grey.shape[0]/2-1
        dy = y[i]//voxel_size + image_grey.shape[0]/2-1
        dz = (max_z - z[i])/scale*255
        image_grey[int(dy),int(dx)] = max(dz,image_grey[int(dy),int(dx)])


    cv2.imwrite(savepath[0],image_grey)



    sum_z = np.zeros((224*bigger,224*bigger),dtype=np.float32)
    cnt_z = np.zeros((224*bigger,224*bigger),dtype=np.float32)
    sum_x = np.zeros((224*bigger,224*bigger),dtype=np.float32)
    sum_y = np.zeros((224*bigger,224*bigger),dtype=np.float32)
    for i in range(len(x)):
        dx = x[i]//voxel_size + image_xyz.shape[0]/2-1
        dy = y[i]//voxel_size + image_xyz.shape[0]/2-1
        cnt_z[int(dy),int(dx)] += 1
        sum_z[int(dy),int(dx)] += z[i]
        sum_x[int(dy),int(dx)] += x[i]
        sum_y[int(dy),int(dx)] += y[i]


    for i in range(len(x)):
        dx = x[i]//voxel_size + image_grey.shape[0]/2-1
        dy = y[i]//voxel_size + image_grey.shape[0]/2-1
        image_xyz[int(dy),int(dx)] = [sum_x[int(dy),int(dx)]/cnt_z[int(dy),int(dx)],sum_y[int(dy),int(dx)]/cnt_z[int(dy),int(dx)],sum_z[int(dy),int(dx)]/cnt_z[int(dy),int(dx)]]

    # upsamle
    for i in range(int(image_xyz.shape[0])):
        print(i)
        if not image_xyz[i].any()==0:
            continue

        for j in range(image_xyz.shape[0]):
            if not image_xyz[:,j].any()==0:
                continue
            if image_xyz[i,j].all() == 0:
                i1 = 0
                i2 = 0
                j1 = 0
                j2 = 0
                for i1 in range(i,image_xyz.shape[0]):
                    if not image_xyz[i1,j].all() == 0:
                        break
                if i1 >= image_xyz.shape[0]-1:
                    continue
                for i2 in range(i,0,-1):
                    if not image_xyz[i2,j].all() == 0:
                        break
                if i2 == 1:
                    continue
                for j1 in range(j,image_xyz.shape[0]):
                    if not image_xyz[i,j1].all() == 0:
                        break
                if j1 >=image_xyz.shape[0]-1:
                    continue
                for j2 in range(j,0,-1):
                    if not image_xyz[i,j2].all() == 0:
                        break
                if j2 == 1:
                    continue

                new_z = image_xyz[i1,j][2]/(i1-i)+ image_xyz[i2,j][2]/(i-i2)+image_xyz[i,j1][2]/(j1-j)+image_xyz[i,j2][2]/(j-j2)
                new_z = new_z /(1/(i1-i)+1/(i-i2)+1/(j1-j)+1/(j-j2))
                image_xyz[i,j] = [(j-(image_grey.shape[0]/2-1))*voxel_size,(i-(image_grey.shape[0]/2-1))*voxel_size,new_z]



    tiff.imwrite(savepath[1], image_xyz)


    if len(abnormal_index)==0:
        image_gt = np.zeros((224*bigger,224*bigger,1),dtype=np.float32)
        cv2.imwrite(savepath[2],image_gt)
        return
    voxel_size = voxel_size * bigger
    bigger = 1
    point_stack = np.asarray(pcd.points)
    point_stack = point_stack[abnormal_index]
    x = point_stack[:, 0]
    y = point_stack[:, 1]
    z = -direction * point_stack[:, 2]


    image_gt = np.zeros((224*bigger,224*bigger,1),dtype=np.float32)

    for i in range(len(x)):
        dx = x[i]//voxel_size + image_gt.shape[0]/2-1
        dy = y[i]//voxel_size + image_gt.shape[0]/2-1

        image_gt[int(dy),int(dx)] = 255
    cv2.imwrite(savepath[2],image_gt)





def rotate2align(pcd, order=[0,1,2], t=10):

    w,v = PCA(pcd)

    point_cloud_vector1 = v[:, 0]  
    point_cloud_vector2 = v[:, 1] 
    point_cloud_vector3 = v[:, 2]  


    R = np.array([point_cloud_vector1,point_cloud_vector2,point_cloud_vector3])
    R0 = np.zeros((3,3),dtype=np.float64)
    R0[order[0]][0] = 1
    R0[order[1]][1] = 1
    R0[order[2]][2] = 1

    R = np.matmul(R,R0)
    pcd.rotate(R)


    point_stack = np.asarray(pcd.points)
    x = point_stack[:, 0]
    y = point_stack[:, 1]
    z = point_stack[:, 2]
    center = np.array(((np.min(x) + np.max(x))/2,(np.min(y) + np.max(y))/2, (np.min(z) + np.max(z))/2),dtype=np.float32)
    pcd.translate(-center)
    
    pcd.translate([0,0,t])

    return pcd


def single_process(path,savepath,order,direction):
    if path.split('.')[-1] == 'txt':
        input_points=np.loadtxt(path,dtype=np.float32)
        pcd = o3d.geometry.PointCloud()
        abnormal_index = input_points[:,3]==1
        pcd.points = o3d.utility.Vector3dVector(input_points[:,0:3]) 



    point_stack = np.asarray(pcd.points)
    x = point_stack[:, 0]
    y = point_stack[:, 1]
    z = point_stack[:, 2]
    center = np.array(((np.min(x) + np.max(x))/2,(np.min(y) + np.max(y))/2, (np.min(z) + np.max(z))/2),dtype=np.float32)
    pcd.translate(-center)

    if direction == 1:
        pcd = rotate2align(pcd,order,t=-30)
    else:
        pcd = rotate2align(pcd,order,t=30)

    project2image(pcd,abnormal_index,savepath,direction,bigger=2)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess MVTec 3D-AD')
    parser.add_argument('--root_path', type=str,
                        default='./Real3D-AD-PCD/',
                        help='The root path of the Real3D')
    parser.add_argument('--save_path', type=str,
                        default='./real3d-2.5d/',
                        help='The save path')

    # NOTE: You should run the preprocessing.py first

    args = parser.parse_args()
    root_path = args.root_path
    save_path = args.save_path

    class_name = ['airplane','candybar','car','chicken','diamond','duck','fish','gemstone','seahorse','shell','starfish','toffees']


    for name in class_name:
        path = os.path.join(root_path, name, "gt")
        print(path)
        txt_paths = glob.glob(path + "/*_cut.txt")
        para_file = './param/'+name+'.txt'
        input_points=np.loadtxt(para_file,dtype=np.int64)
        with ProcessPoolExecutor() as executor:
            for txt_path in txt_paths:
                savepath = []
                abmormal_category = txt_path.split('.')[-2].split('_')[-2]
                filename =  txt_path.split('\\')[-1].split("_")[0]
                print(filename)


                orders ={input_points[i,0]:input_points[i,2:5] for i in range(len(input_points))}
                directions = {input_points[i,0]:input_points[i,1] for i in range(len(input_points))}

                order = orders[int(filename)]
                direction = directions[int(filename)]
                os.makedirs(save_path+name+"/test/"+abmormal_category+"/rgb/",exist_ok=True)
                os.makedirs(save_path+name+"/test/"+abmormal_category+"/xyz/",exist_ok=True)
                os.makedirs(save_path+name+"/test/"+abmormal_category+"/gt/",exist_ok=True)
                
                savepath.append(save_path+name+"/test/"+abmormal_category+"/rgb/"+filename+".png")
                savepath.append(save_path+name+"/test/"+abmormal_category+"/xyz/"+filename+".tiff")
                savepath.append(save_path+name+"/test/"+abmormal_category+"/gt/"+filename+".png")

                
                # single_process(txt_path,savepath,order,direction)
                executor.submit(single_process, txt_path,savepath,order,direction)



