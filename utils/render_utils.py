import copy
import os
import cv2
import numpy as np
import open3d as o3d
import mvtec3d_util as mvt_util
import json
from sklearn.decomposition import PCA

class MultiViewRender():
    COLOR_X = o3d.visualization.PointColorOption.XCoordinate
    COLOR_Y = o3d.visualization.PointColorOption.YCoordinate
    COLOR_Z = o3d.visualization.PointColorOption.ZCoordinate
    COLOR_NORM = o3d.visualization.PointColorOption.Normal
    COLOR_FPFH = 4
    COLOR_UNIFORM = 5
    COLOR_RGB = 6
    def __init__(self, parameters_path,
                 x_angles=[0., -np.pi / 12, np.pi / 12],
                 y_angles=[0., -np.pi / 12, np.pi / 12],
                 z_angles=[0., -np.pi / 12, np.pi / 12],
                 color=None):
        '''
        Initialize a multi view render for data process
        Args:
            parameters_path: the path to camera parameters
            x_angles: the angles we would like to rotate, be sure the first of x_angles is 0
            y_angles: the angles we would like to rotate, be sure the first of y_angles is 0
            z_angles: the angles we would like to rotate, be sure the first of z_angles is 0
            color: to be added further. Control the rendered color of images.
        '''
        assert x_angles[0] == 0
        assert y_angles[0] == 0
        assert z_angles[0] == 0

        super(MultiViewRender, self).__init__()

        # read the camera intrinsic given by mvtec3d
        # however, we found the parameter not useful
        # so here we only use the height and width
        # self.camera_intrinsic, self.H, self.W = self.read_camera_parameters(parameters_path)
        self.W, self.H = 224, 224 # fixed to 448
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.W, height=self.H, visible=False)
        self.angles = self.get_viewpoints(x_angles, y_angles, z_angles)

        self.color_option = color
        if color in [self.COLOR_X, self.COLOR_Y, self.COLOR_Z, self.COLOR_NORM]:
            self.vis.get_render_option().point_color_option = color
        elif color == self.COLOR_FPFH: # suggest to calculate color outsize this class
            self.pca = PCA(n_components=3)


    def calculate_fpfh_features(self, pcd):
        voxel_size = 0.5
        radius_normal = voxel_size * 2
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid
        (radius=radius_feature, max_nn=100))
        fpfh = pcd_fpfh.data.T

        return fpfh

    def read_camera_parameters(self, path):
        '''
        Read the camera parameters of mvtec3d category
        Args:
            path:

        Returns:

        '''
        with open(path, 'r') as f:
            parameters = json.load(f)
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(parameters['image_width'], parameters['image_height'],
                                                             1. / parameters['focus'], 1. / parameters['focus'],
                                                             parameters['cx'], parameters['cy'])
        return camera_intrinsic, parameters['image_height'], parameters['image_width']

    def rotate_render(self, pcd, rotate_angle, ref_points):
        '''
        Rotate a point cloud with the desired angle and then render it to image
        Args:
            pcd:
            rotate_angle:

        Returns:

        '''
        # rotate pcd
        R = o3d.geometry.get_rotation_matrix_from_xyz(rotate_angle)
        pcd_temp = copy.deepcopy(pcd)
        pcd_temp.rotate(R, pcd_temp.get_center())

        ref_points_temp = copy.deepcopy(ref_points)
        ref_points_temp.rotate(R, ref_points_temp.get_center())

        vis = self.vis
        # render and calculate 3d to 2d pairs
        vis.add_geometry(pcd_temp)
        image = vis.capture_screen_float_buffer(do_render=True)
        points2d = calculate_points2d(vis, np.asarray(ref_points_temp.points).T)
        vis.clear_geometries()

        # convert to rgb
        image = cv2.cvtColor(np.asarray(image) * 255, cv2.COLOR_RGB2BGR)

        return image, points2d

    def get_viewpoints(self, x_angles, y_angles, z_angles):
        '''
        Get the full angle list of all viewpoints.
        Args:
            x_angles:
            y_angles:
            z_angles:

        Returns:

        '''
        angles = []
        for x in x_angles:
            for y in y_angles:
                for z in z_angles:
                    angles.append([x, y, z])
        return angles

    def multiview_render(self, pcd:o3d.geometry.PointCloud, rgb, ref_points):
        '''
        Render a point cloud with the selected viewpoints.
        Args:
            pcd:

        Returns:

        '''
        image_list, ponints_list = [], []

        if self.color_option == self.COLOR_FPFH:
            fpfh = self.calculate_fpfh_features(pcd)
            colors = self.pca.fit_transform(fpfh)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif self.color_option == self.COLOR_UNIFORM:
            pcd.paint_uniform_color((0.5,0.5,0.5))
        elif self.color_option == self.COLOR_RGB:
            assert rgb is not None
            pcd.colors = o3d.utility.Vector3dVector(rgb)
        for angle in self.angles:
            image, points2d = self.rotate_render(pcd, angle, ref_points)
            image_list.append(image)
            ponints_list.append(points2d)

        return image_list, ponints_list


def warpImage(image1, points1, points2, backgroud_color=(255, 255, 255)):
    '''
    Warp image1 to image2 using the paired points
    Args:
        image1:
        points1:
        points2:
        backgroud_color:

    Returns:

    '''
    image2 = np.ones_like(image1)
    image2 = image2 * backgroud_color

    H, W = image1.shape[0], image1.shape[1]

    pos1s = points1.astype(int)
    pos2s = points2.astype(int)

    pos1s[0, :] = np.minimum(np.maximum(pos1s[0, :], 0), W - 1)
    pos1s[1, :] = np.minimum(np.maximum(pos1s[1, :], 0), H - 1)
    pos2s[0, :] = np.minimum(np.maximum(pos2s[0, :], 0), W - 1)
    pos2s[1, :] = np.minimum(np.maximum(pos2s[1, :], 0), H - 1)

    image2[np.round(pos2s[1, :]).astype(int), np.round(pos2s[0, :]).astype(int)] = \
        image1[np.round(pos1s[1, :]).astype(int), np.round(pos1s[0, :]).astype(int)]

    return image2


def calculate_points2d(vis, pcd):
    '''
    Project a point cloud into an image plane,
    Args:
        vis:
        pcd:

    Returns:

    '''
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    intrinsics = param.intrinsic.intrinsic_matrix
    extrinsics = param.extrinsic

    rvec = cv2.Rodrigues(extrinsics[:3, :3])[0]
    tvec = extrinsics[:3, 3:]

    points2d, _ = cv2.projectPoints(pcd, rvec, tvec, intrinsics, None)

    return points2d[:, 0, :].T


def read_pcd(path):
    organized_pc = mvt_util.read_tiff_organized_pc(path)
    unorganized_pc = mvt_util.organized_pc_to_unorganized_pc(organized_pc)
    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
    o3d_pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return o3d_pc


def draw_points3d_on_image(image, points):
    '''
    Draw the projected point cloud on the image plane
    Args:
        image:
        points:

    Returns:

    '''
    image_temp = image.copy()
    h = points[1, :]
    w = points[0, :]
    h_size = image.shape[0]
    w_size = image.shape[1]
    h = np.minimum(np.maximum(h, 0), h_size - 1)
    w = np.minimum(np.maximum(w, 0), w_size - 1)
    image_temp[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
    image_temp[np.round(h).astype(int), np.round(w).astype(int), 1] = 255

    return image_temp

def normalize_colors(colors):
    return (colors - colors.min()) / (colors.max() - colors.min())
