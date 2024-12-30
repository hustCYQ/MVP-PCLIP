import os

# import loguru
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *

from torch.utils.data import DataLoader
import numpy as np


def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]

def real3d_classes():
    return [
        "airplane",
        "candybar",
        "car",
        "chicken",
        "diamond",
        "duck",
        "fish",
        "gemstone",
        "seahorse",
        "shell",
        "starfish",
        "toffees",
    ]

class MultiViewDataset(Dataset):
    def __init__(self, dataset_path, transform, target_transform, mode='test',dataset_name = "mvtec3d", views = 9):

        if dataset_name == "mvtec3d":
            self.ALL_CLASS = mvtec3d_classes()
        if dataset_name == "real3d":
            self.ALL_CLASS = real3d_classes()


        self.gt_transform = target_transform
        self.rgb_transform = transform
        self.views = views

        
        self.img_paths = []
        self.multiview_path = []
        self.cls_name = []
        self.gt_paths = []
        self.labels = []
        for cls_name in self.ALL_CLASS:

            read_path = os.path.join(dataset_path, cls_name, 'test')
            img_paths, multiview_path, gt_paths, labels = self.load_dataset(read_path)  # self.labels => good : 0, anomaly : 1


            self.img_paths.extend(img_paths)
            self.multiview_path.extend(multiview_path)
            self.gt_paths.extend(gt_paths)
            self.labels.extend(labels)
            self.cls_name.extend([cls_name]*len(img_paths))


    def get_cls_names(self):
        return self.ALL_CLASS


    def load_dataset(self,read_path):
        img_paths = []
        multiview_path = []
        gt_tot_paths = []
        tot_labels = []
        print(read_path)
        defect_types = os.listdir(read_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(read_path, defect_type, 'rgb') + "/*.png")
                rgb_paths.sort()
                xyz_roots = [s.replace('rgb', 'xyz')[:-4] for s in rgb_paths]

                img_paths.extend(rgb_paths)
                multiview_path.extend(xyz_roots)

                gt_tot_paths.extend([0] * len(rgb_paths))
                tot_labels.extend([0] * len(rgb_paths))
            else:
                rgb_paths = glob.glob(os.path.join(read_path, defect_type, 'rgb') + "/*.png")
                gt_paths = glob.glob(os.path.join(read_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                gt_paths.sort()
                xyz_roots = [s.replace('rgb', 'xyz')[:-4] for s in rgb_paths]

                img_paths.extend(rgb_paths)
                multiview_path.extend(xyz_roots)

                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(rgb_paths))

        assert len(img_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        # loguru.logger.info(f'img tot paths: {img_tot_paths}')
        return img_paths, multiview_path, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb_path = self.img_paths[idx]
        xyz_root = self.multiview_path[idx]
        gt = self.gt_paths[idx]
        label = self.labels[idx]
        cls_name = self.cls_name[idx]
        img = Image.open(rgb_path).convert('RGB')


        img = self.rgb_transform(img) 

        resized_organized_pc, features, view_images, view_positions, gt_index= self.read_xyz(xyz_root)

        if gt == 0:
            gt = torch.zeros(
                [1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        return {'img': (img, resized_organized_pc, features, view_images, view_positions,gt_index), 'img_mask': gt[:1], 'cls_name': cls_name, 'anomaly': label,'img_path': rgb_path}


    def read_xyz(self, xyz_root):

        features = np.load(os.path.join(xyz_root, 'fpfh.npy'))
        organized_pc = read_tiff_organized_pc(os.path.join(xyz_root, 'xyz.tiff'))

        view_image_paths = glob.glob(xyz_root + "/view_*.png")
        view_position_paths = glob.glob(xyz_root + "/view_*.npy")

        view_image_paths.sort()
        view_position_paths.sort()
        

        # 选择视图
        idx = [0,3,6,9,12,15,18,21,24]
        idx = idx[0:self.views]
        view_image_paths = [view_image_paths[id] for id in idx]
        view_position_paths = [view_position_paths[id] for id in idx]


        view_images = [self.rgb_transform(Image.open(image_path).convert('RGB')) for image_path in view_image_paths]
        view_positions = [np.load(position_path) for position_path in view_position_paths]


        resized_organized_pc = resize_organized_pc(organized_pc)


        unorganized_pc = organized_pc_to_unorganized_pc(resized_organized_pc.permute(1,2,0).numpy())
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        gt_index = nonzero_indices
        

        return resized_organized_pc, features, view_images, view_positions,gt_index



def get_data_loader( dataset_path, transform, target_transform, aug_rate, class_name = 'bagel',mode='test', k_shot=0, save_dir=None, obj_name=None, batch_size=1):
    dataset = MultiViewDataset(dataset_path=dataset_path, transform = transform,target_transform=target_transform,aug_rate=aug_rate, class_name=class_name,mode=mode)
    if mode in ['train']:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False,
                                 pin_memory=True)
    elif mode in ['test']:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False,
                                 pin_memory=True)
    else:
        raise NotImplementedError

    return data_loader


def unorganized_data_to_organized(organized_pc, none_zero_data_list):
    '''

    Args:
        organized_pc:
        none_zero_data_list:

    Returns:

    '''
    if not isinstance(none_zero_data_list, list):
        none_zero_data_list = [none_zero_data_list]

    for idx in range(len(none_zero_data_list)):
        none_zero_data_list[idx] = none_zero_data_list[idx].squeeze().detach().cpu().numpy()

    organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy() # H W (x,y,z)
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

    full_data_list = []

    for none_zero_data in none_zero_data_list:
        if none_zero_data.ndim == 1:
            none_zero_data = np.expand_dims(none_zero_data,1)
        full_data = np.zeros((unorganized_pc.shape[0], none_zero_data.shape[1]), dtype=none_zero_data.dtype)
        full_data[nonzero_indices, :] = none_zero_data
        full_data_reshaped = full_data.reshape((organized_pc_np.shape[0], organized_pc_np.shape[1], none_zero_data.shape[1]))
        full_data_tensor = torch.tensor(full_data_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
        full_data_list.append(full_data_tensor)

    return full_data_list




    
