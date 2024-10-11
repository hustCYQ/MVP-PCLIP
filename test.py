import os
import cv2
import json
import torch
import random
import logging
import argparse
import numpy as np
from PIL import Image
from skimage import measure
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise

import open_clip
from model import LinearLayer
from prompt_ensemble import encode_text_with_prompt_ensemble
from prompt_ensemble_ATP import encode_text_with_prompt_ensemble_ATP,PromptLearner
from data.multi_view_data import *
from utils.visz_utils import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    if len(fprs)>0:
        fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
        pro_auc = auc(fprs, pros[idxes])
    else:
        pro_auc = 0
    return pro_auc


def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    txt_path = os.path.join(save_path, 'log.txt')

    # clip
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, img_size, pretrained=args.pretrained,viprompt=args.viprompt)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)



    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    # seg
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)


    checkpoint = torch.load(args.checkpoint_path)
    linearlayer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                              len(features_list), args.model).to(device)

    linearlayer.load_state_dict(checkpoint["para1"])

    linearlayer2 = LinearLayer(model_configs['embed_dim'], model_configs['embed_dim'],args.views,args.model).to(device)
    linearlayer2.load_state_dict(checkpoint["para2"])

    model.load_state_dict(checkpoint["para3"])



    linearlayer.eval()
    linearlayer2.eval()
    model.eval()

    
    if args.use_text_prompt:
        promptlearner = PromptLearner(model,args.n_ctx_general,args.n_ctx_special).to(device)
        promptlearner.load_state_dict(checkpoint["para4"])
        promptlearner.eval()

  
    
    # dataset
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])

    test_data = MultiViewDataset(dataset_path=args.data_path, transform=preprocess, target_transform=transform,
                                mode='test',dataset_name = args.dataset,views=args.views)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)


    obj_list = test_data.get_cls_names()


    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        if not args.use_text_prompt:
            text_prompts = encode_text_with_prompt_ensemble(model, obj_list, tokenizer, device)
        else:
            text_prompts = encode_text_with_prompt_ensemble_ATP(model,promptlearner, obj_list, device)

    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['gt_sp'] = []
    results['pr_sp'] = []
    from tqdm import tqdm
    for items in tqdm(test_dataloader):
        
        image = items['img'][0].to(device)
        resized_organized_pc = items['img'][1]
        features = items['img'][2].to(device)
        view_images = items['img'][3]
        view_positions = items['img'][4]
        cls_name = items['cls_name']
        gt_index = items['img'][5]
        

        results['cls_names'].append(cls_name[0])
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

        gt_mask_clone = gt_mask.squeeze().numpy().copy() #[1,1,224,224]

        if args.dataset == 'real3d' or args.dataset == 'mvtec3d':
            gt_mask = gt_mask.squeeze(0).permute(1,2,0)
            unorganized_gt_mask = gt_mask.reshape(gt_mask.shape[0] * gt_mask.shape[1], gt_mask.shape[2])
            gt_mask = unorganized_gt_mask[gt_index, :].unsqueeze(0)

        results['imgs_masks'].append(gt_mask)  # px
        results['gt_sp'].append(items['anomaly'].item())

        with torch.no_grad(), torch.cuda.amp.autocast():

            pcd_features_list = []
            ALL_patch_tokens = []
            ALL_image_features=[]
            text_features = []
            for cls in cls_name:
                text_features.append(text_prompts[cls])
            text_features = torch.stack(text_features, dim=0)

        
            for img_idx in range(0,len(view_images)):
                view_img = view_images[img_idx]
                view_img = view_img.to(device)
                image_features, patch_tokens = model.encode_image(view_img, features_list)
                ALL_patch_tokens.append(patch_tokens)
                ALL_image_features.append(image_features)

            # pixel
            image_size = 224
            for idx in range(len(ALL_patch_tokens)):
                patch_tokens = ALL_patch_tokens[idx]
                patch_tokens = linearlayer(patch_tokens)
                position = view_positions[idx].to(device)

                for layer in range(0,len(patch_tokens)):
                    patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                    feature_map = patch_tokens[layer]
                    B, L, C = feature_map.shape
                    H = int(np.sqrt(L))
                    feature_map = F.interpolate(feature_map.permute(0, 2, 1).view(B, C, H, H),
                                                size=image_size, mode='bilinear', align_corners=True)
                    per_feature_map = feature_map[:, :, position[0,1,:], position[0,0,:]].permute(2,1,0)
                    pcd_features_list.append(per_feature_map)


            pcd_features_list = torch.cat(pcd_features_list,2) #[点数, 768, 9]
            pcd_features_list = torch.mean(pcd_features_list,2) #[点数, 768]
            pcd_features_list = pcd_features_list / pcd_features_list.norm(dim=-1,keepdim=True)
            pcd_features_list = pcd_features_list.unsqueeze(0) #[1, 点数, 768]

            con_feature = pcd_features_list @ text_features  #[1, 点数, 2]
            con_feature = torch.softmax(con_feature,2)


            # sample
            pr_sp_mv = []
            ALL_image_features = linearlayer2(ALL_image_features)
            for image_features in ALL_image_features:
                image_features = image_features / image_features.norm(dim=-1,keepdim=True)
                text_probs = (100.0 * image_features @ text_features[0]).softmax(dim=-1)
                pr_sp_mv.append(text_probs)
            pr_sp_mv = torch.cat(pr_sp_mv,0)
            pr_sp_mv = torch.mean(pr_sp_mv,0)
            results['pr_sp'].append(pr_sp_mv[1].cpu().item())


            anomaly_map = unorganized_data_to_organized(resized_organized_pc, [con_feature])[0].to(device)
            anomaly_map = torch.softmax(anomaly_map, dim=1)
            anomaly_map = anomaly_map.cpu().numpy()
            anomaly_map_clone = anomaly_map[0,1,:,:].copy()
            
            if args.dataset == 'real3d' or args.dataset == 'mvtec3d':
                results['anomaly_maps'].append([con_feature[0,:,1].unsqueeze(1).cpu().numpy()])
            else:
                results['anomaly_maps'].append([anomaly_map[0,1,:,:]])

            # visualization
            # path = items['img_path']
            # cls = path[0].split('/')[-3]
            # filename = path[0].split('/')[-1]
            # vis = cv2.cvtColor(cv2.resize(cv2.imread(path[0]), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
            # mask = normalize(anomaly_map[0][1])
            # vis = apply_ad_scoremap(vis, mask)
            # vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
            # save_vis = os.path.join(save_path, 'imgs', cls_name[0], cls)
            # if not os.path.exists(save_vis):
            #     os.makedirs(save_vis)
            # # cv2.imwrite(os.path.join(save_vis, filename), vis)
            # # print(os.path.join(save_vis, filename))


            # resized_organized_pc = resized_organized_pc.squeeze(0).permute(1,2,0)
            # image = image.squeeze(0).permute(1,2,0)
            # plot_sample_o3d([resized_organized_pc.cpu().numpy()], [image.cpu().numpy()], {'x': [anomaly_map_clone]}, [gt_mask_clone], os.path.join(save_vis, filename))


    # metrics
    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    f1_sp_ls = []
    f1_px_ls = []
    aupro_ls = []
    ap_sp_ls = []
    ap_px_ls = []
    # for obj in mvtec3d_classes():
    if args.dataset== "mvtec3d":
        all_classes = mvtec3d_classes()
    if args.dataset== "real3d":
        all_classes = real3d_classes()
    for obj in all_classes:
        table = []
        gt_px = []
        pr_px = []
        gt_sp = []
        pr_sp = []
        pr_sp_tmp = []
        table.append(obj)
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes].squeeze(1).numpy())
                pr_px.append(results['anomaly_maps'][idxes])
                pr_sp_tmp.append(np.max(results['anomaly_maps'][idxes]))
                gt_sp.append(results['gt_sp'][idxes])
                pr_sp.append(results['pr_sp'][idxes])
        gt_px_clone = gt_px
        pr_px_clone = pr_px
        if args.dataset == 'real3d' or args.dataset == 'mvtec3d':
            tmp_px = []
            for xi in gt_px:
                for j in (xi[0]).ravel():
                    tmp_px.append(j)
            gt_px = np.array(tmp_px)

            tmp_px = []
            for xi in pr_px:
                for j in (xi[0]).ravel():
                    tmp_px.append(j)
            pr_px = np.array(tmp_px)
        else:
            gt_px = np.array(gt_px)
            pr_px = np.array(pr_px)
            # gt_px_clone = np.array(gt_px_clone)
            # pr_px_clone = np.array(pr_px_clone)
        gt_sp = np.array(gt_sp)
        pr_sp = np.array(pr_sp)


        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        auroc_sp = roc_auc_score(gt_sp, pr_sp)
        ap_sp = average_precision_score(gt_sp, pr_sp)
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        # f1_sp
        precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
        # f1_px
        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        # aupro
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        if args.dataset == 'real3d' or args.dataset == 'mvtec3d':
            aupro = 0
        else:
            aupro = cal_pro_score(gt_px, pr_px)
            # aupro = 0



        table.append(str(np.round(auroc_px * 100, decimals=1)))
        table.append(str(np.round(f1_px * 100, decimals=1)))
        table.append(str(np.round(ap_px * 100, decimals=1)))
        table.append(str(np.round(aupro * 100, decimals=1)))
        table.append(str(np.round(auroc_sp * 100, decimals=1)))
        table.append(str(np.round(f1_sp * 100, decimals=1)))
        table.append(str(np.round(ap_sp * 100, decimals=1)))

        table_ls.append(table)
        auroc_sp_ls.append(auroc_sp)
        auroc_px_ls.append(auroc_px)
        f1_sp_ls.append(f1_sp)
        f1_px_ls.append(f1_px)
        aupro_ls.append(aupro)
        ap_sp_ls.append(ap_sp)
        ap_px_ls.append(ap_px)

    # logger
    table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(f1_px_ls) * 100, decimals=1)), str(np.round(np.mean(ap_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(aupro_ls) * 100, decimals=1)), str(np.round(np.mean(auroc_sp_ls) * 100, decimals=1)),
                     str(np.round(np.mean(f1_sp_ls) * 100, decimals=1)), str(np.round(np.mean(ap_sp_ls) * 100, decimals=1))])
    results = tabulate(table_ls, headers=['objects', 'auroc_px', 'f1_px', 'ap_px', 'aupro', 'auroc_sp',
                                          'f1_sp', 'ap_sp'], tablefmt="pipe")
    logger.info("\n%s", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MVP-PCLIP for Zero-Shot Point Cloud Anomaly Detection", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="data/multi_view_mvtec_3d_anomaly_detection", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/mvtec', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default=None, help='path to save results')
    # model
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-L-14-336.json', help="model configs")
    parser.add_argument("--dataset", type=str, default='mvtec3d', help="test dataset name")
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6,12,18,24], help="features used")
    parser.add_argument("--image_size", type=int, default=224, help="image size")

    parser.add_argument("--use_vision_prompt",  type=bool, default= True)
    parser.add_argument("--use_text_prompt",  type=bool, default= True)
    parser.add_argument("--n_ctx_general",  type=int, default= 16)
    parser.add_argument("--n_ctx_special",  type=int, default= 8)
    parser.add_argument("--n_KLVP",  type=int, default= 1)
    parser.add_argument("--views",  type=int, default= 9)

    args = parser.parse_args()
    import time
    random.seed(time.time())



    if args.use_vision_prompt == True:
        args.viprompt = {'nums':24,'tokens':args.n_KLVP,'list':args.features_list}
    else:
        args.viprompt = None
    test(args)
