import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
import argparse
from torch.utils.data import DataLoader
from datetime import datetime
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import logging
from tqdm import tqdm
import copy


import open_clip
from data.multi_view_data import *
from model import LinearLayer
from loss import FocalLoss, BinaryDiceLoss
from prompt_ensemble import encode_text_with_prompt_ensemble
from prompt_ensemble_ATP import encode_text_with_prompt_ensemble_ATP,PromptLearner

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    # configs
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, 'log.txt')  

    # model configs
    features_list = args.features_list
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)

    # clip model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, image_size, pretrained=args.pretrained,viprompt=args.viprompt)
    model.to(device)
    model.eval()
    model2 = copy.deepcopy(model)
    for name, param in model.named_parameters():
        if 'prompt' in name:
            continue
        param.requires_grad_(False)


    # for name, param in model.named_parameters():
    #     print(name,param.requires_grad)
    for name, param in model2.named_parameters():
        param.requires_grad_(False)

    tokenizer = open_clip.get_tokenizer(args.model)

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
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

    # transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    # datasets


    train_data = MultiViewDataset(dataset_path=args.train_data_path, transform=preprocess, target_transform=transform,
                                mode='train', dataset_name = args.dataset,views=args.views)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)



    # linear 
    trainable_layer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                                  len(args.features_list), args.model).to(device)
    trainable_layer2 = LinearLayer(model_configs['embed_dim'], model_configs['embed_dim'],args.views,args.model).to(device)
    
    if not args.use_text_prompt:
        optimizer = torch.optim.Adam([{'params': trainable_layer.parameters()},{'params': model.parameters(),"lr":0.0005},
                        {'params':trainable_layer2.parameters(),"lr":0.0005}],lr=learning_rate, betas=(0.5, 0.999))
    else:
        promptlearner = PromptLearner(model2,args.n_ctx_general,args.n_ctx_special).to(device)
        optimizer = torch.optim.Adam([{'params': trainable_layer.parameters()},{'params':promptlearner.parameters(),"lr":0.005},{'params': model.parameters(),"lr":0.0005},
                        {'params':trainable_layer2.parameters(),"lr":0.0005}],lr=learning_rate, betas=(0.5, 0.999))

 

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_class = nn.CrossEntropyLoss()



    for epoch in range(epochs):
        loss_list = []
        iter_idx = 0


        loss = 0
        for items in tqdm(train_dataloader):
            with torch.cuda.amp.autocast():
                obj_list = train_data.get_cls_names()
                if not args.use_text_prompt:
                    text_prompts = encode_text_with_prompt_ensemble(model, obj_list, tokenizer, device)
                else:
                    text_prompts = encode_text_with_prompt_ensemble_ATP(model,promptlearner, obj_list, device)   

            iter_idx += 1

            image = items['img'][0].to(device)
            resized_organized_pc = items['img'][1]
            features = items['img'][2].to(device)
            view_images = items['img'][3]
            view_positions = items['img'][4]
            gt_index = items['img'][5]
            
            cls_name = items['cls_name']

            pcd_features_list = []
            ALL_patch_tokens = []
            ALL_image_features = []
            with torch.cuda.amp.autocast():
                
                text_features = []
                for cls in cls_name:
                    text_features.append(text_prompts[cls])
                text_features = torch.stack(text_features, dim=0)

                for img_idx in range(0,len(view_images)):
                    view_img = view_images[img_idx]
                    view_img = view_img.to(device)
                    position = view_positions[img_idx]

                    image_features, patch_tokens = model.encode_image(view_img, features_list)
                    ALL_image_features.append(image_features)
                    ALL_patch_tokens.append(patch_tokens)

                for idx in range(len(ALL_patch_tokens)):
                    patch_tokens = ALL_patch_tokens[idx]
                    patch_tokens = trainable_layer(patch_tokens)
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


                pcd_features_list = torch.cat(pcd_features_list,2) 
                pcd_features_list = torch.mean(pcd_features_list,2) 
                pcd_features_list = pcd_features_list / pcd_features_list.norm(dim=-1,keepdim=True)
                pcd_features_list = pcd_features_list.unsqueeze(0) 

                con_feature = pcd_features_list @ text_features  
                con_feature = torch.softmax(con_feature,2)
                con_feature = con_feature.permute(0,2,1)
                
                # sample
                pr_sp_mv = []
                ALL_image_features = trainable_layer2(ALL_image_features)
                for image_features in ALL_image_features:
                    image_features = image_features / image_features.norm(dim=-1,keepdim=True)
                    text_probs = (100.0 * image_features @ text_features[0]).softmax(dim=-1)
                    pr_sp_mv.append(text_probs)
                pr_sp_mv = torch.cat(pr_sp_mv,0)
                pr_sp_mv = torch.mean(pr_sp_mv,0)
                gt_class = torch.Tensor([0,0]).to(device)
                gt_class[items['anomaly']] = 1

                gt_class = gt_class.unsqueeze(0)
                pr_sp_mv = pr_sp_mv.unsqueeze(0)


                gt = items['img_mask'].squeeze().to(device)
                gt[gt > 0.5], gt[gt <= 0.5] = 1, 0
                gt = gt.unsqueeze(2)
                unorganized_gt = gt.reshape(gt.shape[0] * gt.shape[1], gt.shape[2])
                gt_points = unorganized_gt[gt_index, :]

                gt_points = gt_points.unsqueeze(0)
                loss += loss_focal(con_feature,gt_points)
                loss += loss_dice(con_feature[:, 1, :], gt_points)


                if iter_idx % 1 == 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_list.append(loss.item())
                    loss = 0
                    torch.cuda.empty_cache()



        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            ckp_path = os.path.join(save_path, 'epoch_' + str(epoch + 1) + '.pth')
            if not args.use_text_prompt:
                torch.save({'para1': trainable_layer.state_dict(),
                        'para2': trainable_layer2.state_dict(),
                        'para3': model.state_dict()}, ckp_path)
            else:
                torch.save({'para1': trainable_layer.state_dict(),
                        'para2': trainable_layer2.state_dict(),
                        'para3': model.state_dict(),
                        'para4': promptlearner.state_dict(),
                        }, ckp_path) 








if __name__ == '__main__':
    parser = argparse.ArgumentParser("MVP-PCLIP for Zero-Shot Point Cloud Anomaly Detection", add_help=True)
    # path
    parser.add_argument("--train_data_path", type=str, default="/home/chengyuqi/code/VAND-APRIL-GAN/data/mixup", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./exps/mixup', help='path to save checkpoints')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-L-14-336.json', help="model configs")
    parser.add_argument("--dataset", type=str, default='mixup', help="train dataset name")

    # Model Setting
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12,18, 24], help="features used")
    # hyper-parameter
    parser.add_argument("--epoch", type=int, default=3, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")

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


    train(args)

