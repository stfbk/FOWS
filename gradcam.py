import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
# -------------------------------------------------------- #
# https://github.com/jacobgil/pytorch-grad-cam/tree/master #
# pip install pip install grad-cam
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# -------------------------------------------------------- #
import numpy as np
import pandas as pd
import cv2
# import matplotlib.pyplot as plt
# import random
# import torchvision.models as models
# import torch.nn as nn
from torch.utils.data import DataLoader
import os
# from PIL import Image
import argparse
# from torch.utils.data import Dataset
from utilscripts.get_trn_tst_model import *
from utilscripts.customDataset import FaceImagesDataset
from utilscripts.get_trn_tst_model import *

# ------------------------------------------------------------------ #
# -- Compute Grad-CAM for a given model and (subset of a) dataset -- #
# ------------------------------------------------------------------ #
# Example usage:
# python gradcam.py --model mnetv2 --train_dataset fows_occ --test_dataset fows_no_occ --ft --cam_method "gradcam++" --num-layers 1 --tags "mnetv2_fows_occ_FT_vs_fows_no_occ"
# ------------------------------------------------------------------ #

def compute_gradacm(model, test_dataloader, device, model_name, exp_results_path, cam_method, num_layers= 1, gotcha = False):
    model.eval()
    # correct = 0
    # total = 0

    # Define the target layer for Grad-CAM
    if 'mnetv2' in model_name:
        if num_layers == 1:
            target_layers = model.features[-1]
        elif num_layers == 2:
            target_layers = [model.features[-2], model.features[-1]]
        elif num_layers == 3:
            target_layers = [model.features[-3], model.features[-2], model.features[-1]]
        else:
            raise ValueError(f"Unsupported number of target layers: {num_layers}")
        # target_layer = model.features[-1]
    
    elif 'effnetb4' in model_name:
        if num_layers == 1:
            target_layers = [model.features[-2]]  # Last Conv2dNormActivation layer as in effnetb4_dfdc
        elif num_layers == 2:
            target_layers = [model.features[-2], model.features[-1]]  # Last two layers
        elif num_layers == 3:
            target_layers = [model.features[-3], model.features[-2], model.features[-1]]  # Last three layers
        else:
            raise ValueError(f"Unsupported number of target layers: {num_layers}")

    elif 'xception' in model_name:
        if num_layers == 1:
            target_layers = [model.conv4]
        elif num_layers == 2:
            target_layers = [model.conv3, model.conv4]
        elif num_layers == 3:
            target_layers = [model.conv2, model.conv3, model.conv4]
        else:
            raise ValueError(f"Unsupported number of target layers: {num_layers}")

    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    # Initialize the CAM-method
    if cam_method == 'gradcam':
        cam = GradCAM(model=model, target_layers=target_layers)
    elif cam_method == 'gradcam++':
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    elif cam_method == 'eigencam':
        cam = EigenCAM(model=model, target_layers=target_layers)
    elif cam_method == 'scorecam':
        cam = ScoreCAM(model=model, target_layers=target_layers)
    else:
        raise ValueError(f"Unsupported CAM method: {cam_method}")

    results = {
        'image_path': [],
        'label': [],
        'prediction': [],
    }

    for i, (images, labels, image_paths) in enumerate(test_dataloader):
        print(f"Batch {i}") # print the batch number -> 1 image per batch (batch_size = 1)
        print(images.size()) # (batch_size, 3, 224, 224)
        print(labels.size()) # (batch_size, 1)

        print("len(images): ", len(images))
        print("len(labels): ", len(labels))
        print("len(image_paths): ", len(image_paths))
        # Enable gradients
        images.requires_grad = True

        # Forward pass
        output = model(images.to(device))

        # Get the predicted class
        _, predicted = torch.max(output.data, 1)
        print("img_path: ", image_paths)
        print("labels: ", labels)
        print("predicted: ", predicted)
        
        # if gotcha: 
        #     # if user_id == '42': 
        #         if 'original' in image_paths[0]:
        #             user_id = image_paths[0].split('/')[-4]
        #             if user_id == '42':
        #                 frame_id = image_paths[0].split('/')[-1].split('.')[0]
        #                 challenge_id = image_paths[0].split('/')[-2]
        #                 algo_id = image_paths[0].split('/')[-3]
        #                 swap_id = None
        #             else: continue
        #         else:
        #             user_id = image_paths[0].split('/')[-5]
        #             if user_id == '42':
        #                 frame_id = image_paths[0].split('/')[-1].split('.')[0]
        #                 swap_id = image_paths[0].split('/')[-2]
        #                 challenge_id = image_paths[0].split('/')[-3]
        #                 algo_id = image_paths[0].split('/')[-4]
        #             else: 
        #                 continue
        # else: 
        # get info to save the image from image_paths[0]
        frame_id = image_paths[0].split('/')[-1].split('.')[0]
        challenge_id = image_paths[0].split('/')[-2]
        algo_id = image_paths[0].split('/')[-3]
        user_id = image_paths[0].split('/')[-4]
        # swap_id = None

        
        # ---------------------------------------------------------------------------------------------------------- #
        grayscale_cam = cam(input_tensor=images.to(device), targets=[BinaryClassifierOutputTarget(labels.item())])[0]
        # as discussed here: https://github.com/jacobgil/pytorch-grad-cam/issues/325 
        # BinaryClassifierOutputTarget -> if the net has only one output with a sigmoid
        # ---------------------------------------------------------------------------------------------------------- #
        # Read the original image
        rgb_img = cv2.imread(image_paths[0])
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = np.float32(rgb_img) / 255
        rgb_img_resized = cv2.resize(rgb_img, (224, 224))

        # print rgb_img_resized.shape
        print("rgb_img_resized", rgb_img_resized.shape) # (224, 224, 3)
        # print the greyscale_cam heatmap shape
        print("grayscale_cam", grayscale_cam.shape) # (224, 224)
        # 

        # Overlay the heatmap on the original image
        cam_image = show_cam_on_image(rgb_img_resized, grayscale_cam, use_rgb=True)
        cam_subfolders_path = f"{exp_results_path}/{user_id}/{algo_id}/"
        os.makedirs(cam_subfolders_path, exist_ok=True)
       

        # if gotcha: 
        #     if swap_id: 
        #         # print(swap_id)
        #         cam_image_path = f"{cam_subfolders_path}/{challenge_id}_{swap_id}_{frame_id}.png"
        #         results['image_path'].append(f"{user_id}_{algo_id}_{challenge_id}_{swap_id}_{frame_id}")
        #         results['label'].append(labels.cpu().numpy())
        #         results['prediction'].append(predicted.cpu().numpy())
        #         # print(cam_image_path)
        #         # 
        #     else:
        #         cam_image_path = f"{cam_subfolders_path}/{challenge_id}_{frame_id}.png"
        #         results['image_path'].append(f"{user_id}_{algo_id}_{challenge_id}_{frame_id}")
        #         results['label'].append(labels.cpu().numpy())
        #         results['prediction'].append(predicted.cpu().numpy())
        # else: 
        # print("no swap_id")
        cam_image_path = f"{cam_subfolders_path}/{challenge_id}_{frame_id}.png"
        results['image_path'].append(f"{user_id}_{algo_id}_{challenge_id}_{frame_id}")
        results['label'].append(labels.cpu().numpy())
        results['prediction'].append(predicted.cpu().numpy())
            
        # save gradcam
        cv2.imwrite(cam_image_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        print(f"Grad-CAM image saved to: {cam_image_path}")
        # 


    print("len(results['image_path'])", len(results['image_path']))
    print("len(results['label'])", len(results['label']))
    print("len(results['prediction'])", len(results['prediction']))
    print(f"Batch {i}") # should be 14 
    # breakpoint()

    print("Grad-CAM testing completed!")
    results_table = pd.DataFrame(results)
    print("results: \n", results_table)

    # save the results table in a log file called results.log in the same folder as the gradcam result
    results_table.to_csv(f"{exp_results_path}/results.log", index=False)


    
def get_args_parse():
    parser = argparse.ArgumentParser(description='Grad-CAM testing')
    parser.add_argument('--model', type=str, default='mnetv2', help='Model name')
    parser.add_argument('--train_dataset', type=str, default='fows_occ', help='Dataset used for training')
    parser.add_argument('--test_dataset', type=str, default='fows_occ', help='Dataset used for testing')
    parser.add_argument('--tl', action='store_true', help='Use transfer learning model')
    parser.add_argument('--ft', action='store_true', help='Use fine-tuned model')
    parser.add_argument('--tags', type=str, default='mnetv2_fows_occ_FT_vs_fows_no_occ', help='Target type')
    parser.add_argument('--load_baseline', action = 'store_true', help='Load a baseline model' )
    parser.add_argument('--cam_method', type=str, default='gradcam++', choices=['gradcam', 'gradcam++', 'eigencam', 'scorecam'], help='CAM method')
    parser.add_argument('--num-layers', type=int, default=1, help = 'Set num of target layers for CAM analysis')

    return parser

def main():

    # Parse the arguments
    parser = get_args_parse() # get the arguments from the command line 
    args, unknown = parser.parse_known_args() # parse the known arguments and ignore the unknown ones
    print(args)
    gotcha = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    print("model:", args.model)
    print("ft: ", args.ft)
    print("tl: ", args.tl)
    print("train_dataset: ", args.train_dataset)
    print("test_dataset: ", args.test_dataset)
    print("num_layers: ", args.num_layers)
    

    # loading model
    if args.load_baseline :
        # load baseline_weights
        model_path = './model_weights'
        model, model_name = load_baseline_weight(args, model_path)
    else:
        model_path = './results' # where trained models are saved
        model, model_name = load_model_from_path(args, model_path)

    
    model.to(device)
    print("Model loaded!")
    # print("model_path: ", pretrained_model_path)
    print(f"model_name: {model_name}")
    print(model)

    # Define the transformation for the test dataset 
    test_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # select the test dataset (i.e. a random subset of the fows dataset)
    # NOTE: update the dataset paths accordingly
    if args.test_dataset == 'fows_occ':
        test_dataset = FaceImagesDataset('path_to_dataset', test_transform)
    elif args.test_dataset == 'fows_no_occ':
        test_dataset = FaceImagesDataset('path_to_dataset', test_transform)
    # elif args.dataset == 'gotcha_occ':
    #     test_dataset = FaceImagesDataset('path_to_dataset', test_transform)
    #     gotcha = True
    # elif args.dataset == 'gotcha_no_occ':
    #     test_dataset = FaceImagesDataset('path_to_dataset', test_transform)
    #     gotcha = True

    else: 
        raise ValueError(f"Unsupported dataset name: {args.dataset}")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    
    if args.tags:
        model_name = args.tags 
    else:
        model_name = args.model + '_' + args.train_dataset + '_' + ('_TL' if args.tl else '_FT') + '_vs_' + args.test_dataset
        # ie. mnetv2_fows_occ_TL

    if args.num_layers == 1:
        exp_results_path = f'./results/gradcam/{model_name}_{args.cam_method}/'
    else: 
        exp_results_path = f'./results/gradcam/{model_name}_{args.cam_method}_{args.num_layers}_layers/'
    os.makedirs(exp_results_path, exist_ok=True)
    
    compute_gradacm(model, test_dataloader, device, args.model, exp_results_path, args.cam_method, args.num_layers, gotcha) 

    print("done")

if __name__ == '__main__':
    main()