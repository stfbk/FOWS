# example code of running the proeprocessing code using the json file from the dlc swaps dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import os
import sys
import timm # for using the XceptionNet model (pretrained) # pip install timm
import yaml
import glob
import matplotlib.pyplot as plt
# from scipy.special import expit
import glob
import numpy as np
import cv2
# import mediapy as media
import mediapipe as mp
from PIL import Image
# from demo_utils import *

from preprocessing.frame_and_faces_extraction import *

# ---------------------------------------------------------- #

print("TEST DLC VIDEOS")
dlc_vid_root = '../FOWS_demo/demo_videos/dlc/'
dlc_occ_root = '../FOWS_demo/user_faces/dlc_occ/'
dlc_no_occ_root = '../FOWS_demo/user_faces/dlc_no_occ/'

json_occ = '../FOWS_demo/demo_videos/dlc/dlc_occ_frames.json'
json_no_occ = '../FOWS_demo/demo_videos/dlc/dlc_no_occ_frames.json'

# dlc_vid_root points to the directory containing user_XXXXXX folders
# dlc_faces_root points to the base directory where user_XXXXXX output folders will be created

# ------------------------------------------------------------------------------ #
# create the folder structure for the output of the face extraction #

occ_video_paths = []
occ_output_paths = []
no_occ_video_paths = []
no_occ_output_paths = []

# Iterate through each user's video directory within dlc_vid_root
user_video_dirs = extract_subfolders(dlc_vid_root)

for user_video_dir in user_video_dirs:
    # Extract user_id from the user's video directory path
    user_id = os.path.basename(os.path.normpath(user_video_dir))

    # Construct the full output path for this specific user
    dlc_occ_faces_user_path = os.path.join(dlc_occ_root, user_id)
    dlc_no_occ_faces_user_path = os.path.join(dlc_no_occ_root, user_id)

    # Create the user-specific directory if it doesn't exist
    if not os.path.exists(dlc_occ_faces_user_path):
        print(f"Creating user directory: {dlc_occ_faces_user_path}")
        os.makedirs(dlc_occ_faces_user_path)
    else:
        print(f"User directory already exists: {dlc_occ_faces_user_path}")

    # Now create challenge subfolders inside the user directory
    print(f"Creating/Ensuring challenge subfolders in: {dlc_occ_faces_user_path}")
    create_subfolders(dlc_occ_faces_user_path)


    # Create the user-specific directory if it doesn't exist
    if not os.path.exists(dlc_no_occ_faces_user_path):
        print(f"Creating user directory: {dlc_no_occ_faces_user_path}")
        os.makedirs(dlc_no_occ_faces_user_path)
    else:
        print(f"User directory already exists: {dlc_no_occ_faces_user_path}")

    # Now create challenge subfolders inside the user directory
    print(f"Creating/Ensuring challenge subfolders in: {dlc_no_occ_faces_user_path}")
    create_subfolders(dlc_no_occ_faces_user_path)

    # Collect video paths for this user
    # Iterate through challenge subfolders within the user's video directory
    for challenge_video_dir in extract_subfolders(user_video_dir):
        videos_in_challenge = extract_files(challenge_video_dir)
        if videos_in_challenge:
            occ_video_paths.extend(videos_in_challenge)
            no_occ_video_paths.extend(videos_in_challenge)
            # print(videos_in_challenge)
            # Corresponding output paths for this user's challenge
            challenge_id = os.path.basename(os.path.normpath(challenge_video_dir))
            user_occ_output_path = os.path.join(dlc_occ_faces_user_path, challenge_id)
            user_no_occ_output_path = os.path.join(dlc_no_occ_faces_user_path, challenge_id)
            # Extend output paths with the correct user_challenge_output_path for each video in the challenge
            # Assuming one video per challenge, but if multiple, this needs adjustment
            occ_output_paths.extend([user_occ_output_path] * len(videos_in_challenge))
            no_occ_output_paths.extend([user_no_occ_output_path] * len(videos_in_challenge))

# Sort both lists to ensure they match correctly, assuming a consistent naming convention
occ_video_paths = sort_paths(occ_video_paths)
occ_output_paths = sort_paths(occ_output_paths)

print(occ_video_paths)
print(occ_output_paths)

no_occ_video_paths = sort_paths(no_occ_video_paths)
no_occ_output_paths = sort_paths(no_occ_output_paths)

print(no_occ_video_paths)
print(no_occ_output_paths)

# ------------------------------------------------------------------------------ #
# start face extraction

print("TEST DLC VIDEOS - OCC")
extract_faces_from_videos_json(occ_video_paths, occ_output_paths, json_occ)
print("done with OCC videos")

print("\nsanity check: ")
# add sanity check to see if the number of frames extracted from the original videos is correct (390 frames)
for user_subfolder in extract_subfolders('../FOWS_demo/user_faces/dlc_occ'):
    user_id_name = os.path.basename(user_subfolder)
    print(f"\nChecking user folder: {user_id_name}")
    for challenge_subfolder in extract_subfolders(user_subfolder):
        challenge_id_name = os.path.basename(challenge_subfolder)
        print(f"  Checking challenge: {challenge_id_name}")
        files_in_challenge = extract_files(challenge_subfolder)
        if files_in_challenge:
            num_files = len(files_in_challenge)
            print(f"    Number of frames extracted: {num_files}")
            if num_files != 100:
                print(f"    ERROR: number of frames extracted from the facedancer videos is not correct")
            else:
                print("    Number of frames extracted from the facedancer videos is correct")
        else:
            print("    No files found in this challenge folder.")


# ------------------------------------------------------------------------------ #
print("TEST DLC VIDEOS - NO OCC")
extract_faces_from_videos_json(no_occ_video_paths, no_occ_output_paths, json_no_occ)
print("done with NO OCC videos")

print("\nsanity check: ")
# add sanity check to see if the number of frames extracted from the original videos is correct (390 frames)
for user_subfolder in extract_subfolders('../FOWS_demo/user_faces/dlc_no_occ'):
    user_id_name = os.path.basename(user_subfolder)
    print(f"\nChecking user folder: {user_id_name}")
    for challenge_subfolder in extract_subfolders(user_subfolder):
        challenge_id_name = os.path.basename(challenge_subfolder)
        print(f"  Checking challenge: {challenge_id_name}")
        files_in_challenge = extract_files(challenge_subfolder)
        if files_in_challenge:
            num_files = len(files_in_challenge)
            print(f"    Number of frames extracted: {num_files}")
            if num_files != 100:
                print(f"    ERROR: number of frames extracted from the facedancer videos is not correct")
            else:
                print("    Number of frames extracted from the facedancer videos is correct")
        else:
            print("    No files found in this challenge folder.")

