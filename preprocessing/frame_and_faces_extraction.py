
"""
This code shows how the face detection and extraction was performed as first preprocessing step in the FOWS dataset.

Given the path to the videos for a subset of the dataset, i.e. the original videos, of a certain user, 
the code detects the faces via mediapipe's face detector.
Then, from the face bounding box computed by mediapipe face detector, the coordinates are enlarged of a 30%, to include more face infomation. 
The enlarged bbox are used to extract the user faces from the videos and save the images in the destination folder.

This procedure is repeated for all algorithms and for all users in the dataset.
The code reports an example of the preprocessing for user_85808.

Please note that the preprocessing was done in this way to allow us to keep under control the face extraction for each user and each algorithm.
The code might be optimized for preprocessing the whole user or the whole dataset.
"""


from genericpath import exists
import os
import mediapipe as mp
import cv2
# import numpy as np
import re
import json

# mediapipe face detection model
face_detection_model = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
# model_selection=0 -> use the short model for the face detection
# min_detection_confidence=0.5 -> min confidence threshold to detect a face

# make the facedetector run in CPU mode

# ----------------------------------------- #
# # mp face detector 
# BaseOptions = mp.tasks.BaseOptions
# FaceDetector = mp.tasks.vision.FaceDetector
# FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
# VisionRunningMode = mp.tasks.vision.RunningMode

# # Create an FaceDetector object.
# face_base_options = BaseOptions(model_asset_path='./face_landmark_model/blaze_face_short_range.tflite')
# face_options = FaceDetectorOptions(
#     base_options=face_base_options,
#     running_mode=VisionRunningMode.VIDEO,
#     min_detection_confidence=0.88, # min confidence threshold to detect a face (fine-tuned)
#     min_suppression_threshold=0.3 # min non-max suppression threshold to detect multiple faces in an image 
#     )
# # with FaceDetector.create_from_options(options) as detector:
# face_detection_model = FaceDetector.create_from_options(face_options)
# ----------------------------------------- #

def sort_paths(path):
    # given a path to a folder, sort the files in the folder by the number after 'hand_occlusion' or 'obj_occlusion' in the folder
    path = sorted(path, key=lambda x: int(re.search(r'\d+', x).group()))
    return path


def get_file_name(path):
    return os.path.basename(path)

def create_subfolders(path):
    # create a subfolder for each challenge
    os.makedirs(path + '/hand_occlusion_1')
    os.makedirs(path + '/hand_occlusion_2')
    os.makedirs(path + '/hand_occlusion_3')
    os.makedirs(path + '/obj_occlusion_1')
    os.makedirs(path + '/obj_occlusion_2')
    os.makedirs(path + '/obj_occlusion_3')

def create_subfolders_demo(path):
    # create the subfolder for the demo
    os.makedirs(path + '/hand_occlusion')
    os.makedirs(path + '/obj_occlusion')

# function to extract all the subfolders in a directory
def extract_subfolders(directory):
    # return subfolders following the order of the subfolders in the directory (not sorted)
    # follows the alphabetical order of the subfolders in the directory
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    return subfolders

# function to extract all the files in a directory (following the order of the files in the directory)
def extract_files(directory):
    videos = [f.path for f in os.scandir(directory) if f.is_file()]
    return videos

def face_detection(frame, face_detection_model):
    # convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # detect faces in the frame
    results = face_detection_model.process(frame_rgb)
    return results

def extract_face_from_frame(frame, face_bbox):
    # extract face from frame
    x, y, w, h = face_bbox
    face = frame[y:y+h, x:x+w]
    return face


def extract_faces_from_videos(vid_paths, out_paths):

    """
        Extract faces from videos using mediapipe face detection model
        input:
            - vid_paths: list of video paths
            - out_paths: list of output paths where the extracted faces will be saved

        output:
            - the extracted faces are saved to the corresponding output paths

        NOTE: the number of frames extracted from the original videos should be 390 
        (fixed, corresponding to 13 seconds of guiding video)


        Bounding Box (bbox) enlargement (gist):
        - if a face has been detected
        - compute the face bbox coordinates:  bboxC = detection.location_data.relative_bounding_box
        - compute the image width and height: ih, iw, _ = image.shape
        Then compute the bbox coordinates in pixels:
            - x, y, w, h computed from the relative bounding box coordinates (0.0 to 1.0)
            - x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            - bboxC.xmin * iw -> x coordinate of the top-left corner of the bbox (xmin * image_width)
            - bboxC.ymin * ih -> y coordinate of the top-left corner of the bbox
            - bboxC.width * iw -> width of the bbox
            - bboxC.height * ih -> height of the bbox
        -> x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

        - enlarge the face bounding box by 30% (15% to the left and 15% to the top of the bbox)
            x = max(0, x - int(0.15 * w)) # max(0, x - 0.15 * w) -> x coordinate of the top-left corner of the bbox
            y = max(0, y - int(0.15 * h)) # max(0, y - 0.15 * h) -> y coordinate of the top-left corner of the bbox
            # the 0.15 * w and 0.15 * h are used to go 15% to the left and 15% to the top of the bbox
            w = min(iw, w + int(0.3 * w)) # min(iw, w + 0.3 * w) -> width of the bbox
            # w * 1.3 = w + 0.3 * w
            # iw = image width (in pixels) is the maximum width of the image
            # take the min because the width of the bbox cannot be bigger than the width of the image
            # the min function is used to avoid the bbox to go out of the image
            # remember that the bbox is a rectangle that contains the face in the image
            h = min(ih, h + int(0.3 * h)) # min(ih, h + 0.3 * h) -> height of the bbox
        
        - extract face from frame: face = extract_face_from_frame(image, (x, y, w, h))
        - the face is then saved to the corresponding output path
        - the process is repeated for all frames in the video (until 390 frames are extracted)
    """
    
    for vid, out in zip(vid_paths, out_paths):
        print('extracting faces from video:', vid)  # print video name
        print('saving faces to:', out)  # print output path
        # Path to video file
        vidObj = cv2.VideoCapture(vid)
        # Used as counter variable
        count = 0
        # checks whether frames were extracted
        success = 1

        while success and count < 390: # until extracted 390 frames (13 seconds of guiding video)
            # vidObj object calls read function extract frames
            success, image = vidObj.read()
            results = face_detection(image, face_detection_model)
            if results.detections: # if there are faces detected in the frame
                for detection in results.detections:
                    # compute the bbox coordinates
                    bboxC = detection.location_data.relative_bounding_box
                    # get the image width and height
                    ih, iw, _ = image.shape
                    # get the bbox coordinates in pixels
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    # enlarge face bbox by 30%
                    x = max(0, x - int(0.15 * w)) 
                    y = max(0, y - int(0.15 * h)) 
                    # the 0.15 * w and 0.15 * h are used to go 15% to the left and 15% to the top of the bbox
                    w = min(iw, w + int(0.3 * w)) 
                    h = min(ih, h + int(0.3 * h)) # min(ih, h + 0.3 * h) -> height of the bbox

                    # # enlarge face bbox by 30% while keeping the aspect ratio
                    # scale_factor = 1.3
                    # new_w = int(w * scale_factor)
                    # new_h = int(h * scale_factor)
                    # x = max(0, x - (new_w - w) // 2)
                    # y = max(0, y - (new_h - h) // 2)
                    # w = min(iw - x, new_w)
                    # h = min(ih - y, new_h)
                    
                    # extract face from frame
                    face = extract_face_from_frame(image, (x, y, w, h))

                    # check if the frame is in the index of the frames to extract

                    # save face
                    cv2.imwrite(out + "/frame%d.jpg" % count, face)
                    count += 1
        print("done")
        vidObj.release()
        cv2.destroyAllWindows()

def extract_faces_from_videos_json(vid_paths, out_paths, frames_json_path):
    
    
    """
        Extract faces from videos using mediapipe face detection model
        input:
            - vid_paths: list of video paths
            - out_paths: list of output paths where the extracted faces will be saved
            - frames_json_path: path to the json file with the index of the frames to extract (optional)
                - the json file should have the following format:
                {
                    "training": {
                        "user_id": {
                            "algorithm_name": {
                                "challenge_id": [index of the frames to extract]
                            }
                        }
                    }
                    "testing": { **same format as training** }
                }

                -> the json file in the ../dataset/ folder will be used as reference for extracting the frames from the video

        output:
            - the extracted faces are saved to the corresponding output paths

        NOTE: the number of frames extracted from the original videos should be 390 
        (fixed, corresponding to 13 seconds of guiding video)


        Bounding Box (bbox) enlargement (gist):
        - if a face has been detected
        - compute the face bbox coordinates:  bboxC = detection.location_data.relative_bounding_box
        - compute the image width and height: ih, iw, _ = image.shape
        Then compute the bbox coordinates in pixels:
            - x, y, w, h computed from the relative bounding box coordinates (0.0 to 1.0)
            - x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            - bboxC.xmin * iw -> x coordinate of the top-left corner of the bbox (xmin * image_width)
            - bboxC.ymin * ih -> y coordinate of the top-left corner of the bbox
            - bboxC.width * iw -> width of the bbox
            - bboxC.height * ih -> height of the bbox
        -> x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

        - enlarge the face bounding box by 30% (15% to the left and 15% to the top of the bbox)
            x = max(0, x - int(0.15 * w)) # max(0, x - 0.15 * w) -> x coordinate of the top-left corner of the bbox
            y = max(0, y - int(0.15 * h)) # max(0, y - 0.15 * h) -> y coordinate of the top-left corner of the bbox
            # the 0.15 * w and 0.15 * h are used to go 15% to the left and 15% to the top of the bbox
            w = min(iw, w + int(0.3 * w)) # min(iw, w + 0.3 * w) -> width of the bbox
            # w * 1.3 = w + 0.3 * w
            # iw = image width (in pixels) is the maximum width of the image
            # take the min because the width of the bbox cannot be bigger than the width of the image
            # the min function is used to avoid the bbox to go out of the image
            # remember that the bbox is a rectangle that contains the face in the image
            h = min(ih, h + int(0.3 * h)) # min(ih, h + 0.3 * h) -> height of the bbox
        
        - extract face from frame: face = extract_face_from_frame(image, (x, y, w, h))
        - the face is then saved to the corresponding output path
        - the process is repeated for all frames in the video (until 390 frames are extracted)
    """

    if frames_json_path:
        # load the json file with the index of the frames to extract
        with open(frames_json_path, 'r') as f:
            frames_index = json.load(f)

    else:
        print("No JSON file with the index of the frames to extract")
        exit()

    for vid, out in zip(vid_paths, out_paths):
        print('extracting faces from video:', vid)  # print video name
        print('saving faces to:', out)  # print output path
        # user_id/algo_name/challenge_id/frame%d.jpg -> get the user id, algorithm name and challenge id from the output path
        ## get user info from out_path
        # get info on the output path - use these info to extract frames from json
        user_id = vid.split('/')[-3] # get the user id from the video path
        # algo = out.split('/')[-2] # get the algorithm name from the output path
        challenge_id = vid.split('/')[-2] # get the challenge id from the output path
        print(f"user_id: {user_id} - algo: DLC - challenge_id: {challenge_id}")
        # --------------------------------------- #
        # Path to video file
        vidObj = cv2.VideoCapture(vid)
        # Used as counter variable
        count = 0
        # checks whether frames were extracted
        success = 1


        frames_list = frames_index[user_id][challenge_id]
        # convert the list of frames names (e.g., frame170.jpg) to a list of frame numbers (e.g., 170)
        frames_to_extract = [int(frame.split('frame')[1].split('.jpg')[0]) for frame in frames_list]
        print(f"frames to extract from the json file for user {user_id}, algorithm DLC, challenge {challenge_id}:", frames_to_extract)
        print(frames_to_extract)

        # sanity check - 100 frames per challenge
        if frames_to_extract and len(frames_to_extract) != 100:
            print("ERROR: number of frames to extract from the json file is not correct (!= 100)")
        elif frames_to_extract:
            print("number of frames to extract from the json file is correct (= 100)")


        while success and count < 390: # until extracted 390 frames (13 seconds of guiding video)
            # vidObj object calls read function extract frames
            success, image = vidObj.read()
            results = face_detection(image, face_detection_model)
            if results.detections and count in frames_to_extract: # if there are faces detected in the frame
                for detection in results.detections:
                    # compute the bbox coordinates
                    bboxC = detection.location_data.relative_bounding_box
                    # get the image width and height
                    ih, iw, _ = image.shape
                    # get the bbox coordinates in pixels
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    # enlarge face bbox by 30%
                    x = max(0, x - int(0.15 * w)) 
                    y = max(0, y - int(0.15 * h)) 
                    # the 0.15 * w and 0.15 * h are used to go 15% to the left and 15% to the top of the bbox
                    w = min(iw, w + int(0.3 * w)) 
                    h = min(ih, h + int(0.3 * h)) # min(ih, h + 0.3 * h) -> height of the bbox

                    # # enlarge face bbox by 30% while keeping the aspect ratio
                    # scale_factor = 1.3
                    # new_w = int(w * scale_factor)
                    # new_h = int(h * scale_factor)
                    # x = max(0, x - (new_w - w) // 2)
                    # y = max(0, y - (new_h - h) // 2)
                    # w = min(iw - x, new_w)
                    # h = min(ih - y, new_h)
                    
                    # extract face from frame
                    face = extract_face_from_frame(image, (x, y, w, h))

                    # check if the frame is in the index of the frames to extract

                    if frames_to_extract and count in frames_to_extract: # if the frame is not in the index of the frames to extract, skip it
                        # extract face only for the frames in the frams_to_extract_list
                        print(f"frame {count} - face detected in image {get_file_name(vid)}")
                        cv2.imwrite(out + "/frame%d.jpg" % count, face)
                    elif not frames_to_extract: # if there is no json file with the index of the frames to extract, extract face from all the frames with face detected
                        print(f"frame {count} - face detected in image {get_file_name(vid)}")
                        # save face
                        cv2.imwrite(out + "/frame%d.jpg" % count, face)
                    else:
                        print(f"frame {count} - face detected but frame not in the index of the frames to extract, skipping frame {count}")

            count += 1

        print("done")
        vidObj.release()
        cv2.destroyAllWindows()


def sanity_check_faces_extraction(faces_paths):
    """
    Sanity check to see if the number of frames extracted from the original videos is correct (390 frames)
    """
    print("\nsanity check: ") 
    # add sanity check to see if the number of frames extracted from the videos is correct (390 frames)
    for sub in extract_subfolders(faces_paths):
        print("sub:", sub)
        # print(len(extract_files(sub)))
        if len(extract_files(sub)) != 390: # check if all frames extracte from the video
            print("ERROR: number of frames extracted from the real videos is not correct (!= 390)")
        else:
            print("number of frames extracted from the real videos is correct (= 390)")


# -------------------------------------------------------------------------------------------------------------- #
# Example of face extraction from a video for user_85808
# process repeated for all users and all algotihrms (GHOST, SimSwap, FaceDancer)
# user_85808
# extract faces from original videos
# faces_path = './user_faces/user_85808/original_faces/'
# subfolders = extract_subfolders(faces_path)
# #  ../user_faces/user_85808/original_faces/hand_occlusion_1 - hand_occ_2 - hand_occ_3 - obj_occ_1 - obj_occ_2 - obj_occ_3
# print("subfolders:", subfolders)
# vid_path = './original_videos/user_85808_short/'
# videos = extract_files(vid_path)
# print("videos:", videos)

# print("\n user_85808")
# print("\nextracting faces from original videos")
# extract_faces_from_videos(videos, subfolders) # extract faces from all swapped videos and save them to the corresponding subfolders
# print("\nsanity check: ")
# for sub in subfolders:
#     print("sub:", sub)
#     print(len(extract_files(sub)))
#     if len(extract_files(sub)) != 390:
#         print("ERROR: number of frames extracted from the original videos is not correct")
#     else:
#         print("number of frames extracted from the original videos is correct")

# print("\nextracting faces from swapped videos")
# print("\nghost")
# ghost_vid = './swapped_videos/GHOST/user_85808/'
# ghost_faces = './user_faces/user_85808/ghost_faces/'
# extract_faces_from_videos(extract_files(ghost_vid), extract_subfolders(ghost_faces))
# print("\nsanity check: ")
# for sub in extract_subfolders(ghost_faces):
#     print("sub:", sub)
#     print(len(extract_files(sub)))
#     if len(extract_files(sub)) != 390:
#         print("ERROR: number of frames extracted from the ghost videos is not correct")
#     else:
#         print("number of frames extracted from the ghost videos is correct")

# print("\nsimswap")
# simswap_vid = './swapped_videos/SimSwap/user_85808/'
# simswap_faces = './user_faces/user_85808/simswap_faces/'
# extract_faces_from_videos(extract_files(simswap_vid), extract_subfolders(simswap_faces))
# print("\nsanity check: ")
# for sub in extract_subfolders(simswap_faces):
#     print("sub:", sub)
#     print(len(extract_files(sub)))
#     if len(extract_files(sub)) != 390:
#         print("ERROR: number of frames extracted from the simswap videos is not correct")
#     else:
#         print("number of frames extracted from the simswap videos is correct")

# print("\nfacedancer")
# facedancer_vid = './swapped_videos/FaceDancer/user_85808/'
# facedancer_faces = './test/user_faces/user_85808/facedancer_faces/'
# if not os.path.exists(facedancer_faces):
#   print("path doesn't exists")
#   create_subfolders(facedancer_faces)
# else:
#   print("subfolders already exist")
#   extract_subfolders(facedancer_faces)


# extract_faces_from_videos(extract_files(facedancer_vid), extract_subfolders(facedancer_faces))
# print("\nsanity check: ")
# # add sanity check to see if the number of frames extracted from the original videos is correct (390 frames)
# for sub in extract_subfolders(facedancer_faces):
#     print("sub:", sub)
#     print(len(extract_files(sub)))
#     if len(extract_files(sub)) != 390:
#         print("ERROR: number of frames extracted from the facedancer videos is not correct")
#     else:
#         print("number of frames extracted from the facedancer videos is correct")
# -------------------------------------------------------------------------------------------------------------- #


# print("\nfacedancer")
# facedancer_vid = '../dataset/test/user_85808/'
# facedancer_faces = '../dataset/test/test_faces/'
# if not os.path.exists(facedancer_faces):
#   print("path doesn't exists")
#   create_subfolders(facedancer_faces)
# else:
#   print("subfolders already exist")
#   extract_subfolders(facedancer_faces)


# extract_faces_from_videos(extract_files(facedancer_vid), extract_subfolders(facedancer_faces))
# print("\nsanity check: ")
# # add sanity check to see if the number of frames extracted from the original videos is correct (390 frames)
# for sub in extract_subfolders(facedancer_faces):
#     print("sub:", sub)
#     print(len(extract_files(sub)))
#     if len(extract_files(sub)) != 390:
#         print("ERROR: number of frames extracted from the facedancer videos is not correct")
#     else:
#         print("number of frames extracted from the facedancer videos is correct")