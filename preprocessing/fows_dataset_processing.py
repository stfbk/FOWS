"""
Given a path to the dataset (user faces) navigate the folder structure until the images are found
Process all the images only in the folder with the name hand_occlusion or obj_occlusion

Apply mediapipe face detection to the images: 
- if face detected -> save the image in the face folder
- if face not detected -> save the image in the no_face folder

The values of min_detection_confidence and min_suppression_threshold are fine-tuned to detect the face in the images.

The images are saved in the same folder structure as the original dataset.

NOTE: the code reported below is a snippet for the 'hand_occlusion' folder for the original dataset. 
The same code can be used for the 'obj_occlusion' folder by changing the path to the dataset.
You can also use the code to process the other algorithms by changing the path to the dataset (i.e. 'simswap_faces/obj_occlusion').

All folders have the same structure: user_id -> algo_name -> challenge_name (hand_occlusion or obj_occlusion).
The processed images are saved in the same folder structure as the original dataset, with the addition of the face and no_face folders.
-> new folder structure: user_id -> algo_name -> challenge_name -> face / no_face

-------------------------------------------------------------------------------------------------------------------------------------
In FOWS the challenges occlude the face less than the ones in GOTCHA. Therefore, we can just detect if the face is visible or not in 
the image. If the face is not detected, it is likely that the face is occluded by a hand or an object. 
The face detection confidence was set to 0.88 since face clearly visible in the image (fine-tuned).

gist: if face not detected, probably there is a hand / obj occluding it -> check if face detected   
"""


import os
import re
import cv2
# import shutil # to move the images
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def sort_paths(paths):
    # sort the paths in numerical order
    paths.sort(key=lambda f: int(re.sub('\D', '', f))) # sort the paths in numerical order
    return paths

def get_file_name(path):
    return path.split("\\")[-1].split(".")[0] # get the file name without the extension

def face_detection(img_path, save_path):
    img = cv2.imread(img_path)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    mp_image = mp.Image.create_from_file(img_path)
    # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # face detection
    face_results = face_detector.detect(mp_image)

    # gist: if face not detected, probably there is a hand / obj -> check if face detected
    if face_results.detections: # face detected
        print(f"Face detected in image {get_file_name(img_path)}")
        cv2.imwrite(save_path + '/face/' + get_file_name(img_path) + '.jpg', img)

    
    else: # mp do not detect face -> there should be a hand / obj occluding the face
        print(f"NO face detected in image {get_file_name(img_path)}")
        cv2.imwrite(save_path + '/no_face/' + get_file_name(img_path) + '.jpg', img)
        # move the image to the no_face subfolder
        # os.
        # shutil(save_path + '/no_face/' + get_file_name(img_path) + '.jpg', img_path)

# ---------------------------------------------------------- #


# mp face detector 
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create an FaceDetector object.
face_base_options = BaseOptions(model_asset_path='./face_landmark_model/blaze_face_short_range.tflite')
face_options = FaceDetectorOptions(
    base_options=face_base_options,
    running_mode=VisionRunningMode.IMAGE,
    min_detection_confidence=0.88, # min confidence threshold to detect a face (fine-tuned)
    min_suppression_threshold=0.3 # min non-max suppression threshold to detect multiple faces in an image 
    )
# with FaceDetector.create_from_options(options) as detector:
face_detector = FaceDetector.create_from_options(face_options)

# test image modification directly in the folder
dataset_path = './path_to_dataset/user_faces/'

save_path = './user_faces_preprocesed/'
# list of all the subdirectories in the directory
directories = os.listdir(dataset_path)


# list all subdir in numerical order
directories = sort_paths(directories)
print(directories)

# ---------------------------------------------------------- #

# preprocess the images in the hand_occlusion and obj_occlusion folders
# save the images in the face and no_face folders
# output path (processed images): user_faces_preprocessed/user_id/algo_name/challenge_name/face or no_face

for subdir in directories: 
    # subdir is the user_id 
    path = os.path.join(dataset_path, subdir)
    user_name = subdir
    print('user:', user_name)
    for root, dirs, files in os.walk(path):
        if '_faces\hand_occlusion_' in root: 
            challenge_name = 'hand_occlusion'
            hand_occ_path = save_path + '/hand_occlusion'
            if not os.path.exists(hand_occ_path):
                os.makedirs(hand_occ_path + '/face')
                os.makedirs(hand_occ_path + '/no_face')
            else:
                print("folder already exists")
                continue
            # print("root:", root)
            # root is the path to the challenge folder (hand_occlusion_1, hand_occlusion_2, hand_occlusion_3) for each algorithm
            if dirs and root.split('\\')[-1] != user_name: 
                print("algo:", root.split('\\')[-1])   
            elif files:
                # print("files:", files)
                name = root.split('\\')[-2:]
                folder_name = user_name + '\\' + name[0] + '\\' + name[1] # user_id\algo_name\challenge_name
                print('folder_name:', folder_name) 

                # print('root:', root)
                files = os.listdir(root)
                files_sorted = sort_paths(files)
                files_path = [os.path.join(root, file) for file in files_sorted]

                # start face and hand detection
                print("start face detection in folder:", folder_name)
                for file in files_path:
                    face_detection(file, hand_occ_path + folder_name)
            else: 
                continue
        elif  '_faces\obj_occlusion_' in root:
            challenge_name = 'obj_occlusion'
            obj_occ_path = save_path + '/obj_occlusion'
            if not os.path.exists(obj_occ_path):
                os.makedirs(obj_occ_path + '/face')
                os.makedirs(obj_occ_path + '/no_face')
            else:
                print("folder already exists")
                continue
            
            if dirs and root.split('\\')[-1] != user_name: 
                print("algo:", root.split('\\')[-1])

            elif files:
                name = root.split('\\')[-2:]
                folder_name = user_name + '\\' + name[0] + '\\' + name[1]
                print('folder_name:', folder_name)

                files = os.listdir(root)
                files_sorted = sort_paths(files)
                files_path = [os.path.join(root, file) for file in files_sorted]
                print("start face detection in folder:", folder_name)
                for file in files_path:
                    face_detection(file, obj_occ_path + folder_name)

        else:
            continue

print("done!")