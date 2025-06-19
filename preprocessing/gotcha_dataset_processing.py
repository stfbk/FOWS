"""
This code navigates the gotcha's fake images folder structure and process the images

The folder structure in gothca is as follows:
- algo_id/user_id/challenge_id/swapped_user_id
- each user is swapped with different targets and performs multiple challenges
- we only selected the hand_occlusion and obj_occlusion challenges -> challenge_id: 2 and 5

In order to preserve the gothca dataset structure, the code will copy the folder structure in the destination path.
However, it will add the occ / no_occ folders inside the destination path.
-> i.e. DFL/0/2/0 -> DFL/0/2/0/occ / no_occ 

inside the occ and no_occ folders, the code will move the processed images based on the face detection results

The face is detected using the mediapipe face detector with a confidence threshold of 0.9 and a non-max suppression threshold of 0.3
In gotcha the face is clearly visible and get occluded by either a hand or an object in the selcted challenges.
Therefore, if the face is not detected, it is likely that the face is occluded by a hand or an object.
The images are moved to the no_face folder in this case.

The values used for the face detector have been fine-tuned to best detect the face in the images.
"""


import os
import mediapipe as mp
import cv2
import shutil # to move the images

def sort_paths(path):
    """
    input: path to the images
    output: imgs paths sorted in numerical order based on the frame number
    """
    path = sorted(path, key = lambda x : int(x)) # sort by number
    return path

def copy_folder_structure(src_path, dst_path):
    """
    input: 
    - src_path: path to the gotcha dataset
    - dst_path: path to where to save the processed images

    output:
    - copy the folder structure to the destination path
    """
    separator, det, no_det = '', '', ''

    if 'hand_occlusion' in dst_path:
        # print("hand occlusion destination path")
        
        separator = os.sep + '2' + os.sep
        det = 'hand'
        no_det = 'no_hand'

    elif 'obj_occlusion' in dst_path:
        # print("obj occlusion destination path")
        separator = os.sep + '5' + os.sep
        det = 'face'
        no_det = 'no_face'

    else:
        return # break the loop if the destination path is not correct
    
    for dirpath, dirnames, filenames in os.walk(src_path):
        # dirpath = path to the directory
        # dirnames = list of directories in the directory
        # filenames = list of files in the directory
        rel_path = os.path.relpath(dirpath, src_path) # 0\2\0
        rel_path = rel_path.replace("\\", os.sep)  # replace backslashes with the correct path separator
       

        if separator in rel_path:
            structure = os.path.join(dst_path, rel_path) 
            print("copying structure", rel_path)

            if not os.path.isdir(structure):
                os.makedirs(structure) # to create multiple directories
                os.makedirs(structure + '/'+det)
                os.makedirs(structure + '/'+no_det)
                print(f"created folder structure in {dst_path}!")
            else:
                print("Folder does already exits!")
        else:
            continue


def gotcha_fake_detection(src_path, dst_path):
    """
        copy the gotcha folder structure to the destination path
        add the detection / no_detection folders inside the destination path

        input:
        - src_path: path to the gotcha dataset
        - dst_path: path to where to save the processed images

        output:
        - move the images to the corresponding folder based on the face detection results
    """
    separator = ''
    det = ''
    no_det = ''

    # set the separation and the detection / no_detection folders
    if 'hand_occlusion' in dst_path:
        # print("hand occlusion destination path")
        
        separator = os.sep + '2' + os.sep # /2/ in the path (os.sep = /)
        det = 'hand'
        no_det = 'no_hand'

    elif 'obj_occlusion' in dst_path:
        # print("obj occlusion destination path")
        separator = os.sep + '5' + os.sep
        det = 'face'
        no_det = 'no_face'

    else:
        return # break the loop if the destination path is not correct

    # start processing the images
    for dirpath, dirnames, filenames in os.walk(src_path):
        rel_path = os.path.relpath(dirpath, src_path) # 0\2\0
        rel_path = rel_path.replace("\\", os.sep)  # replace backslashes with the correct path separator

        # if os.sep + '2' + os.sep in rel_path:
        if separator in rel_path:
            print("rel_path", os.path.relpath(dirpath, src_path)) # 0\2\0
            structure = os.path.join(dst_path, rel_path) # ./test_gotcha_processing/fake/hand_occlusion/0/2/0
            print("structure", structure)
            # # copy the directory structure to the destination path
            if os.path.isdir(structure): 
                print("folder structure exists in dst_path!")

                # start processing the images
                for image in filenames:
                    # get the path to the image
                    file_path = os.path.join(dirpath, image)
                    # call the function to process the image
                    face_detection(file_path, structure, det, no_det)
            else:
                print("folder structure does not exist in dst_path!")
                break
 
        else:
            continue

def get_file_name(path):
    """
        get the file name without the extension from the file path
    """
    return path.split("\\")[-1].split(".")[0] # get the file name without the extension

def face_detection(img_path, save_path, det, no_det):
    """
        detect the face in the image and move the image to the corresponding folder
        if the face is detected, move the image to the face folder (no occlusion)
        if the face is not detected, move the image to the no_face folder (occlusion)

        input:
        - img_path: path to the image
        - save_path: path to where to save the processed images
        - det: folder name for the face detection
        - no_det: folder name for no face detection

        output:
        - move the image to the corresponding folder based on the face detection results
    """
    # det, no_det = 'face', 'no_face' (for face detection)
    img = cv2.imread(img_path)
    mp_image = mp.Image.create_from_file(img_path)
    face_results = face_detector.detect(mp_image)

    if face_results.detections:
        # move the image in the face folder
        print("face detected", get_file_name(img_path))
        path = save_path + '/' + det +'/'+ get_file_name(img_path) + '.jpg'
        shutil.move(img_path, path)
        print(f"moved {get_file_name(img_path)} to {det} folder!")
        
    else:
        print("no face detected", get_file_name(img_path))
        path = save_path + '/' + no_det +'/' + get_file_name(img_path) + '.jpg'
        shutil.move(img_path, path)
        print(f"moved {get_file_name(img_path)} to {no_det} folder!")


# ------------------------------------------------------------------------------ #
# Example of code usage for the DFL subset of the gotcha dataset #
# How to use the code:
# 1. define the path to the gotcha dataset
# 2. define the path where to save the processed images
# 3. copy the folder structure to the destination path
# 4. start processing the images
# repeat the steps for the hand and object occlusion
# ------------------------------------------------------------------------------ #
# mediapipe face detector initialization #
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create an FaceDetector object.
face_base_options = BaseOptions(model_asset_path='./face_landmark_model/blaze_face_short_range.tflite')
face_options = FaceDetectorOptions(
    base_options=face_base_options,
    running_mode=VisionRunningMode.IMAGE,
    min_detection_confidence=0.9, # min confidence threshold to detect a face 
    min_suppression_threshold=0.3 # min non-max suppression threshold to detect multiple faces in an image 
    )
# with FaceDetector.create_from_options(options) as detector:
face_detector = FaceDetector.create_from_options(face_options)

# path to the dataset
data_path =  './_gotcha_dataset_/DFL'
# path to where to save the processed images
hand_save_path = './_gotcha_dataset_/test/hand_occlusion' 
face_save_path = './_gotcha_dataset_/test/obj_occlusion'

# ----------------------------------------------------- #
# hand occlusion #
print("--------- Hand occlusion ---------")
print("--------- copying the folder structure ---------")
copy_folder_structure(data_path, hand_save_path)
print("done!")

print("--------- start processing the images ---------")
print("hand_save_path", hand_save_path)
gotcha_fake_detection(data_path, hand_save_path)
print("done!")
# ----------------------------------------------------- #
# obj occlusion #
print("--------- Object occlusion ---------")
print("--------- copying the folder structure ---------")
copy_folder_structure(data_path, face_save_path)
print("done!")

print("--------- start processing the images ---------")
print("face_save_path", face_save_path)
gotcha_fake_detection(data_path, face_save_path)
print("done!")
