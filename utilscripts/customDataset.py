import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2



# Define a custom dataset class
class FaceImagesDataset(Dataset):
    def __init__(self, dataset_dir, transform, training = True):
        """
            directory: str, path to the directory containing the images
            transform: data augmentation to be applied to the images
            milan_aug: bool, flag to apply the Milan dataset data augmentations (albumentation)
            challenge: str, the challenge name to load the data from
            algo: str, the algorithm name to load the data from

            This class is used to load the images from the directory and apply the data augmentation to the images.
            Applies the labels to the images based on the directory structure.
            If the images are in the 'original' folder, the label is 'original' and if the images are in the other folders, the label is 'swap'.

            @NOTE:
                - Add the option to load the data from the challenge (i.e. only hand_occlusion_1 images for all algorithms)
                    - test generalization capabilities of different challenges
                        - hand_occlusion_1, hand_occlusion_2, hand_occlusion_3 (hand occlusion)
                        - obj_occlusion_1, obj_occlusion_2, obj_occlusion_3 (object occlusion)
                        -> binary classification (original vs fake)
                            -> train on hand_occlusion and test on obj_occlusion (and vice versa) for all algorithms
                - Add the option to load the data from the algorithm (i.e. all images challenges from GHOST and original)
                    - test generalization capabilities of different algorithms
                        - GHOST, SimSwap, FaceDancer vs original (binary classification)
                - IMPORTANT: for now only the challenge or the algo can be defined, not both! If both are defined an error is raised.
        """

        self.directory = dataset_dir
        self.image_files = []
        self.labels = [] # 'original', 'swap'
        self.transform = transform  
        self.training = training # if we are training or testing the model

        # # print("training: ", self.training)
        self.process_dataset()
        
        # label mapping 
        self.label_map = {
                'original': 0, # pos_class -> the one we want to detect
                'swap': 1 # neg_class
            }
        
        # count the number of images labeled 'original' and 'swap' in the dataset
        original_count = self.labels.count('original')
        swap_count = self.labels.count('swap')
        print(f"Original images: {original_count}")
        print(f"Swap images: {swap_count}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx] 
        label = self.labels[idx]
        label = self.label_map[label]
        label_to_tensor = torch.tensor(label, dtype=torch.long) # convert the label to a tensor, torch.long is the data type for the label (integer type)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # if self.transform:
        #     image = self.transform(image)
        # # else:
            # image = transforms.ToTensor()(image) # convert the image to a tensor
        # image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)  # Convert to PyTorch tensor
        return image, label_to_tensor, img_path


    def process_dataset(self):
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    if 'original' in full_path:
                        self.image_files.append(full_path)
                        self.labels.append('original')
                    else:
                        self.image_files.append(full_path)
                        self.labels.append('swap')
                else: continue