## add script for code used in the demo 
## slight variation wrt the one used for trn/tst models

import os
import sys
import torch
import torch.nn as nn
from torchvision import models
import timm  # for xception model 

# ----------------------------------------------------------------------------------------------- #
# TODO: update code to load models (use args)
# ----------------------------------------------------------------------------------------------- #

# ------------------------------------------------------- #
# functions to load the baseline models #
# ------------------------------------------------------- #
def get_model_path(args, model_weights_path):

    if args.model not in ['mnetv2', 'effnetb4', 'xception']: 
        print(f"Model {args.model} is not supported")
        exit()
    
    if args.train_dataset not in ['fows_occ', 'fows_no_occ', 'gotcha_occ', 'gotcha_no_occ']:
        print(f"Dataset {args.train_dataset} is not valid")
        exit()

    
    # model_str = args.tags if args.tags else args.model + '_' + args.train_dataset + '_' + ('TL' if args.tl else 'FT')
    model_str = args.model + '_' + args.train_dataset + '_' + ('TL' if args.tl else 'FT')
    print("Model tags: ", args.tags)
    print("Model string: ", model_str)


    model_path = ''
    found = False
    for root, dirs, files in os.walk(model_weights_path):
      for file in files:
        if file.endswith('.pth') and model_str.lower() in file.lower():
          found = True
          print(os.path.join(root, file))
          model_path = os.path.join(root, file)
          break
 
    if not found:
        print(f"NO model {model_str} found in {model_weights_path}")
        exit()

    print(model_path)

    return model_path

def load_baseline_weight(args, model_weights_path):
    print("Loading baseline model weights ...")
    pretrained_model_path = get_model_path(args, model_weights_path)

    if pretrained_model_path:
        print("model path: ", pretrained_model_path)
        print("loading the model...")
    else:
        print("no pretrained model found")
        exit()


    # if 'mnetv2' in model_name.lower():
    if args.model == 'mnetv2':
        print("Loading MobileNetV2 model")
        # model = mobilenet_v2(pretrained=True)
        model = models.mobilenet_v2(weights = 'MobileNet_V2_Weights.IMAGENET1K_V2')
        model_name = 'MobileNetV2' # add the model name to the model object
        # Replace the classifier layer
        model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face
        
        best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
        # breakpoint()
        if 'model' in best_ckpt.keys():
            model.load_state_dict(best_ckpt['model'])
        else:
            model.load_state_dict(best_ckpt)
        
        # model.to(device)

    # elif 'effnetb4' in model_name.lower():
    elif args.model == 'effnetb4':
        print("Loading EfficientNetB4 model")
        model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1')
        model_name = 'EfficientNet_B4' # add the model name to the model object
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real of swap face

        best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
        # breakpoint()
        if 'model' in best_ckpt.keys():
            model.load_state_dict(best_ckpt['model'])
        else:
            model.load_state_dict(best_ckpt)
        


    elif args.model == 'xception':
        print("Loading pretrained XceptionNet model...")
        # load the xceptionet model
        # pip install timm
        # import timm
        model = timm.create_model('xception', pretrained=True, num_classes=1) # only 1 output -> prob of real of swap face
        model_name = 'XceptionNet' # add the model name to the model object
        # load the pre-trained model for testing (load model weights)
        best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
        if 'model' in best_ckpt.keys():
            model.load_state_dict(best_ckpt['model'])
        else:
            model.load_state_dict(torch.load(pretrained_model_path))


    else:
        print("Model not supported")
        sys.exit()

    return model, model_name    


# ------------------------------------------------------- #
# functions to load the trained model from the saved path #
# ------------------------------------------------------- #
def get_pretrained_path(args, model_path):
    # Check if the model is valid
    if args.model not in ['mnetv2', 'effnetb4', 'xception']:
        print(f"Model {args.model} is not supported")
        exit()
    # Check if the dataset is valid
    if args.train_dataset not in ['fows_occ', 'fows_no_occ', 'gotcha_occ', 'gotcha_no_occ']:
        print(f"Dataset {args.train_dataset} is not valid")
        exit()

    # Get the model path based on the args.tags
    model_str = args.model + '_' + args.train_dataset + '_' + ('TL' if args.tl else 'FT')
    # args.tags if args.tags else 
    print("Model tags: ", args.tags)
    print("Model string: ", model_str)

    if args.tl:
        models_folder = model_path+'/TL/' 
    else:
        models_folder = model_path+'/FT/' 
    pretrained_model_path = ''
    model_dir = models_folder

    # Check if the model path exists
    if not os.path.exists(models_folder):
        print(f"Model path {models_folder} does not exist")
        exit()

    # navigate all model folders subdirectories and check if the model_info is in any of them
    found = False
    for root, dirs, files in os.walk(models_folder):
        for dir in dirs:
            # print(f"Checking directory: {dir}")
            if model_str.lower() in dir.lower():  # check if the directory name is the same as the model_info
                # find the .pth file in the directory
                print(f"Found directory: {dir} with same name as {model_str}")
                model_dir = os.path.join(root, dir)
                print("model_dir:", model_dir)
                breakpoint()
                for sub_root, sub_dirs, sub_files in os.walk(model_dir):
                    for file in sub_files:
                        if file.endswith('best_checkpoint.pth'): # and args.tags.lower() in model_dir.lower():
                            found = True
                            pretrained_model_path = os.path.join(sub_root, file)
                            print(f"Pretrained model path: {pretrained_model_path}")
                            break
                break # uncomment this if you want to stop searching aFTer finding the first directory

    if not found:
        print(f"NO model {args.model} found in {model_dir}")
        exit()

    print(f"Pretrained model path: {pretrained_model_path}")

    return pretrained_model_path

def load_model_from_path(args, model_weights_path):
    print("Loading trained model ...")
    pretrained_model_path = get_pretrained_path(args, model_weights_path)
    # pretrained_model_path = get_pretrained_path(model_name, trn_strategy, dataset, model_weights_path)
    if pretrained_model_path:
        print("model path: ", pretrained_model_path)
        print("loading the model...")
    else:
        print("no pretrained model found")
        exit()


    # if 'mnetv2' in model_name.lower():
    if args.model == 'mnetv2':
        print("Loading MobileNetV2 model")
        # model = mobilenet_v2(pretrained=True)
        model = models.mobilenet_v2(weights = 'MobileNet_V2_Weights.IMAGENET1K_V2')
        model_name = 'MobileNetV2' # add the model name to the model object
        # Replace the classifier layer
        model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face
 
        best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
        # breakpoint()
        if 'model' in best_ckpt.keys():
            model.load_state_dict(best_ckpt['model'])
        else:
            model.load_state_dict(best_ckpt)
        
        # model.to(device)

    # elif 'effnetb4' in model_name.lower():
    elif args.model == 'effnetb4':
        print("Loading EfficientNetB4 model")
        model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1')
        model_name = 'EfficientNet_B4' # add the model name to the model object
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real of swap face

        best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
        # breakpoint()
        if 'model' in best_ckpt.keys():
            model.load_state_dict(best_ckpt['model'])
        else:
            model.load_state_dict(best_ckpt)
        


    elif args.model == 'xception':
        print("Loading pretrained XceptionNet model...")
        # load the xceptionet model
        model = timm.create_model('xception', pretrained=True, num_classes=1) # only 1 output -> prob of real of swap face
        model_name = 'XceptionNet' # add the model name to the model object
        # load the pre-trained model for testing (load model weights)
        best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
        if 'model' in best_ckpt.keys():
            model.load_state_dict(best_ckpt['model'])
        else:
            model.load_state_dict(torch.load(pretrained_model_path))


    else:
        print("Model not supported")
        sys.exit()

    return model, model_name

# ------------------------------------------------------------- #

def get_backbone(args):
    if args.model == 'mnetv2': 
        print("Loading MobileNetV2 model")

        if args.tl:
            print("Transfer learning - Freezing all layers except the classifier")
            model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V2')
            print("Transfer learning - Freezing all layers except the classifier")
            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False
            
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face
            model_name = 'MobileNetV2_TL' # add the model name to the model object
        
        elif args.ft:
            print("Fine-Tuning")
            model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V2')
            print("Transfer learning - Freezing all layers except the classifier")
            
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face
            model_name = 'MobileNetV2_FT' # add the model name to the model object

        else:
            print("Loading pre-trained ImageNet model")
            model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V2')
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face
            model_name = 'MobileNetV2' # add the model name to the model object

    elif args.model == 'effnetb4':
        print("Loading EfficientNetB4 model")
        if args.tl: # and args.resume != '':
            print("Transfer learning - Freezing all layers except the classifier")
            model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1') # correct way to call pre-trained model
            # https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b4.html
            print("Transfer learning - Freezing all layers except the classifier")
            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False
            
            # Replace the classifier layer
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real or swap face
            # last layer of EfficientNetB4 is a Linear layer (classifier) with 1000 outputs (for ImageNet) -> change it to 1 output
            model_name = 'EfficientNet_B4' # add the model name to the model object
            
        elif args.ft: 
            print("Fine-Tuning")
            model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1') # correct way to call pre-trained model
            # Replace the classifier layer
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real or swap face
            model_name = 'EfficientNet_B4_FT'
            
        else: 
            print("Loading EfficientNetB4 pre-trained ImageNet model")
            model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1') # correct way to call pre-trained model
            # Replace the classifier layer
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
            # model.to(device)
            model_name = 'EfficientNet_B4' # add the model name to the model object
       
        print("Model loaded!")

    elif args.model == 'xception':
        print("Loading XceptionNet model")
        if args.tl:
            print("Transfer learning - Freezing all layers except the classifier")
            model = timm.create_model('xception', pretrained=True) #, num_classes=1) # only 1 output -> prob of real or swap face
            model_name = 'XceptionNet' # add the model name to the model object
            
            # Freeze all layers except the classifier
            for param in model.parameters():
                param.requires_grad = False

            # Replace the classifier layer
            in_features = model.get_classifier().in_features
            model.fc = nn.Linear(in_features, 1)  # modify the last layer of the classifier to have 1 output -> prob of real or swap face

            # Ensure the classifier layer is trainable
            for param in model.fc.parameters():
                param.requires_grad = True
           

        elif args.ft: 
            print("Fine Tuning")
            model = timm.create_model('xception', pretrained=True) #, num_classes=1) # only 1 output -> prob of real or swap face
            model_name = 'XceptionNet' # add the model name to the model object
            # Replace the classifier layer
            in_features = model.get_classifier().in_features
            model.fc = nn.Linear(in_features, 1)  # modify the last layer of the classifier to have 1 output -> prob of real or swap face

        else: 
            print("Loading XceptionNet pre-trained ImageNet model")
            model = timm.create_model('xception', pretrained=True) #, num_classes=1) # only 1 output -> prob of real or swap face
            model_name = 'XceptionNet' # add the model name to the model object
            # print("Transfer learning - Freezing all layers except the classifier")
            # # Freeze all layers except the classifier
            # for param in model.parameters():
            #     param.requires_grad = False

            # Replace the classifier layer
            in_features = model.get_classifier().in_features
            model.fc = nn.Linear(in_features, 1)  # modify the last layer of the classifier to have 1 output -> prob of real or swap face

            # Ensure the classifier layer is trainable
            for param in model.fc.parameters():
                param.requires_grad = True
           
            # check the model gradient
            # check_model_gradient(model)
        # else:
        #     # retraining the model with one output class 
        #     model = timm.create_model('xception', pretrained=True, num_classes=1) # only 1 output -> prob of real or swap face
        #     model_name = 'XceptionNet' # add the model name to the model object

        

        # model.to(device)
        # print("Model loaded!")

        print(model)
        # exit()
        # check the model gradient
        # check_model_gradient(model)


    else:
        print("Model not supported")
        exit()

    return model, model_name
