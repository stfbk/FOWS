import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, SequentialSampler
import os
from utilscripts.customDataset import FaceImagesDataset
from utilscripts.train_val_test import *
import argparse # for command line arguments
from utilscripts.logger import * # import the logger functions
import timm # for using the XceptionNet model (pretrained)
import random
from utilscripts.get_trn_tst_model import *

# -------------------------------------------------------------------------------- #
# test the models 
# Example command to run the script:
# python test.py --model mnetv2 --train_dataset fows_occ --test_dataset fows_no_occ --tl --tags MnetV2_fows_occ_TL_vs_fows_no_occ
# --model = model name
# --tags = info about the test, e.g. MnetV2_fows_occ_TL_vs_fows_no_occ 
#       -> will also be the name of the folder where the results will be saved
# ------------------------------------------------------------------------------------------- #
# Create the argument parser
def get_args_parse():
    parser = argparse.ArgumentParser(description='Model Training and Testing')
    # Add arguments

    # model parameters
    parser.add_argument('--model', type=str, default='mnetv2', help='Define which trained model to load, e.g. mnetv2_fows_occ')
    parser.add_argument('--train_dataset', type=str, default='fows_occ', help='Name of the dataset used to train the model')
    parser.add_argument('--test_dataset', type=str, default='fows_occ', help='Test dataset')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--thresh', type = float, default = 0.5, help = 'Threshold for binary classification.')
    parser.add_argument('--save-log', action='store_false', help='Save the model output logs')
    parser.add_argument('--tl', action = 'store_true', help='Use the re-trained version of the model (transf learning)')
    parser.add_argument('--ft', action = 'store_true', help='Fine-Tuning the model')
    parser.add_argument('--data-aug', type=str, default='fows', help='Data augmentation to use for training and testing') 
    # parser.add_argument('--model_path', type = str, default = './model_weights/', help = 'Path to the trained models folder')
    parser.add_argument('--load_baseline', action = 'store_true', help='Load a baseline model' )
    parser.add_argument('--tags', type=str, default='', help='Name of the folder from which to load the trained model. The folder name contains info about the model, training setting, dataset, test setting (FT/TL), etc.')
    
    return parser


def load_partial_state_dict(model, state_dict):
    model_dict = model.state_dict()
    # Filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    # Overwrite entries in the existing state dict
    model_dict.update(state_dict)
    # Load the new state dict
    model.load_state_dict(model_dict)

def init_seed():
    # --------------------------------------------- #
    # init_seed def taken from DFB code
    # Set the random seed for reproducibility
    random.seed(42)
    # np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        # torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        print("fixed random seed!")
    # --------------------------------------------- #

def check_model_gradient(model):
    # Check if the gradient is active for the FC layer
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, requires_grad: {param.requires_grad}")
        # if 'fc' in name:
        #     print(f"Layer: {name}, requires_grad: {param.requires_grad}")
        # else:
        #     print(f"Layer: {name}, requires_grad: {param.requires_grad}")

# ----------------------------------------- #
# main function to train and test the model #
# ----------------------------------------- #
def main(): 
    # Parse the arguments
    parser = get_args_parse() # get the arguments from the command line 
    args, unknown = parser.parse_known_args() # parse the known arguments and ignore the unknown ones
    # initialize the random seed
    init_seed()

    # if args.scratch and args.model == 'mobilenetv2':
    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # ----------------------- #
    # --- BASELINE MODELS --- #
    # ----------------------- #

    print("args.tl: ", args.tl)
    print("args.ft: ", args.ft)
    print("args.load_baseline: ", args.load_baseline)
    # print("args.gen: ", args.gen)

    print("thresh: ", args.thresh)
    # 

    print("args.model: ", args.model)
    print("args.train_dataset: ", args.train_dataset)
    print("args.test_dataset: ", args.test_dataset)
    print("args.data_aug: ", args.data_aug)

    if not args.tags:
        args.tags = args.model + '_' + args.train_dataset + '_' + ('TL' if args.tl else 'FT') + '_vs_' + args.test_dataset
    print("args.tags: ", args.tags)


    # loading model
    if args.load_baseline:
        print("Loading baseline model weights...")
        # load baseline_weights
        model_path = './model_weights'
        if not os.path.exists(model_path):
            print(f"Model path {model_path} does not exist")
            exit()
        
        model, model_name = load_baseline_weight(args, model_path)

    else:
        print("Loading model weights from the results folder...")
        model_path = './results' # where trained models are saved
        
        if not os.path.exists(model_path):
            print(f"Model path {model_path} does not exist")
            exit()

        model, model_name = load_model_from_path(args, model_path)

    
    model.to(device)
    print("Model loaded!")
    # print("model_path: ", pretrained_model_path)
    print(f"model_name: {model_name}")
    print(model)

    
    
    if args.test_dataset == 'fows_occ':
        test_dir = './dataset/fows_occlusion/testing/'
        
    elif args.test_dataset == 'fows_no_occ':
        test_dir = './dataset/fows_no_occlusion/testing/'

    else:
        print("Dataset not supported")
        exit()

    print("dataset:", args.test_dataset)
    print("test_dir:", test_dir)
    
    # --------------------------------- #
    # Define the dataset and dataloaders
    # --------------------------------- #
    if args.data_aug == 'fows':
        # Transformations for testing data
        test_transform = transforms.Compose([
            transforms.Resize((256,256)), 
            # BICUBIC is used for EfficientNetB4 -> check the documentation
            # BICUBIC vs BILINEAR -> https://discuss.pytorch.org/t/what-is-the-difference-between-bilinear-and-bicubic-interpolation/20920 
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # test dataloader 
        # test_dataset = FaceImagesDataset(test_dir, test_transform)
        test_dataset = FaceImagesDataset(dataset_dir=test_dir, transform=test_transform, training = False) #, challenge=args.trn_challenge, algo=args.trn_algo)
        print("test_dataset: ", len(test_dataset))
        # get the test_dataset labels
        dataset_labes = test_dataset.label_map
        print(dataset_labes)
        sampler_test = SequentialSampler(test_dataset) # load the dataset in the order it is in the folder
        test_dataloader = DataLoader(test_dataset, 
                                    batch_size= args.batch_size, #64, 
                                    sampler=sampler_test,
                                    num_workers = 3) # shuffle=False)
    else:
        print("data augmentation not supported")
        exit()


    print("data augmentations: ", args.data_aug)
    # print(test_transform)

    # --------------------------------------------------------- #
    # ------------------- Testing the model ------------------- #
    # --------------------------------------------------------- #

    # print(f'Testing the pretrained model {model_name} on the {args.dataset} dataset...')
    print(f'Testing the model {args.model}_{args.train_dataset} on the {args.test_dataset} dataset...')

    # creating a folder where to save the testing results 
    if args.tl:
            exp_results_path = f'./results/results_TL/{args.tags}/testing' # i.e. ./results/EfficientNetB4_FF_no_occ_focal_loss/testing
    elif args.ft:
            exp_results_path = f'./results/results_FT/{args.tags}/testing'
    else:
        exp_results_path = f'./results/{args.tags}/testing'

    # ----------------------------------- #
    # Evaluate the model on the test data #
    # ----------------------------------- #
    # if not gotcha:
    # print(f"Testing the {args.model} on the {args.dataset} dataset")

    test_accuracy, balanced_test_acc, test_accuracy_original, test_accuracy_simswap, test_accuracy_ghost, test_accuracy_facedancer, TPR, TNR, auc_score, ap_score, eer, auc_best_threshold, eer_threshold = test_one_epoch(
        model, test_dataloader, device, model_name, exp_results_path, args.tags, args.thresh
    )

    # print(f"Test results of the pretrained model {model_name} on the {args.dataset} dataset with thresh {args.thresh}")
    print(f"Evaluation results of the {args.model}_{args.train_dataset} model on the {args.test_ataset} dataset with thresh {args.thresh}")
    print("Dataset labels: ", dataset_labes)
    print(f"Data augmentation: {args.data_aug}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Balanced Test Accuracy: {balanced_test_acc:.4f}")
    print(f"Test Accuracy Original: {test_accuracy_original:.4f}")
    print(f"Test Accuracy SimSwap: {test_accuracy_simswap:.4f}")
    print(f"Test Accuracy Ghost: {test_accuracy_ghost:.4f}")
    print(f"Test Accuracy FaceDancer: {test_accuracy_facedancer:.4f}")
    print(f"Average Precision: {ap_score:.4f}")
    print(f"TPR (sensitivity): {TPR:.4f}")
    print(f"TNR (specificity): {TNR:.4f}")
    print(f"AUC: {auc_score:.4f}")
    print(f"AUC thresh: {auc_best_threshold:.4f}")
    print(f"EER: {eer:.4f}")
    print(f"EER thresh: {eer_threshold:.4f}")

    # log the results to a log file
    if args.save_log:
        print(f"logging results - {args.tags}")
        # set up logger
        log_path = log_path = exp_results_path+ f'/exp_logs/testing/output.log'
        print("log_path: ", log_path)
        log = create_logger(log_path)
        log.info(f"Evaluation results of the {args.model}_{args.train_dataset} model on the {args.test_dataset} dataset with thresh {args.thresh}")
        log.info(f"Dataset test dir: {test_dir}")
        log.info(f"Model weights: {model_path}")
        log.info(f"FT: {args.ft}, TL: {args.tl}, GEN: {args.gen}")
        log.info(f"Dataset labels: {dataset_labes}")
        log.info(f"Data augmentation: {args.data_aug}")
        log.info(f"Test Accuracy: {test_accuracy:.4f}")
        log.info(f"Balanced Test Accuracy: {balanced_test_acc:.4f}")
        log.info(f"Test Accuracy Original: {test_accuracy_original:.4f}")
        log.info(f"Test Accuracy SimSwap: {test_accuracy_simswap:.4f}")
        log.info(f"Test Accuracy Ghost: {test_accuracy_ghost:.4f}")
        log.info(f"Test Accuracy FaceDancer: {test_accuracy_facedancer:.4f}")
        log.info(f"Average Precision: {ap_score:.4f}")
        log.info(f"TPR (recall): {TPR:.4f}")
        log.info(f"TNR (specificity): {TNR:.4f}")
        log.info(f"AUC: {auc_score:.4f}")
        log.info(f"AUC thresh: {auc_best_threshold:.4f}")
        log.info(f"EER: {eer:.4f}")
        log.info(f"EER thresh: {eer_threshold:.4f}")
        log.info("Note the labels/preds/imgs_paths are in the json file")

    # ------------------------------------------------------------------------------------- #
    # else:
    # print("Testing GOTCHA dataset!")

    #     # test_accuracy, balanced_test_accuracy, test_accuracy_original, test_accuracy_dfl, test_accuracy_fsgan, TPR, TNR, auc_score, ap_score, eer, labels_collection, prediction_collection, img_path_collection = gotcha_test(model, test_dataloader, device, model_name, exp_results_path, args.tags, args.thresh) 
        
    #     # test_accuracy, balanced_test_acc, test_accuracy_original, test_accuracy_dfl, test_accuracy_fsgan, \
    #     # TPR, TNR, auc_score, ap_score, eer, labels_list, img_path_collection, prob_original, \
    #     # prob_dfl, prob_fsgan, auc_best_threshold, eer_threshold 
    #     #  img_path_collection, labels_collection, prediction_collection,
    #     test_accuracy, balanced_test_acc, test_accuracy_original, test_accuracy_dfl, test_accuracy_fsgan, TPR, TNR, auc_score, ap_score, eer, auc_best_threshold, eer_threshold = gotcha_test(model, test_dataloader, device, model_name, exp_results_path, args.tags, args.thresh)
        
    #     # print(f"Test results of the pretrained model {model_name} on the {args.dataset} dataset with thresh {args.thresh}")
    #     print(f"Test results of the pretrained model {args.model} on the {args.dataset} dataset with thresh {args.thresh}")
    #     print("Dataset labels: ", dataset_labes)
    #     print(f"Dataset test dir: {test_dir}")
    #     print(f"Test Accuracy: {test_accuracy:.4f}")
    #     print(f"Balanced Test Accuracy: {balanced_test_acc:.4f}")
    #     print(f"Test Accuracy Original: {test_accuracy_original:.4f}")
    #     print(f"Test Accuracy DFL: {test_accuracy_dfl:.4f}")
    #     print(f"Test Accuracy FSGAN: {test_accuracy_fsgan:.4f}")
    #     # print(f"Test Accuracy FaceDancer: {test_accuracy_facedancer:.4f}")
        
    #     # print(f"AUC Score: {auc:.4f}") 
    #     print(f"Average Precision: {ap_score:.4f}")
    #     print(f"TPR (sensitivity): {TPR:.4f}")
    #     print(f"TNR (specificity): {TNR:.4f}")
    #     print(f"AUC: {auc_score:.4f}")
    #     print(f"AUC thresh: {auc_best_threshold:.4f}")
    #     print(f"EER: {eer:.4f}")
    #     print(f"EER thresh: {eer_threshold:.4f}")

    #     # # log results to wandb
    #     # if args.wandb:
    #     #     run.tags = run.tags + ('Testing',) # add testing tag
            
    #     #     run.log({'Test Accuracy': test_accuracy})
    #     #     run.log({'Balanced Test Accuracy': balanced_test_acc})
    #     #     run.log({'Test Accuracy Original': test_accuracy_original}) 
    #     #     run.log({'Test Accuracy DFL': test_accuracy_dfl}) 
    #     #     run.log({'Test Accuracy FSGAN': test_accuracy_fsgan})
            
    #     #     run.log({'AP': ap_score})
    #     #     run.log({'TPR': TPR})
    #     #     run.log({'TNR': TNR})
    #     #     run.log({'AUC': auc_score})
    #     #     run.log({'EER': eer})
            

    #     # log the results to a log file
    #     if args.save_log:
    #         print(f"logging results - {args.tags}")
    #         # set up logger
    #         log_path = log_path = exp_results_path+ f'/exp_logs/testing/output.log'
    #         log = create_logger(log_path)
    #         # log.info("")
    #         # log.info(f"Test results of the pretrained model {model_name} on the {args.dataset} dataset with thresh {args.thresh}")
    #         log.info(f"Test results of the pretrained model {args.model} on the {args.dataset} dataset with thresh {args.thresh}")
    #         log.info(f"Model weights: {pretrained_model_path}")
    #         log.info(f"Dataset labels: {dataset_labes}")
    #         log.info(f"Dataset test dir: {test_dir}")
    #         log.info(f"Data augmentation: {args.data_aug}")
    #         log.info(f"Test Accuracy: {test_accuracy:.4f}")
    #         log.info(f"Balanced Test Accuracy: {balanced_test_acc:.2f}")
    #         log.info(f"Test Accuracy Original: {test_accuracy_original:.4f}")
    #         log.info(f"Test Accuracy DFL: {test_accuracy_dfl:.4f}")
    #         log.info(f"Test Accuracy FSGAN: {test_accuracy_fsgan:.4f}")
            
    #         # log.info(f"AUC: {auc:.4f}")
    #         log.info(f"Average Precision: {ap_score:.4f}")
    #         log.info(f"TPR (recall): {TPR:.4f}")
    #         log.info(f"TNR (specificity): {TNR:.4f}")
    #         log.info(f"AUC: {auc_score:.4f}")
    #         log.info(f"EER: {eer:.4f}")
    #         log.info("Note the labels/preds/imgs_paths are in the json file")
    #         log.info("")
    # ------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    main()
