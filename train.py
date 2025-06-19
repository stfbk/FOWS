import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, random_split
import os
from utilscripts.customDataset import FaceImagesDataset
from utilscripts.train_val_test import *
import argparse # for command line arguments
from utilscripts.logger import *
from focalLoss import FocalLoss
import timm # for using the XceptionNet model (pretrained)
from utilscripts.get_trn_tst_model import get_backbone
import random
import datetime


# -------------------------------------------------------------------------------------------------------------------------------------------------------- #
# training the model
# example command to train the model
# python train.py --model "mnetv2" --train_dataset fows_occ --ft --tags mnetv2_fows_occ_FT
# model -> defines the model backbone used for training
# train_dataset is the dataset used for training (fows_occ, fows_no_occ)
# ft -> fine-tuning the model
# tl -> transfer learning the model
# tags -> defines the name of the folder where to save the model and the logs
# -------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Create the argument parser
def get_args_parse():
    parser = argparse.ArgumentParser(description='Model Training and Testing')
    # model parameters
    parser.add_argument('--model', type=str, default='mnetv2', help='Model to use for training and testing')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate') # default for adam and adamw
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use for training')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay for the optimizer') # default for adamw
    parser.add_argument('--patience', type=int, default=3, help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--loss', type=str, default='focal', help='Loss function to use for training')
    parser.add_argument('--thresh', type = float, default = 0.5, help = 'Threshold for binary classification.')
    parser.add_argument('--save-log', action='store_false', help='Save the model output logs')
    parser.add_argument('--save-model', action = 'store_false', help='Save the model in a folder with the same name as the training tags.')
    parser.add_argument('--data-aug', type=str, default='fows', help='Data augmentation to use for training and testing') 
    parser.add_argument('--tl', action='store_true', help='Use transfer learning for the model') # use transfer learning for the model
    parser.add_argument('--ft', action='store_true', help='Fine-Tuning the model') # use transfer learning for the model
    parser.add_argument('--train_dataset', type=str, default='fows_occ', help='Name of the training dataset')
    # parser.add_argument('--wandb', action='store_true', help='Use wandb for logging') # if not specified in the command as --wandb the value is set to False
    parser.add_argument('--tags', type=str, default='', help='Name of the folder where to save the model. The folder name contains info about the model, training setting, testing_dataset, test setting (FT/TL), etc.')

    return parser


def check_model_name(model_path):
    i = 1
    while os.path.exists(model_path): # check if the new graph name already exists -> if it does, increment i 
        # while stops when the new graph name does not exist 
        # new_graph_name = f'{graph_name}_{i}.png'
        i += 1
        model_name = model_path.split('/')[-1].split('.')[0] # get the model name from the model path (remove .pth)
        model_path = model_path.replace(model_name, f'{model_name}_{i}') # replace the model name with the new model name
    return model_path



def init_seed():
    # --------------------------------------------- #
    # taken from: https://github.com/SCLBD/DeepfakeBench/blob/main/training/train.py
    # Set the random seed for reproducibility
    random.seed(42)
    # np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
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

def main(): 
    # ----------------------------------------- #
    # main function to train and test the model #
    # ----------------------------------------- #
    # Parse the arguments
    parser = get_args_parse() # get the arguments from the command line 
    args, unknown = parser.parse_known_args() # parse the known arguments and ignore the unknown ones

    init_seed() # init random seed for reproducibility

    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print("thresh: ", args.thresh)

    if not args.tags:
        args.tags = args.model + '_' + args.train_dataset + '_' + ('TL' if args.tl else 'FT')
    print("tags: ", args.tags)

    # ---------------------- #
    # --- LOAD THE MODEL --- #
    # ---------------------- #
    model, model_name = get_backbone(args) # load the model backbone to train (FT/TL) 
    model.to(device)
    print("Model Loaded!")
    print(model)
    check_model_gradient(model)
    # breakpoint()

    if args.train_dataset == 'fows_occ':
        train_dir = './dataset/fows_occ/training/'
        # test_dir = '/media/data/rz_dataset/users_face_occlusion/testing/'
        # args.data_aug = ''
        

    elif args.train_dataset == 'fow_no_occ':
        train_dir = './dataset/fows_no_occlusion/training/'
        # test_dir = '/media/data/rz_dataset/user_faces_no_occlusion/testing/'

    else:
        print("Dataset not supported")
        exit()

    print("dataset:", args.train_dataset)
    print("train_dir", train_dir)
    # print("train_dir:", train_dir)
    # print("test_dir:", test_dir)



    num_epochs = args.num_epochs
    print("num_epochs: ", args.num_epochs)
    # lr = args.lr # 0.001 = 1e-3 (default value)
    # wheight_decay = args.weight_decay # 1e-5 (default value)
    patience = args.patience # 3 (number of epochs to wait for improvement before stopping)
    best_val_loss = float('inf')  # Initialize best validation loss
    early_stopping_counter = 0  # Counter to keep track of non-improving epochs

    # ---------------------------------- #
    # ------ Define the optimizer ------ #
    # ---------------------------------- #

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr= args.lr) #, weight_decay= args.weight_decay) # add weight decay

    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr= args.lr, weight_decay= args.weight_decay) 
        # default values:
        # lr = 1e-3 ()
        # weight_decay = 1e-2 (try with differnet values, 1e-5, 1e-4, 1e-3)

    else: 
        print("Optimizer not supported")
        exit()
        
    print("optimizer: ", args.optimizer)
    # breakpoint()
    # ---------------------------------- #
    # ---- Define the loss function ---- # 
    # ---------------------------------- #
    # Focal loss
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2) 
        # def values from pytorch documentation (https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html)
    elif args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss() # used by milan models
    elif args.loss == 'bce_loss':
        criterion = nn.BCELoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        print("Loss function not supported")
        exit()

    print("Loss Function: ", criterion)
    # ()

    if args.save_model and args.tl:
        # model_path = './model/' + args.tags +'_train' + '/' + 'best_model.pth'
        save_model_folder = f'./results/TL/{args.tags}/training/' + model_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # save_model_name = 'best_model.pth'
        # os.makedirs(save_model_folder, exist_ok=True)
        # model_path = check_model_name(save_model_folder, save_model_name)
    elif args.save_model and args.ft:
        # model_path = './model/' + args.tags +'_train' + '/' + 'best_model.pth'
        save_model_folder = f'./results/FT/{args.tags}/training/' + model_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # save_model_name = 'best_model.pth'
        # os.makedirs(save_model_folder, exist_ok=True)
    else:
        # model_path = './model/' + args.tags +'_train' + '/' + 'best_model.pth'
        save_model_folder = f'../results/{args.tags}/training/' + model_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # save_model_name = 'best_model.pth'
    
    os.makedirs(save_model_folder, exist_ok=True)

    # print("model folder created")
    # print("model path:", model_path)


    # --------------------------------- #
    # Define the dataset and dataloaders
    # --------------------------------- #
    # if args.model == 'mobilenetv2':

    generator = torch.Generator().manual_seed(42) # fix generator for reproducible results (hopefully good)

    batch_size = args.batch_size # 32

    if args.data_aug == 'fows':
        train_transform = transforms.Compose([
            # base transforms
            transforms.RandomResizedCrop((224,224)), #interpolation = BICUBIC), # extracts random crop from image (i.e. 300x300) and rescale it to 224x224
            transforms.RandomHorizontalFlip(), # helps the training
            # augmentations
            transforms.RandomRotation((-5,5)), # rotate the image by a random angle between -5 and 5 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # training and validation dataset
        train_dataset = FaceImagesDataset(train_dir, train_transform)
        train_size = int(0.8 * len(train_dataset)) # 80% of the dataset for training  -> 9600 fows imgs in training
        val_size = len(train_dataset) - train_size # 20% of the dataset for validation -> 2400 fows imgs in testing/validation

        # split the dataset into training and validation
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator)
        # returns two datasets: train_dataset and val_dataset

        # test with samplers for the dataloader
        sampler_train = RandomSampler(train_dataset)
        sampler_val   = SequentialSampler(val_dataset)

        # Define the dataloaders
        train_dataloader = DataLoader(train_dataset,batch_size=batch_size, sampler=sampler_train, drop_last = True, num_workers = 3) # batch_sampler = batch_sampler_train) #
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler_val, drop_last = False, num_workers = 3)

    else:
        print("data augmentation not supported")
        exit()

    print("data augmentations: ", args.data_aug)
    # print(train_transform)
    
    # -------------------------------------------------------- #
    # Train the model
    # -------------------------------------------------------- #
    

    print(f"Training the model {model_name}...")

    
    # # creating a folder where to save the testing results 
    if args.tl:
        exp_results_path = f'./results/results_TL/{args.tags}/training' # i.e. ./results/EfficientNetB4_FF_no_occ_focal_loss/testing
    elif args.ft: 
        exp_results_path = f'./results/results_FT/{args.tags}/training'
    else: 
        exp_results_path = f'../results/results/{args.tags}/training'

    if args.save_log:
        print(f"logging results - {args.tags}")
        # set up logger
        log_path = exp_results_path + '/exp_log/output.log'
        log = create_logger(log_path)
        log.info(f"Training the {args.tags} model on the {args.train_dataset} dataset with threshold {args.thresh}")

        # print some info on the model architecture
        log.info("Model informations:")
        log.info(f"Model Name: {model_name}")
        # log.info(f"Pretrained model weights: {pretrained_model_path if pretrained_model_path != '' else 'ImageNet pre-trained model weights'}")
        log.info(f"Optimizer: {args.optimizer}")
        log.info(f"Loss function: {criterion}")
        log.info(f"Learning rate: {args.lr}")
        log.info(f"Batch size: {batch_size}")
        log.info(f"Number of epochs: {num_epochs}")
        # log.info(f"Resume training from epoch: ", checkpoint['epoch']+1) if args.resume else print("training model from scratch")
        log.info(f"Early Stopping-Patience: {patience}")
        log.info(f"Dataset: {args.train_dataset}")
        log.info(f"Dataset path: {train_dir}")
        log.info(f"Data augmentation: {args.data_aug}")
        log.info(f"{train_transform}")
        # log.info(f"Model path: {model_path}")
        log.info(f"Training the model...")
        

    start_epoch = 0
    # if args.resume:
    #     if 'optimizer' in checkpoint and 'epoch' in checkpoint and 'criterion' in checkpoint:
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #         start_epoch = checkpoint['epoch'] + 1 
    #         criterion = checkpoint['criterion']
    #     log.info(f"Resume training from epoch: {start_epoch}")
    
    # else:
    #     print("Training the model from scratch (epoch 0).")
    #     # log.info("Training the model from scratch (epoch 0)")

    # Train the model
    for epoch in range(start_epoch, args.num_epochs):
        
        # if not gotcha:
            train_loss, train_accuracy, train_accuracy_original, train_accuracy_simswap, train_accuracy_ghost, train_accuracy_facedancer = train_one_epoch(model, criterion, optimizer, train_dataloader, device, args.thresh)
            
            # lr_scheduler.step() # update the learning rate based on the validation loss

            val_loss, val_accuracy, val_accuracy_original, val_accuracy_simswap, val_accuracy_ghost, val_accuracy_facedancer, balanced_test_acc, TPR, TNR, auc, eer, ap_score = validate_one_epoch(model, criterion, val_dataloader, device, exp_results_path, epoch, args.thresh) #, thresh = 0.6)
            
            # print train and validation info
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}")
            print('--------------------------------------------------------------------')
            # print some more tain adn val info for each method
            print(f"Train Accuracy Original: {train_accuracy_original:.4f}, Train Accuracy SimSwap: {train_accuracy_simswap:.4f}, Train Accuracy Ghost: {train_accuracy_ghost:.4f} Train Accuracy FaceDancer: {train_accuracy_facedancer:.4f}")
            print(f"Validation Accuracy Original: {val_accuracy_original:.4f}, Validation Accuracy SimSwap: {val_accuracy_simswap:.4f}, Validation Accuracy Ghost: {val_accuracy_ghost:.4f}, Validation Accuracy FaceDancer: {val_accuracy_facedancer:.4f}")
            print(f"Balanced Accuracy: {balanced_test_acc:.4f}, AP: {ap_score:.4f}, TPR: {TPR:.4f}, TNR: {TNR:.4f}, AUC: {auc:.4f}, EER: {eer:.4f}")
            print('--------------------------------------------------------------------')

            if args.save_log:
                log.info(f'Epoch: {epoch+1}/{num_epochs}:'),
                log.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}")
                log.info(f"Train Accuracy Original: {train_accuracy_original:.4f}, Train Accuracy SimSwap: {train_accuracy_simswap:.4f}, Train Accuracy Ghost: {train_accuracy_ghost:.4f} Train Accuracy FaceDancer: {train_accuracy_facedancer:.4f}")
                log.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}")
                log.info(f"Validation Accuracy Original: {val_accuracy_original:.4f}, Validation Accuracy SimSwap: {val_accuracy_simswap:.4f}, Validation Accuracy Ghost: {val_accuracy_ghost:.4f}, Validation Accuracy FaceDancer: {val_accuracy_facedancer:.4f}")
                log.info(f"Balanced Accuracy: {balanced_test_acc:.4f}, AP: {ap_score:.4f}, TPR: {TPR:.4f}, TNR: {TNR:.4f}, AUC: {auc:.4f}, EER: {eer:.4f}")
                log.info('--------------------------------------------------------------------')
            

            # save checkpoint at each epoch
            # checkpoint_paths = [save_model_folder + '/checkpoint.pth' ]

            # At the end of each epoch, check the validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"best val_loss: {best_val_loss:.4f}")
                print('--------------------------------------------------------------------')
                early_stopping_counter = 0

                # Save the best model
                # checkpoint_paths.append(save_model_folder + '/best_checkpoint.pth')
                best_model_path = save_model_folder + '/best_checkpoint.pth'
                # -------------------------------------------------- #
                # save all model info
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'criterion': criterion,
                    'optimizer': optimizer.state_dict(),
                    # 'train_loss': train_loss,
                    # 'train_accuracy': train_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'args': args
                }, best_model_path)
                print(f"best checkpoint saved in {best_model_path}")
                log.info(f"Best checkpoint saved in: {best_model_path}")
                log.info('--------------------------------------------------------------------')
                # -------------------------------------------------- #
                
            else:
                
                # update early stopping condition
                early_stopping_counter += 1
                print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
                print('--------------------------------------------------------------------')
                log.info(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
                log.info('--------------------------------------------------------------------')
                if early_stopping_counter >= patience:
                    print('Early stopping')
                    log.info("Early Stopping")
                    log.info('--------------------------------------------------------------------')
                    break
            
            
        # # ------- #
        # # GOTCHA! #
        # # ------- #
        # else:

        #     train_loss, train_accuracy, train_accuracy_original, train_accuracy_dfl, train_accuracy_fsgan = gotcha_train_one_epoch(model, criterion, optimizer, train_dataloader, device, args.thresh)
            
        #     # lr_scheduler.step() # update the learning rate based on the validation loss

        #     val_loss, val_accuracy, balanced_test_acc, val_accuracy_original, val_accuracy_dfl, val_accuracy_fsgan, TPR, TNR, auc_score, ap_score, eer = gotcha_validate(model, criterion, val_dataloader, device, args.thresh) #, thresh = 0.6)
            
        #     # print train and validation info
        #     print(f"Epoch {epoch+1}/{num_epochs}:")
        #     print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}")
        #     print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}")
        #     print('--------------------------------------------------------------------')
        #     # print some more tain adn val info for each method
        #     print(f"Train Accuracy Original: {train_accuracy_original:.4f}, Train Accuracy DFL: {train_accuracy_dfl:.4f}, Train Accuracy FSGAN: {train_accuracy_fsgan:.4f}")
        #     print(f"Validation Accuracy Original: {val_accuracy_original:.4f}, Validation Accuracy DFL: {val_accuracy_dfl:.4f}, Validation Accuracy FSGAN: {val_accuracy_fsgan:.4f}")
        #     print(f"Balanced Accuracy: {balanced_test_acc:.4f}, AP: {ap_score:.4f}, TPR: {TPR:.4f}, TNR: {TNR:.4f}, AUC: {auc_score:.4f}, EER: {eer:.4f}")
        #     print('--------------------------------------------------------------------')

        #     # log all train and vall info to wandb
        #     if args.wandb:
        #         run.log({'Train Loss': train_loss, 'Train Accuracy': train_accuracy})
        #         run.log({'Validation Loss': val_loss, 'Validation Accuracy': val_accuracy})
        #         run.log({'Epoch': epoch+1})
        #         run.log({'Train Accuracy Original': train_accuracy_original, 'Train Accuracy DFL': train_accuracy_dfl, 'Train Accuracy FSGAN': train_accuracy_fsgan})
        #         run.log({'Validation Accuracy Original': val_accuracy_original, 'Validation Accuracy DFL': val_accuracy_dfl, 'Validation Accuracy FSGAN': val_accuracy_fsgan})
            
        #     if args.save_log:
        #         log.info(f'Epoch: {epoch+1}/{num_epochs}:'),
        #         log.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}")
        #         log.info(f"Train Accuracy Original: {train_accuracy_original:.4f}, Train Accuracy DFL: {train_accuracy_dfl:.4f}, Train Accuracy FSGAN: {train_accuracy_fsgan:.4f}")
        #         log.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}")
        #         log.info(f"Validation Accuracy Original: {val_accuracy_original:.4f}, Validation Accuracy DFL: {val_accuracy_dfl:.4f}, Validation Accuracy FSGAN: {val_accuracy_fsgan:.4f}")
        #         log.info(f"Balanced Accuracy: {balanced_test_acc:.4f}, AP: {ap_score:.4f}, TPR: {TPR:.4f}, TNR: {TNR:.4f}, AUC: {auc_score:.4f}, EER: {eer:.4f}")
        #         log.info('--------------------------------------------------------------------')
                        
        #     # save checkpoint at each epoch
        #     checkpoint_paths = [save_model_folder + '/checkpoint.pth' ] 

        #     # At the end of each epoch, check the validation loss
        #     if val_loss < best_val_loss:
        #         best_val_loss = val_loss
        #         print(f"best val_loss: {best_val_loss:.4f}")
        #         print('--------------------------------------------------------------------')
        #         early_stopping_counter = 0

        #         # Save the best model
        #         # checkpoint_paths.append(save_model_folder + '/best_checkpoint.pth')
        #         best_model_path = save_model_folder + '/best_checkpoint.pth'
        #         # save all model info
        #         torch.save({
        #             'epoch': epoch,
        #             'model': model.state_dict(),
        #             'criterion': criterion,
        #             'optimizer': optimizer.state_dict(),
        #             # 'train_loss': train_loss,
        #             # 'train_accuracy': train_accuracy,
        #             'val_loss': val_loss,
        #             'val_accuracy': val_accuracy,
        #             'args': args
        #         }, best_model_path)
        #         print(f"best checkpoint saved in: {best_model_path}")
        #         log.info(f"Best checkpoint saved in: {best_model_path}")
        #         log.info('--------------------------------------------------------------------')
                
        #     else:
        #         early_stopping_counter += 1
        #         print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
        #         print('--------------------------------------------------------------------')
        #         log.info(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
        #         log.info('--------------------------------------------------------------------')
        #         if early_stopping_counter >= patience:
        #             print('Early stopping')
        #             log.info("Early Stopping")
        #             log.info('--------------------------------------------------------------------')
        #             break

if __name__ == '__main__':
    main()
