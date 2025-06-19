import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import json


def train_one_epoch(model, criterion, optimizer, train_dataloader, device, thresh = 0.5):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print_freq = 30
    
    # keep track of the number of correct predictions for each class
    correct_original, correct_simswap, correct_ghost, correct_facedancer = 0, 0, 0, 0
    total_original, total_simswap, total_ghost, total_facedancer = 0, 0, 0, 0

    print("threshold: ", thresh)

    for batch_idx, (images, labels, paths) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.float().to(device)
        img_paths = paths

        optimizer.zero_grad() 
        outputs = model(images) 
        outputs = outputs.squeeze(1) 
        loss = criterion(outputs, labels)    
        running_loss += loss.item() 
        loss.backward() 
        optimizer.step() 
        prob = torch.sigmoid(outputs) 
       
        preds = (prob > thresh).float()
        total += labels.size(0) 
        correct += (preds == labels).sum().item() 

        for i in range(len(img_paths)):
            if 'original' in img_paths[i]:
                total_original += 1
                if preds[i] == labels[i]: 
                    correct_original += 1
            elif 'simswap' in img_paths[i]:
                total_simswap += 1
                if preds[i] == labels[i]:
                    correct_simswap += 1
            elif 'ghost' in img_paths[i]:
                total_ghost += 1
                if preds[i] == labels[i]:
                    correct_ghost += 1
            elif 'facedancer' in img_paths[i]:
                total_facedancer += 1
                if preds[i] == labels[i]:
                    correct_facedancer += 1

        if batch_idx % print_freq == 0:
            print(f"Batch {batch_idx}/{len(train_dataloader)}, Loss: {running_loss / (batch_idx + 1):.4f}, Accuracy: {100 * correct / total:.4f}")

    train_loss = running_loss / len(train_dataloader)
    train_accuracy = 100 * correct / total

    train_accuracy_original = (correct_original / total_original * 100) if total_original != 0 else -1
    train_accuracy_ghost = (correct_ghost / total_ghost * 100) if total_ghost != 0 else -1
    train_accuracy_simswap = (correct_simswap / total_simswap * 100) if total_simswap != 0 else -1
    train_accuracy_facedancer = (correct_facedancer / total_facedancer * 100) if total_facedancer != 0 else -1

    return train_loss, train_accuracy, train_accuracy_original, train_accuracy_simswap, train_accuracy_ghost, train_accuracy_facedancer 

def validate_one_epoch(model, criterion, val_dataloader, device, exp_results_path, epoch, thresh = 0.5):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    y_true = [] 
    y_score = [] 
    y_pred = [] 

    if hasattr(model, 'get_attention'):
        att_imgs_path = exp_results_path + f'/att_imgs/epoch_{epoch}/'
        if not os.path.exists(att_imgs_path):
            os.makedirs(att_imgs_path, exist_ok=True)
    else:
        print("the model has no self-attention")

    correct_original, correct_simswap, correct_ghost, correct_facedancer = 0, 0, 0, 0
    total_original, total_simswap, total_ghost, total_facedancer = 0, 0, 0, 0
    print("threshold: ", thresh)
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(val_dataloader):
            images = images.to(device)

            labels = labels.float().to(device)
            img_paths = paths
            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() 
            prob = torch.sigmoid(outputs) 
            preds = (prob > thresh).float()
            total += labels.size(0) 
            correct += (preds == labels).sum().item() 
            
            y_true.append(labels) 
            y_score.append(prob) 
            y_pred.append(preds) 

            for i in range(len(img_paths)):
                if 'original' in img_paths[i]:
                    total_original += 1
                    if preds[i] == labels[i]: 
                        correct_original += 1

                elif 'simswap' in img_paths[i]:
                    total_simswap += 1
                    if preds[i] == labels[i]:
                        correct_simswap += 1

                elif 'ghost' in img_paths[i]:
                    total_ghost += 1
                    if preds[i] == labels[i]:
                        correct_ghost += 1

                elif 'facedancer' in img_paths[i]:
                    total_facedancer += 1
                    if preds[i] == labels[i]:
                        correct_facedancer += 1


    val_loss = running_loss / len(val_dataloader)
    val_accuracy = 100 * correct / total

    val_accuracy_original = (correct_original / total_original * 100) if total_original != 0 else -1
    val_accuracy_ghost = (correct_ghost / total_ghost * 100) if total_ghost != 0 else -1
    val_accuracy_simswap = (correct_simswap / total_simswap * 100) if total_simswap != 0 else -1
    val_accuracy_facedancer = (correct_facedancer / total_facedancer * 100) if total_facedancer != 0 else -1

    labels_collection = torch.cat(y_true, dim=0) 
    probs_collection = torch.cat(y_score, dim=0) 
    labels_list = labels_collection.cpu().numpy() 
    probs_list = probs_collection.cpu().numpy() 
    pred_list = torch.cat(y_pred, dim=0).cpu().numpy() 
    # ------------------------------------------------ #
    # compute the balanced test accuracy
    balanced_test_acc = metrics.balanced_accuracy_score(labels_list, pred_list)
    print(f"Balanced Test Accuracy: {balanced_test_acc:.4f}")

    cm = metrics.confusion_matrix(labels_list, pred_list)
    tn, fp, fn, tp = cm.ravel() 
    TPR = tp / (tp + fn) 
    TNR = tn / (tn + fp) 
    print(f"TPR (recall): {TPR:.4f}, TNR (specificity): {TNR:.4f}")
    # ------------------------------------------------ #
    # compute the ROC curve
    fpr_roc, tpr_roc, _ = metrics.roc_curve(labels_list, probs_list, pos_label=1) 
    auc_score = metrics.roc_auc_score(labels_list, probs_list)
    auc = metrics.auc(fpr_roc, tpr_roc)
    print(f'ROC-AUC score: {auc_score}') 
    print(f'AUC: {auc:.4f}')

    # ------------------------------------------------ #
    # compute Equal Error Rate (EER) - the point where the fpr is equal to the fnr
    # taken from DFB utils.py #
    tpr = tpr_roc
    fpr = fpr_roc
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))] if not np.isnan(fpr).all() else 1
    print("Equal Error Rate (EER): ", eer)
    ap_score = metrics.average_precision_score(labels_list, probs_list)

    return val_loss, val_accuracy, val_accuracy_original, val_accuracy_simswap, val_accuracy_ghost, val_accuracy_facedancer, balanced_test_acc, TPR, TNR, auc, eer, ap_score

def check_graph_name(folder_path, graph_name):
    graph_path = os.path.join(folder_path, graph_name)
    if os.path.exists(graph_path): 
        i = 1
        file_name = graph_name.split('.png')[0] 
        new_graph_path  = os.path.join(folder_path, f'{file_name}_{i}.png')
        while os.path.exists(new_graph_path): 
            i += 1
            new_graph_path  = os.path.join(folder_path, f'{file_name}_{i}.png')
        return new_graph_path
    else:
        return graph_path

def plot_prob_hist(prob_original, prob_simswap, prob_ghost, prob_facedancer, thresh, plot_path, tags):
    zero = False
    # check if 0.0 is in the data
    if 0.0 in prob_original or 0.0 in prob_facedancer or 0.0 in prob_ghost or 0.0 in prob_simswap:
        print("0.0 is in the data")
        zero = True
    else:
        # zero = False
        # Compute the logarithm of the data with base 10
        base = 10
        prob_or_log = np.log(prob_original) / np.log(base)
        prob_fd_log = np.log(prob_facedancer) / np.log(base)
        prob_gh_log = np.log(prob_ghost) / np.log(base)
        prob_ss_log = np.log(prob_simswap) / np.log(base)
        thresh = np.log(thresh) / np.log(base)

    plt.figure(figsize=(12, 8))
    if zero: 
        plt.hist(prob_original, bins=100, alpha=0.6, label='Original', color='blue', density=True, log=True,)
        plt.hist(prob_simswap, bins=100, alpha=0.6, label='SimSwap', color='orange', density=True, log=True,)
        plt.hist(prob_ghost, bins=100, alpha=0.6, label='Ghost', color='green', density=True, log=True, )
        plt.hist(prob_facedancer, bins=100, alpha=0.6, label='Facedancer', color='red', density=True, log=True,)
    else: 
        plt.hist(prob_or_log, bins=100, alpha=0.6, label='Original', color='blue', log=True,)
        plt.hist(prob_ss_log, bins=100, alpha=0.6, label='SimSwap', color='orange',   log=True,)
        plt.hist(prob_gh_log, bins=100, alpha=0.6, label='Ghost', color='green',   log=True, )
        plt.hist(prob_fd_log, bins=100, alpha=0.6, label='Facedancer', color='red',   log=True,)

    # Add a vertical line for the threshold
    plt.axvline(x=thresh, color='black', linestyle='--', linewidth=2, label='Threshold')
    # linestyle='--' -> dashed line

    # Add labels indicating real and fake regions
    plt.text(thresh / 2, plt.ylim()[1] * 0.6, 'Real', horizontalalignment='center', fontsize=12, color='black')
    plt.text(thresh + (1 - thresh) / 2, plt.ylim()[1] * 0.6, 'Fake', horizontalalignment='center', fontsize=12, color='black')
    plt.legend(loc='best')

    # Remove y-axis numbers
    plt.yticks([])
    plt.xticks([])

    plt.savefig(plot_path)
    plt.close()

def plot_eer(fpr, fnr, eer, path):
    """
    Plot the Equal Error Rate (EER) graph.

    Parameters:
    - fpr: False Positive Rate (FPR) values.
    - fnr: False Negative Rate (FNR) values.
    - eer: Equal Error Rate (EER) value.
    - eer_threshold: Threshold at which EER occurs.
    - path: Path to save the plot image.
    """

    fig, ax1 = plt.subplots()

    # Plot FPR on the primary y-axis
    ax1.plot(fpr, label='FPR (False Positive Rate)', color='blue')
    ax1.set_xlabel('Threshold Index')
    ax1.set_ylabel('FPR', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis for FNR
    ax2 = ax1.twinx()
    ax2.plot(fnr, label='FNR (False Negative Rate)', color='orange')
    ax2.set_ylabel('FNR', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Find the index where FPR and FNR are closest to each other
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_index]
    print(eer_index)

    # Annotate the EER point
    ax1.scatter([eer_index], [eer], color='red', label=f'EER = {eer:.2f}') 
    ax1.text(eer_index, eer, f'({eer_index}, {eer:.2f})', color='red', fontsize=12, ha='right')

    # Add legends
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    plt.title('Equal Error Rate (EER)')
    plt.savefig(path)  
    plt.show()
    plt.close() 


def plot_real_vs_fake_predictions(prob_original, prob_ghost,  thresh, plot_path, tags): #prob_simswap, prob_ghost, prob_facedancer,
    """
    Plot real vs fake predictions in a graph with both real and fake points.
    
    Parameters:
    - prob_original: List of probabilities for original images.
    - prob_simswap: List of probabilities for SimSwap images.
    - prob_ghost: List of probabilities for Ghost images.
    - prob_facedancer: List of probabilities for FaceDancer images.
    - thresh: Threshold value to separate real and fake predictions.
    - plot_path: Path to save the plot image.
    - tags: Tags to be used in the plot title.


    graph will have:
    - x-axis: frame number
    - y-axis: model prediction (or model score)
    """

    # Plot real (original) points
    plt.scatter(prob_original, np.zeros_like(prob_original), label='Original', color='blue', alpha=0.6)
    
    # Plot fake (SimSwap, Ghost, FaceDancer) points
    # plt.scatter(prob_simswap, np.ones_like(prob_simswap), label='SimSwap', color='orange', alpha=0.6)
    plt.scatter(prob_ghost, np.ones_like(prob_ghost) * 2, label='Ghost', color='green', alpha=0.6)
    # plt.scatter(prob_facedancer, np.ones_like(prob_facedancer) * 3, label='FaceDancer', color='red', alpha=0.6)
    
    # Add a vertical line for the threshold
    plt.axvline(x=thresh, color='black', linestyle='--', linewidth=2, label='Threshold')
    
    # Add labels and title
    plt.xlabel('Frames')
    plt.ylabel('Score')
    plt.title(f'Real vs Fake Predictions ({tags})')
    
    # Set y-ticks to show class names
    # plt.yticks([0, 1, 2, 3], ['Original', 'SimSwap', 'Ghost', 'FaceDancer'])

    # y-axis -> number of frames (number of predictions in the list)
    plt.yticks(rotation=45)
    plt.ylim(0, 1)  # Set x-axis limits to [0, 1] for probabilities

    # x-axis -> num of frames = len(prob_original)
    # num_frames = max(len(prob_original), len(prob_ghost)) # should all be the same length

    plt.legend()
    
    # Save the plot
    plt.savefig(plot_path)


def test_one_epoch(model, test_dataloader, device, model_name, exp_results_path, tags, thresh = 0.5):
    model.eval()
    # running_loss = 0.0
    correct = 0
    total = 0
    
    correct_original, correct_simswap, correct_ghost, correct_facedancer = 0, 0, 0, 0
    total_original, total_simswap, total_ghost, total_facedancer = 0, 0, 0, 0
    print("threshold: ", thresh)

    labels_collection = []
    original_labels, facedancer_labels, ghost_labels, simswap_labels = [], [], [], []
    original_path, facedancer_path, ghost_path, simswap_path = [], [], [], []

    y_true = [] # true binary labels
    y_score = [] # target scores 
    y_pred = []

    i = 0

    # probabilites for each class
    prob_original = []
    prob_simswap = []
    prob_ghost = []
    prob_facedancer = []

    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(test_dataloader):
            images = images.to(device) 
            labels = labels.float().to(device) 
            img_path = paths
            outputs = model(images) # forward pass through the model
            outputs = outputs.squeeze(1) # remove the extra dimension from the outputs (batch_size, 1) to (batch_size,)
            prob = torch.sigmoid(outputs) # get the probabilities (apply the sigmoid function to the outputs) -> model outputs a single value between 0 and 1, which is the probability of the image being fake (1) or real (0)
            
            preds = (prob > thresh).float() # get the predictions based on the threshold -- if prob > thresh, then the prediction is 1 (fake), else 0 (real)
            total += labels.size(0) 
            correct += (preds == labels).sum().item() 
            
            y_true.append(labels) 
            y_score.append(prob) 
            y_pred.append(preds) 
            
            for i in range(len(img_path)):
                if 'original' in img_path[i]:
                    total_original += 1
                    prob_original.append(prob[i].item())
                    original_labels.append(labels[i].item())
                    original_path.append(img_path)
                    if preds[i] == labels[i]:
                        correct_original += 1
                    
                elif 'simswap' in img_path[i]:
                    total_simswap += 1
                    prob_simswap.append(prob[i].item())
                    simswap_labels.append(labels[i].item())
                    simswap_path.append(img_path)
                    if preds[i] == labels[i]:
                        correct_simswap += 1

                elif 'ghost' in img_path[i]:
                    total_ghost += 1
                    prob_ghost.append(prob[i].item())
                    ghost_labels.append(labels[i].item())
                    ghost_path.append(img_path)
                    if preds[i] == labels[i]:
                        correct_ghost += 1

                elif 'facedancer' in img_path[i]:
                    total_facedancer += 1
                    prob_facedancer.append(prob[i].item())
                    facedancer_labels.append(labels[i].item())
                    facedancer_path.append(img_path)
                    if preds[i] == labels[i]:
                        correct_facedancer += 1
    
    
    prob_his_dir = os.path.join(exp_results_path, 'prob-hist')
    os.makedirs(prob_his_dir, exist_ok=True)
    prob_hist_path = check_graph_name(prob_his_dir, f"{tags}_prob_hist_{thresh}_thresh.png")
    print("prob_hist_path: ", prob_hist_path)
    plot_prob_hist(prob_original, prob_simswap, prob_ghost, prob_facedancer, thresh, prob_hist_path, tags)

    # compute the accuracy for each class
    test_accuracy = 100 * correct / total

    #calculate the accuracy for each class
    test_accuracy_original = (100 * correct_original / total_original) if total_original != 0 else -1
    test_accuracy_simswap = (100 * correct_simswap / total_simswap) if total_simswap != 0 else -1
    test_accuracy_ghost = (100 * correct_ghost / total_ghost) if total_ghost != 0 else -1
    test_accuracy_facedancer = (100 * correct_facedancer / total_facedancer) if total_facedancer != 0 else -1

    print("\nOriginal")
    print("correct_original: ", correct_original)
    print("total_original: ", total_original)
    print("original_acc: ", test_accuracy_original)

    print("\nSimSwap")
    print("correct_simswap: ", correct_simswap)
    print("total_simswap: ", total_simswap)
    print("simswap_acc: ", test_accuracy_simswap)

    print("\nGhost")
    print("correct_ghost: ", correct_ghost)
    print("total_ghost: ", total_ghost)
    print("ghost_acc: ", test_accuracy_ghost)

    print("\nFacedancer")
    print("correct_facedancer: ", correct_facedancer)
    print("total_facedancer: ", total_facedancer)
    print("facedancer_acc: ", test_accuracy_facedancer)
    
    labels_collection = torch.cat(y_true, dim=0) 
    probs_collection = torch.cat(y_score, dim=0) 
    labels_list = labels_collection.cpu().numpy() 
    probs_list = probs_collection.cpu().numpy() 
    pred_list = torch.cat(y_pred, dim=0).cpu().numpy() 

    if len(np.unique(labels_list)) == 2:
        balanced_test_acc = metrics.balanced_accuracy_score(labels_list, pred_list) 
        print(f"Balanced Test Accuracy: {balanced_test_acc:.4f}")

        # ------------------------------------------------ #

        cm = metrics.confusion_matrix(labels_list, pred_list)
        tn, fp, fn, tp = cm.ravel() 
        cm_display = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(labels_list, pred_list)).plot()
        conf_matrix_dir = os.path.join(exp_results_path, 'conf-matrix') 
        os.makedirs(conf_matrix_dir, exist_ok=True)
        conf_matrix_path = check_graph_name(conf_matrix_dir, f"{tags}_conf_matrix.png")
        cm_display.figure_.savefig(conf_matrix_path)
        # ------------------------------------------------ #
        TPR = tp / (tp + fn) 
        TNR = tn / (tn + fp) 
        print(f"TPR (recall): {TPR:.4f}, TNR (specificity): {TNR:.4f}")
        # ------------------------------------------------ #
        # compute the ROC curve
        fpr_roc, tpr_roc, thresholds = metrics.roc_curve(labels_list, probs_list, pos_label=1) #, pos_label= 0) 
        
        auc_score = metrics.roc_auc_score(labels_list, probs_list)
        auc = metrics.auc(fpr_roc, tpr_roc)
        print(f'ROC-AUC score: {auc_score}') 
        print(f'AUC: {auc:.4f}')

        # compute best AUC threshold (youden's J statistic)
        youden_j = tpr_roc - fpr_roc
        auc_best_threshold = thresholds[np.argmax(youden_j)]
        print(f"Best AUC threshold: {auc_best_threshold:.4f}")


        # # plot the ROC curve with AUC score
        display_roc = metrics.RocCurveDisplay(fpr=fpr_roc, tpr=tpr_roc, roc_auc=auc_score, estimator_name=model_name)
        # # get model name from wandb_tags (e.g., 'bceWLL_test')
        display_roc.plot() # plot the ROC curve with the AUC score

        # # save the ROC curve plot
        roc_plot_dir = os.path.join(exp_results_path, 'roc-curve')  #'/home/rz/rz-test/bceWLL_test/results/roc-curve/'+model_name
        os.makedirs(roc_plot_dir, exist_ok=True)
        roc_plot_path = check_graph_name(roc_plot_dir, f"{tags}_roc_curve_auc.png")
        display_roc.figure_.savefig(roc_plot_path)

        # ------------------------------------------------ #
        # compute Equal Error Rate (EER) - the point where the fpr is equal to the fnr
        # taken from DFB utils.py #
        tpr = tpr_roc
        fpr = fpr_roc
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))] if not np.isnan(fpr).all() else 1
        print("Equal Error Rate (EER): ", eer)
        # get the threshold at which the EER occurs
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] if not np.isnan(fpr).all() else 1
        print(f"EER threshold: {eer_threshold:.4f}")
        eer_plot_dir = os.path.join(exp_results_path, 'eer-plot')
        os.makedirs(eer_plot_dir, exist_ok=True)
        eer_plot_path = check_graph_name(eer_plot_dir, f"{tags}_eer_plot.png")
        plot_eer(fpr, fnr, eer, eer_plot_path)
        # --------------------------------------------- #
        # compute the average precision score
        ap_score = metrics.average_precision_score(labels_list, probs_list)
        # compute precision and recall
        precision, recall, _ = metrics.precision_recall_curve(labels_list, probs_list) # returns: precision, recall, thresholds
        # # compute the precision-recall curve and plot it
        display_pr = metrics.PrecisionRecallDisplay(precision=precision, recall=recall, average_precision = ap_score, estimator_name=model_name)
        display_pr.plot() 

        # save the precision-recall curve plot
        pr_plot_dir = os.path.join(exp_results_path, 'pr-curve') 
        os.makedirs(pr_plot_dir, exist_ok=True)
        pr_plot_path = check_graph_name(pr_plot_dir, f"{tags}_prc_curve_auc.png")
        
        display_pr.figure_.savefig(pr_plot_path)
    else:
        print("Only one class present in the data, cannot compute the balanced accuracy, ROC curve, EER, and AUC score")
        balanced_test_acc, TPR, TNR, auc_score, eer, ap_score = -1, -1, -1, -1, -1, -1
        
    # Save the results in a dictionary format for better organization and readability
    preds_dict = {
        "note": "Note the labels/preds/imgs_paths are ordered as the Acc results (Original, FaceDancer, Ghost, SimSwap)",
        "predictions": {
            "original": prob_original,
            "facedancer": prob_facedancer,
            "ghost": prob_ghost,
            "simswap": prob_simswap,
        },
    }

    labels_dict = {
        "note": "Note the labels/preds/imgs_paths are ordered as the Acc results (Original, FaceDancer, Ghost, SimSwap)",
        "labels": {
            "original": original_labels,
            "facedancer": facedancer_labels,
            "ghost": ghost_labels,
            "simswap": simswap_labels,
        },
    }

    image_paths_dict = {
        "note": "Note the labels/preds/imgs_paths are ordered as the Acc results (Original, FaceDancer, Ghost, SimSwap)",
        "image_paths": {
            "original": original_path,
            "facedancer": facedancer_path,
            "ghost": ghost_path,
            "simswap": simswap_path,
        },
    }

    # plot original vs fake predictions in a graph with both real and fake points



    # Optionally, save the dictionary to a JSON file for easier logging and retrieval
    results_json_path = os.path.join(exp_results_path, f"{tags}_preds.json")
    with open(results_json_path, "w") as json_file:
        json.dump(preds_dict, json_file, indent=4)

    results_json_path = os.path.join(exp_results_path, f"{tags}_labels.json")
    with open(results_json_path, "w") as json_file:
        json.dump(labels_dict, json_file, indent=4)

    results_json_path = os.path.join(exp_results_path, f"{tags}_img_paths.json")
    with open(results_json_path, "w") as json_file:
        json.dump(image_paths_dict, json_file, indent=4)

    return test_accuracy, balanced_test_acc, test_accuracy_original, test_accuracy_simswap, test_accuracy_ghost, test_accuracy_facedancer, TPR, TNR, auc_score, ap_score, eer, auc_best_threshold, eer_threshold



# -------------------------------------------------------------------------------- #
# Training function for the Gotcha dataset
# -------------------------------------------------------------------------------- #
def gotcha_train_one_epoch(model, criterion, optimizer, train_dataloader, device, thresh = 0.5):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print_freq = 30
    
    # keep track of the number of correct predictions for each class
    correct_original, correct_dfl, correct_fsgan = 0, 0, 0
    total_original, total_dfl, total_fsgan = 0, 0, 0

    print("threshold: ", thresh)

    for batch_idx, (images, labels, paths) in enumerate(train_dataloader):

        images = images.to(device)
        labels = labels.float().to(device)
        img_paths = paths

        optimizer.zero_grad() 
        outputs = model(images) 
        outputs = outputs.squeeze(1) 
        loss = criterion(outputs, labels)      
        running_loss += loss.item() 
        loss.backward() 
        optimizer.step() 

        prob = torch.sigmoid(outputs)
       
        preds = (prob > thresh).float()
        
        total += labels.size(0) 
        correct += (preds == labels).sum().item() 
        # ------------------------------------------------------------ #
        # keep track of the number of correct predictions for each class
        # check if the selected image has 'original_faces' in the path
        for i in range(len(img_paths)):
            if 'original' in img_paths[i]:
                total_original += 1
                if preds[i] == labels[i]: # if the prediction is correct (the predicted label is equal to the true label)
                    correct_original += 1
            elif 'DFL' in img_paths[i]:
                total_dfl += 1
                if preds[i] == labels[i]:
                    correct_dfl += 1
            elif 'FSGAN' in img_paths[i]:
                total_fsgan += 1
                if preds[i] == labels[i]:
                    correct_fsgan += 1
            
        if batch_idx % print_freq == 0:
            print(f"Batch {batch_idx}/{len(train_dataloader)}, Loss: {running_loss / (batch_idx + 1):.4f}, Accuracy: {100 * correct / total:.4f}")

    train_loss = running_loss / len(train_dataloader)
    train_accuracy = 100 * correct / total

    train_accuracy_original = (correct_original / total_original * 100) if total_original != 0 else -1
    train_accuracy_dfl = (correct_dfl / total_dfl * 100) if total_dfl != 0 else -1
    train_accuracy_fsgan = (correct_fsgan / total_fsgan * 100) if total_fsgan != 0 else -1

    return train_loss, train_accuracy, train_accuracy_original, train_accuracy_dfl, train_accuracy_fsgan

# -------------------------------------------------------------------------------- #
# Validation function for the Gotcha dataset
# -------------------------------------------------------------------------------- #

def gotcha_validate(model, criterion, val_dataloader, device, thresh = 0.5):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    y_true = [] # true binary labels (list of tensors)
    y_score = [] # target (predicted by the classifier) scores (list of tensors)
    y_pred = [] # predicted labels (list of tensors)

    
    correct_original, correct_dfl, correct_fsgan = 0, 0, 0
    total_original, total_dfl, total_fsgan = 0, 0, 0
    print("threshold: ", thresh)
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(val_dataloader):
            images = images.to(device)
            labels = labels.float().to(device)
            # labels = labels.to(device)
            img_paths = paths

            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() # put here to be sure loss is reduced to one item
            prob = torch.sigmoid(outputs) # get the probabilities (apply the sigmoid function to the outputs)
            preds = (prob > thresh).float()
            total += labels.size(0) 
            correct += (preds == labels).sum().item() 
            # ------------------------------------------------------------ #
            # keep track of true labels and predicted labels
            
            y_true.append(labels) # true binary labels (list of tensors)
            y_score.append(prob) # target (predicted by the classifier) scores (list of tensors)
            y_pred.append(preds) # predicted labels (list of tensors)

            # running_loss += loss.item()

            # keep track of the number of correct predictions for each class
            # check if the selected image has 'original_faces' in the path
            for i in range(len(img_paths)):
                if 'original' in img_paths[i]:
                    total_original += 1
                    if preds[i] == labels[i]: # if the prediction is correct (the predicted label is equal to the true label)
                        correct_original += 1
                elif 'DFL' in img_paths[i]:
                    total_dfl += 1
                    if preds[i] == labels[i]:
                        correct_dfl += 1
                elif 'FSGAN' in img_paths[i]:
                    total_fsgan += 1
                    if preds[i] == labels[i]:
                        correct_fsgan += 1
                

    val_loss = running_loss / len(val_dataloader)
    val_accuracy = 100 * correct / total

    val_accuracy_original = (correct_original / total_original * 100) if total_original != 0 else -1
    val_accuracy_dfl = (correct_dfl / total_dfl * 100) if total_dfl != 0 else -1
    val_accuracy_fsgan = (correct_fsgan / total_fsgan * 100) if total_fsgan != 0 else -1

    # return val_loss, val_accuracy, val_accuracy_original, val_accuracy_dfl, val_accuracy_fsgan, thresh
    # concatenate the list of tensors along the first dimension (stack the tensors)
    labels_collection = torch.cat(y_true, dim=0) # concatenate the list of tensors along the first dimension (stack the tensors)
    # print(labels_collection[0:2])
    probs_collection = torch.cat(y_score, dim=0) # concatenate the list of tensors along the first dimension (stack the tensors)
    # print(prediction_collection[0:2])
    # convert to numpy array (tensor -> numpy array), then give it to roc_curve
    labels_list = labels_collection.cpu().numpy() # convert to numpy array -> true binary labels (ground truth, 0 or 1)
    probs_list = probs_collection.cpu().numpy() # convert to numpy array -> predicted scores (probabilities)
    pred_list = torch.cat(y_pred, dim=0).cpu().numpy() # convert to numpy array -> predicted labels (0 or 1)
    # ------------------------------------------------ #
    # compute the balanced test accuracy
    balanced_test_acc = metrics.balanced_accuracy_score(labels_list, pred_list) #, adjusted=True) # needs numpy array results, not cuda tensors like y_true, y_pred
    # adjusted = True -> rescale the balanced acc to the range [0, 1] (random performance = 0), 
    # adjusted = False -> return the unadjusted balanced acc [random performance = 0.5] 
    # in general, the closer the balanced accuracy is to 1, the better the model
    print(f"Balanced Test Accuracy: {balanced_test_acc:.4f}")
    # breakpoint()
    # ------------------------------------------------ #
    # compute the confusion matrix
    # save the confusion matrix plot as an image
    cm = metrics.confusion_matrix(labels_list, pred_list)
    print("cm.shape:", cm.shape)
    # get the tp, fp, tn, fn values from the confusion matrix
    tn, fp, fn, tp = cm.ravel() # confusion matrix for all classes, ravel() flattens the array
    # ------------------------------------------------ #
    TPR = tp / (tp + fn) # sensitivity, recall ->  fraction of positive predictions out of all positive instances in the data set.
    TNR = tn / (tn + fp) # specificity -> fraction of negative predictions out of all negative instances in the data set.
    print(f"TPR (recall): {TPR:.4f}, TNR (specificity): {TNR:.4f}")
    # bal_acc = (TPR + TNR) / 2
    # print(f"Balanced Accuracy: {bal_acc:.4f}")
    # ------------------------------------------------ #
    # compute the ROC curve
    fpr_roc, tpr_roc, _ = metrics.roc_curve(labels_list, probs_list) #, pos_label= 0) 
     
    auc_score = metrics.roc_auc_score(labels_list, probs_list)
    auc = metrics.auc(fpr_roc, tpr_roc)
    print(f'ROC-AUC score: {auc_score}') 
    print(f'AUC: {auc:.4f}')

    # ------------------------------------------------ #
    # compute Equal Error Rate (EER) - the point where the fpr is equal to the fnr
    # taken from DFB utils.py #
    tpr = tpr_roc
    fpr = fpr_roc
    fnr = 1 - tpr
    # eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] if not np.isnan(fpr).all() else 1
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))] if not np.isnan(fpr).all() else 1
    print("Equal Error Rate (EER): ", eer)
    # --------------------------------------------- #
    # compute the average precision score
    ap_score = metrics.average_precision_score(labels_list, probs_list)

    return val_loss, val_accuracy, balanced_test_acc, val_accuracy_original, val_accuracy_dfl, val_accuracy_fsgan, TPR, TNR, auc_score, ap_score, eer

def check_graph_name(folder_path, graph_name):
    # check if the graph_name already exists, 
    # if it does, add a number to the end of the graph_name
    # e.g., graph_name = 'loss' -> 'loss_1', 'loss_2', 'loss_3', ...
    graph_path = os.path.join(folder_path, graph_name)
    # print("graph_path", graph_path)

    if os.path.exists(graph_path): # if the graph_name already exists -> it means that the graph has been saved before
        i = 1
        file_name = graph_name.split('.png')[0] # remove the file extension -> /home/rz/rz-test/bceWLL_test/roc-curve/' + model.name + "_" + tags + '_roc_curve_auc
        new_graph_path  = os.path.join(folder_path, f'{file_name}_{i}.png')
        while os.path.exists(new_graph_path): # check if the new graph name already exists -> if it does, increment i 
            # while stops when the new graph name does not exist 
            # new_graph_name = f'{graph_name}_{i}.png'
            i += 1
            new_graph_path  = os.path.join(folder_path, f'{file_name}_{i}.png')
               
        # print("new_graph_path", new_graph_name)
        return new_graph_path
    # if the graph_path doesn't exists, return it 
    else:
        return graph_path

def gotcha_plot_prob_hist(prob_original, prob_dfl, prob_fsgan, thresh, plot_path, tags):
    zero = False
    # check if 0.0 is in the data
    if 0.0 in prob_original or 0.0 in prob_dfl or 0.0 in prob_fsgan:
        print("0.0 is in the data")
        zero = True
    else:
        # zero = False
        # Compute the logarithm of the data with base 10
        base = 10
        prob_or_log = np.log(prob_original) / np.log(base)
        prob_dfl_log = np.log(prob_dfl) / np.log(base)
        prob_fsgan_log = np.log(prob_fsgan) / np.log(base)

    # Plot the histogram of the predicted probabilities for each class
    # plt.figure(figsize=(8, 8))
    plt.figure(figsize=(12, 8))
    if zero: 
        plt.hist(prob_original, bins=100, alpha=0.6, label='Original', color='blue', density=True, log=True,)
        plt.hist(prob_dfl, bins=100, alpha=0.6, label='SimSwap', color='orange', density=True, log=True,)
        plt.hist(prob_fsgan, bins=100, alpha=0.6, label='Ghost', color='green', density=True, log=True, )
        # plt.hist(prob_facedancer, bins=100, alpha=0.6, label='Facedancer', color='red', density=True, log=True,)
    else: 
        plt.hist(prob_or_log, bins=100, alpha=0.6, label='Original', color='blue', log=True,)
        plt.hist(prob_dfl_log, bins=100, alpha=0.6, label='SimSwap', color='orange',   log=True,)
        plt.hist(prob_fsgan_log, bins=100, alpha=0.6, label='Ghost', color='green',   log=True, )

    # density=True: This parameter normalizes the histogram so that the area under the histogram sums to 1. 
    # This effectively converts the y-axis from frequency counts to probability density.

    # Add a vertical line for the threshold
    plt.axvline(x=thresh, color='black', linestyle='--', linewidth=2, label='Threshold')
    # linestyle='--' -> dashed line

    # Add labels indicating real and fake regions
    plt.text(thresh / 2, plt.ylim()[1] * 0.6, 'Real', horizontalalignment='center', fontsize=12, color='black')
    plt.text(thresh + (1 - thresh) / 2, plt.ylim()[1] * 0.6, 'Fake', horizontalalignment='center', fontsize=12, color='black')

    # Place the legend outside the plot
    plt.legend(loc='best')

    # Remove y-axis numbers
    plt.xticks([])
    plt.yticks([])

    # Save the histogram
    plt.savefig(plot_path)
    plt.close()


def gotcha_test(model, test_dataloader, device, model_name, exp_results_path, tags, thresh = 0.5):

    model.eval()
    correct = 0
    total = 0
    
    # keep track of the number of correct predictions for each class
    correct_original, correct_dfl, correct_fsgan = 0, 0, 0
    total_original, total_dfl, total_fsgan = 0, 0, 0
    print("threshold: ", thresh)

    labels_collection = []
    original_labels, dfl_labels, fsgan_labels = [], [], []
    original_imgs_path, dfl_imgs_path, fsgan_imgs_path = [], [], []

    y_true = [] # true binary labels
    y_score = [] # target scores 
    y_pred = []

    i = 0

    # probabilites for each class
    prob_original = []
    prob_dfl = []
    prob_fsgan = []

    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(test_dataloader):
            images = images.to(device) # torch tensor
            labels = labels.float().to(device) # torch tensor
            img_path = paths

            # print("labels type", type(labels))
            # print("imgs type", type(images))
            
            ## -------------------------------------------------------------------------- ##
            outputs = model(images) # forward pass through the model -> outputs are logits (raw scores) for each image
            
            # output = torch.squeeze(outputs)
            outputs = outputs.squeeze(1) # remove the extra dimension from the outputs (batch_size, 1) to (batch_size,)
            # what's the extra dimension? is 
            
            # ------------------------------------------------------------ #
            # compute predictions for monitoring (BCEwithLogitsLoss does not require rounding the outputs to 0 or 1)
            # BCEwithLogitsLoss outputs probabilities and does not require rounding the outputs to 0 or 1
            prob = torch.sigmoid(outputs) # get the probabilities (apply the sigmoid function to the outputs) --> model outputs a single value between 0 and 1, which is the probability of the image being fake (1) or real (0)
            # change name of prob to outputs_prob or 
            # ------------------------------------------------------------ #
            
            preds = (prob > thresh).float() # get the predictions based on the threshold -- if prob > thresh, then the prediction is 1 (fake), else 0 (real)
            # ------------------------------------------------------------ #
            # predicted = torch.round(outputs).squeeze() # get the predicted labels (round the outputs to 0 or 1)
            total += labels.size(0) # get the total number of labels
            correct += (preds == labels).sum().item() # get the number of correct labels
            # ------------------------------------------------------------ #
            # keep track of true labels and predicted labels
            
            y_true.append(labels) # true binary labels (list of tensors)
            y_score.append(prob) # model scores, predicted probabilitiy (after sigmoid) of the positive class (list of tensors)
            y_pred.append(preds) # predicted labels (0 or 1) (list of tensors)
            
            # ------------------------------------------------------------ #
            # define the plot for the histogram of the predicted probabilities
            # ------------------------------------------------------------ #
            # keep track of the number of correct predictions for each class
            # check if the selected image has 'original_faces' in the path
            for i in range(len(img_path)):
                if 'original' in img_path[i]:
                    total_original += 1
                    prob_original.append(prob[i].item())
                    original_labels.append(labels[i].item())
                    original_imgs_path.append(img_path)
                    if preds[i] == labels[i]:
                        correct_original += 1
                    
                elif 'DFL' in img_path[i]:
                    total_dfl += 1
                    prob_dfl.append(prob[i].item())
                    dfl_labels.append(labels[i].item())
                    dfl_imgs_path.append(img_path)
                    if preds[i] == labels[i]:
                        correct_dfl += 1

                elif 'FSGAN' in img_path[i]:
                    total_fsgan += 1
                    prob_fsgan.append(prob[i].item())
                    fsgan_labels.append(labels[i].item())
                    fsgan_imgs_path.append(img_path)
                    if preds[i] == labels[i]:
                        correct_fsgan += 1

    # --------------------------------------------------------------------------------------------- #    
    # Call the plot_prob_hist function to plot the histogram of the predicted probabilities
    prob_his_dir = os.path.join(exp_results_path, 'prob-hist') #'/home/rz/rz-test/bceWLL_test/results/prob-hist/'+model_name
    os.makedirs(prob_his_dir, exist_ok=True)
    prob_hist_path = check_graph_name(prob_his_dir, f"{tags}_prob_hist_{thresh}_thresh.png")
    print("prob_hist_path: ", prob_hist_path)
    gotcha_plot_prob_hist(prob_original, prob_dfl, prob_fsgan, thresh, prob_hist_path, tags)
    # breakpoint()
    # --------------------------------------------------------------------------------------------- #

    # compute the accuracy for each class
    test_accuracy = 100 * correct / total

    #calculate the accuracy for each class
    test_accuracy_original = (100 * correct_original / total_original) if total_original != 0 else -1
    test_accuracy_dfl = (100 * correct_dfl / total_dfl) if total_dfl != 0 else -1
    test_accuracy_fsgan = (100 * correct_fsgan / total_fsgan) if total_fsgan != 0 else -1
    

    print("\nOriginal")
    print("correct_original: ", correct_original)
    print("total_original: ", total_original)
    print("original_acc: ", test_accuracy_original)

    print("\nDFL")
    print("correct_DFL: ", correct_dfl)
    print("total_DFL: ", total_dfl)
    print("DFL_acc: ", test_accuracy_dfl)

    print("\nFSGAN")
    print("correct_fsgan: ", correct_fsgan)
    print("total_fsgan: ", total_fsgan)
    print("fsgan_acc: ", test_accuracy_fsgan)
    
    

    # concatenate the list of tensors along the first dimension (stack the tensors)
    labels_collection = torch.cat(y_true, dim=0) # concatenate the list of tensors along the first dimension (stack the tensors)
    # print(labels_collection[0:2])
    probs_collection = torch.cat(y_score, dim=0) # concatenate the list of tensors along the first dimension (stack the tensors)
    # print(prediction_collection[0:2])
    # convert to numpy array (tensor -> numpy array), then give it to roc_curve
    labels_list = labels_collection.cpu().numpy() # convert to numpy array -> true binary labels (ground truth, 0 or 1)
    probs_list = probs_collection.cpu().numpy() # convert to numpy array -> predicted scores (probabilities)
    pred_list = torch.cat(y_pred, dim=0).cpu().numpy() # convert to numpy array -> predicted labels (0 or 1)
    
    # ------------------------------------------------ #
    # Check if both classes are present in y_true
    if len(np.unique(labels_list)) == 2:
        # compute the balanced test accuracy
        balanced_test_acc = metrics.balanced_accuracy_score(labels_list, pred_list)
        print(f"Balanced Test Accuracy: {balanced_test_acc:.4f}")
        # compute the confusion matrix
        # save the confusion matrix plot as an image
        cm = metrics.confusion_matrix(labels_list, pred_list)
        # get the tp, fp, tn, fn values from the confusion matrix
        tn, fp, fn, tp = cm.ravel() # confusion matrix for all classes, ravel() flattens the array
        cm_display = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(labels_list, pred_list)).plot()
        conf_matrix_dir = os.path.join(exp_results_path, 'conf-matrix')
        os.makedirs(conf_matrix_dir, exist_ok=True)
        conf_matrix_path = check_graph_name(conf_matrix_dir, f"{tags}_conf_matrix.png")
        cm_display.figure_.savefig(conf_matrix_path)
        # ------------------------------------------------ #
        TPR = tp / (tp + fn) # sensitivity, recall 
        TNR = tn / (tn + fp) # specificity 
        print(f"TPR (recall): {TPR:.4f}, TNR (specificity): {TNR:.4f}")
        # ------------------------------------------------ #
        # compute the ROC curve
        fpr_roc, tpr_roc, thresholds = metrics.roc_curve(labels_list, probs_list) #, pos_label= 0) 
        
        auc_score = metrics.roc_auc_score(labels_list, probs_list)
        auc = metrics.auc(fpr_roc, tpr_roc)
        print(f'ROC-AUC score: {auc_score}') 
        print(f'AUC: {auc:.4f}')
        # compute best AUC threshold (youden's J statistic)
        youden_j = tpr_roc - fpr_roc
        auc_best_threshold = thresholds[np.argmax(youden_j)]
        print(f"Best AUC threshold: {auc_best_threshold:.4f}")

        # # plot the ROC curve with AUC score
        display_roc = metrics.RocCurveDisplay(fpr=fpr_roc, tpr=tpr_roc, roc_auc=auc_score, estimator_name=model_name)
        # # get model name from wandb_tags (e.g., 'bceWLL_test')
        display_roc.plot() # plot the ROC curve with the AUC score

        # # save the ROC curve plot
        roc_plot_dir = os.path.join(exp_results_path, 'roc-curve')  #'/home/rz/rz-test/bceWLL_test/results/roc-curve/'+model_name
        os.makedirs(roc_plot_dir, exist_ok=True)
        roc_plot_path = check_graph_name(roc_plot_dir, f"{tags}_roc_curve_auc.png")
        display_roc.figure_.savefig(roc_plot_path)

        # ------------------------------------------------ #
        # compute Equal Error Rate (EER) - the point where the fpr is equal to the fnr
        # taken from DFB utils.py #
        tpr = tpr_roc
        fpr = fpr_roc
        fnr = 1 - tpr
        # eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] if not np.isnan(fpr).all() else 1
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))] if not np.isnan(fpr).all() else 1
        print("Equal Error Rate (EER): ", eer)
        # get the threshold at which the EER occurs
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] if not np.isnan(fpr).all() else 1
        print(f"EER threshold: {eer_threshold:.4f}")

        eer_plot_dir = os.path.join(exp_results_path, 'eer-plot')
        os.makedirs(eer_plot_dir, exist_ok=True)
        eer_plot_path = check_graph_name(eer_plot_dir, f"{tags}_eer_plot.png")
        plot_eer(fpr, fnr, eer, eer_plot_path)
        # --------------------------------------------- #
        # compute the average precision score
        ap_score = metrics.average_precision_score(labels_list, probs_list)
        # compute precision and recall
        precision, recall, _ = metrics.precision_recall_curve(labels_list, probs_list) # returns: precision, recall, thresholds
        # # compute the precision-recall curve and plot it
        display_pr = metrics.PrecisionRecallDisplay(precision=precision, recall=recall, average_precision = ap_score, estimator_name=model_name)
        display_pr.plot() 

        # save the precision-recall curve plot
        pr_plot_dir = os.path.join(exp_results_path, 'pr-curve') 
        os.makedirs(pr_plot_dir, exist_ok=True)
        pr_plot_path = check_graph_name(pr_plot_dir, f"{tags}_prc_curve_auc.png")
        display_pr.figure_.savefig(pr_plot_path)
    else: 
        print("Only one class present in the data, cannot compute the balanced accuracy, ROC curve, EER, and AUC score")
        balanced_test_acc, TPR, TNR, auc_score, eer, ap_score = -1, -1, -1, -1, -1, -1
    
    # Save the results in a dictionary format for better organization and readability
    preds_dict = {
        "note": "Note the labels/preds/imgs_paths are ordered as the Acc results (original, DFL, FSGAN)",
        "predictions": {
            "original": prob_original,
            "DFL": prob_dfl,
            "FSGAN": prob_fsgan,
        },
    }

    labels_dict = {
        "note": "Note the labels/preds/imgs_paths are ordered as the Acc results (original, DFL, FSGAN)",
        "labels": {
            "original": original_labels,
            "DFL": dfl_labels,
            "FSGAN": fsgan_labels,
        },
    }

    image_paths_dict = {
        "note": "Note the labels/preds/imgs_paths are ordered as the Acc results (original, DFL, FSGAN)",
        "imgs_path": {
            "original": original_imgs_path,
            "DFL": dfl_imgs_path,
            "FSGAN": fsgan_imgs_path,
        },

    }

    # Optionally, save the dictionary to a JSON file for easier logging and retrieval
    results_json_path = os.path.join(exp_results_path, f"{tags}_preds.json")
    with open(results_json_path, "w") as json_file:
        json.dump(preds_dict, json_file, indent=4)

    results_json_path = os.path.join(exp_results_path, f"{tags}_labels.json")
    with open(results_json_path, "w") as json_file:
        json.dump(labels_dict, json_file, indent=4)

    results_json_path = os.path.join(exp_results_path, f"{tags}_img_paths.json")
    with open(results_json_path, "w") as json_file:
        json.dump(image_paths_dict, json_file, indent=4)

    return test_accuracy, balanced_test_acc, test_accuracy_original, test_accuracy_dfl, test_accuracy_fsgan, TPR, TNR, auc_score, ap_score, eer, auc_best_threshold, eer_threshold