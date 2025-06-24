import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def calculate_bonecot_metrics(fold_dir, task_name='benign_malignant'):
    """
    Calculate BoneCoT AUROC and AUPRC metrics from a fold directory.
    
    Args:
        fold_dir (str): Path to the fold directory containing pred_npz folder
        task_name (str): Name of the task to evaluate (default: 'benign_malignant')
    
    Returns:
        tuple: (overall_auroc, overall_auprc)
    """    
    auroc_list = []
    auprc_list = []
    
    for fold_num in range(5):
        real_fold_dir = fold_dir.replace("fold_0", f"fold_{fold_num}")
        pred_file_dir = os.path.join(real_fold_dir, "pred_npz")
        if not os.path.exists(pred_file_dir):
            continue
        
        if 'bone_lesion' in task_name:
            # For bone lesion task, use test_epoch_0.npz directly
            pred_file_path = os.path.join(pred_file_dir, "test_epoch_0.npz")
            if not os.path.exists(pred_file_path):
                continue
            
            pred_dict = np.load(pred_file_path, allow_pickle=True)['arr_0'].item()
            pred_list = []
            label_list = []
            for study_id in pred_dict.keys():
                for series_id in pred_dict[study_id].keys():
                    for image_name, image_pred in pred_dict[study_id][series_id].items():
                        pred_list.append(image_pred['pred'])
                        label_list.append(image_pred['label'])
            
            auroc = roc_auc_score(label_list, pred_list)
            auprc = average_precision_score(label_list, pred_list)
            auroc_list.append(auroc)
            auprc_list.append(auprc)
        else:
            # For other tasks, find the best metrics across all epochs
            best_auroc = 0
            best_auprc = 0
            
            for pred_file_name in os.listdir(pred_file_dir):
                if pred_file_name.endswith('.npz'):
                    pred_file_path = os.path.join(pred_file_dir, pred_file_name)
                    pred_dict = np.load(pred_file_path, allow_pickle=True)['arr_0'].item()
                    pred_dict = pred_dict[task_name]
                    pred_list = []
                    label_list = []
                    for study_id in pred_dict.keys():
                        for series_id in pred_dict[study_id].keys():
                            for image_name, image_pred in pred_dict[study_id][series_id].items():
                                pred_list.append(image_pred['pred'])
                                label_list.append(image_pred['label'])
                        
                    # Calculate BoneCoT AUROC and AUPRC metrics
                    auroc = roc_auc_score(label_list, pred_list)
                    auprc = average_precision_score(label_list, pred_list)
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_auprc = auprc
            
            auroc_list.append(best_auroc)
            auprc_list.append(best_auprc)
    
    if len(auroc_list) == 0:
        return 0, 0
    
    overall_auroc = np.mean(auroc_list)
    overall_auprc = np.mean(auprc_list)
    return overall_auroc, overall_auprc
