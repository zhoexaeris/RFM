"""Model testing script for evaluating trained models on test datasets.

This module provides functionality to test trained models on both real and fake image
datasets, calculating various performance metrics including AUC, accuracy, precision,
recall, F1-score, and TPR (True Positive Rate) at different thresholds.
"""

import torch
from utils.utils import Eval, calRes
from pretrainedmodels import xception
from utils.custom_dataset import CustomDataset
from torch.utils.data import DataLoader
import argparse
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def test_model(model_path, dataset_path, batch_size=32, device="cuda:0"):
    """Test a trained model on real and fake image datasets.

    This function loads a trained model and evaluates its performance on both real
    and fake image datasets. It calculates various metrics including AUC, accuracy,
    precision, recall, F1-score, and TPR at different thresholds.

    Args:
        model_path (str): Path to the trained model file.
        dataset_path (str): Path to the dataset directory.
        batch_size (int, optional): Batch size for testing. Defaults to 32.
        device (str, optional): Device to run the model on. Defaults to "cuda:0".

    Returns:
        None: Results are printed to console.
    """
    # Initialize model
    model = xception(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Initialize dataset
    dataset = CustomDataset(folder_path=dataset_path)
    
    # Get test sets
    testsetR = dataset.getTestsetR()
    TestsetList, TestsetName = dataset.getsetlist(real=False, setType=2)
    
    # Create data loaders
    testdataloaderR = DataLoader(
        testsetR,
        batch_size=batch_size,
        num_workers=4
    )
    
    testdataloaderList = []
    for tmptestset in TestsetList:
        testdataloaderList.append(
            DataLoader(
                tmptestset,
                batch_size=batch_size,
                num_workers=4
            )
        )
    
    # Loss function
    lossfunc = torch.nn.CrossEntropyLoss()
    
    print("\nTesting Results:")
    print("----------------------------------------")
    
    # Test on real images
    loss_r, y_true_r, y_pred_r = Eval(model, lossfunc, testdataloaderR)
    
    # Test on fake images
    sumAUC = sumTPR_2 = sumTPR_3 = sumTPR_4 = 0
    sumACC = sumPrecision = sumRecall = sumF1 = 0
    
    for i, tmptestdataloader in enumerate(testdataloaderList):
        loss_f, y_true_f, y_pred_f = Eval(model, lossfunc, tmptestdataloader)
        
        # Combine real and fake predictions
        y_true_combined = torch.cat((y_true_r, y_true_f))
        y_pred_combined = torch.cat((y_pred_r, y_pred_f))
        
        # Calculate metrics
        ap, acc, AUC, TPR_2, TPR_3, TPR_4 = calRes(y_true_combined, y_pred_combined)
        
        # Convert to numpy for sklearn metrics
        y_true_np = y_true_combined.cpu().numpy()
        y_pred_np = y_pred_combined.argmax(dim=1).cpu().numpy()
        
        # Calculate additional metrics
        precision = precision_score(y_true_np, y_pred_np)
        recall = recall_score(y_true_np, y_pred_np)
        f1 = f1_score(y_true_np, y_pred_np)
        
        print(f"\nResults for {TestsetName[i]}:")
        print(f"AUC: {AUC:.6f}")
        print(f"Accuracy: {acc:.6f}")
        print(f"Precision: {precision:.6f}")
        print(f"Recall: {recall:.6f}")
        print(f"F1-Score: {f1:.6f}")
        print(f"TPR_2: {TPR_2:.6f}")
        print(f"TPR_3: {TPR_3:.6f}")
        print(f"TPR_4: {TPR_4:.6f}")
        
        # Sum metrics for averaging
        sumAUC += AUC
        sumACC += acc
        sumPrecision += precision
        sumRecall += recall
        sumF1 += f1
        sumTPR_2 += TPR_2
        sumTPR_3 += TPR_3
        sumTPR_4 += TPR_4
    
    if len(testdataloaderList) > 1:
        print("\nAverage Results:")
        print(f"AUC: {sumAUC/len(testdataloaderList):.6f}")
        print(f"Accuracy: {sumACC/len(testdataloaderList):.6f}")
        print(f"Precision: {sumPrecision/len(testdataloaderList):.6f}")
        print(f"Recall: {sumRecall/len(testdataloaderList):.6f}")
        print(f"F1-Score: {sumF1/len(testdataloaderList):.6f}")
        print(f"TPR_2: {sumTPR_2/len(testdataloaderList):.6f}")
        print(f"TPR_3: {sumTPR_3/len(testdataloaderList):.6f}")
        print(f"TPR_4: {sumTPR_4/len(testdataloaderList):.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the trained model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--dataset_path', type=str, default=r"D:\.THESIS\datasets\sample_data",
                        help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='Device to use (cuda:0 or cpu)')
    
    args = parser.parse_args()
    
    test_model(args.model_path, args.dataset_path, args.batch_size, args.device) 