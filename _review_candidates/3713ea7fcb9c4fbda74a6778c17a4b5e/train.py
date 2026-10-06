import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve

import wandb
from torch import nn
from torch.utils.data import DataLoader

from Model.QCNN import QCNN
from Model.WDCNN import WDCNN
from Model.QCNNResNet import QCNNResNet
from Model.MSQAResNet import MSQAResNet
from QDCA_BiGRU_Res import QDCA_BiGRU_Res
from utils.DatasetLoader import CustomTensorDataset
from utils.Preprocess import prepro
from utils.train_function import group_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

# Select Model Architecture
def select_model(chosen_model):
    if chosen_model == 'wdcnn':
        model = WDCNN()
    if chosen_model == 'qcnn':
        model = QCNN()
    if chosen_model == 'qcnnresnet':
        model = QCNNResNet()
    if chosen_model == 'msqaresnet':
        model = MSQAResNet()
    if chosen_model == 'qdca_bigru_res':
        model = QDCA_BiGRU_Res()
    return model

# Definition of Training, Validation and Testing functions; 
def train(config, dataloader):
    net = select_model(config.chosen_model)
    if use_gpu:
        net = net.cuda()
    if config.chosen_model == 'qdca_bigru_res':
        group = group_parameters(net)
        optimizer = torch.optim.SGD([
            {"params": group[0], "lr": config.lr},  # weight_r
            {"params": group[1], "lr": config.lr * config.alpha},  # weight_g
            {"params": group[2], "lr": config.lr * config.alpha},  # weight_b
            {"params": group[3], "lr": config.lr},  # bias_r
            {"params": group[4], "lr": config.lr * config.alpha},  # bias_g
            {"params": group[5], "lr": config.lr * config.alpha},  # bias_b
            {"params": group[6], "lr": config.lr},
            {"params": group[7], "lr": config.lr},
        ], lr=config.lr, momentum=0.9, weight_decay=5e-4)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-8)
    wandb.watch(net, log="all")
    max_acc = 0.0
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    loss_func = nn.CrossEntropyLoss()
    for e in range(config.epochs):
        for phase in ['train', 'validation']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            loss_total = 0.0
            total = 0
            correct = 0
            for step, (x, y) in enumerate(dataloader[phase]):
                x = x.type(torch.float)
                y = y.type(torch.long).view(-1)
                if use_gpu:
                    x, y = x.cuda(), y.cuda()
                with torch.no_grad() if phase == 'validation' else torch.enable_grad():
                    y_hat = net(x)
                    loss = loss_func(y_hat, y)
                    loss_total += loss.item()
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    y_predict = y_hat.argmax(dim=1)
                    total += y.size(0)
                    correct += (y_predict == y).sum().item()
                    if step % 20 == 0 and phase == 'train':
                        print('Epoch:%d, Step [%d/%d], Loss: %.4f'
                            % (e + 1, step + 1, len(dataloader[phase].dataset), loss_total / (step + 1)))
            loss_total = loss_total / len(dataloader[phase])
            acc = correct / total
            if phase == 'train':
                train_loss.append(loss_total)
                train_acc.append(acc)
                wandb.log({
                    "Train Accuracy": 100. * acc,
                    "Train Loss": loss_total
                })
            else:
                valid_loss.append(loss_total)
                valid_acc.append(acc)
                wandb.log({
                    "Validation Accuracy": 100. * acc,
                    "Validation Loss": loss_total
                })
                if acc >= max_acc:
                    max_acc = acc
                    if not os.path.exists('Models'):
                        os.makedirs('Models')
                    torch.save(net.state_dict(), 'Models/best_checkpoint.pth')
                    print('Model saved at epoch %d' % (e + 1))
                    if os.path.exists('Models/best_checkpoint.pth'):
                        artifact = wandb.Artifact('model', type='model')
                        artifact.add_file('Models/best_checkpoint.pth')
                        wandb.log_artifact(artifact)
                    else:
                        print("Warning: Model file not found, artifact not logged.")
                scheduler.step() 
            print('%s accuracy: %.4f, error: %.4f' % (phase, acc, loss_total))
    return net


def inference(config, dataloader):
    net = select_model(config.chosen_model)
    if not os.path.exists('Models/best_checkpoint.pth'):
        raise FileNotFoundError("Model file not found at 'Models/best_checkpoint.pth'")
    state_dict = torch.load('Models/best_checkpoint.pth', weights_only=True)
    net.load_state_dict(state_dict)
    if use_gpu:
        net = net.cuda()
    net.eval()
    y_list, y_predict_list, y_scores = [], [], []
    table_data = []
    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            x = x.type(torch.float)
            y = y.type(torch.long).view(-1)
            if use_gpu:
                x, y = x.cuda(), y.cuda()
            y_hat = net(x)
            y_predict = y_hat.argmax(dim=1)
            y_list.extend(y.detach().cpu().numpy())
            y_predict_list.extend(y_predict.detach().cpu().numpy())
            y_scores.extend(torch.softmax(y_hat, dim=1).detach().cpu().numpy())
            # table_data.append([step, y.item(), y_predict.item(), y_scores[step].tolist()])
            table_data.append([step, y.cpu().numpy().tolist(), y_predict.cpu().numpy().tolist(), y_scores[step].tolist()])
        y_scores = np.array(y_scores)
        y_true_binary = np.eye(np.max(y_list) + 1)[y_list]  
        fpr, tpr, _ = roc_curve(y_true_binary.ravel(), y_scores.ravel())
        precision, recall, _ = precision_recall_curve(y_true_binary.ravel(), y_scores.ravel())
        cnf_matrix = confusion_matrix(y_list, y_predict_list)
        recall = recall_score(y_list, y_predict_list, average="macro")
        precision = precision_score(y_list, y_predict_list, average="macro")
        table = wandb.Table(columns=["Sample Index", "True Labels", "Predicted Labels", "Prediction Scores"], data=table_data)
        wandb.log({"Predictions Table": table})

        F1 = f1_score(y_list, y_predict_list, average="macro")
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        TN = TN.astype(float)
        FPR = np.mean(FP / (FP + TN))
        
        wandb.log({
            "F1 Score": F1,
            "FPR": FPR,
            "Recall": recall,
            "Precision": precision,
            "Confusion Matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_list,
                preds=y_predict_list,
                class_names=[str(i) for i in range(cnf_matrix.shape[0])]
            ),
            "ROC Curve": wandb.plot.roc_curve(
                y_true=y_list,
                y_probas=y_scores,
                labels=[str(i) for i in range(cnf_matrix.shape[0])]
            ),
            "Precision-Recall Curve": wandb.plot.pr_curve(
                y_true=y_list,
                y_probas=y_scores,
                labels=[str(i) for i in range(cnf_matrix.shape[0])]
            )
        })
        
        print("Confusion Matrix:\n", cnf_matrix)
        return F1, FP, FN, TP, TN, FPR, recall, precision


def main(config):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    random_seed(config.seed)
    path = os.path.join('data', config.chosen_data)
    # train set, number denotes each category has 750 samples
    train_X, train_Y, valid_X, valid_Y = prepro(d_path=path,
                                                length=2048,
                                                number=750,
                                                normal=False,
                                                enc=True,
                                                enc_step=28,
                                                snr=config.snr,
                                                property='Train',
                                                noise=config.add_noise
                                                )
    # test set, number denotes each category has 250 samples
    test_X, test_Y = prepro(d_path=path,
                            length=2048,
                            number=250,
                            normal=False,
                            enc=True,
                            enc_step=28,
                            snr=config.snr,
                            property='Test',
                            noise=config.add_noise
                            )
    train_X, valid_X, test_X = train_X[:, np.newaxis, :], valid_X[:, np.newaxis, :], test_X[:, np.newaxis, :]
    train_dataset = CustomTensorDataset(torch.tensor(train_X, dtype=torch.float), torch.tensor(train_Y))
    valid_dataset = CustomTensorDataset(torch.tensor(valid_X, dtype=torch.float), torch.tensor(valid_Y))
    test_dataset = CustomTensorDataset(torch.tensor(test_X, dtype=torch.float), torch.tensor(test_Y))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    data_loaders = {
        "train": train_loader,
        "validation": valid_loader
    }
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    
    for i in range(min(5, train_X.shape[0])):  
        plt.figure(figsize=(10, 4))
        plt.plot(train_X[i, 0, :])
        plt.title(f"Sample Train Signal {i} (SNR={config.snr}, Label={train_Y[i]})")
        plt.xlabel("Time Index")
        plt.ylabel("Signal Amplitude")
        wandb.log({f"Train Signal Sample {i}": wandb.Image(plt)})
        plt.close()
    
    for i in range(min(5, test_X.shape[0])):  
        plt.figure(figsize=(10, 4))
        plt.plot(test_X[i, 0, :])
        plt.title(f"Sample Test Signal {i} (SNR={config.snr}, Label={test_Y[i]})")
        plt.xlabel("Time Index")
        plt.ylabel("Signal Amplitude")
        wandb.log({f"Test Signal Sample {i}": wandb.Image(plt)})
        plt.close()
        
    for i in range(min(5, test_X.shape[0])):  
        plt.figure(figsize=(10, 4))
        for channel in range(test_X.shape[1]):
            plt.plot(test_X[i, channel, :], label=f"Channel {channel}")
        plt.title(f"Sample Test Signal {i} (SNR={config.snr}, Label={test_Y[i]})")
        plt.xlabel("Time Index")
        plt.ylabel("Signal Amplitude")
        plt.legend()
        wandb.log({f"Test Signal Multi-Channel Sample {i}": wandb.Image(plt)})
        plt.close()
        
    plt.figure(figsize=(10, 4))
    plt.hist(test_Y.astype(int), bins=int(np.max(test_Y)) + 1, edgecolor='black')
    plt.title("Distribution of Test Labels")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    wandb.log({"Test Label Distribution": wandb.Image(plt)})
    plt.close()
        
    train(config, data_loaders)
    inference(config, test_loader)
    print("Train X shape:", train_X.shape)
    print("Valid X shape:", valid_X.shape)
    print("Test X shape:", test_X.shape)



if __name__ == '__main__':
    # wandb initialization, you need to create a wandb account and enter the username in 'entity'
    wandb.init(project="bearinganomaly_Detection", entity="alinaderi119-upwork", mode='disabled')
    
    # WandB – Config is a variable that holds and saves hypermarkets and inputs
    config = wandb.config  # Initialize config
    config.no_cuda = True  # disables CUDA training
    config.log_interval = 200  # how many batches to wait before logging training status
    config.seed = 42  # random seed (default: 42)
    
    # Hyperparameters, lr and alpha need to fine-tune
    config.batch_size = 64  # input batch size for training (default: 64)
    config.epochs = 200  # number of epochs to train (default: 100)
    config.lr = 0.01  # learning rate (default: 0.001)
    config.alpha = 0.003 # scale factor alpha
    
    # noisy condition
    config.add_noise = True
    config.snr = -6
    
    # dataset and model
    config.chosen_data = '0HP'
    config.chosen_model = 'qdca_bigru_res'  # wdcnn or qcnn or qcnnresnet
    main(config)
    wandb.finish()





