import os, random, json, datetime
import torch
import gdown
import numpy as np
import pandas as pd

from pathlib import Path
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


############################## Runtime ##################################

def init_seeds(state):
    seed = int(time()) if state['seed'] == -1 else state['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True 
    state['seed'] = seed
    return state

def calc_stats(state, scores, label):
    state[f'{label}_scores_min'] = torch.min(scores)
    state[f'{label}_scores_mean'] = torch.mean(scores)
    state[f'{label}_scores_sigma'] = torch.std(scores)
    state[f'{label}_scores_max'] = torch.max(scores)
    return state

def get_best_metrics(csv_path, model='wideresnet40'):
    accuracy, precision, f1_score = 0.000, 0.000, 0.000
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        accuracy = df['accuracy'].max()
        f1_score = df['f1_score'].max()
        precision = df['precision'].max()

    return accuracy, precision, f1_score

def calc_measures(state, labels, predictions):
    state['aupr'] = average_precision_score(labels.cpu().numpy(), predictions.cpu().numpy())
    state['auroc'] = roc_auc_score(labels.cpu().numpy(), predictions.cpu().numpy())
    fpr, tpr, thresholds = roc_curve(labels.cpu().numpy(), predictions.cpu().numpy())
    state['fpr_95_tpr'] = fpr[np.where(tpr >= 0.95)[0][0]]
    state['threshold'] = thresholds[np.where(tpr >= 0.95)[0][0]]
    return state

def download_weights(path):
    weight_path = f'{path}/weights'
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    weights = [
        {'file_id': '1o800onuSobylqzsdKEOftYG_qWWXLLfC', 'file_name': 'cifar10_flows.pt'},
        {'file_id': '1nxTKxWUwj1cHGx55ubRZDIedZ1dxL-cO', 'file_name': 'cifar10_wideresnet40.pt'},
        {'file_id': '1oLXZPLvO4xx7b_X7VAkV0mVR8s04gX1G', 'file_name': 'cifar10_resnet18.pt'},
        {'file_id': '1oECfEFNqfFZWuzWsh0SI8f5csuzVFRAe', 'file_name': 'cifar10_mobilenetv2.pt'},
        {'file_id': '1oG0mz6bFV9GiBScIRcUHBTGj4pmwdq2q', 'file_name': 'cifar10_densenet121.pt'},
        {'file_id': '1o9Pd98404gwFME2IWklSguB7-_f60oIF', 'file_name': 'cifar100_flows.pt'},
        {'file_id': '1nzHhVw-hzmliSw4Zevyy3AncYYGU_j6C', 'file_name': 'cifar100_wideresnet40.pt'},
        ]
    for weight in weights:
        if not os.path.exists(f'{weight_path}/{weight["file_name"]}'):
            print(f"Downloading {weight['file_name']} weights from Google Drive...\n")
            gdown.download(f"https://drive.google.com/uc?id={weight['file_id']}", output=os.path.join(weight_path, weight['file_name']), quiet=False)
            print('Download completed!')
        else:
            print(f"{weight['file_name']} weights already downloaded and verified: {os.path.join(weight_path, weight['file_name'])}")

def increment_path(path, sep='-'):
    path = Path(path)  
    if path.exists():
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  
            if not os.path.exists(p):  
                break
        path = Path(p)

    print('\nResults will be saved to: %s' % path)

    return path

def convert_values_to_string(dct):
    for key, value in dct.items():
        if isinstance(value, dict):
            convert_values_to_string(value)
        elif not isinstance(value, (int, float)):
            dct[key] = str(value)
    return dct


############################## Testing ##################################

def init_classifier(state, cls, device=None):
    if not os.path.isfile(state['classifier_weight']):
        error_msg = f"Pre-trained [{state['classifier_model'].upper()}] not found: {state['classifier_weight']}"
        raise ValueError(error_msg)

    device = torch.device("cuda" if not state["no_cuda"] and torch.cuda.is_available() else "cpu") if device is None else device

    try:
        checkpoint = torch.load(state['classifier_weight'], map_location=device)
        model_name = checkpoint if state['classifier_pretrained'] else checkpoint['cls_state_dict']
        cls.load_state_dict(model_name)
        cls.to(device)
    except FileNotFoundError:
        raise ValueError(f"File not found: {state['classifier_weight']}")
    except (KeyError, RuntimeError):
        raise ValueError(f"Failed to load weights from {state['classifier_weight']}")

    print(f"\nPre-trained [{state['classifier_model'].upper()}] found: {state['classifier_weight']}")

    return state, cls, device

def init_test(state, nfs, device=None):
    if not os.path.isfile(state['flows_weight']):
        error_msg = f"Pre-trained [{state['flows_model'].upper()}] not found: {state['flows_weight']}"
        raise ValueError(error_msg)

    try:
        checkpoint = torch.load(state['flows_weight'], map_location=device)
        nfs.load_state_dict(checkpoint['nfs_state_dict'])
        nfs.to(device)
    except FileNotFoundError:
        raise ValueError(f"File not found: {state['flows_weight']}")
    except (KeyError, RuntimeError):
        raise ValueError(f"Failed to load weights from {state['flows_weight']}")

    print(f"Pre-trained [{state['flows_model'].upper()}] found: {state['flows_weight']}")

    save_path = increment_path(Path(f'{state["save_path"]}/test/{state["id_dataset"]}-{state["ood_dataset"]}'))
    print(f"Evaluating on [{state['id_dataset'].upper()}] as the in-distribution, and [{state['ood_dataset'].upper()}] as the out-of-distribution dataset ... \n")
    
    prepare_directory_and_save_state(state, save_path)

    return state, nfs

def prepare_directory_and_save_state(state, save_path):
    os.makedirs(save_path, exist_ok=True)
    state = convert_values_to_string(state)
    state['save_path'] = str(save_path)
    state['exp_date'] = datetime.datetime.now().strftime('%d.%m.%Y - %H:%M:%S')
    with open(os.path.join(save_path, 'exp.json'), 'w') as f:
        json.dump(state, f, indent=4)

def save_results(state, df_results):
    df_results.to_csv(os.path.join(state['save_path'], 'results.csv'), index=False)
    grouped = df_results.groupby(['method', 'cls', 'ind', 'ood'])
    mean_df = grouped.mean()
    std_df = grouped.std()

    std_df = std_df.rename(columns={col: f'{col}_std' for col in std_df.columns})
    result = pd.concat([mean_df, std_df], axis=1)
    for col in mean_df.columns:
        result[col] = result[col].map('{:.2f}'.format) + ' ± ' + result[col+'_std'].map('{:.2f}'.format)
    result = result.reset_index()
    averaged_df = result[['method', 'cls', 'ind', 'ood', 'fpr95', 'auroc', 'aupr', 'infr']]
    averaged_df.to_csv(os.path.join(state['save_path'], 'results_avg.csv'), index=False)



############################## Training ##################################

def init_flows(state, nfs, device, optimizer=None):
    state['start'], save_path = 0, ''
    if os.path.isfile(state['flows_weight']):
        try:
            checkpoint = torch.load(state['flows_weight'], map_location=device)
            nfs.load_state_dict(checkpoint['nfs_state_dict'])
            nfs.to(device)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                state['start'] = int(checkpoint['epoch']) + 1
                current_path = Path(state['flows_weight']).resolve()
                save_path = current_path.parent
        except:
            pass
        
        print(f"Pre-trained [{state['flows_model'].upper()}] found: {state['flows_weight']}")
    
    if state['use_cuda']:
        nfs = nfs.cuda()
    if optimizer is not None:
        if state['start'] == 0 and save_path == '':
            save_path = increment_path(Path(f'{state["save_path"]}/train/{state["id_dataset"]}-{state["flows_model"]}'))
            os.makedirs(save_path)
        if not os.path.exists(os.path.join(save_path, 'log.csv')):
            with open(os.path.join(save_path, 'log.csv'), 'w') as f:
                header = ('epoch, train_loss, eval_loss, test_loss\n')
                f.write(header)

        if os.path.exists(os.path.join(save_path, 'exp.json')):
            os.remove(os.path.join(save_path, 'exp.json')) 

        state = convert_values_to_string(state)
        state['save_path'] = str(save_path)
        state['exp_date'] = datetime.datetime.now().strftime('%d.%m.%Y - %H:%M:%S')
        with open(os.path.join(save_path, 'exp.json'), 'w') as f:
            json.dump(state, f, indent=4)

        print(f"Training [{state['flows_model'].upper()}] on [{state['id_dataset'].upper()}] from epoch [{int(state['start'])+1}] ... \n")

        return state, nfs, optimizer
    else:
        save_path = increment_path(Path(f'{state["save_path"]}/test/{state["id_dataset"]}-{state["ood_dataset"]}'))
        print(f"Evaluating on [{state['id_dataset'].upper()}] as the in-distribution, and [{state['ood_dataset'].upper()}] as the out-of-distribution dataset ... \n")
        os.makedirs(save_path)
        state = convert_values_to_string(state)
        state['save_path'] = str(save_path)
        state['exp_date'] = datetime.datetime.now().strftime('%d.%m.%Y - %H:%M:%S')
        with open(os.path.join(save_path, 'exp.json'), 'w') as f:
            json.dump(state, f, indent=4)

        return state, nfs

def log_flows(state, epoch):
    with open(os.path.join(state['save_path'], 'log.csv'), 'a') as f:
        result = (f"{epoch:3d},{state['train_loss']:4.0f},{state['eval_loss']:4.0f},{state['test_loss']:4.0f}\n")
        f.write(result)

    log = (f"Epoch {epoch:3d} | Train {state['train_loss']:4.0f} | Eval {state['eval_loss']:4.0f} | Test {state['test_loss']:4.0f}")    
    print(log)

def save_checkpoint(epoch, state, nfs, optimizer, filename):
    torch.save({
        'nfs_state_dict': nfs.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'seed': state['seed'],
    }, os.path.join(state['save_path'], filename))

def save_flows(state, epoch, nfs, optimizer):
    # Save the model of the current epoch
    if epoch % state['save_freq'] == 0:
        save_checkpoint(epoch, state, nfs, optimizer, 'last.pt')
        # If it's the third save, delete the second previous one
        if epoch > state['save_freq']:
            os.remove(os.path.join(state['save_path'], f'epoch-{epoch-2*state["save_freq"]}.pt'))
        # If it's the last epoch, save with the specific epoch name
        if epoch == state['epochs']:
            save_checkpoint(epoch, state, nfs, optimizer, f'epoch-{epoch}.pt')

        


