from __future__ import print_function

import torch
import torch.nn as nn

from torch.optim import Adam

from torch.utils.tensorboard import SummaryWriter

from models.mobilenet import MobileNetV2
from models.resnet import ResNet18
from models.densenet import DenseNet121
from models.wideresnet import WideResNet40

from models.flows import Flows, LogLikelihood

from datasets.dataset import CustomDataset

from utils.config import get_args
from utils.general import  init_seeds, calc_stats, init_classifier, init_flows, save_flows, log_flows


def run(**kwargs):
    state = init_seeds({key: value for key, value  in kwargs.items()})
    dset = CustomDataset(state)

    cls = None 
    if state['classifier_model'] == 'wideresnet40':
        cls = WideResNet40(dset.num_classes) 
    elif state['classifier_model'] == 'mobilenetv2':
        cls = MobileNetV2(dset.num_classes)
    elif state['classifier_model'] == 'resnet18':
        cls = ResNet18(dset.num_classes)
    elif state['classifier_model'] == 'densenet121':
        cls = DenseNet121(dset.num_classes)
    state, cls, device = init_classifier(state, cls)

    nfs = Flows(state, cls.dims_out)
    criterion = LogLikelihood(state) 
    optimizer = Adam(nfs.parameters(), lr=state['learning_rate'], betas=tuple(state['betas']), eps=state['eps'], weight_decay=state['weight_decay'])
    state, nfs, optimizer = init_flows(state, nfs, device, optimizer)
    
    writer = SummaryWriter(log_dir=state['save_path'])

    for epoch in range(state['start'], state['epochs']):
        cls.eval()
        nfs.train()
        train_loss = 0.0
        for idx, (data, _ ) in enumerate(dset.train_loader):
            data = data.cuda()
            with torch.no_grad():
                _, features = cls(data)
            optimizer.zero_grad()
            z, log_jac_det = nfs(features.detach())
            _, nll = criterion(z, log_jac_det)
            train_loss += nll.item() 
            loss = nll / cls.dims_out
            loss.backward()
            nn.utils.clip_grad_norm_(nfs.parameters(), state['max_norm']) 
            optimizer.step()
            writer.add_scalar("loss/train", nll.item(), epoch+idx)
            train_ll.append(ll)
        
        state['train_loss'] = train_loss / len(dset.train_loader)

        nfs.eval()
        eval_loss = 0.0
        for idx, (data, _ ) in enumerate(dset.eval_loader):
            data = data.cuda()
            with torch.no_grad():
                _, features = cls(data)
                z, log_jac_det = nfs(features)
            _, nll = criterion(z, log_jac_det)
            eval_loss += nll.item()
            writer.add_scalar("loss/eval", nll.item(), epoch+idx)
            eval_ll.append(ll)
        
        state['eval_loss'] = eval_loss / len(dset.eval_loader)

        test_loss = 0.0
        for idx, (data, _) in enumerate(dset.test_loader):
            data = data.cuda()
            with torch.no_grad():
                _, features = cls(data)
                z, log_jac_det = nfs(features)
            _, nll = criterion(z, log_jac_det)
            test_loss += nll.item()
            writer.add_scalar("loss/test", nll.item(), epoch+idx)
            test_ll.append(ll)
            
        state['test_loss'] = test_loss / len(dset.test_loader)

        log_flows(state, epoch)
        save_flows(state, epoch, nfs, optimizer)

    writer.flush()
    writer.close()
    

def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = get_args()
    main(opt)
