import os
import random
import sys
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from utils import FlowDataset
import torch.utils.data as data
from Parallel_Hybrid_Residual_Networks import PHRNet


parser = argparse.ArgumentParser(description='Parameters for my model')
parser.add_argument('--sqe_rate', type=int, default=3, help='not use The squeeze rate of CovBlockAttentionNet')
parser.add_argument('--sqe_kernel_size', type=int, default=2, help='not use The kernel size of CovBlockAttentionNet')
parser.add_argument('--resnet_layers', type=int, default=4, help='Number of layers of week and day data in ResNet')
parser.add_argument('--res_kernel_size', type=int, default=3, help='ResUnit kernel size')
parser.add_argument('--ext_dim', type=int, default=28, help='The dim of external data')
parser.add_argument('--use_ext', type=bool, default=True, help='Whether use external data')
parser.add_argument('--trend_day', type=int, default=7, help='The length of trend and leak data')
parser.add_argument('--current_day', type=int, default=5, help='The length of current data')
parser.add_argument('--epochs', type=int, default=150, help='Epochs of train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size of dataloader')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of optimizer')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight_decay of optimizer')
parser.add_argument('--load', type=bool, default=False, help='Whether load checkpoint')
parser.add_argument('--check_point', type=int, default=False, help='Checkpoint')
parser.add_argument('--data_name', type=str, default='TaxiBJ', help='Train data name')
parser.add_argument('--gpu_num', type=int, default=7, help='Choose which GPU to use')
parser.add_argument('--test_num', type=str, default='9', help='Just for test')


def load_data():
    train_dataset = FlowDataset(data_name, data_type='train', use_ext=use_ext)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = FlowDataset(data_name, data_type='val', use_ext=use_ext)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = FlowDataset(data_name, data_type='test', use_ext=use_ext)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_h_w():
    data_h = 0
    data_w = 0
    if data_name == 'TaxiBJ':
        data_h = 32
        data_w = 32
    elif data_name == 'BikeNYC':
        data_h = 16
        data_w = 8
    elif data_name == 'TaxiNYC':
        data_h = 15
        data_w = 5
    return data_h, data_w


def train(load_sign):
    train_loader, val_loader, test_loader = load_data()
    data_h, data_w = get_h_w()
    model = PHRNet(sqe_rate=sqe_rate, sqe_kernel_size=sqe_kernel_size, resnet_layers=resnet_layers, res_kernel_size=res_kernel_size,
                      data_h=data_h, data_w=data_w, use_ext=use_ext, trend_len=trend_len, current_len=current_len)
    model.to(device)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    show_parameter(model)
    if load_sign:
        model, optimizer = load_checkpoint(model, optimizer)
    train_len = len(train_loader)
    test_count = 1
    for i in range(epochs):
        for _, batch_data in enumerate(train_loader, 1):
            optimizer.zero_grad()
            x_data = batch_data[0]
            ext_data = batch_data[2]
            x_data = x_data.clone().detach()
            ext_data = ext_data.clone().detach()
            x_data = x_data.to(device)
            y_data = batch_data[1].to(device)
            ext_data = ext_data.to(device)
            y_hat = model(x_data, ext_data)
            loss = criterion(y_hat, y_data)
            loss.backward()
            optimizer.step()
            sys.stdout.write("\rTRAINDATE:  Epoch:{}\t\t loss:{} res train:{}".format(i, loss.item(), train_len - _))
        test_model(i, model, criterion, val_loader, test_loader)
        if test_count % 5 == 0:
            save_checkpoint(model, i, optimizer)
        test_count += 1
        stepLR.step()


def test_model(i, model, criterion, val_loader, test_loader):
    global min_rmse
    model_path = './model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with torch.no_grad():
        model.eval()
        val_RMSE, val_MAE, loss = cal_rmse(model, criterion, val_loader)
        print('\n')
        print('\tVALIDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t\tMAE:     {} \t loss:{}'.format(i, val_RMSE, val_MAE, loss))
        mess = '\tVALIDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t\tMAE:     {} \t loss:{}'.format(i, val_RMSE, val_MAE, loss)
        logger.info(str(mess))
        test_RMSE, test_MAE, loss = cal_rmse(model, criterion, test_loader)
        print('\tTESTDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t\tMAE:     {} \t loss:{}'.format(i, test_RMSE, test_MAE, loss))
        mess = '\tTESTDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t\tMAE:     {} \t loss:{}'.format(i, test_RMSE, test_MAE, loss)
        logger.info(str(mess))
        if test_RMSE < min_rmse:
            min_rmse = test_RMSE
            path = os.path.join(model_path, model_save.format(data_name, test_key))
            torch.save(model.state_dict(), path)


def cal_rmse(model, criterion, data_loader):
    rmse_total_loss = []
    mae_total_loss = []
    model.eval()
    count_all_pred_val = 0
    with torch.no_grad():
        for _, batch_data in enumerate(data_loader):
            x_data = batch_data[0].to(device)
            y_data = batch_data[1].to(device)
            ext_data = batch_data[2].to(device)
            y_hat = model(x_data, ext_data)
            criterion_loss = criterion(y_hat, y_data)
            y_real = inverse_mmn(y_data).float()
            y_hat_real = inverse_mmn(y_hat).float()
            rmse_loss = (y_hat_real - y_real) ** 2
            mae_loss = abs(y_hat_real - y_real)
            rmse_total_loss.append(rmse_loss.sum().item())
            mae_total_loss.append(mae_loss.sum().item())
            count_all_pred_val += y_hat_real.shape.numel()
    RMSE = np.sqrt(np.array(rmse_total_loss).sum() / count_all_pred_val)
    MAE = np.array(mae_total_loss).sum() / count_all_pred_val
    return RMSE, MAE, criterion_loss.item()


def inverse_mmn(img):
    data_max = 1292
    data_min = 0
    if data_name == 'TaxiBJ':
        data_max = 1292
        data_min = 0
    elif data_name == 'BikeNYC':
        data_max = 267
        data_min = 0
    elif data_name == 'TaxiNYC':
        data_max = 1852
        data_min = 0
    elif data_name == 'PoolBJ':
        data_max = 3082
        data_min = 0
    img = (img + 1.) / 2.
    img = 1. * img * (data_max - data_min) + data_min
    return img


def show_parameter(model):
    par = list(model.parameters())
    s = sum([np.prod(list(d.size())) for d in par])
    print("Parameter of PHRNet:", s)


def set_logger():
    global logger
    log_path = './run_log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(log_path, log_save.format(data_name, test_key)))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def save_checkpoint(model, epoch, optimizer):
    check_save_path = './checkpoints/'
    check_path = os.path.join(check_save_path, data_name)
    if not os.path.exists(check_save_path):
        os.makedirs(check_save_path)
    if not os.path.exists(check_path):
        os.makedirs(check_path)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    check_model_path = os.path.join(check_path, check_save.format(data_name, epoch, test_key))
    torch.save(checkpoint, check_model_path)


def load_checkpoint(model, optimizer):
    global epochs, lr
    check_path = './model/'
    check_model_path = os.path.join(check_path, check_save.format(data_name, check_point, test_key))
    train_state = torch.load(check_model_path)
    model.load_state_dict(train_state['model_state_dict'])
    optimizer.load_state_dict(train_state['optimizer_state_dict'])
    epochs = epochs - train_state['epoch'] - 1
    return model, optimizer


if __name__ == '__main__':
    min_rmse = 16.0001
    seed = 7687
    seed = random.randint(1, 10000)
    args = parser.parse_args()
    test_num = args.test_num
    gpu_num = args.gpu_num
    log_save = 'run_{}_log_{}.log'
    model_save = 'model_{}_parameter_{}.pkl'
    check_save = 'model_{}_{:03d}_{}.pt'
    test_key = 'test' + test_num
    #  global parameters
    device = torch.device(gpu_num)
    torch.manual_seed(seed)
    # model parameters
    sqe_rate = args.sqe_rate
    sqe_kernel_size = args.sqe_kernel_size
    resnet_layers = args.resnet_layers
    res_kernel_size = args.res_kernel_size
    use_ext = args.use_ext
    ext_dim = args.ext_dim
    trend_len = args.trend_day
    current_len = args.current_day
    # train parameters
    lr = args.lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    epochs = args.epochs
    load = args.load
    check_point = args.check_point
    data_name = args.data_name
    logger = logging.getLogger(__name__)
    set_logger()
    change = 'change trend day:{} current day:{}'.format(trend_len, current_len)
    print(args)
    print('single dilation cnn running on: {}, changes: {}, seed: {}'.format(test_key, change, seed))
    train(load)