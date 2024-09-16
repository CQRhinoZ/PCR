import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from utils import FlowDataset
import torch.utils.data as data
import seaborn as sns
import matplotlib.pyplot as plt
from Parallel_Hybrid_Residual_Networks import PHRNet


'''
use temp code rmse 17.26
seed 6540
lr 7e-4
sqe_rate 2
'''

parser = argparse.ArgumentParser(description='Parameters for my model')
parser.add_argument('--sqe_rate', type=int, default=3, help='The squeeze rate of CovBlockAttentionNet')
parser.add_argument('--sqe_kernel_size', type=int, default=3, help='The kernel size of CovBlockAttentionNet')
parser.add_argument('--resnet_layers', type=int, default=4, help='Number of layers of week and day data in ResNet')
parser.add_argument('--res_kernel_size', type=int, default=3, help='ResUnit kernel size')
parser.add_argument('--ext_dim', type=int, default=28, help='The dim of external data')
parser.add_argument('--use_ext', type=bool, default=True, help='Whether use external data')
parser.add_argument('--trend_day', type=int, default=7, help='The length of trend and leak data')  # default 7
parser.add_argument('--current_day', type=int, default=5, help='The length of current data')  # default 4
parser.add_argument('--epochs', type=int, default=150, help='Epochs of train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size of dataloader')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of optimizer')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight_decay of optimizer')
parser.add_argument('--load', type=bool, default=False, help='Whether load checkpoint')
parser.add_argument('--check_point', type=int, default=False, help='Checkpoint')
parser.add_argument('--data_name', type=str, default='TaxiBJ', help='Train data name')
parser.add_argument('--gpu_num', type=int, default=4, help='Choose which GPU to use')
parser.add_argument('--test_num', type=str, default='10', help='Just for test')


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
    elif data_name == 'TaxiCQ':
        data_h = 20
        data_w = 25
    return data_h, data_w


def test():
    # model_path = "./model/model_TaxiCQ_parameter_test1.pkl"
    # model_path = "./pre_train_model/model_BikeNYC_RMSE4.99.pkl"
    model_path = "./pre_train_model/model_TaxiBJ_RMSE15.942.pkl"
    data_h, data_w = get_h_w()
    train_loader, val_loader, test_loader = load_data()
    model = PHRNet(sqe_rate=sqe_rate, sqe_kernel_size=sqe_kernel_size, resnet_layers=resnet_layers,
                      res_kernel_size=res_kernel_size,
                      data_h=data_h, data_w=data_w, use_ext=use_ext, trend_len=trend_len, current_len=current_len, ext_dim=ext_dim)
    model.load_state_dict(torch.load(model_path), False)
    model.to(device)
    criterion = nn.L1Loss()
    show_parameter(model)
    test_model(model, criterion, val_loader, test_loader)


def test_model(model, criterion, val_loader, test_loader):
    with torch.no_grad():
        model.eval()
        # val_RMSE, val_MAE, loss = cal_rmse(model, criterion, val_loader)
        # print('\n')
        # print('\tVALIDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t\tMAE:     {} \t loss:{}'.format(0, val_RMSE, val_MAE, loss))
        test_RMSE, test_MAE, loss = cal_rmse(model, criterion, test_loader)
        print('\tTESTDATE'.ljust(12), '\tEpoch:{}\t\tRMSE:     {} \t\tMAE:     {} \t loss:{}'.format(0, test_RMSE, test_MAE, loss))


def cal_rmse(model, criterion, data_loader):
    rmse_total_loss = []
    mae_total_loss = []
    model.eval()
    count_all_pred_val = 0
    min_rmse_loss = np.inf
    with torch.no_grad():
        for _, batch_data in enumerate(data_loader):
            x_data = batch_data[0].to(device)
            y_data = batch_data[1].to(device)
            ext_data = batch_data[2].to(device)
            day = batch_data[3]
            y_hat = model(x_data, ext_data)
            # y_hat, y_data = filter_res(y_hat, y_data, day)
            y_real = inverse_mmn(y_data).float()
            y_hat_real = inverse_mmn(y_hat).float()
            rmse_loss = (y_hat_real - y_real) ** 2
            mae_loss = abs(y_hat_real - y_real)
            if y_hat.shape[0] != 8:
                if rmse_loss.sum().item() < min_rmse_loss:
                    min_rmse_loss = rmse_loss.sum().item()
                    best_res = y_hat_real
                    best_true = y_real
            rmse_total_loss.append(rmse_loss.sum().item())
            mae_total_loss.append(mae_loss.sum().item())
            count_all_pred_val += y_hat_real.shape.numel()
    t = best_res - best_true
    for i in range(32):
        v_max = 1292
        show(best_true[i][0][0].cpu(), 'true_in' + str(i), v_max, 0)
        show(best_true[i][1][0].cpu(), 'true_out' + str(i), v_max, 0)
        show(best_res[i][0][0].cpu(), 'res_in' + str(i), v_max, 0)
        show(best_res[i][1][0].cpu(), 'res_out' + str(i), v_max, 0)
        show(t[i][0][0].cpu(), 'error_in' + str(i), 100, 0)
        show(t[i][1][0].cpu(), 'error_out' + str(i), 100, 0)
    RMSE = np.sqrt(np.array(rmse_total_loss).sum() / count_all_pred_val)
    MAE = np.array(mae_total_loss).sum() / count_all_pred_val
    return RMSE, MAE, 0


def filter_res(predict, true, days):
    import datetime
    predict = predict.cpu().detach().numpy()
    true = true.cpu().detach().numpy()
    days = days.cpu().detach().numpy()
    res_pre = []
    res_true = []
    for pre, tru, day in zip(predict, true, days):
        text = str(day)

        # 提取日期信息
        year = int(text[:4])
        month = int(text[4:6])
        day = int(text[6:8])

        date = datetime.date(year, month, day)
        # print(date.weekday())
        if date.weekday() == 0:
            res_pre.append(pre)
            res_true.append(tru)
    return torch.tensor(np.array(res_pre)), torch.tensor(np.array(res_true))


def inverse_mmn(img):
    data_max = 1292
    data_min = 0
    if data_name == 'TaxiBj':
        data_max = 1292
        data_min = 0
    elif data_name == 'BikeNYC':
        data_max = 267
        data_min = 0
    elif data_name == 'TaxiNYC':
        data_max = 1852
        data_min = 0
    elif data_name == 'TaxiCQ':
        data_max = 2000
        data_min = 0
    img = (img + 1.) / 2.
    img = 1. * img * (data_max - data_min) + data_min
    return img


def show_parameter(model):
    par = list(model.parameters())
    s = sum([np.prod(list(d.size())) for d in par])
    print("Parameter of 3DRTCN:", s)


def show(data, name, v_max, v_min):
    fig = plt.figure()
    # data = inverse_mmn(data)
    sns.heatmap(data=data, square=True, cmap='RdBu_r', vmin=v_min, vmax=v_max)
    plt.savefig('./picture/' + name + '.png')
    fig.clear()


if __name__ == '__main__':
    args = parser.parse_args()
    test_num = args.test_num
    gpu_num = args.gpu_num
    device = torch.device(gpu_num)
    torch.manual_seed(7687)
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
    test()