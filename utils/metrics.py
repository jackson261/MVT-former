import numpy as np
import torch

def RSE(pred, true):
    
    return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - true.mean(dim=0)) ** 2))

def CORR(pred, true):
    
    u = ((true - true.mean(dim=0)) * (pred - pred.mean(dim=0))).sum(dim=0)
    d = torch.sqrt(((true - true.mean(dim=0)) ** 2 * (pred - pred.mean(dim=0)) ** 2).sum(dim=0))
    return (u / d).mean(dim=-1)

def MAE(pred, true):
    
    return torch.mean(torch.abs(pred - true))


def MSE(pred, true):
    
    return torch.mean((pred - true) ** 2)


def RMSE(pred, true):
    
    return torch.sqrt(MSE(pred, true))


def MAPE(pred, true):
    
    epsilon = 1e-5  # 避免除以零
    return torch.mean(torch.abs((pred - true) / (true + epsilon)))


def MSPE(pred, true):
   
    epsilon = 1e-5  # 避免除以零
    return torch.mean(((pred - true) / (true + epsilon)) ** 2)


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
