"""
    对角高斯策略 pytorch 实现
    1. 生成一个对角高斯分布的动作
    2. 计算对角高斯分布的对数概率
    3. 计算对角高斯分布的熵
    4. 计算对角高斯分布的 KL 散度
"""

import torch 

def generate_action(mu, log_std, nosise=None, dim=1):
    """
    生成dim维度的对角高斯分布的动作
    :param mu: 均值
    :param log_std: 对数标准差
    :param noise: 噪声
    :param dim: 维度
    :return: 动作
    """
    std = torch.exp(log_std)
    if noise is None:
        noise = torch.randn_like(mu)
    action = mu + std * noise
    return action
    

def generate_diagonal_gaussian(mu, log_std, noise=None):
    """
    生成一个对角高斯分布的动作
    :param mu: 均值
    :param log_std: 对数标准差
    :param noise: 噪声
    :return: 动作
    """
    if noise is None:
        noise = torch.randn_like(mu)
    std = torch.exp(log_std)
    action = mu + std * noise
    return action

def calculate_diagonal_gaussian_log_prob(mu, log_std, action):
    """
    计算对角高斯分布的对数概率
    :param mu: 均值
    :param log_std: 对数标准差
    :param action: 动作
    :return: 对数概率
    """
    std = torch.exp(log_std)
    log_prob =  -0.5 *  ((action - mu) ** 2 / (std ** 2) + 2 * log_std + torch.log(torch.tensor(2 * torch.pi)))
    return log_prob.sum(dim=-1)