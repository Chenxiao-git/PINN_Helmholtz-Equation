# 基于物理信息的神经网络（PINNs）求解偏微分方程

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10%2B-red.svg)](https://pytorch.org/get-started/locally/)

本项目使用物理信息神经网络（PINNs）求解二维非齐次亥姆霍兹方程 (Helmholtz equation)，支持标准激活函数和自适应激活函数两种模式，实现完整的训练流程与结果可视化。

## 📋 问题描述

求解以下偏微分方程系统：

微分方程 $\Delta u + u = q(x,y)$ 定义在 
区域 $\Omega = [-1,1] \times [-1,1]$ 上，
边界条件为 $u|_{\partial\Omega} = 0$
