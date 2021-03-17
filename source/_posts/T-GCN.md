
---
title: GCN交通领域论文汇总
mathjax: true
tags:
  - 论文笔记
  - 深度学习
  - 交通预测
categories:
  - 论文
---

<!--more-->
# GCN: A Temporal Graph ConvolutionalNetwork for Traffic Prediction
## methodology

### problem Definiton

1. road network: $G = (V,E)$,描述道路网络的拓扑结构 

   adjancency matrix:$A \in R^{N \times N}$ 描述邻接矩阵之间的关系

2. feature matrix: $X^{N \times P}$ 描述节点的特征数矩阵

   $X_t \in R^{N \times i}$  代表在i时刻的路段速度

问题：在G的拓扑结构的前提下，通过前一段时间的特征预测下一段时间的特征问题

$[X_{t+1}, ..., X_{t+T}] = f(G;(X_{t-n},...,X_{t-1}, X_t)$

## Overview

T-GCN = GCN + GRU

1. n time内的历史数据，作为输入，其中gcn捕捉道路得拓扑结构
2. 将空间特征输入GRU模型，获取时间特征，最后通过全连接层得到结果

![image-20200820005656918](https://i.loli.net/2021/03/17/yf5HOajZWvIRPEn.png)

## Methodology

### Spatial Dependence Modeling

$$f(X,A) = \sigma(\hat{A} Relu(\hat{A}XW_0)W_1)$$

$X$：特征矩阵

$\hat{A}$：空间特征提取过程

$W_0,W_1$: 两层的权重矩阵

### Temporal Dependence Modeling 

![image-20200820200012908](https://i.loli.net/2021/03/17/hALpW61igKvDxIE.png)

采用了GRU模型

$h_{t-1}$： 隐藏状态

$x_t$：t时刻的交通信息

$r_t$：重置门

$u_t$：更新门

$c_t$：当前时间交通信息的存储

### Temporal Graph Convolutional Network

![image-20200820200019712](https://i.loli.net/2021/03/17/HMeIcmKDiA6uCQ7.png)

$u_t = \sigma(W_u[f(A,X_t),h_{t-1}] + b_u)$

$r_t = \sigma(W_r[f(A,X_t),h_{t-1}] + br)$

 $c_t = tanh(W_c[f(A,X_t),(r_t*h_{t-1})] + b_c)$

$h_t = u_t* h_{t-1} + (1-u_t)*c_t$

### Loss Function

$loss = ||Y_t - \hat{Y_t}|| + \lambda L_{reg}$

为了防止过拟合引入了L2正则化项

# GCN-GAN: A Non-linear Temporal Link Prediction Model for Weighted Dynamic Networks

GCN-GAN网络处理对于权重动态网络的link-prediction任务

## Problem definition

$G = {G_1,G_2,...,G_\tau}$

$G_t = (V,E_t,W_t)$

其中研究对象的图均为无向图，且节点一致（不产生新节点）

$A_t \in R^{|V| \times |V|}$ 描述邻接矩阵的静态空间结构，同时令$(A_t)_{ij} = (A_t)_{ji} = W_t{i,j}$

**任务描述**：即通过$\{A_{\tau-l},A_{\tau-l+1},...,A_\tau\}$，预测$A_{\tau+1}$

$\tilde{A}_{\tau+1} = f(A_{\tau-1},A_{\tau-l+1}..., A_\tau)$

## Methodology

结构图

![image-20200906203119640](https://i.loli.net/2021/03/17/9NLz3ufYZOr5tnd.png)i) Graph Convolutional Network

ii) Long Short-Term Memory

iii) Generative Adversarial Nets(GAN)

1. 通过GCN获取本地的图的拓扑结构
2. 利用GCN获取得到的图信息，作为LSTM的输入
3. 通过全连接的判别网络进行动态网络的预测

### The GCN Hidden Layer

$X = GCN(Z,A) = f(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}ZW), A\in R^{N \times N}, Z\in R^{N \times M}$

其中$Z$为顶点的特征矩阵，$\hat{A} = A +I_N$ 

### The LSTM Hidden Layer

介绍LSTM的细节

### The Generative Adversarial Network

为了解决动态网络边缘权重的稀疏性和宽值范围问题

$min_Gmax_D (E_{x\sim p_{data}x}[logD(x)] +E_{z\sim p_z}[1-D(G_z)])$

其中$x$作为训练集的输入数据，$z$代表由一确定的可能性分布$p(z)$噪声生成

a) The Discriminative Network D

将$G$的输出$\tilde{A}_{\tau+1}$和真实数据$A_{\tau+1}$一维化后作为输入

$D(A') = (\sigma(a'W_h^D+ b_h^D)W_o^D + b_o^D)$

需要将$A_{\tau+1}$进行归一化

b) The Generative Network G

GCN中将$A^\tau_{\tau-1}$以及噪声Z作为输入，其中$Z \sim U(0,1)$

$\tilde{A}_{\tau+1} = G(Z,A^\tau_{\tau-l})$通过逆过程获取最终的预测结果

### Model Optimization

预训练过程更新G的相关参数

loss function: $min_{\theta_G}h(\theta_G;Z,A^{\tau-1}_{\tau-l-1},A_\tau) = ||A_\tau-G(Z,A^{\tau-1}_{\tau-l-1})||_F^2 + \lambda/2 ||\theta_G||^2_2$

利用梯度下降更新D的相关参数

loss function: $min_{\theta_D}h_D(\theta_D;D,A^{\tau-1}_{\tau-l-1},A_\tau) = E[D(A_\tau)] - E[D(G(Z,A^{\tau-1}_{\tau-l-1}))]$

对于D更新完之后，利用D的参数更新G

loss function: $min_{\theta_G}h(\theta_G;Z,A^{\tau-1}_{\tau-l-1}) = -E[D(G(Z,A^{\tau-1}_{\tau-l-1}))]$

# A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting

T-GCN的改进模型引入了注意力的相关机制

### Attention Model

采用 soft attention 模型，学习每个时刻交通信息的重要性，产生context vector 可以表示交通状况全局变化趋势

suppose：$x_i(i=1,2,...)$表示$i$时刻的时间序列

1. 通过CNNs/RNNs计算不同时刻的隐藏状态$h_i(i=1,2,...n)$
2. 设计scoring function 计算每个隐藏层的得分/权重
3. attention function· 计算context vector($C_t $)来描述交通全局变量信息
4. 使用context vector 获取最终的结果

$$e_i = w_{(2)}(w_{(1)}H + b_{(1)}) + b_{(2)} $$ 标注：公式有问题 正确表达应该是$E = w_{(2)}(w_{(1)}H + b_{(1)}) + b_{(2)}$其中$E = (e_1,e_2,...e_n)$

$\alpha_i = \frac{exp(e_i)}{\sum^n_{k=1}exp(e_k)}$

其中$\alpha_i$表示每个特征的权重，通过归一化计算获得

$C_t = \sum^n_{i=1}\alpha_i * h_i$

### AT3-GCN Model

![image-20200913210128356](https://i.loli.net/2021/03/17/8iZcpAYFknSaEKC.png)

GCN 计算得到的隐藏状态作为注意力模型的输入

# Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting
## Methodology

### Problem Definition

multi-step traffic forecasting problem

$\mathcal{X} = \{X_{:,0},X_{:,1},...X_{:,t},...\}$

$X_{:,t} = \{x_{1,t},x_{2,t},...x_{i,t}...,x_{N,t}\}^T$ 代表的是$t$时刻的特征合集

target：$\{X_{:,t+1},X_{:,t+2},...X_{:,t+\tau}\} = \mathcal{F}_\theta(X_{:,t},X_{:,t-1},....,X_{:,t-T+1};\mathcal{G})$

### Node Adaptive Parameter Learning

GCN可以很好的被近似为一阶Chebyshev 多项式展开，生成的高维的GCN可以表示为

$Z = (I_N + D^{-1/2}AD^{-1/2})X\Theta + b$

从单个Node的角度来说，GCN的本质可以看成从$X^i \in R^{1\times C} \to Z^i \in R^{1\times F}$，在所有节点共享 $\Theta$以及$b$

sharing parameter可以有效的减少参数数量。对于道路预测问题，道路之间的联系是一个动态的模式，所以传统的GCN不是最佳的解决方案，两个相邻的节点在不同的时刻由于不同的因素产生出来的模式也不尽相同。

对于$\Theta \in R^{N \times C \times F}$来说，当$N$过大时难以近似甚至会导致过拟合问题。

GCN 结合 节点自适应参数学习模型

NAPL学习两个小参数据矩阵

1. a node-embedding matrix $E_{\mathcal{G}} \in R^{N\times d}$, 其中$d$是代表embedding dimension，其中$d << N;2$

2. 权重池$W_{\mathcal{G}} \in R^{d\times C \times F}$,其中$\Theta = E_{\mathcal{G}}·W_{\mathcal{G}}$，从单个节点($i$)的角度来说，相当于是根据node embedding $E_{\mathcal{G}}^i$从整个共享权重池$W_{\mathcal{G}}$中提取参数$\Theta^i$

   $Z = (I_N + D^{-1/2}AD^{-1/2})XE_{\mathcal{G}}W_{\mathcal{G}} + E_{\mathcal{G}}b_{\mathcal{G}}$

   从改进公式来看，主要是将学习全局的过程，拆分为每个节点的embedding dimension内的参数的学习，能防止参数学习的过拟合问题等。

### Data Adaptive Graph Generation

针对于邻接矩阵$A$的定义，传统方式可以分为两种 1）  距离函数，根据节点之间的物理距离进行定义 2）根据计算节点之间的相似度进行定义

提出**Data Adaptive Graph Generation**，从数据推断隐藏的相互关系。

1. 对每个节点定义了一个可学习节点嵌入字典，$E_A \in R^{N\times d_e}$，其中$d_e$定义了节点嵌入的维数，根据节点的相似度定义

   $D^{-1/2}AD^{-1/2} = softmax(ReLU(E_A·E^T_A))$

2. $E_A$不断的更新，从而显示不同交通信息序列的隐藏关系，来得到图卷积的邻接关系

   $Z = (I_N + softmax(ReLU(E_A · E_A^T)))X\Theta$

时间部分采用了GRU的模型架构

# Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting

 ## Preliminaries

### Traffice Networks

$G = (V,E,A)$，$V$代表节点数目，$E$代表节点之间的连通性，$A \in \Bbb{R}^{N \times N}$ 代表G的邻接矩阵

### Traffic Flow Forecasting

$x^{c,i}_t \in \Bbb{R}$代表node-$i$，在时间$t$的第$c$个feature

 $X_t = (x_t^1,x_t^2,...,x_t^N)^T \in \Bbb{R}^{N\times F}$在$t$时刻的节点的所有特征

$\mathcal{X} = (X_1,X_2,...X_\tau)^T \in \Bbb{R}^{N\times F \times \tau}$代表在$\tau$时间切片上所有节点的所有特征

$y_t^i = x^{f,i}_t \in \Bbb{R}$代表节点$i$在时间$t$时刻的流量

**Problem**：给定$\mathcal{X}$，预测未来的流量序列$Y = (y^1,y_2,...y^N)^T \in \Bbb{R}^{N\times T_p}$

其中$y^i = (y_{\tau+1}^i,y_{\tau+2}^i,...,y_{\tau+T_p}^i) \in \Bbb{R}^{T_p}$

## Attention Based Spatial-Temporal Graph Convolutional Networks

![image-20200915095136063](https://i.loli.net/2021/03/17/L31qBfnYUjolwZc.png)

1. The recent segmen

   $\mathcal{X}_h = (X_{t_0-T_h + 1},X_{t_0-T_h + 2},....X_{t_0}) \in \Bbb{R}^{N\times F \times T_h}$

2. The daily-periodic segment 

   $\mathcal{X}_d = (X_{t_0 - (T_d/T_p)*q + 1},...,X_{t_0 - (T_d/T_p)*q + T_p},X_{t_0 - (T_d/T_p-1)*q + 1},...X_{t_0 - (T_d/T_p-1)*q + T_p}...X_{t_0 - q + 1},...X_{t_0 - q + T_p}) \in \Bbb{R}^{N\times F \times T_d}$

3. The weekly-periodic segment

   $\mathcal{X}_w = (X_{t_0 - 7*(T_w/T_p)*q + 1},...,X_{t_0 - 7*(T_w/T_p)*q + T_p},X_{t_0 -7* (T_w/T_p-1)*q + 1},...X_{t_0 - 7*(T_w/T_p-1)*q + T_p}...X_{t_0 -7* q + 1},...X_{t_0 -7* q + T_p}) \in \Bbb{R}^{N\times F \times T_w}$

![image-20200915100716655](https://i.loli.net/2021/03/17/wEbqIJLA4FclVZN.png)\

### Spatial-Temporal Attention

#### Spatial attention

空间注意力部分

$S = V_s · \sigma((\mathcal{X}_h^{(r-1)}W_1)W_2(W_3\mathcal{X}_h^{(r-1)})^T + b_s)$

$S'_{i,j} = \frac{exp(S_{i,j})}{\sum^N_{j=1}exp(S_{i,j})}$

其中$\mathcal{X}_h^{(r-1)} = (X_1,X_2,...X_{T_{r-1}}) \in \Bbb{R}^{N\times C_{r-1} \times T_{r-1}}$，是第$r$次的spatial-temporal block

$S_{i,j}$代表节点$i,j$之间的相关关系

$C_{r-1}$是在$r$层layer输入数据的维度，当$r=1$时，$C_0 = F$

$T_{r-1}$是在$r$层layer输入数据的时间维度，当$r=1$，$T_0=T_h$(recent component)

#### Temporal attention

时间注意力部分

$E = V_e·\sigma(((\mathcal{X}_h^{(r-1)})^TU_1)U_2(U_3\mathcal{X}_h^{(r-1)})+b_e)$

$E_{i,j}' = \frac{exp(E_{i,j})}{\sum_{j=1}^{T_{r-1}}exp(E_{i,j})}$

$E_{i,j}$代表时间$i,j$之间的相关关系

#### Spatial-Temporal Convolution

![image-20200915111531618](https://i.loli.net/2021/03/17/ywtljfLhHQoIxZU.png)

# Origin-Destination Matrix Prediction via Graph Convolution: a New Perspective of Passenger Demand Modeling

设计出租车的需求预测，即OD矩阵的预测

## Preliminaries

### Defintions

**Grid**：$G = \{g_1,g_2,...g_n\}$

**Time slot**：$\{Slot_1,Slot_2,...,Slot_t\}$

**OD Matrix**：$M \in \Bbb{N}^{G\times G}$，其中$m_{i,j}$代表从$g_i$到$g_j$的需求量

**target**：$M^{t+1} = f(M_1,M_2,...,M_{t})$

## Solution

![image-20200916150622021](https://i.loli.net/2021/03/17/v9fwdMchY2H5jPT.png)

### Grid Embedding 

备注：尝试用注意力机制进行替换

#### Geographical Neighborhood

地理邻居关系：$\Phi_i = \{g_j|dis(g_i,g_j)<L\}$，定义网格$g_i$的地理邻居节点集合

#### Semantic Neighborhood

语义上的网络而言，存在订单时候则代表两个网格之间存在语义之间的邻居关系

$\Omega^i_{t'} = \{g_j|m_{i,j}>0 || m_{j,i} > 0, m_{i,j} \in M_{t'}, m_{j,i} \in M_{t'}\}$

#### Pre-Weighted Aggregator for Grid Embedding 

此处是训练了一个聚合函数学习如何从网格的邻居节点选取特征信息

$v_i = \sigma(W·MEAN(\{v_i'\}\bigcap \{v_j', g_j \in N_i\}))$

该文提出了Pre-Weighted Aggregator，有选择性的选择更重要的邻居网格进行grid embedding

$r_{t'}^i = \sigma(W_g ·(f^i_{t'} + \sum_{g_j \in \Phi_i} \frac{dis(g_i,g_j)}{\sum dis(g_i,g_j)}f^j_{t'}))$  自身的特征+邻居节点的距离紧密度*邻居节点的特征

对于地理邻域的节点进行的属性聚合

$s_{t'}^i = \sigma(W_s ·(f^i_{t'} + \sum_{g_j \in \Omega_i} \frac{degree(g_i,g_j)}{\sum degree(g_i,g_j) + \epsilon}f^j_{t'}))$

对于语义上邻域的节点属性聚合

其中grid Embedding 使用的最终结果为：$v^i_{t'} = [r^i_{t'},s^i_{t'}]$

### Multi-Task Learning

![image-20200916220603834](https://i.loli.net/2021/03/17/e78cmXRQWJVxoP2.png)

#### Periodic-Skip LSTM

$h_t =  LSTM(x_t,h_{t-1})$

由于仅仅利用前一个小时训练得到的特征容易有误差，所以此处采用了skip的方式

$h_t = LSTM(v_t^i,h^i_{t-p})$

#### Main Task: Predicting the OD Matrix

decoder： $\hat{m}_{i,j} = (W_m h_t^i)^T h_t^j$

Loss Function: $\mathcal{L}_{ODMP} = \frac{1}{|M_{t+1}|\times N}\sum^N_{n=1}||M_{t+1} - \hat{M}_{t+1}||$

#### Two Subtasks: Predicting the In- and Out-Degress

将模型分割成out和in两种流量的模式

$\hat{p}_i = w_{in}^Th_t^i$

$\hat{q}_i = w_{out}^Th_t^i$

$\mathcal{L}_{IN} = \frac{1}{|G|\times N} \sum^N_{n=1} \sum_{g_i \in G}(p_{i,n},\hat{p}_{i,n})^2$

$\mathcal{L}_{OUT} = \frac{1}{|G|\times N} \sum^N_{n=1} \sum_{g_i \in G}(q_{i,n},\hat{q}_{i,n})^2$

#### Loss Function

$\mathcal{L}_{GEML} = \eta \mathcal{L}_{ODMP} + \eta_{in}\mathcal{L}_{IN} + \eta_{out}\mathcal{L}_{OUT}$

#### Optimization Strategy

采用了SGD的优化方法，其中采用了Adam的策略

# Representation Learning on Graphs: Methods and Applications

## Introduction

基于图机器学习的主要目标是将图上的信息整合入机器学习模型中，embedding 

即需要将图部分的信息编码成特征向量信息

学习嵌入点或者整张图在低维的映射$\Bbb{R}^d$，优化该映射从而能在embedding space反应最初图的结构信息

同之前图表示的区别在于，之前的图表示是利用图统计等相关手段进行图信息的预处理过程，而之后的图表示学习是将这一过程作为学习过程中的一部分

![image-20201012205948918](https://i.loli.net/2021/03/17/uGPD6YROhxrkVMW.png)

encode node  -> decode neighborhodd / decode node label

## Embedding nodes

### overview of approaches: An encoder-decoder perspective

$$ENC: V \to \Bbb{R}^d$$

$$DEC: \Bbb{R}^d \times \Bbb{R}^d \to \Bbb{R}^+$$

目标是优化编码器和解码器映射，以最大程度地减少此重构过程中的loss

$$DEC(ENC(v_i), ENC(v_j)) = DEC(z_i, z_j) \approx s_\mathcal{G}(v_i, v_j) $$

loss function： $\mathcal{L} = \sum_{(v_i,v_j) \in \mathcal{D}} l(DEC(z_i,z_j),s_\mathcal{G}(v_i,v_j))$

four methodological components

1. **A pairwise similarity function**
2. **An encoder function**
3. **A decoder function**
4. **A loss function**

### Shallow embedding approaches

大多数节点嵌入算法都依赖于我们所说的浅层嵌入。对于这些浅层嵌入方法，将节点映射到矢量嵌入的编码器功能只是一个“嵌入查找”

$ENC(v_i) = Zv_i$

#### Factorization-based approaches

基于矩阵分解方法进行降维

**Laplacian eigenmaps**： decoder：$DEC(z_i, z_j) = ||z_i - z_j||^2_2$

loss function : $\mathcal{L} = \sum_{(v_i,v_j) \in \mathcal{D}} DEC(z_i, z_j) · s_\mathcal{G}(v_i,v_j)$

**Inner-product methods**：decoder： $DEC(z_i, z_j) = z_i^Tz_j$

loss function: $\mathcal{L} = \sum_{(v_i, v_j)\in \mathcal{D}} ||DEC(z_i,z_j) - s_\mathcal{G}(v_i, v_j)||^2_2$

### Random walk approaches

### Generalized encoder-decoder architectures

#### Neighborhood autoencoder methods

![image-20201013151604019](https://i.loli.net/2021/03/17/UZJg6wP7DAC2FRa.png)

#### Neighborhood aggregation and convolutional encoders

![image-20201013151615833](https://i.loli.net/2021/03/17/SqGXO3hHIDiolA7.png)

## Embedding subgraphs

 **Goal**: encode a set of nodes and edges into a low-dimensional vector embedding 

![image-20201013151549677](https://i.loli.net/2021/03/17/m6BWVOoqGyQ51bL.png)

 ### Sets of node embeddings and convolutional approaches

#### Sum-based approaches

将节点嵌入的总和作为子图的表示

$z_\mathcal{S} = \sum_{v_i \in \mathcal{S}}z_i$

Dai等人提出的聚合方法

$$\eta_{i,j}^k = \sigma(W_\xi^k · COMBINE(x_i, AGGREGATE(\eta_{l,k}^{k-1}, \forall v_l \in \mathcal{N}(v_l) / v_j)))$$

$$z^i= \sigma(W_\mathcal{V}^k · COMBINE(x_i, AGGREGATE(\{\eta_{i,j}^K, \forall v_l \in \mathcal{N}(v_i)\})))$$

### Graph neutal networks

 $h_i^k =   \sum_{v_j \in \mathcal{N}(v_i)}h(h_j, x_i, x_j)$

Li et al 提出的GRU对于GNN的改造方式

$$h_i^k = GRU(h_i^{k-1}, \sum_{v_j \in \mathcal{N}(v_i)}Wh_j^{k-1})$$

# GATED GRAPH SEQUENCE NEURAL NETWORKS

将GNN的传播过程使用GRU进行替换，有空再看

# Graph2Seq：Graph to sequence Learning with attention-based neural networks

![image-20201013173839844](https://i.loli.net/2021/03/17/NUHZ4BquDxnhJFO.png)

## Graph-to-Sequence Model

### Node embedding generation



**key**： 将整个过程分为三个部分，Node Embedding, Graph Embedding, Node Attention to Sequence Decoder

可以借鉴的地方为采用了Node Attention方法，该部分将graph embedding得到的图表示向量重新映射至$(z_0,z_1, ... z_n)$的节点上，结合注意力的机制进行注意力的重新计算

其余后续补充

# Graph Attention Auto-Encoders

![image-20201013210906456](https://i.loli.net/2021/03/17/B5ea7xYCbOLTENM.png)

## Architecture

Graph attention auto-encoder 网络结构

### Encoder

$e_{ij}^{(k)} = Sigmod(v_s^{(k)^T}\sigma(W^{(k)}h_i^{(k-1)}) + v_r^{(k)^T}\sigma(W^{(k)}h_j^{(k-1)}))$

$\alpha_{ij}^{(k)} = \frac{exp(e_{ij}^{(k)})}{\sum_{l\in \mathcal{N}_i}exp(e_{il}^{(k)})}$

${\rm h}_i^{(k)} = \sum_{j\in \mathcal{N}_i}\alpha^{(k)}_{ij}\sigma(W^{(k)}h_j^{(k-1)})$

### Decoder

$\hat{\alpha}_{ij}^{(k)} = \frac{exp(\hat{e}_{ij}^{(k)})}{\sum_{l\in \mathcal{N}_i}exp(\hat{e}_{il}^{(k)})}$

$\hat{e}_{ij}^{(k)} = Sigmod(\hat{v}_s^{(k)^T}\sigma(\hat{W}^{(k)}\hat{h}_i^{(k-1)}) + \hat{v}_r^{(k)^T}\sigma(\hat{W}^{(k)}\hat{h}_j^{(k-1)}))$

${\rm \hat{h}}_i^{(k-1)} = \sum_{j\in \mathcal{N}_i}\hat{\alpha}^{(k)}_{ij}\sigma(\hat{W}^{(k)}\hat{h}_j^{(k)})$

### Loss Function

$\sum^N_{i=1} || \rm{x_i} -\hat{x}_i ||_2 -\lambda\sum_{j\in \mathcal{N}_i} log(\frac{1}{1+exp(-h^T_ih_j)})$

后面的损失项是为了防止由于节点特征的相似性问题，导致没有edge节点之间的连接，减小图的结构性损失



**总结**： 解码器部分，通过预测节点的特征数据，解码得到$E(v_i,v_j)$的数值，采用GAT相近的解码方式。

是否需要单独对解码器进行训练？

