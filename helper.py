import os
import random
import numpy as np
import re
import torch
import matplotlib.pyplot as plt
# 绘图函数
def show_figure(fignum=1,loss=[],acc=[]):
    plt.figure(figsize=(8,3))
    plt.suptitle('Figure '+str(fignum))
    # 打印损失值
    plt.subplot(121)
    plt.ylabel('Loss')
    plt.plot(loss[0],label='Train Loss')
    plt.plot(loss[1],label='Valid Loss')
    plt.legend()
    plt.grid()
    # 打印正确率
    plt.subplot(122)
    plt.ylabel('ACC')
    plt.plot(acc[0],label='Train Acc')
    plt.plot(acc[1],label='Valid Acc')
    plt.legend()
    plt.grid()
    plt.show()


def read_arrays_from_file(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    
    matches = re.findall(r'\[([\s\d.eE+-]+)\]', data)    
    arrays = [np.fromstring(match, sep=' ') for match in matches]
    if all(len(arr) == len(arrays[0]) for arr in arrays):
        return np.array(arrays)
    else:
        return arrays  # 返回列表形式以支持不同长度的数组

"""
按照训练集和测试集 8 : 2的比例划分数据集
划分后的数据集位于原目录下train_data 和 test_data文件夹下
"""
def Split_Train_Test():
    
    H_file_path = 'H_conn_eigenvectors.txt'
    MDD_file_path = 'MDD_conn_eigenvectors.txt'
    H_arrays = read_arrays_from_file(H_file_path)
    MDD_arrays = read_arrays_from_file(MDD_file_path)
    ones = np.ones(shape=(len(H_arrays)))
    zeros = np.zeros(shape=(len(MDD_arrays)))
    datas = np.vstack((H_arrays, MDD_arrays))
    labels = np.hstack((ones,zeros))
    
    train_rate = 0.8
    train_picknumber = int(len(datas) * train_rate)
    total_numbers = list(range(len(datas)))
    random.shuffle(total_numbers)
    train_data = datas[total_numbers[0: train_picknumber]]
    train_label = labels[total_numbers[0: train_picknumber]]
    test_data = datas[total_numbers[train_picknumber:]]
    test_label = labels[total_numbers[train_picknumber:]]

    return torch.from_numpy(train_data).float(), torch.from_numpy(train_label).float(), torch.from_numpy(test_data).float(), torch.from_numpy(test_label).float()

# 定义测试函数
def test_epoch(net, data_loader, criterion,device):  
    net.eval() #指定当前模式为测试模式  
    test_batch_num = len(data_loader)  
    total_loss = 0  
    correct = 0  
    sample_num = 0  
    #指定不进行梯度变化  
    with torch.no_grad():  
        for batch_idx, (data, target) in enumerate(data_loader):  
            data = data.unsqueeze(1)
            data = data.to(device).float()  
            target = target.to(device).long()   
            output = net(data)  
            loss = criterion(output, target)  
            total_loss += loss.item( )  
            prediction = torch.argmax(output, 1)  
            correct += (prediction == target).sum().item()  
            sample_num += len(prediction)  
    loss = total_loss / test_batch_num  
    acc = correct / sample_num  
    return loss,acc

# 定义训练函数
def train_epoch(net, data_loader, optimizer,criterion,device):      
    net.train() #指定当前为训练模式  
    train_batch_num = len(data_loader) #记录共有多少个batch   
    total_1oss = 0 #记录Loss  
    correct = 0 #记录共有多少个样本被正确分类  
    sample_num = 0 #记录样本总数  
      
    #遍历每个batch进行训练  
    for batch_idx, (data,target) in enumerate (data_loader):  
        #将数据放入指定的device中  
        data = data.unsqueeze(1)
        data = data.to(device).float()  
        #将数据标签放入指定的device中  
        target = target.to(device).long()  
        #将当前梯度清零  
        optimizer.zero_grad()  
        #使用模型计算出结果
        output = net(data)  
        #计算损失  
        loss = criterion(output, target.squeeze())  
        #进行反向传播  
        loss.backward()  
        optimizer.step()  
        #累加loss  
        total_1oss += loss.item( )  
        #找出每个样本值最大的idx,即代表预测此数据属于哪个类别  
        prediction = torch.argmax(output, 1)
        correct += (prediction == target).sum().item()  
        #累加当前的样本总数  
        sample_num += len(prediction)  
    #计算平均oss与准确率  
    loss = total_1oss / train_batch_num  
    acc = correct / sample_num  
    return loss, acc  