import torch
import os
from torch.utils.data import DataLoader
from torch import nn, optim
from model import *
from Dataset import *
def weight_init(m):
    if isinstance(m,nn.Conv2d):
        # 使用 He 初始化 (kaiming normal)，适合 ReLU 激活函数
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)  # 如果有偏置，初始化为 0
    elif isinstance(m,nn.Linear):
        # 使用 Xavier 初始化（也叫 Glorot 初始化），适合激活函数是 sigmoid 或 tanh
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)  # 如果有偏置，初始化为 0
    elif isinstance(m,nn.BatchNorm2d):
        # 批归一化层初始化
        nn.init.constant_(m.weight, 1)  # 权重初始化为 1
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # 偏置初始化为 0
def main():
    """
    进行模型的训练，计算测试集的准确率，打印出测试集前十个数据的测试结果
    :return:
    """
    torch.manual_seed(42)
    trans = transforms.Compose([
        transforms.Resize((224, 224)),  # 统一尺寸
        transforms.ToTensor(),  # 转换为 PyTorch Tensor
        # transforms.Normalize(mean=[...], std=[...])  # 归一化（如需）
    ])
    batch_size=16
    tarin_iter, test_iter = dataset_loader('./data_list.txt', batch_size=16,transform=trans)
    start_epoch,end_epoch=0,20
    lr=0.005
    net = MultiViewNet(num_classes=4).cuda()
    net.apply(weight_init)
    loss=nn.CrossEntropyLoss().cuda()
    accuracy_pre = -1
    optimizer=torch.optim.SGD(net.parameters(),lr=lr)
    for epoch in range(start_epoch,end_epoch):
        net.train()
        for idx , (batch_samples,batch_labels) in enumerate(tarin_iter):
            batch_samples={key:value.cuda() for key ,value in batch_samples.items()}
            y_hat=net(batch_samples)
            l=loss(y_hat,batch_labels.cuda())
            print('epoch :{}, batch :{}, loss :{:.4f}'.format(epoch, idx, l.sum().item()))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        # 每轮训练结束后进行测试
        net.eval()
        total_correct = 0
        total_num = 0
        with torch.no_grad():
            for batch_samples ,batch_labels in test_iter:
                batch_samples = {key: value.cuda() for key, value in batch_samples.items()}
                y_hat=net(batch_samples)
                correct=d2l.accuracy(y_hat,batch_labels.cuda())
                total_correct+=correct
                total_num+=batch_labels.numel()
            test_accuracy=total_correct/total_num
            print('epoch:', epoch, 'test_accuracy:', test_accuracy)
            if test_accuracy > accuracy_pre:
                accuracy_pre = test_accuracy
                ################################
                fd = open('log_gpu.dat', 'a+')
                fd.write('epoch {}'.format(epoch) + ': ' + str(test_accuracy) + '\n')
                fd.close()
                ################################
                os.makedirs('./weights',exist_ok=True)
                save_path = os.path.join('./weights', 'gpu_best_weights' +'.pth')
                torch.save(net.state_dict(), save_path)
    batch_samples,batch_labels=next(iter(test_iter))
    test_samples= {key:value[:10].cuda() for key ,value in batch_samples.items()}
    #进行预测
    net = MultiViewNet(4).cuda()
    net.load_state_dict(torch.load('weights/gpu_best_weights.pth'))
    net.eval()
    with torch.no_grad():
        output = net(test_samples)
        predicted_labels = torch.argmax(output, dim=1)
        print("预测结果：", predicted_labels.cpu().numpy())
        print("真实标签：", batch_labels[:10])

if __name__ == '__main__':
    main()

