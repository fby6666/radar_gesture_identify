from multiprocessing import Process, Queue
import os, time, random

from pyparsing import empty

from train import *
from torchvision import transforms
import matlab.engine
import cv2
import numpy as np
def img_exist(rt_path, dt_path, at_azimuth_path, at_elevation_path):
    if not os.path.exists(rt_path):
        print("请确保存在一张名为 rt 的图片用于测试！")
    if not os.path.exists(dt_path):
        print("请确保存在一张名为 dt 的图片用于测试！")
    if not os.path.exists(at_azimuth_path):
        print("请确保存在一张名为 at_azimuth 的图片用于测试！")
    if not os.path.exists(at_elevation_path):
        print("请确保存在一张名为 at_elevation 的图片用于测试！")
def normalize_and_convert(data):
    # 将数据标准化到0-255范围，并转换为8位无符号整型
    data_normalized = cv2.normalize(data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return data_normalized
def signal_process():
    """
        信号处理程序，用于处理 Ctrl+C 退出信号
    """
    # 启动MATLAB引擎
    eng = matlab.engine.start_matlab()

    # 定义路径和文件名
    path = r'D:\File\Code\project\radar_gesture_identify\sample.bin'
    # 设置工作目录
    eng.cd(r'./process')
    # 调用MATLAB函数
    RT, VT, AT_FW, AT_FY = eng.signal_process(path, nargout=4)

    # 将MATLAB数组转换为NumPy数组
    RT = np.array(RT)
    VT = np.array(VT)
    AT_FW = np.array(AT_FW)
    AT_FY = np.array(AT_FY)

    # print(RT.shape, VT.shape, AT_FW.shape, AT_FY.shape)
    # 确保数据类型为8位无符号整型，并且值在0到255之间
    RT = normalize_and_convert(RT)
    VT = normalize_and_convert(VT)
    AT_FW = normalize_and_convert(AT_FW)
    AT_FY = normalize_and_convert(AT_FY)
    rt_path = './predict_data/rt/1.jpg'
    dt_path = './predict_data/dt/1.jpg'
    at_azimuth_path = './predict_data/at_azimuth/1.jpg'
    at_elevation_path = './predict_data/at_elevation/1.jpg'
    cv2.imwrite(rt_path, RT)
    cv2.imwrite(dt_path, VT)
    cv2.imwrite(at_azimuth_path, AT_FW)
    cv2.imwrite(at_elevation_path, AT_FY)
    # 关闭MATLAB引擎
    eng.quit()
    return 1
def collect_data(q):
    """
        Producer：采集数据并写入队列
        模拟从雷达采集数据，这里使用一张 dummy.jpg 图片，
        对每个视角都用相同图片进行模拟采集
    """
    flag=0
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # 21-right
    rt_path = './predict_data/rt/1.jpg'
    dt_path = './predict_data/dt/1.jpg'
    at_azimuth_path = './predict_data/at_azimuth/1.jpg'
    at_elevation_path = './predict_data/at_elevation/1.jpg'
    sample = {}
    while True:
        if flag ==0:
            flag=signal_process()
        if flag == 1:
            img_exist(rt_path=rt_path, dt_path=dt_path, at_azimuth_path=at_azimuth_path,
                      at_elevation_path=at_elevation_path)
            sample['rt'] = trans(Image.open(rt_path)).unsqueeze(0)
            sample['dt'] = trans(Image.open(dt_path)).unsqueeze(0)
            sample['at_azimuth'] = trans(Image.open(at_azimuth_path)).unsqueeze(0)
            sample['at_elevation'] = trans(Image.open(at_elevation_path)).unsqueeze(0)
            q.put(sample)
            print("采集数据写入队列")
            flag=0
            time.sleep(0.1)

def predict_data(q):
    net=MultiViewNet(num_classes=8).cuda()
    net.load_state_dict(torch.load('weights/gpu_best_weights.pth',weights_only=True))
    print("模型加载完毕*********************")
    net.eval()
    print("进入测试模式*********************")
    labels=["front","back","left","right","up","down","clockwise","counterclockwise"]
    last_label=""
    empty_flag=False
    while True:
        if not q.empty():
            sample=q.get()
            if sample is None:  # 终止信号
                print("推理进程结束")
                break
            sample={key : value.cuda() for key, value in sample.items()}
            with torch.no_grad():
                output = net(sample)
                predicted = torch.argmax(output, dim=1)
                label=labels[predicted.cpu().numpy()[0]]
                print("推理结果：", label)
                if label!=last_label:
                    print("可以使用新的手势,并且可以使用下个手势")
                    last_label = label
                time.sleep(0.1)
        else:
            if not empty_flag:
                print("正在处理信号中****************")
                print("处理信号过程较为缓慢****************")
                empty_flag = True  # 设为 True，避免重复打印
            time.sleep(0.1)

if __name__ == '__main__':
    # signal_process()
    q = Queue()
    p_collect=Process(target=collect_data,args=(q,))
    p_predict=Process(target=predict_data,args=(q,))
    p_collect.start()
    p_predict.start()
    try:
        # 主进程等待，直到按 Ctrl+C 终止
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("终止实时推理")
        # 退出前可发送结束信号给队列，例如 q.put(None)
        p_collect.terminate()
        p_predict.terminate()
        p_collect.join()
        p_predict.join()