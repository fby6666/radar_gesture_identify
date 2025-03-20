from multiprocessing import Process, Queue
import os, time, random
from train import *
from torchvision import transforms
def img_exist(rt_path, dt_path, at_azimuth_path, at_elevation_path):
    if not os.path.exists(rt_path):
        print("请确保存在一张名为 rt 的图片用于测试！")
    if not os.path.exists(dt_path):
        print("请确保存在一张名为 dt 的图片用于测试！")
    if not os.path.exists(at_azimuth_path):
        print("请确保存在一张名为 at_azimuth 的图片用于测试！")
    if not os.path.exists(at_elevation_path):
        print("请确保存在一张名为 at_elevation 的图片用于测试！")

def collect_data(q):
    """
        Producer：采集数据并写入队列
        模拟从雷达采集数据，这里使用一张 dummy.jpg 图片，
        对每个视角都用相同图片进行模拟采集
    """
    trans=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    #21-right
    rt_path='./predict_data/rt/21.jpg'
    dt_path='./predict_data/dt/21.jpg'
    at_azimuth_path='./predict_data/at_azimuth/21.jpg'
    at_elevation_path='./predict_data/at_elevation/21.jpg'
    img_exist(rt_path=rt_path,dt_path=dt_path,at_azimuth_path=at_azimuth_path,at_elevation_path=at_elevation_path)
    while True:
        sample={}
        sample['rt']=trans(Image.open(rt_path)).unsqueeze(0)
        sample['dt']=trans(Image.open(dt_path)).unsqueeze(0)
        sample['at_azimuth']=trans(Image.open(at_azimuth_path)).unsqueeze(0)
        sample['at_elevation']=trans(Image.open(at_elevation_path)).unsqueeze(0)
        q.put(sample)
        print("采集数据写入队列")
        time.sleep(0.1)
def predict_data(q):
    net=MultiViewNet(4).cuda()
    net.load_state_dict(torch.load('weights/gpu_best_weights.pth'))
    net.eval()
    labels=["front","back","left","right"]
    while True:
        if not q.empty():
            sample=q.get()
            sample={key : value.cuda() for key, value in sample.items()}
            with torch.no_grad():
                output = net(sample)
                predicted = torch.argmax(output, dim=1)
                label=labels[predicted.cpu().numpy()[0]]
                print("推理结果：", label)
                time.sleep(0.1)
        else:
            print("队列为空")
            time.sleep(0.1)

if __name__ == '__main__':
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