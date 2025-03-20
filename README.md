# radar_gesture_identify
**background:**

a project for radar gesture identify

# 1. Install

CUDA 12.6,pytorch==2.5.1

```python
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

d2l

```python
pip install d2l==0.17.6
```

opencv-python

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
```

# 2. how to use

## 2.1 Dataset Structure

```
📂 Dataset/                  # 数据集根目录
│── 📂 front/                # 正面视角
│   ├── 📂 dt/               # 俯视角
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── 📂 rt/               # 右视角
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── 📂 at_azimuth/       # 方位角
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── 📂 at_elevation/     # 俯仰角
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
```

if you want strengthen data,you can run 

```
python strengthen.py
```



## 2.2 data.txt

you need to run img_txt.py

```
python img_txt.py
```

generate a notebook for these Dataset 

```
./dt/0.jpg ./rt/0.jpg ./at_azimuth/0.jpg ./at_elevation/0.jpg 0
...
```

## 3. train and test

```
python train.py
```

It will generate a file log_gpu.dat and weights will generate in "./weights/gpu_best_weights.pth"

​							