import numpy as np
import scipy.linalg as la
from scipy.linalg import eig
from scipy import signal
import cv2

def bin2mat(file_name):
    """
    将二进制文件转换为矩阵形式的复数数据。

    该函数读取特定格式的二进制文件，并将其转换为一个复数矩阵，其中包含了天线接收的数据。
    这对于处理雷达或通信系统接收到的数据特别有用。

    参数:
    file_name: 字符串，指定要读取的二进制文件的文件名。

    返回值:
    adcData: 一个形状为(8, num_chirps // 2 * num_ADCSamples)的复数矩阵。
    """

    # 定义常量
    num_ADCSamples = 256
    num_RX = 4

    # 读取二进制数据
    adcData = np.fromfile(file_name, dtype=np.int16)

    # 计算文件大小和 chirp 数量
    filesize = adcData.shape[0]
    num_chirps = filesize // (2 * num_ADCSamples * num_RX)

    # 初始化临时数组来存储复数数据
    tmp = np.zeros(filesize // 2, dtype=np.complex64)

    # 将二进制数据转换为复数形式
    counter = 0
    for ii in range(0, filesize - 3, 4):
        tmp[counter] = adcData[ii] + 1j * adcData[ii + 2]
        tmp[counter + 1] = adcData[ii + 1] + 1j * adcData[ii + 3]
        counter += 2

    # 重新调整临时数组的形状
    tmp = tmp.reshape((num_ADCSamples * num_RX, num_chirps), order='F').T

    # 初始化adcData矩阵来存储按天线分组的复数数据
    adcData = np.zeros((num_RX, num_chirps * num_ADCSamples), dtype=np.complex64)
    for row in range(num_RX):
        for ii in range(num_chirps):
            adcData[row, ii * num_ADCSamples:(ii + 1) * num_ADCSamples] = tmp[ii, row * num_ADCSamples:(
                                                                                                                   row + 1) * num_ADCSamples]

    # 转置adcData矩阵
    tmp = adcData.T

    # 初始化adcData矩阵来存储最终的复数数据
    adcData = np.zeros((8, num_chirps // 2 * num_ADCSamples), dtype=np.complex64)
    for ii in range(4):
        RxTx = tmp[:, ii].reshape((num_ADCSamples, num_chirps), order='F')
        RxT1 = RxTx[:, ::2]
        RxT2 = RxTx[:, 1::2]
        adcData[2 * ii:2 * ii + 2, :] = np.vstack([RxT1.ravel(order='F'), RxT2.ravel(order='F')])

    return adcData


def at_process(adcData):
    c = 3e8
    f0 = 77e9
    lambda_ = c / f0

    num_ADCSamples = 256
    num_chirps = 64
    num_frame = 32
    num_RX = 8

    data = np.zeros((num_RX, num_ADCSamples, num_chirps, num_frame))
    for nn in range(num_RX):
        index = 0
        for ii in range(num_frame):
            for jj in range(num_chirps):
                data[nn, :, jj, ii] = adcData[nn, index * num_ADCSamples: (index + 1) * num_ADCSamples]
                index += 1

    interval = 3
    num_MTI = num_chirps - interval
    data_MTI = np.zeros((num_RX, num_ADCSamples, num_MTI, num_frame))
    for nn in range(num_RX):
        for ii in range(num_frame):
            for jj in range(num_MTI):
                data_MTI[nn, :, jj, ii] = data[nn, :, jj, ii] - data[nn, :, jj + interval, ii]

    d_base = 0.5 * lambda_
    space_num = 101
    angle = np.linspace(-50, 50, space_num)
    Pmusic1 = np.zeros(space_num)
    Pmusic2 = np.zeros(space_num)
    Pmusic_mn = np.zeros((space_num, num_frame * num_MTI))

    index = 0
    for ii in range(num_frame):
        for jj in range(num_MTI):
            Rxx = np.dot(data_MTI[2:7, :, jj, ii], data_MTI[2:7, :, jj, ii].T) / num_ADCSamples
            # Eigenvalue decomposition
            EV, D =eig(Rxx)
            EVA = np.diag(D)
            I = np.argsort(EVA)
            EV = EV[:, I[::-1]]  # Sort eigenvectors

            # Compute spatial spectrum for each angle
            for iang in range(space_num):
                phim = np.deg2rad(angle[iang])
                a = np.exp(-1j * 2 * np.pi * d_base / lambda_ * np.sin(phim))
                En = EV[:, 1:]  # Noise subspace
                Pmusic1[iang] = 1 / np.abs(np.dot(a.T, np.dot(En, En.T)).dot(a))

            Rxx = np.dot(data_MTI[3:8, :, jj, ii], data_MTI[3:8, :, jj, ii].T) / num_ADCSamples
            # Eigenvalue decomposition
            EV, D = eig(Rxx)
            EVA = np.diag(D)
            I = np.argsort(EVA)
            EV = EV[:, I[::-1]]  # Sort eigenvectors

            # Compute spatial spectrum for each angle
            for iang in range(space_num):
                phim = np.deg2rad(angle[iang])
                a = np.exp(-1j * 2 * np.pi * d_base / lambda_ * np.sin(phim))
                En = EV[:, 1:]  # Noise subspace
                Pmusic2[iang] = 1 / np.abs(np.dot(a.T, np.dot(En, En.T)).dot(a))

            index += 1
            Pmusic_abs = np.abs(Pmusic1 + Pmusic2)
            Pmmax = np.max(Pmusic_abs)
            Pmusic_mn[:, index] = Pmusic_abs / Pmmax  # Normalize

    AT_FW = signal.resample(Pmusic_mn, 32, axis=0)  # Resize using resample

    # Elevation angle
    space_num = 91
    angle = np.linspace(-45, 45, space_num)
    Pmusic = np.zeros(space_num)
    Pmusic_mn = np.zeros((space_num, num_frame * num_MTI))

    index = 0
    for ii in range(num_frame):
        for jj in range(num_MTI):
            Rxx = np.dot(data_MTI[:4, :, jj, ii], data_MTI[:4, :, jj, ii].T) / num_ADCSamples
            # Eigenvalue decomposition
            EV, D = eig(Rxx)
            EVA = np.diag(D)
            I = np.argsort(EVA)
            EV = EV[:, I[::-1]]  # Sort eigenvectors

            # Compute spatial spectrum for each angle
            for iang in range(space_num):
                phim = np.deg2rad(angle[iang])
                a = np.exp(-1j * 2 * np.pi * d_base / lambda_ * np.sin(phim))
                En = EV[:, 1:num_RX - 4]  # Noise subspace
                Pmusic[iang] = 1 / np.abs(np.dot(a.T, np.dot(En, En.T)).dot(a))

            index += 1
            Pmusic_abs = np.abs(Pmusic)
            Pmmax = np.max(Pmusic_abs)
            Pmusic_mn[:, index] = Pmusic_abs / Pmmax  # Normalize

    AT_FY = signal.resample(Pmusic_mn, 32, axis=0)  # Resize using resample

    return AT_FW, AT_FY

if __name__ == '__main__':
    file_name= 'sample.bin'
    adcData=bin2mat(file_name)
    print(adcData.shape)
    AT_FW,AT_FY=at_process(adcData)
    print(AT_FW.shape,AT_FY.shape)