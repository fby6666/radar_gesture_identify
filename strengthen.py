import os
import cv2
import numpy as np


# 加高斯噪声
def add_gaussian_noise(image):
    row, col = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


# 加椒盐噪声
def add_salt_and_pepper_noise(image, prob=0.01):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


# 加泊松噪声
def add_poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


# 随机裁切
def random_crop(image):
    height, width = image.shape
    new_height = int(height * 0.8)
    new_width = int(width * 0.8)
    y = np.random.randint(0, height - new_height)
    x = np.random.randint(0, width - new_width)
    cropped = image[y:y + new_height, x:x + new_width]
    cropped = cv2.resize(cropped, (width, height))
    return cropped


# 弹性拉伸
def elastic_stretch(image):
    stretch_ratio = np.random.choice([0.9, 1.1, 1.2])
    new_height = int(image.shape[0] * stretch_ratio)
    new_width = int(image.shape[1] * stretch_ratio)
    resized = cv2.resize(image, (new_width, new_height))
    resized = cv2.resize(resized, (image.shape[1], image.shape[0]))
    return resized


# 平移变换
def translation(image):
    height, width = image.shape
    dx = int(width * 0.1)
    dy = int(height * 0.1)
    directions = [(dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy)]
    direction = np.random.choice(len(directions))
    M = np.float32([[1, 0, directions[direction][0]], [0, 1, directions[direction][1]]])
    translated = cv2.warpAffine(image, M, (width, height))
    return translated


# 旋转变换
def rotation(image):
    height, width = image.shape
    methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR]
    angles = [10, -10, 30, -30]
    method = np.random.choice(methods)
    angle = np.random.choice(angles)
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (width, height), flags=method)
    return rotated


# 数据增强
def augment_data(data_dir):
    gesture_classes = ['left', 'right', 'front', 'back']
    time_graphs = ['rt', 'dt', 'at_elevation', 'at_azimuth']

    for gesture in gesture_classes:
        gesture_dir = os.path.join(data_dir, gesture)
        for time_graph in time_graphs:
            time_graph_dir = os.path.join(gesture_dir, time_graph)
            # 获取当前文件夹中最后一张图片的编号
            existing_images = [int(os.path.splitext(f)[0]) for f in os.listdir(time_graph_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if existing_images:
                last_index = max(existing_images)
            else:
                last_index = 0

            for img_name in os.listdir(time_graph_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(time_graph_dir, img_name)
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    # 加高斯噪声
                    noisy_gaussian_image = add_gaussian_noise(image)
                    last_index += 1
                    noisy_gaussian_img_name = os.path.join(time_graph_dir, f'{last_index}.jpg')
                    cv2.imwrite(noisy_gaussian_img_name, noisy_gaussian_image)

                    # 加椒盐噪声
                    noisy_salt_and_pepper_image = add_salt_and_pepper_noise(image)
                    last_index += 1
                    noisy_salt_and_pepper_img_name = os.path.join(time_graph_dir, f'{last_index}.jpg')
                    cv2.imwrite(noisy_salt_and_pepper_img_name, noisy_salt_and_pepper_image)

                    # 加泊松噪声
                    noisy_poisson_image = add_poisson_noise(image)
                    last_index += 1
                    noisy_poisson_img_name = os.path.join(time_graph_dir, f'{last_index}.jpg')
                    cv2.imwrite(noisy_poisson_img_name, noisy_poisson_image)

                    # 随机裁切
                    cropped_image = random_crop(image)
                    last_index += 1
                    cropped_img_name = os.path.join(time_graph_dir, f'{last_index}.jpg')
                    cv2.imwrite(cropped_img_name, cropped_image)

                    # 弹性拉伸
                    stretched_image = elastic_stretch(image)
                    last_index += 1
                    stretched_img_name = os.path.join(time_graph_dir, f'{last_index}.jpg')
                    cv2.imwrite(stretched_img_name, stretched_image)

                    # 平移变换
                    translated_image = translation(image)
                    last_index += 1
                    translated_img_name = os.path.join(time_graph_dir, f'{last_index}.jpg')
                    cv2.imwrite(translated_img_name, translated_image)

                    # 旋转变换
                    rotated_image = rotation(image)
                    last_index += 1
                    rotated_img_name = os.path.join(time_graph_dir, f'{last_index}.jpg')
                    cv2.imwrite(rotated_img_name, rotated_image)


if __name__ == "__main__":
    data_dir = "Dataset"
    augment_data(data_dir)
    print("数据增强完成！")



