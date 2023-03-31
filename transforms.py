import numpy as np
import torch
from torchvision.transforms import functional as F
import random
from dataset.datasets import *


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Random_hflip(object):
    """随机左右翻转"""
    def __init__(self, p):
        self.p = p
        
    def __call__(self, image, landmarks):
        if random.random() < self.p:
            image1 = np.ascontiguousarray(np.flip(image, axis=[1]))
            
            new_landmarks = []
            landmark_xy = landmarks.reshape(-1,2)
            for (x, y) in landmark_xy:
                new_landmarks.append(1-x)
                new_landmarks.append(y)
        return image, np.asarray(landmarks)
    
class Random_Noise(object):
    """随机增加噪声"""
    def __init__(self, p, limit):
        self.p = p
        self.limit = limit
    
    def __call__(self, image, landmarks):
        if random.random() < self.p: 
            noise = np.random.uniform(0, self.limit, size=(image.shape[0], image.shape[1])) * 255
            if random.random()>0.5:
                image += noise[:, :, np.newaxis].astype(np.uint8)
            else:
                image -= noise[:, :, np.newaxis].astype(np.uint8)
            image = np.clip(image, 0, 255)
        return image, landmarks
    
class Random_Rotation(object):
    """随机旋转"""
    def __init__(self, p, max_alpha):
        self.p = p
        self.alpha = max_alpha
    def __call__(self, image, landmarks):
        if random.random() < self.p: 
            a = int(random.random() * self.alpha)
            rows, cols = image.shape[:2]
            # M是2*3 旋转+平移矩阵 2*2负责旋转 2*1负责平移
            # 所以将[x, y, 1]与之矩阵乘法 得出[x_new, y_new]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), a, 1)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            # 首先计算出旋转后图片最大外接矩形
            new_w = rows * sin + cols * cos
            new_h = rows * cos + cols * sin
            # 刚说明M最后一列是平移量，旋转不边，只需要平移量增减即可保证原图中心也是新图中心
            M[0, 2] += (new_w - cols) * 0.5
            M[1, 2] += (new_h - rows) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))
            landmarks = landmarks.reshape(-1,2)
            # 图像旋转, 旋转后大小为外接矩形大小
            img_rotated_by_alpha = cv2.warpAffine(image, M, (w, h))
            x_ = (w - cols)/2
            y_ = (h - rows)/2
            newlandmarks = []
            for x,y in landmarks:
                x = x*cols 
                y = y*rows
                newlandmarks.append(x)
                newlandmarks.append(y)
                
            pt = np.asarray(newlandmarks).reshape(-1,2)
            ones = np.ones((pt.shape[0], 1), dtype=float)
            pt = np.concatenate([pt, ones], axis=1).T
            new_pt = np.dot(M, pt).T
            new_pt = (new_pt * np.array([1/w,1/h])) .flatten()
            new_image = cv2.resize(img_rotated_by_alpha,(112,112))
            return new_image, new_pt
            # for x,y in new_pt.reshape(-1,2):
            #     cv2.circle(new_image,(int(x*112),int(y*112)), 1, (0, 0, 255))
                
            
                
            
            # cv2.imshow('0', new_image)
            # cv2.waitKey(0)
       

            
    
    
            
    
transform = Compose([Random_hflip(1.0),
                    #  Random_Noise(1.0, 0.2),
                     Random_Rotation(1.0, 45),
                     ToTensor()])
file_list = './data/test_data/list.txt'
wlfwdataset = WLFWDatasets(file_list,transforms=transform)
dataloader = DataLoader(wlfwdataset,
                        batch_size=1,
                        shuffle=True,
                        num_workers=0,
                        drop_last=False)
for img, landmark, attribute, euler_angle in dataloader:
    print("img shape", img.shape)
    print("landmark size", landmark.size())
    print("attrbute size", attribute)
    print("euler_angle", euler_angle.size())