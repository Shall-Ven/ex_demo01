#打开调试日志
import logging
logging.basicConfig(level=logging.INFO,format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger=logging.getLogger(__name__)

import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data.dataloader as Data
import torch.nn as nn
from torchvision import models
from adversary import Adversary
#from adversarialbox.attacks.saliency import JSMA
from JSMA import JSMA
from pytorch import PytorchModel
import numpy as np
import cv2
#from tools import show_images_diff
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from past.utils import old_div
def show_images_diff(original_img, original_label, adversarial_img, adversarial_label):
    plt.figure()

    plt.subplot(131)
    plt.title('Original')
    plt.imshow(original_img)
    plt.axis('off')

    plt.subplot(132)
    plt.title('Adversarial')
    plt.imshow(adversarial_img)
    plt.axis('off')

    plt.subplot(133)
    plt.title('Adversarial-Original')
    difference = adversarial_img - original_img

    l0 = np.where(difference != 0)[0].shape[0]
    l2 = np.linalg.norm(difference)
    # print(difference)
    print("l0={} l2={}".format(l0, l2))

    # (-1,1)  -> (0,1)
    difference = old_div(difference, abs(difference).max()) / 2.0 + 0.5

    plt.imshow(difference, cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#定义被攻击的图片
image_path="e:\\cow.jpg"

# Define what device we are using
logging.info("CUDA Available: {}".format(torch.cuda.is_available()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#cv2默认读取格式为bgr bgr -> rgb   
orig = cv2.imread(image_path)[..., ::-1]
#转换成224*224
orig = cv2.resize(orig, (224, 224))
adv=None
img = orig.copy().astype(np.float32)

#图像数据标准化
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img /= 255.0
img = (img - mean) / std

#pytorch中图像格式为CHW  
#[224,224,3]->[3,224,224]
img = img.transpose(2, 0, 1)

img = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0)).cpu().numpy()


# Initialize the network
#Alexnet
model = models.alexnet(pretrained=True).to(device).eval()

#print(model)

#设置为不保存梯度值 自然也无法修改
for param in model.parameters():
    param.requires_grad = False

# advbox demo
m = PytorchModel(
    model, None,(-3, 3),
    channel_axis=1)

#实例化JSMA max_iter为最大迭代次数  theta为扰动系数 max_perturbations_per_pixel为单像素最大修改次数
attack = JSMA(m)
attack_config = {
        "max_iter": 2000,
        "fast":True,
        "theta": 0.3,
        "max_perturbations_per_pixel": 7,
        "fast":True,
        "two_pix":False
}


inputs=img
labels = None

print(inputs.shape)

adversary = Adversary(inputs, labels)

#定向攻击
tlabel = 538
adversary.set_target(is_targeted_attack=True, target_label=tlabel)


adversary = attack(adversary, **attack_config)

if adversary.is_successful():
    print(
        'attack success, adversarial_label=%d'
        % (adversary.adversarial_label))

    adv=adversary.adversarial_example[0]

else:
    print('attack failed')


print("jsma attack done")

#格式转换
adv = adv.transpose(1, 2, 0)
adv = (adv * std) + mean
adv = adv * 256.0
adv = np.clip(adv, 0, 255).astype(np.uint8)

show_images_diff(orig,adversary.original_label,adv,adversary.adversarial_label)