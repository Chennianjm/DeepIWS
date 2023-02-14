import os.path
import torchvision.transforms as transforms
from data.image_folder import make_dataset
import cv2
import numpy as np
import scipy.misc
import torch

class SingleDataset():
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)
        self.A_paths = make_dataset(self.dir_A)
        self.A_paths = sorted(self.A_paths)		
        self.transform = transforms.ToTensor()
        self.single = opt.single_image

    def __getitem__(self, index):
        A_path = self.A_paths[index]

        A_img = cv2.imread(A_path, -1)
        A_img = np.float32(A_img[:, :, ::-1]) / 65535.0
        A = torch.tensor(A_img)

        B = torch.unsqueeze(A[:, :, 2], dim=2)

        if self.single == 'A':
            A_original = torch.unsqueeze(A[:, :, 0], dim=2)
            A = torch.cat((A_original, A_original), dim=2)
        elif self.single == 'B':
            A_original = torch.unsqueeze(A[:, :, 1], dim=2)
            A = torch.cat((A_original, A_original), dim=2)
        else:
            A = A[:, :, 0:2]

        A_img = A.numpy()
        B_img = B.numpy()
       
        A_img = patch(A_img)
        B_img = patch(B_img)

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        A_img = (A_img - 0.5) * 2
        B_img = (B_img - 0.5) * 2

        return {'A': A_img, 'B': B_img, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'


def patch(A_img):
    h, w = A_img.shape[0], A_img.shape[1]
    H, W = int(h / 512) * 512, int(w / 512) * 512
    sh, sw = int((h - H) / 2), int((w - W) / 2)
    img = A_img[sh:sh + H, sw:sw + W, :]
    return img


