from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import scipy.io as io
import cv2

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.float32):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1.)/2.
#    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = None
    if image_numpy.shape[2] == 1:
        image_numpy = np.reshape(image_numpy, (image_numpy.shape[0], image_numpy.shape[1])) * 255
        # image_pil = Image.fromarray(image_numpy).convert('L')
        # image_pil.save(image_path)
        cv2.imwrite(image_path, image_numpy)
    else:
        print('The image shape is not gray.')

    # elif image_numpy.shape[2] == 2:
    #     image_numpy = image_numpy * 65535
    #     image_add = np.mean(image_numpy, axis=2).reshape(image_numpy.shape[0], image_numpy.shape[1], 1)
    #     image_numpy = np.uint16(np.concatenate((image_numpy, image_add), axis=2))
    # else:
    #     image_numpy = image_numpy * 65535
    #     image_numpy = np.uint16(image_numpy)
    # cv2.imwrite(image_path, image_numpy)
    # image_pil = Image.fromarray(image_numpy)
    # image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings. Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" % (method.ljust(spacing), processFunc(str(getattr(object, method).__doc__))) for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# def generate_sr(img, mask):
#     img_1, img_2 = torch.unsqueeze(img[:, 0, :, :], dim=1), torch.unsqueeze(img[:, 1, :, :], dim=1)
#     # img_sr = img_1 - mask * img_2
#     img_sr = img_1 - (mask + 1) * (img_2 + 1)
#     img_sr = img_sr / torch.max(torch.max(img_sr, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
#     return img_sr


# def generate_sr(img, mask):
#     img_1, img_2 = torch.unsqueeze(img[:, 0, :, :], dim=1), torch.unsqueeze(img[:, 1, :, :], dim=1)
#     img_sr_1 = 0.5 * ((img_1 + 1) - (mask + 1) * (img_2 + 1))
#     img_sr_1[img_sr_1 < 0] = 0
#     img_sr_normal = img_sr_1 / torch.max(torch.max(img_sr_1, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
#     img_sr = 2 * img_sr_normal - 1
#     return img_sr

def generate_sr(img, mask):
    img_1, img_2 = torch.unsqueeze(img[:, 0, :, :], dim=1), torch.unsqueeze(img[:, 1, :, :], dim=1)
    img_1, img_2, mask = (img_1 + 1.0) / 2.0, (img_2 + 1.0) / 2.0, mask + 1
    img_sr_1 = img_1 - mask * img_2
    img_sr_normal = img_sr_1 / torch.max(torch.max(img_sr_1, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
    img_sr = 2 * (img_sr_normal - 0.5)
    return img_sr
