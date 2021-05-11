import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from skimage import color
from skimage import io
from skimage.util import img_as_ubyte
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import pickle
import os


def tens2np(tens):
    return tens.detach().numpy()[0, ...].transpose((1, 2, 0))


def read_to_pil(img_path):
    out_img = Image.open(img_path)
    if len(np.asarray(out_img).shape) == 2:
        out_img = np.stack([np.asarray(out_img), np.asarray(out_img), np.asarray(out_img)], 2)
        out_img = Image.fromarray(out_img)
    return out_img


def pil_to_gray_np(img, Channels=3):
    lab_img = color.rgb2lab(np.asarray(img))
    if Channels == 1:
        grey_img = lab_img[:, :, 0]
        return grey_img
    elif Channels == 3:
        lab_img[:, :, 1:] = 0
        grey_img_3ch = color.lab2rgb(lab_img) * 255
        return grey_img_3ch


def resize_img(img, HW=(256, 256), resample=3):
    # resample=3 => BILINEAR
    return img.resize((HW[1], HW[0]), resample=resample)


def resize_large_img(img, large_dim=1280):
    if max(img.size) > large_dim:
        width = img.size[0]
        hight = img.size[1]
        if width > hight:
            new_resolution = (large_dim, int(large_dim * hight / width))
        else:
            new_resolution = (int(large_dim * width / hight), large_dim)
        return img.resize(new_resolution, resample=3)
    else:
        return img


def preprocess(pil_img, HW=(256, 256), resample=3):
    pil_img_rs = resize_img(pil_img, HW=HW, resample=resample)
    img_l_rs = pil_to_gray_np(pil_img_rs, Channels=1)
    tens_l_img_rs = torch.Tensor(img_l_rs)[None, None, :, :]

    return (pil_img.size[1], pil_img.size[0]), tens_l_img_rs


def parabola_fn(x):
    stiff_fn = lambda a: -4 * a ** 2 + 4 * a  # more desaturation
    smoother_fn = lambda b: -2.5 * b ** 2 + 2.5 * b + 0.375  # less desaturation
    return smoother_fn(x)


def desaturate(pil_img, out_img):
    np_grey_img = color.rgb2gray(np.asarray(pil_img))
    pixels_weights = np_grey_img[:, :]
    mapped_weights = parabola_fn(pixels_weights)
    hsv_img = color.rgb2hsv(out_img)
    hsv_img[:, :, 1] = np.multiply(hsv_img[:, :, 1], mapped_weights)  # de-saturated
    return color.hsv2rgb(hsv_img)


def postprocess(pil_img, np_ab_img, Desaturate=True):
    np_l_img = pil_to_gray_np(pil_img, Channels=1)
    np_l_img = np.expand_dims(np_l_img, axis=2)
    np_lab_img = np.concatenate([np_l_img, np_ab_img], axis=2)
    np_rgb_img = color.lab2rgb(np_lab_img)
    if Desaturate:
        return desaturate(pil_img, np_rgb_img)
    else:
        return np_rgb_img


def scaleback_ab_tens(HW_orig, out_ab, mode='bilinear'):
    HW = out_ab.shape[2:]
    # call resize function if needed
    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
    else:
        out_ab_orig = out_ab
    return out_ab_orig


def detector(img, save_path):
    if not os.path.exists(save_path):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
        cfg.MODEL.WEIGHTS = 'models/model_final_a3ec72.pkl'
        pred = DefaultPredictor(cfg)
        grey_img = pil_to_gray_np(img, Channels=3)
        outputs = pred(grey_img)
        pickle.dump(outputs, open(save_path, 'wb'))
    else:
        outputs = pickle.load(open(save_path, 'rb'))

    return outputs


def patchFullimg(fullimg, instance, pred_box, mask):
    startx, starty, endx, endy = pred_box
    mask2ch = np.stack([mask, mask], axis=2)
    (H, W, _) = fullimg.shape
    plecedinstance = np.array(fullimg)
    plecedinstance[starty:endy, startx:endx, :] = instance[:, :, :]
    mask2ch = np.array(mask2ch, dtype=bool)
    finalimg = np.multiply(plecedinstance, mask2ch) + np.multiply(fullimg, np.invert(mask2ch))
    return finalimg


def instancesMasks(fullimg_size, outputs):
    (W, H) = fullimg_size
    num_instances = len(outputs["instances"])
    masks = np.zeros(shape=(num_instances, H, W))
    for i in range(num_instances):
        currentmask = outputs["instances"].pred_masks[i, :, :].cpu()
        npcurrentmask = np.uint8(currentmask) * 255
        masks[i, :, :] = npcurrentmask
    return masks


def save_img(save_path, img):
    io.imsave(save_path + '.jpg', img_as_ubyte(img), quality=100)


def segnificat_bboexes_indices(img, bboxes, Threshold=0.002):
    # decreasing threshold keeps more bounding boxes
    W, H = img.size
    return remove_small_bboxes(bboxes, H * W * Threshold)


def remove_small_bboxes(bboxes, min_area):
    indices = []
    for i in range(len(bboxes)):
        startx, starty, endx, endy = bboxes[i]
        w = endx-startx
        h = endy-starty
        area = w*h
        if area > min_area:
            indices.append(i)
    return indices


