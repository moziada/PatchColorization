import cv2
import numpy as np
import os


def drawbboxes(input_dir, save_path, pil_img, pboxes, indices):
    cv2img = np.array(pil_img)
    cv2img = cv2img[:, :, ::-1].copy()
    boxcolor = (0, 0, 255)
    thickness = 1

    for i in indices:
        startx, starty, endx, endy = pboxes[i]
        topleft = (startx, starty)
        bottomright = (endx, endy)
        cv2img = cv2.rectangle(cv2img, topleft, bottomright, color=boxcolor, thickness=thickness)
    # cv2.imwrite(save_path+'_bboxes.jpg', img)
    cv2.imshow('image', cv2img)
