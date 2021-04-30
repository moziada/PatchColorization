import os
from eccv16 import *
from util import *
from visuals import *

input_dir = 'examples'
output_folder = 'output'
imgName = '1533130050-cairo_main.jpg'
save_path = os.path.join(output_folder, imgName.split('.')[0])

pil_img = read_to_pil(os.path.join(input_dir, imgName))
pil_img = resize_large_img(pil_img)

outputs = detector(pil_img, save_path)
masks = instancesMasks(pil_img.size, outputs)
pred_bboxes = outputs["instances"].pred_boxes.to(torch.device('cpu')).tensor.numpy().astype(np.int32)
instances_indecies = segnificat_bboexes_indices(pil_img, pred_bboxes, Threshold=0.001)
#drawbboxes(input_dir, save_path, pil_img, pred_bboxes, instances_indecies)

colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_eccv16.cuda()
(HW_orig, tens_l_img_rs) = preprocess(pil_img, HW=(256, 256))
tens_ab_img_rs = colorizer_eccv16(tens_l_img_rs.cuda()).cpu()
tens_ab_img = scaleback_ab_tens(HW_orig, tens_ab_img_rs)
np_ab_img = tens2np(tens_ab_img)

for i in instances_indecies:
    (HW_orig_instance, tens_l_instance_rs) = preprocess(pil_img.crop(pred_bboxes[i]), HW=(256, 256))
    tens_ab_instance_rs = colorizer_eccv16(tens_l_instance_rs.cuda()).cpu()
    tens_ab_instance = scaleback_ab_tens(HW_orig_instance, tens_ab_instance_rs)
    np_ab_instance = tens2np(tens_ab_instance)
    np_ab_img = patchFullimg(np_ab_img, np_ab_instance, pred_bboxes[i], masks[i])

finalimg = postprocess(pil_img, np_ab_img, Desaturate=True)
save_img(save_path, finalimg)
