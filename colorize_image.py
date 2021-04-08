import os

# -----------py files------------
from eccv16 import *
from util import *

input_dir = 'examples'
output_folder = 'output'
imgName = '1533130050-cairo_main.jpg'
save_path = os.path.join(output_folder, imgName.split('.')[0])

pil_img = read_to_pil(os.path.join(input_dir, imgName))
np_grey_img = pil_to_gray_np(pil_img)
(tens_l_img, tens_l_img_rs) = preprocess_img(pil_img, HW=(256, 256))

outputs = detector(np_grey_img, save_path)
masks = instancesMasks(np_grey_img.shape, outputs)
pred_bboxes = outputs["instances"].pred_boxes.to(torch.device('cpu')).tensor.numpy().astype(np.int32)

colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_eccv16.cuda()

tens_ab_img_rs = colorizer_eccv16(tens_l_img_rs.cuda()).cpu()
tens_ab_img = scaleback_ab_tens(tens_l_img, tens_ab_img_rs)
np_ab_img = tens2np(tens_ab_img)
# saveimg=postprocess_tens(tens_l_rs.detach().numpy()[0, ...].transpose((1, 2, 0)), colorizer_eccv16(tens_l_rs.cuda()).cpu().detach().numpy()[0, ...].transpose((1, 2, 0)))
# io.imsave('out_full_img.jpg', saveimg, quality=100)
for i in range(len(outputs["instances"])):
    # startx, starty, endx, endy = pred_bboxes[i]
    (tens_l_instance, tens_l_instance_rs) = preprocess_img(pil_img.crop(pred_bboxes[i]), HW=(256, 256))
    tens_ab_instance_rs = colorizer_eccv16(tens_l_instance_rs.cuda()).cpu()
    tens_ab_instance = scaleback_ab_tens(tens_l_instance, tens_ab_instance_rs)
    np_ab_instance = tens2np(tens_ab_instance)
    np_ab_img = patchFullimg(np_ab_img, np_ab_instance, pred_bboxes[i], masks[i])
    # io.imsave('out_instance'+str(i)+'.jpg', out_instance.detach().numpy()[0, ...].transpose((1, 2, 0)), quality=100)
    # saveimg=postprocess_tens(full_tens_l_orig.detach().numpy()[0, ...].transpose((1, 2, 0)), fullimg_ab)
    # io.imsave('patched_full_img'+str(i)+'.jpg', saveimg, quality=100)
    # io.imsave('patched_full_img' + str(i) + '.jpg', out_instance.detach().numpy()[0, ...].transpose((1, 2, 0)), quality=100)

np_l_img = tens2np(tens_l_img)
finalimg = postprocess(np_l_img, np_ab_img)
finalimg = desaturate(pil_img, finalimg)
# finalpilimg=Image.fromarray(finalimg.astype(np.uint8))
# finalpilimg.save(save_path + '.' + imgName.split('.')[1])
save_img(save_path, imgName, finalimg)
