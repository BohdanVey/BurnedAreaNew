import tensorboard
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import cv2

writer = SummaryWriter("runs")
img_real = cv2.imread(
    'PSScene4Band_20170309_075258_0e3a_analytic_17fbc73d-00c4-4b08-8c02-ad3dfd5188ce/files/20170309_075258_0e3a_3B_AnalyticMS_clip.tif',
    cv2.IMREAD_COLOR)
img_real = cv2.normalize(img_real, dst=None, alpha=0, beta=2048, norm_type=cv2.NORM_MINMAX)
writer.add_image('Img test 2', img_real.transpose(2, 1, 0))
img_answer = cv2.imread(
    'PSScene4Band_20170309_075258_0e3a_analytic_17fbc73d-00c4-4b08-8c02-ad3dfd5188ce/files/20170309_075258_0e3a_3B_AnalyticMS_clip_mask.tif',
    cv2.IMREAD_COLOR)
img_answer = cv2.normalize(img_answer, dst=None, alpha=0, beta=2048, norm_type=cv2.NORM_MINMAX)
writer.add_image('Img test', img_answer.transpose(2, 1, 0))

writer.close()
print("READY")
