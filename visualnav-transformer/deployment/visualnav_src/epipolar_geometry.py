import cv2
import numpy as np
import os, sys

ROOT_DIR = os.path.abspath("/home/grange/Program/naviwhere/visualnav-transformer/ml-aspanformer")
sys.path.insert(0, ROOT_DIR)

from src.ASpanFormer.aspanformer import ASpanFormer 
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
from demos import demo_utils 
import torch



IMAGE_PATH = '/home/grange/Program/naviwhere/visualnav-transformer/deployment/topomaps/rgbd/sqz_121_10'

def generate_matches(image1, image2):
    config_path = '/home/grange/Program/naviwhere/visualnav-transformer/ml-aspanformer/configs/aspan/outdoor/aspan_test.py'
    weight_path = '/home/grange/Program/naviwhere/visualnav-transformer/ml-aspanformer/weights/indoor.ckpt'
    config = get_cfg_defaults()
    config.merge_from_file(config_path)
    _config = lower_config(config)
    matcher = ASpanFormer(config=_config['aspan'])
    state_dict = torch.load(weight_path, map_location='cpu')['state_dict']
    matcher.load_state_dict(state_dict,strict=False)
    matcher.cuda(),matcher.eval()
    
    img0,img1=image1,image2
    img0_g,img1_g=cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY),cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # img0,img1=demo_utils.resize(img0,1024),demo_utils.resize(img1,1024)
    # img0_g,img1_g=demo_utils.resize(img0_g,1024),demo_utils.resize(img1_g,1024)
    data={'image0':torch.from_numpy(img0_g/255.)[None,None].cuda().float(),
            'image1':torch.from_numpy(img1_g/255.)[None,None].cuda().float()}
    with torch.no_grad():
        matcher(data,online_resize=True)
        corr0,corr1=data['mkpts0_f'].cpu().numpy(),data['mkpts1_f'].cpu().numpy()
    F_hat,mask_F=cv2.findFundamentalMat(corr0,corr1,method=cv2.FM_RANSAC,ransacReprojThreshold=1)
    if mask_F is not None:
        mask_F=mask_F[:,0].astype(bool)
    else:
        mask_F=np.zeros_like(corr0[:,0]).astype(bool)
    print(img0.shape,img1.shape)
    print(corr0)
    num_samples = 1000
    if len(corr0) > num_samples:
        indices = np.random.choice(len(corr0), num_samples, replace=False)
        corr0_sampled = corr0[indices]
        corr1_sampled = corr1[indices]
    else:
        corr0_sampled = corr0
        corr1_sampled = corr1
    
    display = demo_utils.draw_match(img0, img1, corr0_sampled, corr1_sampled)

    matches = np.hstack((corr0_sampled, corr1_sampled))
    return matches

K = np.array([[525.0, 0.0, 319.5],
              [0.0, 525.0, 239.5], 
              [0.0, 0.0, 1.0]])

# 读取图像
img1 = cv2.imread(os.path.join(IMAGE_PATH, 'rgb_source.png'))
img2 = cv2.imread(os.path.join(IMAGE_PATH, 'rgb_target.png'))

matches = generate_matches(img1, img2)

pts1 = matches[:, :2]
pts2 = matches[:, 2:]

# 计算基础矩阵
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# 从基础矩阵恢复旋转矩阵
E = K.T @ F @ K  # 本质矩阵
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

