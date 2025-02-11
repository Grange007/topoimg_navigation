import topic_names
import os
import sys
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from topic_names import DEPTH_CAMERA_RGB_TOPIC, DEPTH_CAMERA_DEPTH_TOPIC

ROOT_DIR = os.path.abspath("/home/grange/Program/naviwhere/visualnav-transformer/LoFTR")
sys.path.insert(0, ROOT_DIR)

import os
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.cm as cm
from cv_bridge import CvBridge

os.sys.path.append("/home/yzc/project/navigation/LoFTR/")  # Add the project directory
from src.loftr import LoFTR, default_cfg
from src.config.default import get_cfg_defaults
try:
    from demo.utils import (AverageTimer, VideoStreamer,
                            make_matching_plot_fast, make_matching_plot, frame2tensor)
except:
    raise ImportError("This demo requires utils.py from SuperGlue, please use run_demo.sh to start this script.")


import rospy

class SimilarityCompare:
    def __init__(self, args):
        self.rgb_image = None
        self.match_counter = []
        self.match_diff = []
        self.increase_flag = False
        self.decrease_flag = False
        self.reverse_flag = False
        self.stop_finding_flag = False
        self.folder_path = os.path.join("/home/yzc/project/navigation/visualnav-transformer/deployment/topomaps/rgbd", args.dir)
        self.match_num = 0
        self.counter = 0
        self.final_counter = 0
        self.twist_msg = Twist()
        self.twist_msg.angular.z = 0.35
        self.color_sub = rospy.Subscriber(DEPTH_CAMERA_RGB_TOPIC, Image, self.color_callback)
        self.timer = rospy.Timer(rospy.Duration(0.2), self.timer_callback)
        self.cmd_pub = rospy.Publisher("/mobile_base/commands/velocity", Twist, queue_size=1)
        self.bridge = CvBridge()
        self.display = None
        # Initialize LoFTR
        self.matcher = LoFTR(config=default_cfg)
        self.matcher.load_state_dict(torch.load("/home/yzc/project/navigation/LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
        self.matcher = self.matcher.eval().cuda()
    
    def color_callback(self, data):
        # print("image received")
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except:
            self.rgb_image = self.bridge.imgmsg_to_cv2(data)
            
    def generate_matches_LoFTR(self, image0, image1):
        vis_range = [0, 2000]
        device = 'cuda'
        img0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        img0 = frame2tensor(img0, device)
        img1 = frame2tensor(img1, device)
        img0 = torch.nn.functional.interpolate(img0, (320, 240), mode='bilinear', align_corners=False)
        img1 = torch.nn.functional.interpolate(img1, (320, 240), mode='bilinear', align_corners=False)
        batch = {'image0': img0, 'image1': img1}
        # batch={'image0':torch.from_numpy(image1/255.)[None,None].cuda().float().squeeze(),
                # 'image1':torch.from_numpy(image2/255.)[None,None].cuda().float().squeeze()}
        self.matcher(batch)
        conf_min = 0
        conf_max = 0
        with torch.no_grad():
            self.matcher(batch)    # batch = {'image0': img0, 'image1': img1}
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
            total_n_matches = len(batch['mkpts0_f'])
            if len(mconf) > 0:
                conf_vis_min = 0.
                conf_min = mconf.min()
                conf_max = mconf.max()
                mconf = (mconf - conf_vis_min) / (conf_max - conf_vis_min + 1e-5)
            alpha = 0
            color = cm.jet(mconf, alpha=alpha)

            text = [
                f'LoFTR',
                '# Matches (showing/total): {}/{}'.format(len(mkpts0), total_n_matches),
            ]
            small_text = [
                f'Showing matches from {vis_range[0]}:{vis_range[1]}',
                f'Confidence Range: {conf_min:.2f}:{conf_max:.2f}'
            ]

            image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            image0 = cv2.resize(image0, (640, 480))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image1 = cv2.resize(image1, (640, 480))
            # print(image0.shape, image1.shape)
            
            # out = make_matching_plot_fast(
            #     image0, image1, mkpts0, mkpts1, mkpts0, mkpts1, color, text,
            #     path=None, show_keypoints=False, small_text=small_text)
            cv2.imshow('LoFTR Matches', image1)
            key = chr(cv2.waitKey(1) & 0xFF)
            return np.hstack((mkpts0, mkpts1))

    # def generate_matches(self, image1, image2):
    #     config_path = '/home/grange/Program/naviwhere/visualnav-transformer/ml-aspanformer/configs/aspan/outdoor/aspan_test.py'
    #     weight_path = '/home/grange/Program/naviwhere/visualnav-transformer/ml-aspanformer/weights/indoor.ckpt'
    #     config = get_cfg_defaults()
    #     config.merge_from_file(config_path)
    #     _config = lower_config(config)
    #     matcher = ASpanFormer(config=_config['aspan'])
    #     state_dict = torch.load(weight_path, map_location='cpu')['state_dict']
    #     matcher.load_state_dict(state_dict,strict=False)
    #     matcher.cuda(),matcher.eval()
        
    #     img0,img1=image1,image2
    #     img0_g,img1_g=cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY),cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    #     # img0,img1=demo_utils.resize(img0,1024),demo_utils.resize(img1,1024)
    #     # img0_g,img1_g=demo_utils.resize(img0_g,1024),demo_utils.resize(img1_g,1024)
    #     data={'image0':torch.from_numpy(img0_g/255.)[None,None].cuda().float(),
    #             'image1':torch.from_numpy(img1_g/255.)[None,None].cuda().float()}
    #     with torch.no_grad():
    #         matcher(data,online_resize=True)
    #         corr0,corr1=data['mkpts0_f'].cpu().numpy(),data['mkpts1_f'].cpu().numpy()
    #     F_hat,mask_F=cv2.findFundamentalMat(corr0,corr1,method=cv2.FM_RANSAC,ransacReprojThreshold=1)
    #     if mask_F is not None:
    #         mask_F=mask_F[:,0].astype(bool)
    #     else:
    #         mask_F=np.zeros_like(corr0[:,0]).astype(bool)
    #     print(img0.shape,img1.shape)
    #     print(corr0)
    #     num_samples = 1000
    #     if len(corr0) > num_samples:
    #         indices = np.random.choice(len(corr0), num_samples, replace=False)
    #         corr0_sampled = corr0[indices]
    #         corr1_sampled = corr1[indices]
    #     else:
    #         corr0_sampled = corr0
    #         corr1_sampled = corr1
        
    #     display = demo_utils.draw_match(img0, img1, corr0_sampled, corr1_sampled)
    #     self.display = display

    #     matches = np.hstack((corr0_sampled, corr1_sampled))
    #     return matches

    def determine_destination(self):
        def less_than_all(list, num):
            for i in list:
                if i > num:
                    return False
            return True

        def greater_than_all(list, num):
            for i in list:
                if i < num:
                    return False
            return True
        
        def greater_than_part(list, num, tolerancce):
            false_cnt = 0
            for i in list:
                if i < num:
                    false_cnt += 1
            if (false_cnt > tolerancce):
                return False
            return True
        
        def less_than_part(list, num, tolerancce):
            false_cnt = 0
            for i in list:
                if i > num:
                    false_cnt += 1
            if (false_cnt > tolerancce):
                return False
            return True
        
        def judge_monotonicity(self):
            if (len(self.match_diff) < 7 or less_than_all(self.match_counter[-7: -1], 85)):
                return "invalid"
            else:
                if greater_than_part(self.match_diff[-7: -1], 0, 2):
                    return "increasing"
                elif less_than_part(self.match_diff[-7: -1], 0, 2):
                    return "decreasing"
                else:
                    return "nonmonotonic"
        
        monotonicity = judge_monotonicity(self)
        if (monotonicity == "invalid"):
            print("Invalid, keep moving")
            return "keep"
        
        if (monotonicity == "nonmonotonic"):
            print("Nonmonotonic, keep moving")
            return "keep"

        if (monotonicity == "increasing"):
            self.increase_flag = True
            print("Increasing, keep moving")
            return "keep"
        if (monotonicity == "decreasing"):
            if (self.increase_flag):
                print("Decreasing, stop")
                return "stop"
            else:
                print("Decreasing, reverse")
                return "reverse"
            
    def reverse_direction(self):
        if self.twist_msg.angular.z > 0:
            self.twist_msg.angular.z = -0.35
        else:
            self.twist_msg.angular.z = 0.35

    def timer_callback(self, event):
        if (self.rgb_image is not None):
            target_color_path = os.path.join(self.folder_path, "rgb_target.png")
            image1 = cv2.imread(target_color_path)
            image2 = self.rgb_image
            # print(image1.shape, image2.shape)
            matches = self.generate_matches_LoFTR(image1, image2)
            self.match_num = matches.shape[0]
            # cv2.imshow("Matches", self.display)
            # cv2.waitKey(1)
            print("match_num: ", self.match_num)
        self.max_counter = 100
        self.counter += 1
        
        if self.stop_finding_flag == True:
            self.final_counter += 1
            if (self.final_counter < 7):
                print("Stop finding flag is on, keep reversing")
            else:
                self.twist_msg = Twist()
                self.cmd_pub.publish(self.twist_msg)
                self.cmd_pub.publish(self.twist_msg)
                print("Stop finding flag is on, stop")
                plt.plot(self.match_counter)
                plt.savefig("match_num_test.png")
                rospy.signal_shutdown("find_pose")
        elif self.counter <= 3:
            print("Initial stage, keep moving")
        
        elif self.counter > 3 and self.counter < self.max_counter:
            self.match_counter.append(self.match_num)
            if (len(self.match_counter) == 1):
                self.match_diff.append(0)
            else:
                self.match_diff.append(self.match_counter[-1] - self.match_counter[-2])
            next_action = self.determine_destination()
            if (next_action == "keep"):
                pass
            elif (next_action == "reverse"):
                if (self.reverse_flag == False):
                    self.reverse_flag = True
                    self.reverse_direction()
            elif (next_action == "stop"):
                self.stop_finding_flag = True
                self.reverse_direction()
                
        else:
            if self.counter == self.max_counter:
                plt.plot(self.match_counter)
                plt.savefig("match_num_test.png")
            self.twist_msg = Twist()
        self.cmd_pub.publish(self.twist_msg)

def main():
    parser = argparse.ArgumentParser(description="publish the transform to target pose")

    parser.add_argument(
        "--model",
        "-m",
        default="vint",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )

    parser.add_argument(
        "--dir",
        "-d",
        default="rgbd",
        type=str,
        help="the directory of the rgbd images"
    )
    args = parser.parse_args()

    rospy.init_node('similarity_compare_node')
    sc = SimilarityCompare(args)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()