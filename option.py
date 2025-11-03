import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        # image directory settings
        self.parser.add_argument('--img1_root', type=str, default=None, help='directory for query image')
        self.parser.add_argument('--img2_root', type=str, default=None, help='directory for train image')
        self.parser.add_argument('--saveroot', type=str, default=None, help='save folder directory')

        # stitching settings
        self.parser.add_argument('--method', type=str, choices=['SIFT', 'AKAZE', 'ORB', 'CenSurE', 'BRISK'], default='SIFT', help='method of image feature point extraction')
        self.parser.add_argument('--ratio', type=float, default=0.75, help='ratio for filtering the correct paired keypoints')
        self.parser.add_argument('--threshold', type=float, default=4.0, help='RANSAC distance')

        # showing and saving settings
        self.parser.add_argument('--show_gray', type=str2bool, default=False, help='show horizontal stitching of grayscale images')
        self.parser.add_argument('--save_keypoints', type=str2bool, default=False, help='save the query and train images with keypoints')
        self.parser.add_argument('--save_match', type=str2bool, default=False, help='save the query and train images after keypoints matching')
        self.parser.add_argument('--save_filter', type=str2bool, default=False, help='save the image after filtering keypoints through the homography matrix operation')
        self.parser.add_argument('--direct_result', type=str2bool, default=True, help='save image stitching result with keypoints using homography matrix directly')
        self.parser.add_argument('--blending_result', type=str2bool, default=True, help='save image stitching result with keypoints after Blending')

        # video settings
        self.parser.add_argument('--video1_root', type=str, default=None, help='directory for query video')
        self.parser.add_argument('--video2_root', type=str, default=None, help='directory for train video')
        self.parser.add_argument('--videosaveroot', type=str, default=None, help='save folder directory for video')
        self.parser.add_argument('--blend', type=str2bool, default=True, help='use blend method to optimize image')

        # panorama settings
        self.parser.add_argument('--filepath', type=str, default=None, help='image file path')
        self.parser.add_argument('--savepath', type=str, default=None, help='stitched image save path')
        self.parser.add_argument('--savepointpath', type=str, default=None, help='stitched image save path')
        self.parser.add_argument('--savematchpath', type=str, default=None, help='stitched image save path')
        # teachering settings
        self.parser.add_argument('--teaching', type=str2bool, default=False, help='teaching mode where the number of input image must be 2')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt