from utils import segment
import json
import cv2
import argparse

if __name__ == '__main__':
    import os
    from glob import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('-root', help='path or root of image')
    parser.add_argument('-size', type=int, default=320)
    parser.add_argument('-weight',
                        default='./weight/fold0-5-500.pth',
                        help='path of a weight file')
    parser.add_argument('-max_bbox_num', type=int, default=10, help='if too many bbox, judge NG')
    parser.add_argument('-prob_file', default='model.json', help='The prob of a single model')
    args = parser.parse_args()

    if os.path.isfile(args.root):
        args.img_pths = [args.root]
    else:
        # args.img_pths = glob(args.root + '/**/*.jpg', recursive=True)
        args.img_pths = glob(args.root + '/**/*grp*.jpg', recursive=True)

    if not os.path.exists('./visualization/'):
        os.makedirs('./visualization/')
    for pth, img in segment(args):
        name = pth.split('/')[-1].split('\\')[-1]
        cv2.imencode('.jpg', img,)[1].tofile('./visualization/' + name)

    # segment(args)


