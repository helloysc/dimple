from utils import main
import json
import argparse

if __name__ == '__main__':
    import os
    from glob import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('-root', help='path or root of image')
    parser.add_argument('-size', type=int, default=416)
    parser.add_argument('-weight',
                        default='./weight/fold0-5-500.pth',
                        help='path of a weight file')
    parser.add_argument('-max_bbox_num', type=int, default=6, help='if too many bbox, judge NG')   # 如果红框有缺口，该怎么处理
    parser.add_argument('-prob_file', default='model.json', help='The prob of a single model')
    args = parser.parse_args()

    if os.path.isfile(args.root):
        args.img_pths = [args.root]
    else:
        # args.img_pths = glob(args.root + '/**/*.jpg', recursive=True)
        args.img_pths = glob(args.root + '/**/*grp*.jpg', recursive=True)

    dic = main(args)
    # for k, v in dic.items():
    #     print(k, round(v, 4))

    lines = [k + ' ' + str(round(v, 4)) + '\n' for k, v in dic.items()]
    with open(args.prob_file.replace('.json', '.txt'), 'w') as f:
        f.writelines(lines)

    json_write = json.dumps(dic, sort_keys=False, indent=4, separators=(',', ': '))
    with open(args.prob_file, 'w') as f:
        f.write(json_write)
