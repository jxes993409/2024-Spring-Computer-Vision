import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    weights = [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.0, 0.9],
        [0.1, 0.4, 0.5],
        [0.8, 0.2, 0.0]
    ]
    JBF = Joint_bilateral_filter(2, 0.1)
    jbf_img = []
    cost = []
    guidances = []
    guidances.append(img_gray)
    bf_img = JBF.joint_bilateral_filter(img_rgb, img_rgb)

    for _, weight in enumerate(weights):
        guidances.append(weight[0] * img_rgb[:, :, 0] + weight[1] * img_rgb[:, :, 1] + weight[2] * img_rgb[:, :, 2])

    for _, guidance in enumerate(guidances):
        jbf_img.append(JBF.joint_bilateral_filter(img_rgb, guidance))
        cost.append(np.sum(np.abs(bf_img.astype(np.int16) - jbf_img[-1].astype(np.int16))))
        # cv2.imwrite("./jbf_{}.png".format(index), jbf_img[index])
        # cv2.imwrite("./guidance_{}.png".format(index), guidance)
    
    print(cost)


if __name__ == '__main__':
    main()