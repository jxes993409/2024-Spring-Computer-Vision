import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

np.random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)

    orb = cv2.ORB_create()

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        kp1, des1 = orb.detectAndCompute(im1,None)
        kp2, des2 = orb.detectAndCompute(im2,None)

        kp1_array, kp2_array = np.array([kp.pt for kp in kp1]).astype(np.int), np.array([kp.pt for kp in kp2]).astype(np.int)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        points1, points2 = kp1_array[[match.queryIdx for match in matches]], kp2_array[[match.trainIdx for match in matches]]

        # TODO: 2. apply RANSAC to choose best H
        probability = 0.99
        outliner_ratio = 0.5
        sample_points = 5
        threshold = 4.0
        min_outliner_count = len(matches)
        N = int(np.log((1 - probability)) / np.log((1 - (1 - outliner_ratio) ** sample_points)))

        for _ in range(N):
            random_idx = np.random.randint(0, len(matches), sample_points)
            H = solve_homography(points2[random_idx], points1[random_idx])

            one = np.ones((points2.shape[0], 1), dtype=np.int)
            points = np.concatenate((points2, one), axis=1)
            transformed_points = np.transpose(np.dot(H, np.transpose(points)))
            transformed_points /= transformed_points[:, 2].reshape(-1, 1)

            x = points1[:, 0]
            y = points1[:, 1]
            x_transformed = transformed_points[:, 0]
            y_transformed = transformed_points[:, 1]

            distance = np.sqrt((x - x_transformed) ** 2 + (y - y_transformed) ** 2)
            curr_outliner_count = len(np.where(distance > threshold)[0])

            if curr_outliner_count < min_outliner_count:
                RANSAC_best_H = H
                min_outliner_count = curr_outliner_count


        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, RANSAC_best_H)

        # TODO: 4. apply warping
        warping(im2, dst, last_best_H, 0, im2.shape[0], 0, w_max, 'b')

    return dst

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)