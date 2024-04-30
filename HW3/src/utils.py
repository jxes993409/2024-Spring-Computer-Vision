import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    # H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.array([
         [u[0][0], u[0][1], 1, 0, 0, 0, -(u[0][0] * v[0][0]), -(u[0][1] * v[0][0]), -v[0][0]],
         [u[1][0], u[1][1], 1, 0, 0, 0, -(u[1][0] * v[1][0]), -(u[1][1] * v[1][0]), -v[1][0]],
         [u[2][0], u[2][1], 1, 0, 0, 0, -(u[2][0] * v[2][0]), -(u[2][1] * v[2][0]), -v[2][0]],
         [u[3][0], u[3][1], 1, 0, 0, 0, -(u[3][0] * v[3][0]), -(u[3][1] * v[3][0]), -v[3][0]],
         [0, 0, 0, u[0][0], u[0][1], 1, -(u[0][0] * v[0][1]), -(u[0][1] * v[0][1]), -v[0][1]],
         [0, 0, 0, u[1][0], u[1][1], 1, -(u[1][0] * v[1][1]), -(u[1][1] * v[1][1]), -v[1][1]],
         [0, 0, 0, u[2][0], u[2][1], 1, -(u[2][0] * v[2][1]), -(u[2][1] * v[2][1]), -v[2][1]],
         [0, 0, 0, u[3][0], u[3][1], 1, -(u[3][0] * v[3][1]), -(u[3][1] * v[3][1]), -v[3][1]]])

    # TODO: 2.solve H with A
    _, _, vt = np.linalg.svd(A)

    H = vt[-1].reshape(3, 3)
    
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x = np.arange(xmin, xmax, 1)
    y = np.arange(ymin, ymax, 1)
    one = np.ones(((xmax - xmin) * (ymax - ymin), 1), dtype=np.int)

    mesh_x, mesh_y = np.meshgrid(x, y)
    mesh_x, mesh_y = mesh_x.flatten(), mesh_y.flatten()
    mesh_matrix = np.concatenate((mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1), one), axis=1)
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    # dst = dst.reshape((h_dst * w_dst), ch)

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates

        # TODO: 6. assign to destination image with proper masking

        pass

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        trans_src = np.transpose(np.dot(H, np.transpose(mesh_matrix)))
        trans_src /= trans_src[:, 2].reshape(-1, 1)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = np.where((trans_src[:, 0] < 0) | (trans_src[:, 0] >= w_dst) | (trans_src[:, 1] < 0) | (trans_src[:, 1] >= h_dst))[0].tolist()

        # TODO: 5.filter the valid coordinates using previous obtained mask
        trans_src = np.delete(trans_src, mask, axis=0)

        # TODO: 6. assign to destination image using advanced array indicing
        dst_x, dst_y = np.floor(trans_src[:, 0]).astype(np.int), np.floor(trans_src[:, 1]).astype(np.int)
        x1, y1 = mesh_x, mesh_y
        x2, y2 = np.clip(x1 + 1, None, xmax - 1), np.clip(y1 + 1, None, ymax - 1)
        weight_x, weight_y = trans_src[:, 0] - dst_x, trans_src[:, 1] - dst_y

        dst[dst_y, dst_x] = ((1.0 - weight_x) * (1.0 - weight_y)).reshape(-1, 1) * src[y1, x1] + \
                            ((weight_x) * (1.0 - weight_y)).reshape(-1, 1) * src[y1, x2] + \
                            ((1.0 - weight_x) * (weight_y)).reshape(-1, 1) * src[y2, x1] + \
                            ((weight_x) * (weight_y)).reshape(-1, 1) * src[y2, x2]

    return dst