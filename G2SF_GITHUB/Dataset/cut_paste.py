""" For the anomaly mask, there is only one connected component
"""
import numpy as np
import cv2 as cv


def cut_paste_scc(n_img, n_pcd, n_pcd_full, n_fg_mask, a_mask, a_img, a_pcd):
    intersect_mask = np.logical_and(n_fg_mask == 255, a_mask == 255)

    """ Most of part is within the foreground region
    """
    if np.sum(intersect_mask) > int(2 / 3 * np.sum(a_mask == 255)):
        n_img = paste_image_interpolation(n_img, a_mask, a_img, a_mask)
        n_pcd = paste_point_cloud(n_pcd, n_pcd_full, a_mask, a_pcd, a_mask)
        return n_img, n_pcd, a_mask
    else:
        result = shift_mask(n_fg_mask, a_mask)

        if not isinstance(result, tuple):
            n_img = paste_image_interpolation(n_img, result, a_img, result)
            n_pcd = paste_point_cloud(n_pcd, n_pcd_full, result, a_pcd, result)
            return n_img, n_pcd, result
        else:
            n_img[result[0] == 255, :] = a_img[result[1] == 255, :]
            n_img = paste_image_interpolation(n_img, result[0], a_img, result[1])
            n_pcd = paste_point_cloud(n_pcd, n_pcd_full, result[0], a_pcd, result[1])
            return n_img, n_pcd, result[0]


def paste_image_interpolation(n_img, n_mask, a_img, a_mask):
    new = n_img.copy().astype(np.float64)
    ratio = np.random.uniform(0, 0.5)
    new[n_mask == 255, :] = n_img[n_mask == 255, :] * ratio + a_img[a_mask == 255, :] * (1 - ratio)
    new[new >= 255] = 255
    new[new <= 0] = 0
    new = new.astype(np.uint8)
    return new


def shift_mask(fg_mask, a_mask):
    img_height, img_width = a_mask.shape
    contours, _ = cv.findContours(a_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    assert len(contours) == 1, "Only support a single connected component in anomaly mask to be shifted"

    M = cv.moments(contours[0])

    if M['m00'] == 0:
        return a_mask
    else:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        x_min, x_max = np.min(contours[0][:, :, 0]), np.max(contours[0][:, :, 0])
        y_min, y_max = np.min(contours[0][:, :, 1]), np.max(contours[0][:, :, 1])

        max_width, max_height = x_max - x_min, y_max - y_min
        center_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        center_mask[int(max_height / 2):img_height - int(max_height / 2), int(max_width / 2):img_width - int(max_width / 2)] = 255
        fg_mask = np.logical_and(fg_mask == 255, center_mask == 255)

        x_coord = np.arange(0, img_width)
        y_coord = np.arange(0, img_height)
        xx, yy = np.meshgrid(x_coord, y_coord)
        xx_fg = xx[fg_mask]
        yy_fg = yy[fg_mask]
        xx_yy_fg = np.stack([xx_fg, yy_fg], axis=-1)

        if xx_yy_fg.shape[0] == 0:
            return a_mask

        aug_mask_shifted_i = np.zeros((img_height, img_width), dtype=np.uint8)
        new_aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)

        idx = np.random.choice(np.arange(xx_yy_fg.shape[0]), 1)
        rand_xy = xx_yy_fg[idx]
        delta_x, delta_y = center_x - rand_xy[0, 0], center_y - rand_xy[0, 1]

        x_min, x_max = np.min(contours[0][:, :, 0]), np.max(contours[0][:, :, 0])
        y_min, y_max = np.min(contours[0][:, :, 1]), np.max(contours[0][:, :, 1])

        aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)
        aug_mask_i[y_min:y_max, x_min:x_max] = 255
        aug_mask_i = np.logical_and(a_mask == 255, aug_mask_i == 255)

        xx_ano, yy_ano = xx[aug_mask_i], yy[aug_mask_i]

        xx_ano_shifted = xx_ano - delta_x
        yy_ano_shifted = yy_ano - delta_y
        outer_points_x = np.logical_or(xx_ano_shifted < 0, xx_ano_shifted >= img_width)
        outer_points_y = np.logical_or(yy_ano_shifted < 0, yy_ano_shifted >= img_height)
        outer_points = np.logical_or(outer_points_x, outer_points_y)

        xx_ano_shifted = xx_ano_shifted[~outer_points]
        yy_ano_shifted = yy_ano_shifted[~outer_points]
        aug_mask_shifted_i[yy_ano_shifted, xx_ano_shifted] = 255

        xx_ano = xx_ano[~outer_points]
        yy_ano = yy_ano[~outer_points]

        new_aug_mask_i[yy_ano, xx_ano] = 255
        return aug_mask_shifted_i, new_aug_mask_i


def paste_point_cloud(pcd0, pcd0_full, mask0, pcd1, mask1, lower=2e-4, upper=2e-3):
    pts0 = pcd0[mask0 == 255, :]
    idx0 = np.nonzero(np.all(pts0 != 0, axis=1))[0]
    centroid0 = np.mean(pts0[idx0], axis=0)

    pts1 = pcd1[mask1 == 255, :]

    idx1 = np.nonzero(np.all(pts1 != 0, axis=1))[0]
    centroid1 = np.mean(pts1[idx1], axis=0)

    new = pts1.copy()
    """ we need to consider the range along z
    """
    nonzero = np.all(pcd0 != 0, axis=-1)
    z_range = pcd0[nonzero][:, -1].max() - pcd0[nonzero][:, -1].min()

    move = np.random.uniform(lower, upper)
    direction = np.random.normal()
    direction = 1 if direction > 0 else -1
    new[idx1] = pts1[idx1] - centroid1 + centroid0 + move * direction

    new_x, new_y = np.zeros((new.shape[0],)), np.zeros((new.shape[0],))
    new_x[idx1] = pcd0_full[mask0 == 255, 0][idx1]
    new_y[idx1] = pcd0_full[mask0 == 255, 1][idx1]
    anomaly = np.array([new_x, new_y, new[:, 2]]).T
    pcd0[mask0 == 255, :] = anomaly
    return pcd0
