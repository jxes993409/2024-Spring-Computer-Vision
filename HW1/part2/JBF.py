
import numpy as np
import cv2

class Joint_bilateral_filter(object):
	def __init__(self, sigma_s, sigma_r):
		self.sigma_r = sigma_r
		self.sigma_s = sigma_s
		self.wndw_size = 6 * sigma_s + 1
		self.pad_w = 3 * sigma_s
		self.spatial_kernel_table = self.Spatial_kernel()

	def Spatial_kernel(self):
		two_times_sigma_s_squared = 2 * self.sigma_s ** 2
		center_array = np.full((self.wndw_size, self.wndw_size, 2), self.pad_w, dtype = np.float64)
		index_array = np.array([[(i, j) for j in range(self.wndw_size)] for i in range(self.wndw_size)])
		spatial_kernel_table = np.exp(-np.sum(np.square(index_array - center_array), axis = 2) / (two_times_sigma_s_squared))

		return spatial_kernel_table

	def joint_bilateral_filter(self, img, guidance):
		BORDER_TYPE = cv2.BORDER_REFLECT
		padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
		padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

		is_gray = False
		if len(guidance.shape) == 2:
			is_gray = True

		height, width, channel = img.shape[0], img.shape[1], img.shape[2]

		output = np.zeros((height, width, channel), dtype = np.float64)
		numerator = np.zeros((height, width, channel), dtype = np.float64)
		denominator = np.zeros((height, width, channel), dtype = np.float64)

		padded_img_normalize = padded_img / 255
		padded_guidance_normalize = padded_guidance / 255
		guidance_normalize = guidance / 255

		two_times_sigma_r_squared = 2 * self.sigma_r ** 2

		for i in range(self.wndw_size):
			for j in range(self.wndw_size):
				if is_gray:
					kernel_exp = np.square(guidance_normalize - padded_guidance_normalize[i: i + height, j: j + width])
				else:
					kernel_exp = np.sum(np.square(guidance_normalize - padded_guidance_normalize[i: i + height, j: j + width]), axis = 2)

				range_kernel_value = np.exp(-kernel_exp / two_times_sigma_r_squared)

				parameter = range_kernel_value * self.spatial_kernel_table[i, j]
				parameter = np.repeat(parameter[:, :, np.newaxis], 3, axis = 2)

				numerator = numerator + parameter * padded_img_normalize[i: i + height, j: j + width]
				denominator = denominator +  parameter

		output = numerator / denominator

		return np.clip(output * 255, 0, 255).astype(np.uint8)