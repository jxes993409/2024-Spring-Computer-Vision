
import numpy as np
import cv2

class Joint_bilateral_filter(object):
	def __init__(self, sigma_s, sigma_r):
		self.sigma_r = sigma_r
		self.sigma_s = sigma_s
		self.wndw_size = 6 * sigma_s + 1
		self.pad_w = 3 * sigma_s
		# self.guidance_wndw = np.zeros((self.wndw_size, self.wndw_size), dtype = np.float64)
		# self.img_wndw = np.zeros((self.wndw_size, self.wndw_size), dtype = np.float64)
		self.spatial_kernel_table = self.Spatial_kernel()
		# self.range_kernel_table = np.zeros((self.wndw_size, self.wndw_size), dtype = np.float64)
	
	# def Range_kernel(self, is_gray):

	#     denominator = 2 * self.sigma_r ** 2

	#     if is_gray:
	#         center_pixel_table = np.full((self.wndw_size, self.wndw_size), self.guidance_wndw[self.pad_w][self.pad_w])
	#         self.range_kernel_table = np.exp(-(self.guidance_wndw - center_pixel_table) ** 2 / (denominator))
	#     else:
	#         center_pixel_table = np.full((self.wndw_size, self.wndw_size, 3), self.guidance_wndw[self.pad_w][self.pad_w])
	#         self.range_kernel_table = np.prod(np.exp(-(self.guidance_wndw - center_pixel_table) ** 2 / (denominator)), axis = 2)

	def Spatial_kernel(self):
		two_times_sigma_s_squared = 2 * self.sigma_s ** 2
		center_array = np.full((self.wndw_size, self.wndw_size, 2), self.pad_w, dtype = np.float64)
		index_array = np.array([[(i, j) for j in range(self.wndw_size)] for i in range(self.wndw_size)])
		spatial_kernel_table = np.prod(np.exp(-(index_array - center_array) ** 2 / (two_times_sigma_s_squared)), axis = 2)

		return spatial_kernel_table

		# upper_left = np.zeros((self.pad_w, self.pad_w), dtype = np.float64)

		# for i in range(self.pad_w):
		#     for j in range(i, self.pad_w):
		#         if j == i:
		#             continue
		#         upper_left[i][j] = np.exp(-((i - self.pad_w) ** 2 + (j - self.pad_w) ** 2) / (two_times_sigma_s_squared))
	  
		# upper_left = np.transpose(upper_left) + upper_left
		# for i in range(self.pad_w):
		#     upper_left[i][i] = np.exp(-2 * (i - self.pad_w) ** 2 / (two_times_sigma_s_squared))

		# bottom_left = np.flipud(upper_left)
		# upper_right = np.fliplr(upper_left)
		# bottom_right = np.flipud(upper_right)
		
		# for i in range(self.pad_w):
		#     value = np.exp(-(i - self.pad_w) ** 2 / (two_times_sigma_s_squared))
		#     self.spatial_kernel_table[i][self.pad_w] = value
		#     self.spatial_kernel_table[self.pad_w][i] = value
		#     self.spatial_kernel_table[self.pad_w][self.wndw_size - 1 - i] = value
		#     self.spatial_kernel_table[self.wndw_size - 1 - i][self.pad_w] = value

		# self.spatial_kernel_table[0: self.pad_w, 0: self.pad_w] = upper_left
		# self.spatial_kernel_table[0: self.pad_w, self.pad_w + 1: self.wndw_size] = upper_right
		# self.spatial_kernel_table[self.pad_w + 1: self.wndw_size, 0: self.pad_w] = bottom_left
		# self.spatial_kernel_table[self.pad_w + 1: self.wndw_size, self.pad_w + 1: self.wndw_size] = bottom_right

		# self.spatial_kernel_table[self.pad_w][self.pad_w] = 1.0

	def joint_bilateral_filter(self, img, guidance):
		BORDER_TYPE = cv2.BORDER_REFLECT
		padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
		padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

		is_gray = False
		if len(guidance.shape) == 2:
			is_gray = True

		height, width, channel = img.shape[0], img.shape[1], img.shape[2]

		output = np.zeros((height, width, channel), dtype = np.float64)
		
		padded_img_normalize = padded_img / 255
		padded_guidance_normalize = padded_guidance / 255

		start_pixel_x, start_pixel_y = self.pad_w, self.pad_w
		end_pixel_x, end_pixel_y = width + self.pad_w, height + self.pad_w

		for pixel_y in range(start_pixel_y, end_pixel_y):
			for pixel_x in range(start_pixel_x, end_pixel_x):
				guidance_wndw = padded_guidance_normalize[pixel_y - self.pad_w: pixel_y + self.pad_w + 1, pixel_x - self.pad_w: pixel_x + self.pad_w + 1]
				img_wndw = padded_img_normalize[pixel_y - self.pad_w: pixel_y + self.pad_w + 1, pixel_x - self.pad_w: pixel_x + self.pad_w + 1]

				two_times_sigma_r_squared = 2 * self.sigma_r ** 2
				if is_gray:
					center_pixel_table = np.full((self.wndw_size, self.wndw_size), guidance_wndw[self.pad_w][self.pad_w])
					range_kernel_table = np.exp(-(guidance_wndw - center_pixel_table) ** 2 / (two_times_sigma_r_squared))
				else:
					center_pixel_table = np.full((self.wndw_size, self.wndw_size, 3), guidance_wndw[self.pad_w][self.pad_w])
					range_kernel_table = np.prod(np.exp(-(guidance_wndw - center_pixel_table) ** 2 / (two_times_sigma_r_squared)), axis = 2)

				parameter = self.spatial_kernel_table * range_kernel_table
				parameter = np.repeat(parameter[:, :, np.newaxis], 3, axis = 2)
				numerator = np.sum(img_wndw * parameter, axis = (0, 1))
				denominator = np.sum(parameter[:, :, 0])
				
				output[pixel_y - self.pad_w][pixel_x - self.pad_w] = numerator / denominator
		
		return np.clip(output * 255, 0, 255).astype(np.uint8)