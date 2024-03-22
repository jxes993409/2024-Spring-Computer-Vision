import numpy as np
import cv2

class Difference_of_Gaussian(object):
	def __init__(self, threshold):
		self.threshold = threshold
		self.sigma = 2**(1/4)
		self.num_octaves = 2
		self.num_DoG_images_per_octave = 4
		self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

	def get_keypoints(self, image):
		### TODO ####
		gaussian_images = []
		dog_images = []

		for octave_index in range(self.num_octaves):
			gaussian_images_per_octave = []
			gaussian_images_per_octave.append(image)
			dog_images_per_octave = []

			for image_index in range(self.num_DoG_images_per_octave):
				# Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
				# - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
				gaussian_images_per_octave.append(cv2.GaussianBlur(gaussian_images_per_octave[0], (0, 0), self.sigma ** (image_index + 1), self.sigma ** (image_index + 1)))
				# Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
				# - Function: cv2.subtract(second_image, first_image)
				dog_images_per_octave.append(cv2.subtract(gaussian_images_per_octave[image_index + 1], gaussian_images_per_octave[image_index]))
				# np.save("dog_array_{}_{}".format(octave_index, image_index), dog_images_per_octave[image_index])

				# cv2.imwrite("./DoG_{}_{}.png".format(octave_index + 1, image_index + 1), (dog_images_per_octave[image_index] - min) / (max - min) * 255)

			gaussian_images.append(gaussian_images_per_octave)
			dog_images.append(dog_images_per_octave)
			image = gaussian_images_per_octave[-1]
			image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation = cv2.INTER_NEAREST)

		# Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
		# Keep local extremum as a keypoint
		keypoints = []

		for octave_index, dog_images_per_octave in enumerate(dog_images):
			for layer in range(1, self.num_DoG_images_per_octave - 1):
				height = dog_images_per_octave[-1].shape[0]
				width = dog_images_per_octave[-1].shape[1]

				# check each pixel is extremum or not
				for pixel_y in range(1, height - 1):
					for pixel_x in range(1, width - 1):
						value = dog_images_per_octave[layer][pixel_y][pixel_x]
						is_extremum = True
						find_maximum = False

						# find is maximum or not
						if value > 0 and value > self.threshold:
							find_maximum = True

						# find is minimum or not
						elif value < 0 and abs(value) > self.threshold:
							find_maximum = False
						else:
							is_extremum = False

						offset_layer = -1
						while offset_layer < 2 and is_extremum:
							offset_y = -1

							while offset_y < 2 and is_extremum:
								offset_x = -1

								while offset_x < 2 and is_extremum:
									if offset_x == 0 and offset_y == 0 and offset_layer == 0:
										offset_x = offset_x + 1
										continue
									else:
										curr_value = dog_images_per_octave[layer + offset_layer][pixel_y + offset_y][pixel_x + offset_x]
										# not maximum
										if find_maximum == False and value > curr_value:
											is_extremum = False
										# not minimum
										elif find_maximum == True and value < curr_value:
											is_extremum = False

									offset_x = offset_x + 1
								offset_y = offset_y + 1
							offset_layer = offset_layer + 1

						if is_extremum:
							keypoints.append([pixel_x * (octave_index + 1), pixel_y * (octave_index + 1)])

		# Step 4: Delete duplicate keypoints
		# - Function: np.unique
		keypoints = np.unique(np.array(keypoints), axis = 1)

		# sort 2d-point by y, then by x
		keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]

		return keypoints
