import sys
import numpy as np
import cv2
import random
from pathlib import Path
import imgaug as ia
import imgaug.augmenters as iaa

OBJECT_IMAGE_SIZE_PERCENT = 0.25
TRAINING_IMAGE_SIZE = 512

# note this algorithn assumes the object is a clean segmented object

def merge_object_container(obj, container, aug_object=0, aug_container=0, clean_object_edges=True):
	"""
	Merge foreground image 'object', with its background image 'container'
	Args:
		obj (cv2 image): image of segmented isolated foreground object to be copied to container image
		container (cv2 image): background image
		aug_object (int): amount of augmentations to perform on obj
		aug_container (int): amount of augmentations to perform on container
		clean_object_edges (bool): if rue clean_edges_of_object function to remove edges that do not belong to object
	Returns:
		list of merged object and container images
	"""
	merged = []

	if apply_blur_to_object:
		obj = clean_edges_of_object(obj)
	
	# augmentation if you need change these lines to what match you needs

	# you may not want to add transformations to backgrounds but this may be used with hands.
	if aug_container:
		containers = apply_transformations(container, aug_container)
	else:
		containers = [container]
	if aug_object:
		objects = apply_transformations(obj, aug_object)
	else:
		objects = [obj]

	# merge images
	for resized_container in containers:
		for resized_object in objects:
			# this line is not as necessary with the clean edges function but still helps a little
			mask = np.all(resized_object > 2, axis=2).reshape(TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE, 1)* [1,1,1]
			merged_image = np.where(mask, resized_object, resized_container)
			merged.append(merged_image)
	return merged

def apply_transformations(image, N=10):
	"""
	Apply Affine/Rotation transformations to image
	Args:
		image (cv2 image): image to perform transforms on
		N (int): number of trnasforms to perform
	Returns:
		List of augmented images
	"""

	images = np.array([image for _ in range(N)], dtype=np.uint8)
	seq = iaa.Sequential([
		iaa.Fliplr(0.5), # horizontal flips
        iaa.Flipud(0.5),
		iaa.Affine(
            scale={"x": (0.75, 1.5), "y": (0.75, 1.5)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-20, 20),
        )])
	images_aug = seq(images=images)
	return images_aug


def resize_object(obj, container_shape, object_size_scale=0.5, keep_aspect_ratio=1):
	"""
	Resize object image to be object_size_scale percent of the container size. Aspect ratio can be preserved if needed
	Args:
		obj (cv2 image): image of object to be resized
		container_shape (tuple,list): contains the height and width of the container image respectively
		object_size_scale (float): scale to determine the object size relative to container size
		keep_aspect_ratio (bool): if True keep the aspect ratio when resizing otherwise, use same size w,h
	Returns:
		centered object that only takes object_size_scale*container_shape[0]*container_shape[1] area of the container image
	"""
	new_object_area = int(object_size_scale*container_shape[0]*container_shape[1])
	resized_object = None

	# resize with consistent aspect ratio
	if np.random.binomial(1,keep_aspect_ratio):
		h,w = obj.shape[:2]
		aspect_ratio = w/h
		area = w*h
		new_h = np.sqrt(new_object_area/aspect_ratio)
		new_w = int(round(new_h * aspect_ratio))
		new_h = int(round(new_h))
		resized_object = cv2.resize(obj, (new_w, new_h))

	# square resize
	else:
		new_h = new_object_area // 2
		new_w = new_h
		resized_object = cv2.resize(obj, (new_w, new_h))

	result = np.zeros((container_shape[0], container_shape[1],3), dtype=np.uint8)
	# compute center offset
	center_offset_h = (container_shape[0] - new_h) // 2
	center_offset_w = (container_shape[1] - new_w) // 2
	# copy img image into center of result image
	result[center_offset_h:center_offset_h+new_h, center_offset_w:center_offset_w+new_w] = resized_object
	return result

def clean_edges_of_object(image, thickness=5):
	"""
	Clean edges of image by removing contour edges found by drawContours
	Args:
		image (cv2 image): image to be cleaned
		thikness (int): thickness of contour lines drawn by drawContours 
	Returns: 
		cleaned image 
	"""
	mask = np.zeros(image.shape, np.uint8)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
  
	# Finding Contours 
	contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   

	# Draw all contours 
	# -1 signifies drawing all contours 
	cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness)
	mask = np.all(mask == 255, axis=2).reshape(image.shape[0], image.shape[1], 1)* [1,1,1]
	return np.where(mask, [0,0,0], image)
	

# testing, first argument is a directory containing foreground, 'objects', and the second argument is the background object. 
# adjust however you need 
if __name__ == '__main__':
	if len(sys.argv) != 3:
		sys.exit(1)
	
	#reshape the container
	container = cv2.imread(sys.argv[2])
	resized_container = cv2.resize(container, (TRAINING_IMAGE_SIZE,TRAINING_IMAGE_SIZE))

	#reshape object and merge with container
	for image in Path(sys.argv[1]).iterdir():
		obj = cv2.imread(str(image))
		resized_objects = resize_object(obj,  resized_container.shape[:2], OBJECT_IMAGE_SIZE_PERCENT)
		merged_images = merge_object_container(resized_objects, resized_container)
		for i,merged_image in enumerate(merged_images):
			cv2.imwrite(image.stem+'_{}.jpg'.format(i) , merged_image)