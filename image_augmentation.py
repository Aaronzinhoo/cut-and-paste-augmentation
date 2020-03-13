import numpy as np 
import cv2
import scipy
import sys
import os
import random
import argparse
import collections
import importlib
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
from PIL import ImageFilter
from scipy.ndimage.filters import gaussian_filter
try:
    from utils.lighting import add_lighting
except Exception as e:
    print(e)
    add_lighting = importlib.import_module('lighting').add_lighting

"""
TODO:
    Tuple checking in square mask and get_center_crop 
    make cropping class to load any crop type based on init var
    Given Input and Output dir, crop images by finding the centroid and expanding around the centroid until a percentage of the pixels are black
"""

class GaussianBlurring(object):
    """Apply gaussian blur to a numpy.ndarray (H x W x C)
    """
    def __init__(self, sigma, random_state=np.random):
        self.sigma = sigma
        self.random_state = random_state

    def __call__(self, image):
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(
                self.sigma, random_state=self.random_state)
        else:
            sigma = self.sigma
        image = gaussian_filter(image, sigma=(sigma, sigma, 0))
        return image

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

def bounds_check(x, bounds=(.10,.50)):
    assert isinstance(bounds, (tuple,list)),"Bounds Must Be a tuple or list"
    lower_bound, upper_bound = bounds[0],bounds[1]
    if x >= lower_bound and x <= upper_bound:
        return x
    return np.random.uniform(lower_bound,upper_bound)

        
def gif_to_jpg(image_path, remove_gif=True):
    """
    Single frame gif to jpg conversion w/ opt to remove gif file
    """
    new_image_name = '{}.jpg'.format(image_path.split('.')[0])
    Image.open(image_path).convert('RGB').save(new_image_name)
    if remove_gif:
        os.remove(image_path)
    return new_image_name

def label_image(image,center):    
    cX,cY = center
    cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(image, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image


def get_center(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # calculate moments of binary image
    gray_image = cv2.medianBlur(gray_image, 5)
    
    # thresh = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                                cv2.THRESH_BINARY,25,5)
    ret,thresh = cv2.threshold(gray_image,10,255,0)
    
    M = cv2.moments(thresh)
    # calculate x,y coordinate of center using moments
    # if image too dark may throw division by zero error
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX,cY

def gaussian_blur(x, radius):
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img

def gaussian_kernel(size, sigma=2., dim=2, channels=3):
    # The gaussian kernel is the product of the gaussian function of each dimension.
    # kernel_size should be an odd number.
    
    kernel_size = 2*size + 1

    kernel_size = [kernel_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
    
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    return kernel


def circle_mask(center,radius,array):
    h,w = center
    nx,ny = array.shape
    y,x = np.ogrid[-w:nx-w,-h:ny-h]
    mask = x*x + y*y <= radius**2
    return sum(array[mask]) == np.sum(mask),mask

def square_mask(center,area,array,threshold=.10):
    """
    Given maksed array, sum its contents within area and check if it contains black pixels
    
    Args:
        center - center of the image passed as tuple
        area   - size of square to check around image
        array  - masked array that contains false if the sum of color channels below 1.0
        threshold - proportion of black pixels we allow in the mask
    """
    # get the center of the array we are interested in
    width, height = 0, 0
    if isinstance(area,(tuple,list)):
        assert len(area)==2,"if threshold is range; must only contain 2 elem [width,height]"
        width,height  = area[0],area[1]
    else:
        width,height  = area,area
    
    center_h,center_w = center
    current_mask = array[center_w-width:center_w+width,center_h-height:center_h+height]
    mask_area=width*2*height*2
    return abs(np.sum(current_mask) - mask_area) <= int(mask_area*threshold), current_mask

def rectangle_mask(center,area,array,threshold=.10):
    """
    Given maksed array, sum its contents within area and check if it contains black pixels
    
    Args:
        center - center of the image passed as tuple
        area   - size of square to check around image
        array  - masked array that contains false if the sum of color channels below 1.0
        threshold - proportion of black pixels we allow in the mask
    """
    pass

def get_crop(image,mask,center,radius_itr=0):
    """
    Apply  When Using the circle crop to get a rectangular crop of the image
    """
    print(mask)
    
    where = np.array(np.where(mask))
    print(where)
    x1, y1 = np.amin(where, axis=1)
    x2, y2 = np.amax(where, axis=1)
    return image[y1:y2-radius_itr, x1:x2-radius_itr]

def crop_image_centroid(image,size_itr=1,min_value=0.0,threshold=.01, method='iter'):
    """
    Given full path to an image, crop the images colored region based on the centroid...
    
    Args:
        image - numpy array
        size_itr - what to increment size of mask area by (circle/square)
        min_value - threshold for pixel colors we arent interested in
        threshold - arg for square mask that dictates porportion of min_value pixels we allow in masked image
        method - method to create the crop using the threshold with either a 'random' centered crop or crop that grows until threshold reached
    """
    
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    mask = np.sum(image,axis=2) > min_value

    #get colored portion of image and crop from original
    try:
        center = get_center(image)
    except ZeroDivisionError as e:
        return None
    if isinstance(threshold,(tuple,list)):
        assert len(threshold)==2,"if threshold is range; must only contain 2 elem"
        threshold = np.random.uniform(threshold[0],threshold[1])
    cX , cY = center
    color_mask = None
    current_mask = None
    if method == 'random':
        crop_x = np.random.uniform(min(25,.50*image.shape[0]),1.0*image.shape[1])
        crop_y = np.random.uniform(min(25,threshold[0]*image.shape[0]),threshold[1]*image.shape[1])
        contains_color, current_mask = square_mask(center,size,mask,threshold=threshold)
        if not contains_color:
            return None
        color_mask = current_mask
    elif method == 'iter':
        for size in range(25,128,size_itr):
            contains_color, current_mask = square_mask(center,size,mask,threshold=threshold)
            if not contains_color:
                break
            color_mask = current_mask
    else:
        print("Method Provided Unknown ... Exiting\n")
        sys.exit(1)
            
    cropped_image = None
    try:
        x_len, y_len = color_mask.shape
        x_len = int(x_len/2)
        y_len = int(y_len/2)
        cropped_image = image[cY-x_len:cY+x_len , cX-y_len:cX+y_len]
    except Exception as e:
        #print(e , "  -  Not enough color pixels   -  image {}".format(image_path))
        return None
    return cropped_image

def _get_bbox_dim(image, all_contours=False):
    """
    returns cropped image that fits largest contour found
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_mask = cv2.inRange(img_gray, 1, 255)
    _ , contours, heirarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if all_contours:
        height, width = img_mask.shape
        min_x, min_y = width, height
        max_x = max_y = 0
        # computes the bounding box for the contour, and draws it on the frame,
        for contour in contours:
            (x,y,w,h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x+w, max_x)
            min_y, max_y = min(y, min_y), max(y+h, max_y)
        print(max_y-min_y,max_x-min_x,min_y,min_x)
        return max_y-min_y,max_x-min_x,min_y,min_x
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    x,y,w,h = cv2.boundingRect(biggest_contour)
    return (h,w,y,x)

def get_largest_bbox(image):
    """
    returns cropped image that fits largest contour found
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
    img_mask = cv2.inRange(img_gray, 1, 255)
    _ , contours, _ = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    x,y,w,h = cv2.boundingRect(biggest_contour)
    return image[y:y+h,x:x+w,:]

def iterative_narrowing_crop(image, iterator=1,min_crop=(50,50)):
    # find largest contour
    # smooth grayscale image for better boundingRect
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray,(5,5),0)
    y_lim, x_lim = image.shape[0],image.shape[1]
    x_top , y_top = 0,0
    y_bot, x_bot = y_lim-1,x_lim-1
    min_rectangle = False
    cropped=False
    while not min_rectangle and not y_bot-y_top < min_crop[0] and not x_bot-x_top<min_crop[1]:
        if np.any(img_gray[y_top,x_top:x_bot] == 0):
            y_top+=iterator
            cropped=True
        if np.any(img_gray[y_bot,x_top:x_bot] == 0):
            y_bot-=iterator
            cropped=True
        if np.any(img_gray[y_top:y_bot,x_top] == 0):
            x_top+=iterator
            cropped=True
        if np.any(img_gray[y_top:y_bot,x_bot] == 0):
            x_bot-=iterator
            cropped=True
        if cropped == True:
            cropped = False
        else:
            min_rectangle = True
    return image[y_top:y_bot,x_top:x_bot,:]

def iterative_twoside_crop(image, iterator=1,min_crop=(80,80) , boundary_thresh = 0.1):
    """
    Crop image by iteratively cropping 2 sides at a time (left or right side and top or bottom side)
        Until the image size == min_crop or threshold of black pixels is met 
            for each pair of sides, choose:
            - side with largest amount of black pixels
            - crop this side out
            check if we made a crop:
            - if crop made, rerun the loop
            - else break and return image cropped at the new dimensions
    Args:
        image: image in form of nd.array
        iterator: size of crop made on given side
        min_crop: minimum size image can be cropped to
        boundary_thresh: percentage of black pixels allowed on boundary of new cropped image
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray,(5,5),0)
    y_lim, x_lim = image.shape[0],image.shape[1]
    x_top , y_top = 0,0
    y_bot, x_bot = y_lim-1,x_lim-1
    min_rectangle = False
    cropped=False
    while not min_rectangle:
        #and top_row > bp_threshold*x_top-x_bot applies black pixel threshold
        top_row = np.sum(img_gray[y_top,x_top:x_bot] == 0)
        bot_row = np.sum(img_gray[y_bot,x_top:x_bot] == 0)
        left_col = np.sum(img_gray[y_top:y_bot,x_top] == 0)
        right_col = np.sum(img_gray[y_top:y_bot,x_bot] == 0)
        row_index = np.argmax([top_row,bot_row])
        col_index = np.argmax([left_col,right_col])
        
        if not (y_bot-y_top-1 <= min_crop[0] or boundary_thresh*(x_bot-x_top) >= top_row) and row_index == 0:
            #print("increasing top_row")
            y_top+=iterator
            cropped=True
        if not (y_bot-y_top-1 <= min_crop[0] or boundary_thresh*(x_bot-x_top) >= bot_row) and row_index == 1:
            #print("decreasing bot row")
            y_bot-=iterator
            cropped=True
        if not (x_bot-x_top-1 <= min_crop[0] or boundary_thresh*(y_bot-y_top) >= left_col) and col_index == 0:
            #print("increasing left col")
            x_top+=iterator
            cropped=True
        if not (x_bot-x_top-1 <= min_crop[0] or boundary_thresh*(y_bot-y_top) >= right_col) and col_index == 1:
            #print("decreasing right col")
            x_bot-=iterator
            cropped=True
        if cropped == True:
            cropped = False
        else:
            min_rectangle = True
    return image[y_top:y_bot, x_top:x_bot,:]

def cut_and_paste(base_image, masking_image, BoxCut= True, transform=True):
    h,w,_ = masking_image.shape
    if h > 350 and w > 350:
        h = 300
        w = 300
        masking_image = cv2.resize(masking_image, (w,h))
    
    # Mask the colored portion of base image with the corresponding pixels on the masking img
    if BoxCut:
        height,width = base_image.shape[:2]
        dims = _get_bbox_dim(base_image)
        resized_image = cv2.resize(masking_image, (dims[1],dims[0]))
        # have probability here to scale the mask
        if transform:
            resized_image = simple_augment_image([resized_image])[0]
        masking_image = np.zeros((height,width,3))
        masking_image[dims[2]:dims[2]+dims[0],dims[3]:dims[3]+dims[1]] = resized_image
    else:
        base_image = cv2.resize(base_image, (w,h))
    reshaped_image = cv2.GaussianBlur(base_image, (3,3), 0)
    cut = np.where(np.all(reshaped_image < 10, axis=-1))
    masking_image[cut] = 0
    return masking_image

# The image has shape (height, width, channels) and dtype uint8.
def simple_augment_image(images):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Flipud(0.5),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.75, 1.25), "y": (0.75, 1.25)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-10, 10),
        )
    ], random_order=True)
    images_aug = seq(images=images)
    return images_aug

def augment_image(image,N=100):
    """
    Given An Image Create Augment N augmented copies of the image
    Since we prefer lists or numpy arrays, it is suggested to use OpenCV for opening images
    """
    images = np.array(
        [image for _ in range(N)],
        dtype=np.uint8)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Flipud(0.5),
        iaa.CropAndPad(percent=(-0.10, 0.10),
                       pad_mode=["constant", "mean","maximum"],
                       pad_cval=(0, 128)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.25,
                      iaa.GaussianBlur(sigma=(0, 0.5))),
        iaa.CoarseDropout((0.0, 0.20), size_percent=(0.02, 0.25)),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.7, 1.50)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        iaa.PiecewiseAffine(scale=(0.01, 0.07)),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.60, 1.25),per_channel=.35),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.75, 1.5), "y": (0.75, 1.5)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-20, 20),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order

    images_aug = seq(images=images)
    return images_aug

def augment_lighting(image, N=30):
    choices = ['spot', 'parallel']
    lighting = np.random.randint(2, size=N)
    images_aug = []
    for index in lighting:
        images_aug.append(add_lighting(image, choices[index]))
    return images_aug
    
    
if __name__ == '__main__':
    res=cut_and_paste(cv2.imread(sys.argv[1]),cv2.imread(sys.argv[2]))
    cv2.imwrite('cut_paste3.jpg', res)
    res=cut_and_paste(cv2.imread(sys.argv[1]),cv2.imread(sys.argv[2]),BoxCut=False)
    cv2.imwrite('cut_paste_wo_box_cut3.jpg', res)
