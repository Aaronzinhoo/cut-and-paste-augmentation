import numpy as np
import cv2
import enum
import random
import argparse
from pathlib import Path
from tqdm import tqdm

from merge_images import crop_and_resize_image
from merge_images import resize_object
from merge_images import merge_object_container

def get_args():
	arg_parser = argparse.ArgumentParser(description="")
	arg_parser.add_argument('--root_object_dir', '-o', help="path to object directory (foreground images)")
	arg_parser.add_argument('--container_dir','-c',help="path to background directory")
	arg_parser.add_argument('--root_merged_dir','-m', help='path for merged images to be stored')
	arg_parser.add_argument('--root_dataset_dir','-d', help='path for merged images to be stored')
	arg_parser.add_argument('--merged_image_size', '-t', nargs=2, type=int, default=[512, 512] ,help='size of merged images')
	arg_parser.add_argument('--train_val_test_split', nargs=2, type=float, default=[0.5, 0.25],
							help='splits for test val and train datasets')
	arg_parser.add_argument('--object_size_scale' , type=float, default=0.25, 
							help='size of background to scale object size to')
	arg_parser.add_argument('--keep_object_aspect_ratio', type=float, default=1, 
							help='proportion of times to keep objects aspect ratio the same after resizing')
	arg_parser.add_argument('--num_containers_per_object', type=int, default=5, help="number of containers to combine with each object")
	arg_parser.add_argument('--aug_container', type=int, default=1, help="number of augmented copies of container")
	arg_parser.add_argument('--aug_object', type=int, default=1, help="number of augmented copies of object")
	arg_parser.add_argument('--aug_light', type=int, default=1, help='number of light augmentations on merged images')
	arg_parser.add_argument('--crop_container_prob', default=0.75, type=float, help='prob that container is randomly cropped')
	arg_parser.add_argument('--container_min_crop_proportion', default=0.1, type=float, help='minimum percent of container sides that are cropped')
	arg_parser.add_argument('--seed', default=42, type=int, help='seed for splitting dataset')
	return arg_parser.parse_args()

class RecycleClasses(enum.Enum):
	class1 = 0
	class2 = 1
	class3 = 2
	class4 = 3
	class5 = 4

def merge_class_container_images(object_dir, container_dir,
								 merged_dir, merged_image_size, 
								 object_size_scale, keep_object_aspect_ratio, num_containers_per_object,
								 aug_container, aug_object,
								 aug_light, crop_container_prob, container_min_crop_proportion):
	training_images = []
	count = 0
	for object_path in tqdm(Path(object_dir).iterdir()):
		containers_to_merge = np.random.choice(container_dir, num_containers_per_object)
		for container_path in containers_to_merge:
			container_image = cv2.imread(str(container_path))
			object_image = cv2.imread(str(object_path))
			if container_image is None or object_image is None:
				print('Skipping broken image')
				continue
			
			# resize objects and containers
			resized_container_image = crop_and_resize_image(container_image, tuple(merged_image_size))
			resized_object_image = resize_object(object_image,  merged_image_size, object_size_scale, keep_object_aspect_ratio)
			
			# augment obj/container => merge images => add light augmentation
			merged_images = merge_object_container(resized_object_image, resized_container_image, 
												   aug_object, aug_container, aug_light)
			for merged_image in merged_images:
				merged_image_name = str(merged_dir / (object_path.stem+'_{}.jpg'.format(count)))
				cv2.imwrite( merged_image_name, merged_image)
				training_images.append(merged_image_name)
				count+=1
	return training_images

def write_label_file(images, label_file):
    # label files contain only absolute paths with associated label
    # copy the images from the labeled folder to the training and validation folders
    with open(label_file,'w+') as f:
        for image_path,label in images:
        	path = Path(image_path)
        	
        	class_name = path.parent.stem
        	image_name = path.stem

        	image_path = str(Path(class_name) / image_name)
        	f.write(image_path +','+str(label)+'\n')

def split_data(classes, root_dataset_dir, splits, shuffle=True):
    """
    Make a dataset that can be loaded into dataloader (FOR TRAINING PURPOSES):
        We use a validation and train label to file to feed to the data loader(pytorch)
        while splitting the images into train and val sets (assuming)

    ****name are irrelevant so long as it follows this structure****
    structure of dir:
        root->
            image_folders-> (aka input dir to main)
                        image_folders_seperated_by_class...-> images
            train-> (program make this)
                ...
            validation->(program make this)
                ...
    """
    # dir creation
    root_dataset_dir = Path(root_dataset_dir)
    root_dataset_dir.mkdir(parents=True,exist_ok=True)
    train_dir = root_dataset_dir / 'train'
    validation_dir = root_dataset_dir / 'validation'
    test_dir = root_dataset_dir / 'test'
    train_dir.mkdir(exist_ok=True)
    validation_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
 
    dataset = {"train":[], "val":[], "test":[]}
    # split data and write to train and validation files
    for i, cls in enumerate(classes.items()):
        class_name = cls[0]
        class_data = cls[1][:]

    	# shuffle paths
        random.shuffle(class_data[1])

        training_cutoff = int(class_data[0]* splits[0])
        validation_cutoff = int(class_data[0]*splits[1])+training_cutoff
        print("TRAINING IMAGES: {} For Label {}".format(training_cutoff,class_name))

        dataset["train"].extend([(image_path,i) for image_path in class_data[1][:training_cutoff]])
        dataset["val"].extend([(image_path,i) for image_path in class_data[1][training_cutoff:validation_cutoff]])
        dataset["test"].extend([(image_path,i) for image_path in class_data[1][validation_cutoff:]])


    # shuffle labels if dataloader is not configured to do so
    if shuffle==True:
    	for data in dataset.values():
    		random.shuffle(data)
    # label files contain only absolute paths with associated label
    # copy the images from the labeled folder to the training and validation folders
    write_label_file(dataset["train"],train_dir / 'train_labels.csv')
    write_label_file(dataset["val"]  ,validation_dir / 'validation_labels.csv')
    write_label_file(dataset["test"] ,test_dir / 'test_labels.csv')


def main(root_object_dir, container_dir, seed,
		 root_dataset_dir, root_merged_dir, train_val_test_split,**kwargs):
	random.seed(seed)
	np.random.seed(seed)
	merged_dir = None
	containers = list(Path(container_dir).iterdir())
	dataset = {}
	for object_dir in tqdm(Path(root_object_dir).iterdir()):
		class_name = object_dir.name
		print('=' * 20, class_name, '=' * 20)
		merged_dir = Path(root_merged_dir) / class_name
		merged_dir.mkdir(parents=True, exist_ok=True)
		class_images = merge_class_container_images(object_dir=object_dir, container_dir=containers, merged_dir=merged_dir,**kwargs)
		dataset[class_name] = [len(class_images), class_images]
	split_data(dataset, root_dataset_dir, train_val_test_split)

if __name__ == '__main__':
	args = get_args()
	main(**vars(args))