import os
import cv2
import sys
import argparse
import random
import importlib
from pathlib import Path
from shutil import copy
import random
from scipy.stats import bernoulli
from image_augmentation import Range, get_largest_bbox, \
    iterative_twoside_crop, augment_image , cut_and_paste , augment_lighting
Base = Path(__file__).resolve().parent

"""
Augment and Split the dataset. Crop if option is provided
"""

parser=argparse.ArgumentParser()
parser.add_argument('root_dir',type=str,help='output dir of images to')
parser.add_argument('--maximum', '-m',type=int,default=-1,
                    help='maximum number images per class')
parser.add_argument('--plain_limit', default = .3, type=float,choices= Range(0.0,1.0),
                    help='maximum percent images for plain class')
parser.add_argument('--training_split', '-t',type=float,choices=Range(0.0,1.0),
                    default=.80,
                    help='training/validation split percentage')
parser.add_argument('--augment',type=int, default=100,
                    help='number of augmented images per image')
parser.add_argument('--subset',type=int, default=1000,
                    help='num of images to augment per class')
parser.add_argument('--light_augment',type=int, default=50,
                    help='number of light augmented images per image')
parser.add_argument('--crop' , '-c' , action='store_true',
                    help='twoside crop images')
parser.add_argument('--lighting' , '-l', default=None, choices=[None,'spot', 'parallel', 'all'],
                    help='add artificial lighting to images for augmentation')
parser.add_argument('--prob_lighting' , '-p' , default=.5, choices=Range(0.0, 1.0),
                    help='add artificial lighting to images for augmentation')
parser.add_argument('--test_aug'  , action='store_true',
                    help='test the augmentation setup using one image')
parser.add_argument('--test_light'  , action='store_true',
                    help='test the lighting setup using one image')
parser.add_argument('--DOI',default='',type=str,help='temp var to aug one class')
args=parser.parse_args()

def create_lit_dataset(file_path, output_dir, N=25):
    images_aug = []
    if file_path.is_dir():
        images = list(file_path.glob('*'))
        for index,path in enumerate(images):
            image = cv2.imread(str(path))
            images_aug.extend(augment_lighting(image, N))
        for i, im in enumerate(images_aug):
                cv2.imwrite(str(output_dir / "{}_{}{}".format(images[int(i/N)].stem,i
                                                        , images[int(i/N)].suffix)), im)
    else:
        image = cv2.imread(str(file_path))
        images_aug.extend(augment_lighting(image, N))
        for index, im in enumerate(images_aug):
            cv2.imwrite(str(output_dir / "{}_{}{}".format(file_path.stem,index
                                                          , file_path.suffix)),im)
            
def create_synth_dataset(base_dir, mask_dir, output_dir, mask_prob=0.15):
    mask_images = list(mask_dir.glob('*'))
    n = len(mask_images)
    for path in base_dir.iterdir():
        if path.is_file():
            # with prob mask_prob, mask with pattern image i.
            masks = bernoulli.rvs(mask_prob, size=n)
            image = cv2.imread(str(path))
            for index, apply_mask in enumerate(masks):
                if apply_mask == 1:
                    masked_image = cut_and_paste(image, cv2.imread(str(mask_images[index])))
                    cv2.imwrite(str(output_dir / "{}_{}{}".format(path.stem, index,
                                                                  path.suffix)), masked_image)
def crop_images(file_path, crop=True):
    """
    Apply twosided crop to either an image or images in a dir; return image(s)
    """
    images=[]
    # testing purposes
    if file_path.is_dir():
        for image in file_path.iterdir():
            # Import your picture
            img = cv2.imread(str(image))
            # can recieve error if the image 
            try:
                img = get_largest_bbox(img)
            except ValueError as e:
                print(e, image)
                sys.exit(1)
            if crop:    
                img = iterative_twoside_crop(img , boundary_thresh=0.3)
            images.append(img)
    else:
        # Import your picture
        images = cv2.imread(str(file_path))
        try:
            images = get_largest_bbox(images)
        except ValueError as e:
            print(e, image)
            sys.exit(1)
        if crop:
            images = iterative_twoside_crop(images, boundary_thresh=0.3)
    return images

def _get_images(class_path ,N=500):
    # get the relative image paths associated with each class path
    paths = [image_path for image_path in class_path.iterdir()]
    random.shuffle(paths)
    return paths[:N]

def write_label_file(images, label_file, class_paths):
    # label files contain only absolute paths with associated label
    # copy the images from the labeled folder to the training and validation folders
    with open(label_file,'w+') as f:
        for image_path,label in images:
            src_dir = class_paths[label]
            abs_path = os.path.join(src_dir,image_path)
            f.write(abs_path +' '+str(label)+'\n')

def split_dataset(input_dir, maximum, training_split,shuffle=True):
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
    # get the names of the classes and create label file
    
    root_dir = os.path.dirname(input_dir)
    class_names = []
    with open(Base / 'label_files' / 'pattern_8_labels.txt', 'r') as labels:
        for i,label in enumerate(labels.readlines()):
            class_names.append((label.strip() , i))
    with open(root_dir+'/pattern_class_names.txt', 'w+') as f:
        for label,_ in class_names:
            f.write(label+'\n')

    #get paths to each class and their respective sizes
    class_paths = [os.path.join(input_dir,class_name) for class_name,_ in class_names]
    class_sizes = [len(os.listdir(class_path)) for class_path in class_paths]
    total_size = sum(class_sizes)

    # make training and validation dirs for the data
    train_dir = os.path.join(root_dir,'train')
    validation_dir = os.path.join(root_dir,'validation')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
 
    training_images = []
    validation_images = []
    # split data and write to train and validation files
    for class_name,i in class_names:
        #get the relative image paths associated with each class path
        paths = os.listdir(class_paths[i])
        # if num_images > maximum alloted images => take only up to maximum images
        # shuffle paths to ensure variability of class images choosen
        random.shuffle(paths)
        #if class is plain then reduce the amount by 1/3 to ensure balanced set
        if class_name == 'plain':
            paths = paths[:int(maximum*.30)]
        else:
            paths = paths[:maximum]
        training_cutoff = int(len(paths)*training_split)
        print("TRAINING IMAGES: {} For Label {}".format(training_cutoff,class_name))
        training_images.extend([(image_path,i) for image_path in paths[:training_cutoff]])
        validation_images.extend([(image_path,i) for image_path in paths[training_cutoff:]])
    # shuffle labels if dataloader is not configured to do so
    if shuffle==True:
        random.shuffle(training_images)
        random.shuffle(validation_images)
    # label files contain only absolute paths with associated label
    # copy the images from the labeled folder to the training and validation folders
    write_label_file(training_images,train_dir+'/train_labels.txt',class_paths)
    write_label_file(validation_images,validation_dir+'/validation_labels.txt',class_paths)

def _test_augmentation(image_path, output=Path('./aug_test1')):
    item = Path(image_path)
    if not output.exists():
        output.mkdir(parents=True)
    crop_images(item, output)
    image_paths = [x for x in output.iterdir()]
    for image_path in image_paths:
        print(image_path)
        aug_images = augment_image(str(image_path))
        for img in aug_images:
            index = ''.join(["{}".format(random.randint(0, 9)) for num in range(12)])
            cv2.imwrite(str(output / "{}_{}{}".format(image_path.stem,
                                                      index, image_path.suffix)), img)
def _test_lightintg(image_path,output=Path('./light_test1'),N=50):
    if not output.exists():
        output.mkdir(parents=True)
    create_lit_dataset(Path(image_path), output, N)
   
if __name__=='__main__':
    #constants
    root_output_dir = Path(args.root_dir)
    lit_root_dir = root_output_dir / 'light_images'
    mask_root_dir = root_output_dir / 'masked_images'
    output_root = root_output_dir / 'pattern_dataset' / 'image_folders'
    
    if args.test_aug:
        _test_augmentation(sys.argv[1])
        sys.exit(0)
    if args.test_light:
        _test_lightintg(sys.argv[1])
        sys.exit(0)
    # get labels of classes
    class_names = []
    with open(Base / 'label_files' / 'pattern_8_labels.txt', 'r') as labels:
        for i,label in enumerate(labels.readlines()):
            class_names.append(label.strip())
            output = output_root / class_names[i]
            if not output.exists():
                output.mkdir(parents=True)
            lit= lit_root_dir / class_names[i]
            if not lit.exists():
                lit.mkdir(parents=True)
            mask = mask_root_dir / class_names[i]
            if not mask.exists():
                mask.mkdir(parents=True)
    
    #create synthesized images
    print("Creating Synth Images")
    for d in (root_output_dir / 'basic_pattern_dataset').iterdir():
        print(d, d.name)
        if d.name != args.DOI:
            continue
        create_lit_dataset(d, lit_root_dir / d.name, N=args.light_augment)
        print("Masking Images")
        create_synth_dataset(root_output_dir / 'base_images',
                             lit_root_dir/ d.name, mask_root_dir / d.name,mask_prob=1)

    # augment and crop images of each class
    print("Augmenting Images")
    for item in mask_root_dir.iterdir():
        if item.is_dir():
            print(item.name)
        if item.name != args.DOI:
            continue
        image_paths = _get_images(item, N=10000)
        print("IMAGES: {} For Label {}".format(len(image_paths),item.name))
        for image_path in image_paths:
            image = crop_images(image_path,args.crop)
            aug_images = augment_image(image, N=args.augment)
            for img in aug_images:
                index = ''.join(["{}".format(random.randint(0, 9)) for num in range(12)])
                cv2.imwrite(str(output_root / item.name / "{}_{}{}".format(image_path.stem,
                                                          index, image_path.suffix)),img)
    # create label files for dataset
    print("Creating Label Files For New Synth Dataset")
    split_dataset(output_root, args.maximum, args.training_split)
    
