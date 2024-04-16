import glob
import random
import numpy as np
from PIL import Image, ImageOps
from scipy import spatial


def load_image(source: str) -> np.ndarray:
    # Opens an image from specified source and returns a numpy array with image rgb data
    with Image.open(source) as img:
        im_arr = np.asarray(img)
    return im_arr


# Load in the image we want to turn into a mosaic
face_im_arr = load_image('grayscale_start.jpeg')
width = face_im_arr.shape[0]
height = face_im_arr.shape[1]

target_res = (100, 100)  # Number of images to make up final image (width, height)
mos_template = face_im_arr[::(height // target_res[0]),
               ::(height // target_res[1])]  # Template to insert the smaller images into

images = []
for file in glob.glob('data/*'):
    im = load_image(file)
    images.append(im)


def resize_image(img: Image, size: tuple) -> np.ndarray:
    # Takes an image and resizes to a given size (width, height) as passed to the size parameter
    resized_img = ImageOps.fit(img, size, Image.LANCZOS, centering=(0.5, 0.5))
    return np.array(resized_img)


mosaic_size = (40, 40)  # Defines size of each mosaic image/tile
images = [resize_image(Image.fromarray(i), mosaic_size) for i in images]  # Resize images to specified mosaic tile size


# Get mean RGB values for each image
images_array = np.asarray(images)
image_values = np.apply_over_axes(np.mean, images_array, [1, 2]).reshape(len(images), 3)
tree = spatial.KDTree(image_values)  # Insert into KDTree


image_idx = np.zeros(target_res, dtype=np.uint32)
for i in range(target_res[0]):
    for j in range(target_res[1]):
        template = mos_template[i, j]
        match = tree.query(template, k=40)
        pick = random.randint(0, 39)  # Pick random image to decrease likelihood of repeated images
        image_idx[i, j] = match[1][pick]

canvas = Image.new('RGB', (mosaic_size[0] * target_res[0], mosaic_size[1] * target_res[1]))
for i in range(target_res[0]):
    for j in range(target_res[1]):
        arr = images[image_idx[j, i]]
        x, y = i * mosaic_size[0], j * mosaic_size[1]
        im = Image.fromarray(arr)
        canvas.paste(im, (x, y))

canvas.save('result.png')
