import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def segmentation_mask(encoded_pixels: str, size=(768, 768)):
    mask = np.zeros(size[0] * size[1])

    encoded_pixels = encoded_pixels.split()
    start_pixels = np.array([(int(x) - 1) for x in encoded_pixels[::2]])
    lengths = np.array([int(x) for x in encoded_pixels[1::2]])
    end_pixels = start_pixels + lengths

    for i in range(start_pixels.shape[0]):
        mask[start_pixels[i]:end_pixels[i]] = 1
    mask = mask.reshape(size).T
    return mask


def get_encoded_pixels_by_img_id(img_id, dataset):
    encoded_pixels = dataset[dataset['ImageId'] == img_id]['EncodedPixels']
    if pd.isna(encoded_pixels.values):
        encoded_pixels = ' '
    encoded_pixels = ' '.join(encoded_pixels)

    return encoded_pixels


def display_image_with_segmentation(img_id, dataset, img_folder):
    encoded_pixels = get_encoded_pixels_by_img_id(img_id, dataset)
    segmentation = segmentation_mask(encoded_pixels)

    img = np.asarray(Image.open(os.path.join(img_folder, img_id)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))

    axes[0].imshow(img)
    axes[0].set_title("Image")

    axes[1].imshow(segmentation)
    axes[1].set_title("Segmentation")

    plt.show()


def fill_ds(ids, df, image_folder, img_size=(256, 256, 3)):
    x = np.zeros((len(ids), img_size[0], img_size[1], img_size[2]), dtype=np.float32)
    y = np.zeros((len(ids), img_size[0], img_size[1], 1), dtype=np.float32)

    for n, img_id in tqdm(enumerate(ids), total=len(ids)):
        img = np.asarray(Image.open(os.path.join(image_folder, img_id))
                         .resize((img_size[0], img_size[1])),
                         dtype=np.float32).reshape(img_size)
        x[n] = img / 255.0

        encoded_pixels = get_encoded_pixels_by_img_id(img_id, df)

        segmentation = segmentation_mask(encoded_pixels)
        segmentation = np.asarray(Image.fromarray(segmentation)
                                 .resize((img_size[0], img_size[1])),
                                 dtype=np.float32).reshape((img_size[0], img_size[1], 1))
        y[n] = segmentation

    return x, y
