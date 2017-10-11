#!/usr/bin/env python
from headers import *

images = npy.load("IMAGES_TO_USE.npy")
labels = npy.load("LABELS_TO_USE.npy")

num_images = images.shape[0]

multiplier = 6

augmented_images = npy.zeros((multiplier*num_images,images.shape[1],images.shape[2],images.shape[3]),dtype=npy.uint8)
augmented_labels = npy.zeros((multiplier*num_images,images.shape[1],images.shape[2]),dtype=npy.uint8)

# Vanilla
augmented_images[:num_images] = copy.deepcopy(images)
augmented_labels[:num_images] = copy.deepcopy(labels)

for i in range(num_images):
    # Horizontal flip
    augmented_images[num_images+i] = npy.fliplr(images[i])
    augmented_labels[num_images+i] = npy.fliplr(labels[i])

    # Rotate by 90
    augmented_images[2*num_images+i] = npy.rot90(images[i])
    augmented_labels[2*num_images+i] = npy.rot90(labels[i])

    # Rotate by 270
    augmented_images[3*num_images+i] = npy.rot90(images[i],k=3)
    augmented_labels[3*num_images+i] = npy.rot90(labels[i],k=3)

    # Rotate the flipped by 90.
    augmented_images[4*num_images+i] = npy.rot90(images[i+num_images])
    augmented_labels[4*num_images+i] = npy.rot90(labels[i+num_images])

    # Rotate the flipped by 270.
    augmented_images[5*num_images+i] = npy.rot90(images[i+num_images],k=3)
    augmented_labels[5*num_images+i] = npy.rot90(labels[i+num_images],k=3)

npy.save("Augmented_Images.npy",augmented_images)
npy.save("Augmented_Labels.npy",augmented_labels)