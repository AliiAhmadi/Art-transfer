#! /bin/python3

import tensorflow as tf
import numpy as np

vgg16 = tf.keras.applications.VGG16(include_top=True, weights='imagenet')

vgg16_npy = {}

for layer in vgg16.layers:
    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
        weights = layer.get_weights()
        if len(weights) == 2:
            W, b = weights
            vgg16_npy[layer.name] = [W, b]
        else:
            print(f"Skipping layer {layer.name} — unexpected weights format.")

np.save('../models/vgg16.npy', vgg16_npy)
print("Saved vgg16.npy!")
