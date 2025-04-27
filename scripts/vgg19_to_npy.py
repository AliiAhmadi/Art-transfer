#! /bin/python3

import tensorflow as tf
import numpy as np

vgg19 = tf.keras.applications.VGG19(include_top=True, weights='imagenet')

vgg19_npy = {}

for layer in vgg19.layers:
    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
        weights = layer.get_weights()
        if len(weights) == 2:
            W, b = weights
            vgg19_npy[layer.name] = [W, b]
        else:
            print(f"Skipping layer {layer.name} — unexpected weights format.")

np.save('../models/vgg19.npy', vgg19_npy)
print("Saved vgg19.npy!")
