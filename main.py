# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 03:08:27 2020

@author: hp
"""

import tensorflow as tf
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from unet_model import unet
from IPython.display import clear_output


SIZE = 128
N_CLASSES = 2

def parse_image(img_path: str) -> dict:
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(img_path, "training", "mask")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask_path = tf.strings.regex_replace(mask_path, r'[0-9]+', "0")

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.where(mask == 255, np.dtype('uint8').type(1), mask)
    return {'image': image, 'segmentation_mask': mask}

train_imgs = tf.data.Dataset.list_files("data/first/training/*.jpg")
val_imgs = tf.data.Dataset.list_files("data/first/validation/*.jpg")
train_set = train_imgs.map(parse_image)
test_set = val_imgs.map(parse_image)
dataset = {"train": train_set, "test": test_set}

def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    input_image = tf.image.resize(datapoint['image'], (SIZE, SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (SIZE, SIZE))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint: dict) -> tuple:
    input_image = tf.image.resize(datapoint['image'], (SIZE, SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (SIZE, SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

train_imgs = glob("data/first/training/*.jpg")
TRAIN_LENGTH = len(train_imgs)

BATCH_SIZE = 64
BUFFER_SIZE = 500
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

def display_sample(display_list):
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

for image, mask in train.take(2):
    sample_image, sample_mask = image, mask
display_sample([sample_image, sample_mask])

model = unet()

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display_sample([image[0], mask[0], create_mask(pred_mask)])
    else:
        display_sample([sample_image, sample_mask,
                        create_mask(model.predict(sample_image[tf.newaxis, ...]))])

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 5
VAL_SUBSPLITS = 2
VALIDATION_STEPS = 1500//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

show_predictions(test_dataset, 3)