# -*- coding: utf-8 -*-
"""
# ===========================================================================
# ===========================================================================
# !==   Valerio Carruba, Safwan Aljbaae                                    ==
# !==   June 2024                                                          ==
# ===========================================================================
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
from PIL import Image
from sys import getsizeof
import copy
import time
import datetime
import tracemalloc

start_time = time.time()
# starting the monitoring
tracemalloc.start()

class_names = ['circulation state', 'libration state']
# read data
def read_data(filename, Images_loc, Labels, set_type):
    data_ast = pd.read_csv(filename,
                           skiprows=0,
                           header=None,
                           delim_whitespace=True,
                           index_col=None,
                           names = Labels,
                           low_memory=False,
                           dtype={'id': np.int64,
                                  'a': np.float64,
                                  'sin_i': np.float64,
                                  'label': np.int64
                                  }
                           )
    n_lines =int(len(data_ast))
    data= data_ast.iloc[0:n_lines, :]
    data_id = list(data_ast.id)
    img = [Images_loc + str("{:07d}".format(ast_id)) + '.png' for ast_id in data_id]
    width, height = Image.open(img[0]).convert('1').size
#    images = [np.array(Image.open(x).convert('1').getdata()).reshape(width, height) for x in img]
    images = [np.array(Image.open(x).convert('L').resize((100, 101))) for x in img]
    im_labels = data_ast.label

    print(f'The size of the variable images is : {getsizeof(images)} bytes')
    print(f'We have {len(images)} images ({width} X {height} pixels) in the ', set_type, ' set, belonging to '
          f'{len(set(im_labels))} classes:')
    for i in range(len(set(im_labels))):
        print(f'   {len([x for x in im_labels if x == i])} asteroids in {class_names[i]} (label: {i})')
        print()
    return images, im_labels, data, data_id

filename_train = './TRAINING/res_status_all'
Images_loc_train = './TRAINING/res_osc_'
names_train = ['id', 'a', 'sin_i', 'label']
set_type = 'training'

train_images, train_labels, train_data, train_id = read_data(filename_train,Images_loc_train, names_train, set_type)

min_pixel = min(list(map(min, train_images[0])))
max_pixel = max(list(map(max, train_images[0])))
print(f'The pixel values of each image vary from {min_pixel} to {max_pixel}')

min_pixel = min(list(map(min, train_images[0])))
max_pixel = max(list(map(max, train_images[0])))
print(f'The pixel values of each image vary from {min_pixel} to {max_pixel}')

filename_test = './TEST/res_status_all'
Images_loc_test = './TEST/res_osc_'
names_test = ['id', 'a', 'sin_i', 'label']
set_type = 'testing'

test_images, test_labels, test_data, test_id = read_data(filename_test,Images_loc_test, names_test, set_type)

filename_val = './VALIDATION/res_status_all'
Images_loc_val = './VALIDATION/res_osc_'
names_val = ['id', 'a', 'sin_i', 'label']
set_type = 'validation'

val_images, val_labels, val_data, val_id = read_data(filename_val, Images_loc_val, names_val, set_type)

# preprocessing the data: rescale the pixels value to range from 0 to 1
train_images = train_images / max_pixel
test_images = test_images / max_pixel
val_images = val_images / max_pixel

###########################################################################
def vision_transformer(img_size, patch_size, num_layers, num_heads, mlp_dim, num_classes):
    model = tf.keras.Sequential()

    # Patch Embedding
    patch_size_x, patch_size_y = patch_size, patch_size
    num_patches_x = img_size // patch_size_x
    num_patches_y = img_size // patch_size_y
    num_patches = num_patches_x * num_patches_y
    model.add(tf.keras.layers.Conv2D(num_patches, (patch_size_x, patch_size_y),
                                     strides=(patch_size_x, patch_size_y), padding='valid',
                                     input_shape=(img_size, img_size, 1)))
    patch_embedding = tf.keras.layers.Reshape((num_patches, model.layers[-1].output_shape[-1]))(model.layers[-1].output)

    # Class Token
    class_token = tf.keras.layers.Embedding(1, model.layers[-1].output_shape[-1])(tf.zeros((1, 1)))
    class_token = tf.keras.layers.Reshape((1, model.layers[-1].output_shape[-1]))(class_token)
    model_input = tf.keras.layers.Concatenate(axis=1)([class_token, patch_embedding])

    # Positional Embedding
    pos_embedding = tf.keras.layers.Embedding(num_patches + 1, model.layers[-1].output_shape[-1])(
        tf.range(num_patches + 1))
    pos_embedding = tf.expand_dims(pos_embedding, axis=0)  # Add batch dimension

    # Transformer Blocks
    x = tf.keras.layers.Add()([model_input, pos_embedding])

    # Multi-Head Attention
    for _ in range(num_layers):
        attention = tf.keras.layers.MultiHeadAttention(num_heads, model.layers[-1].output_shape[-1] // num_heads)(x, x)
        x = tf.keras.layers.Add()([x, attention])
        x = tf.keras.layers.LayerNormalization()(x)
        feed_forward = tf.keras.layers.Dense(mlp_dim, activation='relu')(x)
        feed_forward = tf.keras.layers.Dense(model.layers[-1].output_shape[-1])(feed_forward)
        x = tf.keras.layers.Add()([x, feed_forward])
        x = tf.keras.layers.LayerNormalization()(x)

    # Classification Head
    x = tf.keras.layers.Reshape((num_patches + 1, model.layers[-1].output_shape[-1]))(x)
    x = x[:, 0, :]  # Extract the class token
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(model_input, x)


model = vision_transformer(img_size=100, patch_size=10, num_layers=1, num_heads=1, mlp_dim=1024, num_classes=2)
###########################################################################

model.summary()

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# This checkpoint object will store the model parameters
# in the file "weights.hdf5"
checkpoint = ModelCheckpoint('./weights.hdf5',
                             save_weights_only=True,
                             monitor='accuracy',
                             mode='max',
                             verbose=1,
                             save_best_only=True, )
start_time = time.time()
# starting the monitoring
tracemalloc.start()

# compile the model
# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model, using the checkpoint as a callback
x = model.fit(train_images, train_labels, epochs=12, batch_size=32, validation_data=(val_images, val_labels),
              callbacks=[ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)])
end_time = time.time()
exec_time = datetime.timedelta(seconds=(end_time - start_time))
print(f'\n --- The execution time was: {exec_time} (h:m:s) ---')
# displaying the memory: The output is given in form of (current, peak), i.e, current memory is the memory the code
# is currently using, Peak memory is the maximum space the program used while executing.
print(tracemalloc.get_traced_memory())


# plot diagnostic learning curves
def summarize_diagnostics(x):
    fig = plt.figure()
    figure = fig.add_subplot(211)
    figure.plot(x.epoch, x.history['loss'], color='blue', label='train')
    figure.plot(x.epoch, x.history['val_loss'], color='orange', label='val')
    # plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Cross Entropy Loss: ViT')
    figure = fig.add_subplot(212)
    figure.plot(x.epoch, x.history['accuracy'], color='blue', label='train')
    figure.plot(x.epoch, x.history['val_accuracy'], color='orange', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Classification Accuracy: ViT')
    # plt.show()
    plt.tight_layout()
    fig.savefig('history_model_ViT.png', format='png', dpi=300)
    plt.close(fig)


summarize_diagnostics(x)

predictions = model.predict(test_images)
predict_label = [int(np.argmax(x)) for x in predictions]
predict_acc = [100 * max(x) for x in predictions]

predicted_data = copy.deepcopy(test_data)
predicted_data['predicted_label'] = list(predict_label)
predicted_data.to_csv(r'nu6_pred_data.csv', index=False, header=False, sep=' ', float_format='%.7f')
print()


# show the first images in the test data
def show_images():
    fig = plt.figure(figsize=(8, 12))
    for i in range(50):
        plt.subplot(10, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i])
        color = 'blue'
        plt.xlabel("{} ({:2.0f}%)".format(predict_label[i], predict_acc[predict_label[i]]), color=color, fontsize=10)
        plt.ylabel("{}".format(test_id[i]), color=color, fontsize=10)
    plt.subplots_adjust(hspace=0.3, wspace=0)
    # plt.show()
    fig.savefig('predicted_data.png', format='png', dpi=300)
    plt.close(fig)


show_images()

print("End")
