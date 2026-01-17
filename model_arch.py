# cnn/model_arch.py

import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
Input = tf.keras.Input
ResNet50 = tf.keras.applications.ResNet50

IMG_SIZE = 256
NUM_CLASSES = 5

def ResNet50_UNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = Input(shape=input_shape)
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # Encoder skip connections
    skip1 = base_model.get_layer('conv1_relu').output           # 128x128
    skip2 = base_model.get_layer('conv2_block3_out').output     # 64x64
    skip3 = base_model.get_layer('conv3_block4_out').output     # 32x32
    skip4 = base_model.get_layer('conv4_block6_out').output     # 16x16
    x = base_model.get_layer('conv5_block3_out').output         # 8x8

    # Decoder with transposed convs and skip connections
    x = layers.Conv2DTranspose(512, 3, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip4])
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip3])
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip2])
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip1])
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)

    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)