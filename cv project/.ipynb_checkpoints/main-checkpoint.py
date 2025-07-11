
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU,
                                     Dropout, Input, Concatenate, Dense, Flatten)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.applications import VGG19

# Set GPU growth to avoid memory errors (if using GPU)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load and preprocess noisy image
image_path = "C:\\Users\\meetv\\OneDrive\\Documents\\iocl\\cv project\\noisyImages\\noisy_image.jpg"
noisy_image = cv2.imread(image_path)
noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0,1]
noisy_image = cv2.resize(noisy_image, (128, 128))  # Resize to match model input
noisy_image = np.expand_dims(noisy_image, axis=0)  # Add batch dimension

# ----------------- Generator (Improved U-Net) -----------------
def build_generator():
    inputs = Input(shape=(128, 128, 3))

    # Encoding layers
    x1 = Conv2D(64, (3,3), strides=2, padding='same', activation=LeakyReLU(0.2))(inputs)
    x1 = BatchNormalization()(x1)
    x2 = Conv2D(128, (3,3), strides=2, padding='same', activation=LeakyReLU(0.2))(x1)
    x2 = BatchNormalization()(x2)
    x3 = Conv2D(256, (3,3), strides=2, padding='same', activation=LeakyReLU(0.2))(x2)
    x3 = BatchNormalization()(x3)
    x4 = Conv2D(512, (3,3), strides=2, padding='same', activation=LeakyReLU(0.2))(x3)
    x4 = BatchNormalization()(x4)

    # Decoding layers
    x5 = Conv2DTranspose(256, (3,3), strides=2, padding='same', activation='relu')(x4)
    x5 = BatchNormalization()(x5)
    x5 = Concatenate()([x5, x3])
    x6 = Conv2DTranspose(128, (3,3), strides=2, padding='same', activation='relu')(x5)
    x6 = BatchNormalization()(x6)
    x6 = Concatenate()([x6, x2])
    x7 = Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu')(x6)
    x7 = BatchNormalization()(x7)
    x7 = Concatenate()([x7, x1])
    x8 = Conv2DTranspose(3, (3,3), strides=2, padding='same', activation='sigmoid')(x7)  # Output layer
    
    return Model(inputs, x8, name="Generator")

# ----------------- Discriminator (CNN) -----------------
def build_discriminator():
    inputs = Input(shape=(128, 128, 3))
    
    x = Conv2D(64, (3,3), strides=2, padding='same', activation=LeakyReLU(0.2))(inputs)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (3,3), strides=2, padding='same', activation=LeakyReLU(0.2))(x)
    x = Dropout(0.3)(x)
    x = Conv2D(256, (3,3), strides=2, padding='same', activation=LeakyReLU(0.2))(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)  # Output: real or fake
    
    return Model(inputs, x, name="Discriminator")

# ----------------- VGG19 Feature Loss -----------------
def build_vgg_feature_extractor():
    vgg = VGG19(weights="imagenet", include_top=False)
    vgg.trainable = False
    return Model(inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output)

vgg_feature_extractor = build_vgg_feature_extractor()

def vgg_feature_loss(y_true, y_pred):
    """Perceptual loss using VGG19"""
    y_true_features = vgg_feature_extractor(y_true)
    y_pred_features = vgg_feature_extractor(y_pred)
    return tf.reduce_mean(tf.abs(y_true_features - y_pred_features))  # L1 loss on features

# ----------------- Compile Models -----------------
generator = build_generator()
discriminator = build_discriminator()

generator.compile(optimizer=Adam(0.0002, 0.5), loss=vgg_feature_loss)
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# ----------------- Training Loop -----------------
batch_size = 1
epochs = 5000
real_labels = np.ones((batch_size, 1))
fake_labels = np.zeros((batch_size, 1))

for epoch in range(epochs):
    # Generate fake (denoised) images
    denoised_image = generator.predict(noisy_image)

    # Train discriminator on real and fake images
    d_loss_real = discriminator.train_on_batch(noisy_image, real_labels)
    d_loss_fake = discriminator.train_on_batch(denoised_image, fake_labels)
    d_loss = 0.5 * (d_loss_real + d_loss_fake)

    # Train generator (trying to fool discriminator)
    g_loss = generator.train_on_batch(noisy_image, noisy_image)  # Self-supervised training

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
        # if epoch % 500 == 0:
            # generator.save(f"generator_epoch_{epoch}.h5")  # Save model every 500 epochs

# ----------------- Generate and Save Denoised Image -----------------
denoised_image = generator.predict(noisy_image)

denoised_image = np.squeeze(denoised_image)
noisy_image = np.squeeze(noisy_image)

output_path = "C:\\Users\\meetv\\OneDrive\\Documents\\iocl\\cv project\\denoisedImages\\denoised_image.jpg"
cv2.imwrite(output_path, (denoised_image * 255).astype(np.uint8))

# ----------------- Visualization -----------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(noisy_image)
axes[0].set_title("DeNoised Image")
axes[0].axis("off")
axes[1].imshow(denoised_image)
axes[1].set_title("Noised Image")
axes[1].axis("off")

plt.show()
    