import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, ReLU, Flatten, Dense, Reshape, Input
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os

# Define Generator
def build_generator():
    inputs = Input(shape=(128, 128, 1))
    
    x = Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = ReLU()(x)
    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2DTranspose(128, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(64, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(1, (4, 4), strides=2, padding='same', activation='tanh')(x)
    
    return Model(inputs, x, name='Generator')

# Define Discriminator
def build_discriminator():
    inputs = Input(shape=(128, 128, 1))
    
    x = Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs, x, name='Discriminator')

# Define GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    inputs = Input(shape=(128, 128, 1))
    generated_image = generator(inputs)
    validity = discriminator(generated_image)
    return Model(inputs, validity, name='GAN')

# Initialize Models
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Compile Models
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')

# Training Parameters
batch_size = 16
epochs = 5000

# Placeholder Dataset (Replace with actual dataset loading)
def generate_noisy_data(size):
    clean_images = np.random.rand(size, 128, 128, 1)
    noisy_images = clean_images + 0.1 * np.random.randn(size, 128, 128, 1)
    noisy_images = np.clip(noisy_images, 0, 1)
    return noisy_images, clean_images

# Training Loop
for epoch in range(epochs):
    noisy_images, clean_images = generate_noisy_data(batch_size)
    
    # Train Discriminator
    fake_images = generator.predict(noisy_images)
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    d_loss_real = discriminator.train_on_batch(clean_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train Generator
    g_loss = gan.train_on_batch(noisy_images, real_labels)
    
    if epoch % 1 == 0:
        print(f"Epoch {epoch}/{epochs} - D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

# Display Results
def display_images(generator, num_images=5):
    noisy_images, clean_images = generate_noisy_data(num_images)
    generated_images = generator.predict(noisy_images)
    
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 10))
    for i in range(num_images):
        axes[i, 0].imshow(noisy_images[i].squeeze(), cmap='gray')
        axes[i, 0].set_title("Noisy Image")
        axes[i, 1].imshow(generated_images[i].squeeze(), cmap='gray')
        axes[i, 1].set_title("Denoised Image")
        axes[i, 2].imshow(clean_images[i].squeeze(), cmap='gray')
        axes[i, 2].set_title("Clean Image")
        
        for ax in axes[i]:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Call the function to display images
display_images(generator)

print("Training Complete!")
