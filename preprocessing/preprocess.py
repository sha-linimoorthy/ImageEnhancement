import tensorflow as tf
import os


# Step 1: Define Function to Load and Preprocess Images
def load_and_preprocess_image(image_path):
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # Preprocess image (e.g., resize, normalize)
    image = tf.image.resize(image, [256, 256])
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

# Step 2: Define Function to Create Dataset
def create_dataset(data_dir, batch_size):
    # Get list of image paths
    image_paths = [os.path.join(data_dir, folder, image_file) 
                   for folder in os.listdir(data_dir)
                   for image_file in os.listdir(os.path.join(data_dir, folder))]
    # Create dataset from image paths
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    # Map loading and preprocessing function to each image path
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    # Batch dataset
    dataset = dataset.batch(batch_size)
    return dataset

# Step 3: Denoising Autoencoder Model
def denoising_autoencoder(input_shape):
    # Encoder
    inputs = tf.keras.layers.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(conv1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(conv2)

    # Decoder
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    up1 = tf.keras.layers.UpSampling2D((2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    up2 = tf.keras.layers.UpSampling2D((2, 2))(conv4)
    decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2)  # Adjust output channels as needed

    # Autoencoder model
    autoencoder = tf.keras.models.Model(inputs, decoded)
    return autoencoder

def train_denoising_autoencoder(dataset, epochs=10, batch_size=32):
    # Initialize model
    model = denoising_autoencoder(input_shape=(256, 256, 3))  # Adjust input shape as needed
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    # Custom training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, input_batch in enumerate(dataset):
            # Forward pass
            with tf.GradientTape() as tape:
                reconstructed_images = model(input_batch, training=True)
                # Compute loss
                loss = tf.reduce_mean(tf.square(input_batch - reconstructed_images))
            # Backpropagation
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # Print loss every few steps
            if (step + 1) % 100 == 0:
                print(f"Step {step + 1}, Loss: {loss.numpy():.4f}")

  
# Step 4: Apply Histogram Equalization
def apply_histogram_equalization(image):
    # Apply histogram equalization
    # Convert image to grayscale
    image_gray = tf.image.rgb_to_grayscale(image)
    # Convert to numpy array
    image_gray_np = image_gray.numpy()
    # Convert numpy array to uint8
    image_gray_uint8 = (image_gray_np * 255).astype(np.uint8)
    # Apply histogram equalization
    image_eq = cv2.equalizeHist(image_gray_uint8)
    # Convert back to RGB
    image_eq_rgb = cv2.cvtColor(image_eq, cv2.COLOR_GRAY2RGB)
    return image_eq_rgb

# Step 5: Apply Local Contrast Enhancement
def apply_local_contrast_enhancement(image):
    # Apply local contrast enhancement (e.g., adaptive histogram equalization)
    image_enhanced = tf.image.adjust_contrast(image, contrast_factor=2.0)
    return image_enhanced

# Step 6: Extract Patches
def extract_patches(image, patch_size):
    # Extract patches from the image
    patches = tf.image.extract_patches(
        images=tf.expand_dims(image, axis=0),  # Expand dimensions to add batch dimension
        sizes=[1, patch_size[0], patch_size[1], 1],
        strides=[1, patch_size[0], patch_size[1], 1],
        rates=[1, 1, 1, 1],
        padding='SAME'
    )
    return patches

# Example usage
data_dir = "../image slice-T2"
batch_size = 32
epochs = 10

# Step 1: Create Input and Target Datasets
dataset = create_dataset(data_dir, batch_size)


sample_images = next(iter(dataset))
sample_images = sample_images[:5]  # Extract first 5 samples for visualization

# Visualize original images
plt.figure(figsize=(15, 5))
for i in range(len(sample_images)):
    plt.subplot(1, len(sample_images), i+1)
    plt.imshow(sample_images[i])
    plt.title('Original')
    plt.axis('off')
plt.show()


for step, image in enumerate(sample_images):
    # Step 1: Apply Histogram Equalization
    image_histogram_equalized = apply_histogram_equalization(image)

    # Step 2: Apply Local Contrast Enhancement
    image_local_contrast_enhanced = apply_local_contrast_enhancement(image)

    # Step 3: Extract Patches
    patch_size = (64, 64)
    patches = extract_patches(image, patch_size)

    # Reshape patches for visualization
    patches = tf.reshape(patches, [-1, patch_size[0], patch_size[1], 3])  # Assuming RGB images

    # Plot the preprocessed images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_histogram_equalized)
    plt.title('Histogram Equalization')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image_local_contrast_enhanced)
    plt.title('Local Contrast Enhancement')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(patches[0])  # Plot the first patch
    plt.title('Patched Image')
    plt.axis('off')

    plt.show()

