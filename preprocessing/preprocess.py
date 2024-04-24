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


# Create an empty list to store the patches
patched_images = []
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

    for patch in patches:
        patched_images.append(patch)
    
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

patched_dataset = tf.data.Dataset.from_tensor_slices(patched_images)



class DiffusionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, name=None):
        super(DiffusionBlock, self).__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.conv5 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.conv6 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')

    def call(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(tf.nn.relu(x1))
        x3 = self.conv3(tf.nn.relu(x2))
        x4 = self.conv4(tf.nn.relu(x3))
        x5 = self.conv5(tf.nn.relu(x4))
        x6 = self.conv6(tf.nn.relu(x5))
        return x6

# Step 5: Define Diffusion Probabilistic Model
class DiffusionProbabilisticModel(tf.keras.Model):
    def __init__(self, num_blocks, filters, kernel_size):
        super(DiffusionProbabilisticModel, self).__init__()
        self.num_blocks = num_blocks
        self.blocks = [DiffusionBlock(filters, kernel_size, name=f'diffusion_block_{i}') for i in range(num_blocks)]
        self.final_conv = tf.keras.layers.Conv2D(3, kernel_size, padding='same')

    def call(self, x, noise_level):
        # Apply diffusion blocks
        for i in range(self.num_blocks):
            x = self.blocks[i](x)
            x *= noise_level  # Multiply by noise level

        # Final convolution to generate output image
        x = self.final_conv(x)
        return x
def train_diffusion_probabilistic_model(dataset, ddpm_model, epochs):
    optimizer = tf.keras.optimizers.Adam()  # Define optimizer
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        for step, input_batch in enumerate(dataset):
            noise_level = tf.random.uniform((input_batch.shape[0],), minval=0.1, maxval=0.5)  # Random noise level
            
            with tf.GradientTape() as tape:
                # Forward pass
                output_image = ddpm_model(input_batch, noise_level)
                # Compute loss
                loss = compute_loss(output_image, input_batch, noise_level)
            
            # Backpropagation
            gradients = tape.gradient(loss, ddpm_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, ddpm_model.trainable_variables))
            
            # Print loss every few steps
            if (step + 1) % 100 == 0:
                print(f"Step {step + 1}, Loss: {loss.numpy():.4f}")

            # Plot sample input and output images
            if (step + 1) % 500 == 0:
                plot_sample_images(input_batch, output_image)

def compute_loss(output_image, target_image, noise_level):
    # Define your loss function here (e.g., MSE loss)
    loss = tf.reduce_mean(tf.square(output_image - target_image))
    return loss

def plot_sample_images(input_images, output_images):
    num_samples = min(input_images.shape[0], output_images.shape[0], 5)  # Plot at most 5 samples
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))
    
    for i in range(num_samples):
        axes[i, 0].imshow(input_images[i])
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(output_images[i])
        axes[i, 1].set_title('Output Image')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


num_blocks = 6
filters = 64
kernel_size = (3, 3)
ddpm_model = DiffusionProbabilisticModel(num_blocks, filters, kernel_size)
train_diffusion_probabilistic_model(patched_dataset, ddpm_model, epochs)