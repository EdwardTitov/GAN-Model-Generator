# MNIST GAN (Generative Adversarial Network)

## Overview

This project implements a **Generative Adversarial Network (GAN)** using TensorFlow and Keras to generate handwritten digit images based on the MNIST dataset. The model consists of two neural networks that compete against one another during training:

- **Generator**: Learns to create realistic handwritten digits from random noise.
- **Discriminator**: Learns to distinguish between real MNIST images and generated images.

Through adversarial training, the generator gradually improves until it can produce convincing synthetic handwritten digits.

---

## Model Architecture

### Generator

The generator transforms a 100-dimensional random noise vector into a 28×28 grayscale image.

#### Architecture

| Layer | Output |
|---------|---------|
| Dense | 7 × 7 × 256 |
| Reshape | 7 × 7 × 256 |
| Conv2DTranspose + BatchNorm + ReLU | 7 × 7 × 128 |
| Conv2DTranspose + BatchNorm + ReLU | 14 × 14 × 64 |
| Conv2DTranspose + Tanh | 28 × 28 × 1 |

Key design choices:

- Uses **transposed convolutions** for image upsampling.
- Uses **Batch Normalization** to stabilize training.
- Uses **Tanh activation** in the final layer because training images are normalized to `[-1, 1]`.

---

### Discriminator

The discriminator functions as a binary classifier.

#### Architecture

| Layer | Output |
|---------|---------|
| Conv2D + LeakyReLU | 14 × 14 × 64 |
| Conv2D + LeakyReLU | 7 × 7 × 64 |
| Flatten | 3136 |
| Dense | 1 Logit |

Key design choices:

- Uses **LeakyReLU** to avoid dead neurons.
- Outputs logits directly for compatibility with binary cross entropy loss configured with `from_logits=True`.

---

## Dataset

The model is trained using the MNIST handwritten digit dataset.

### Preprocessing Steps

1. Convert images to `float32`
2. Normalize pixel values from `[0, 255]` to `[-1, 1]`
3. Add a channel dimension to convert shape:

```python
(28, 28) → (28, 28, 1)
```

4. Build an optimized `tf.data.Dataset` pipeline with:

- Shuffle
- Batch
- Cache
- Prefetch

This significantly reduces data loading bottlenecks during training.

---

## Training Process

Training follows the standard GAN procedure:

### Step 1

Generate random noise:

```python
noise = tf.random.normal([batch_size, latent_dim])
```

### Step 2

Generator creates fake images.

### Step 3

Discriminator evaluates:

- Real MNIST images
- Generated images

### Step 4

Calculate losses:

#### Generator Loss

The generator attempts to fool the discriminator:

```python
gen_loss = cross_entropy(
    tf.ones_like(fake_output),
    fake_output
)
```

#### Discriminator Loss

The discriminator attempts to correctly classify:

- Real images as real
- Generated images as fake

```python
disc_loss = disc_real_loss + disc_fake_loss
```

### Step 5

Backpropagation updates both networks using Adam optimizers.

---

## Label Smoothing

To improve training stability, label smoothing is applied:

```python
real_label = 0.9
fake_label = 0.0
```

Instead of labeling real samples as `1.0`, they are labeled as `0.9`.

Benefits:

- Prevents discriminator overconfidence.
- Produces smoother gradients.
- Helps reduce GAN instability.

---

## Performance Optimization with `@tf.function`

One of the most impactful optimizations was decorating the training step with:

```python
@tf.function
```

### Why It Was Added

Without the decorator:

- TensorFlow executes operations eagerly.
- Every operation is interpreted individually.
- Training becomes noticeably slower over hundreds of epochs.

With `@tf.function`:

- TensorFlow converts Python code into a static computation graph.
- Operations are optimized and compiled.
- GPU utilization improves significantly.
- Training throughput increases substantially.

### Challenges Encountered

Adding `@tf.function` was not completely straightforward.

#### Dynamic Batch Sizes

The final batch in a dataset is often smaller than the configured batch size.

A fixed noise tensor such as:

```python
noise = tf.random.normal([batch_size, latent_dim])
```

can create shape mismatches when the final batch contains fewer samples.

To resolve this issue:

```python
current_bs = tf.shape(images)[0]
noise = tf.random.normal([current_bs, latent_dim])
```

The noise tensor dynamically matches the incoming batch size, allowing graph execution to remain stable.

#### Debugging Difficulties

Graph execution makes debugging more difficult because:

- Errors are often reported from generated graph code.
- Python print statements behave differently.
- Tensor values cannot always be inspected directly.

Several iterations were required to identify shape mismatches and ensure compatibility with graph tracing.

---

## Google Drive Integration

A major challenge during development was overcoming local hardware limitations.

Training for:

- 500 epochs
- Frequent image generation
- Checkpoint saving

can consume significant storage and compute resources.

To address this, Google Drive was mounted directly inside Google Colab:

```python
drive.mount("/content/drived")
```

Generated images are automatically saved to a Drive directory:

```python
IMAGE_DIR = "/content/drived/MyDrive/Generative_Image_Project/Generated_Images"
```

### Benefits

- Persistent storage across Colab sessions.
- No reliance on local disk space.
- Generated images remain available even after runtime disconnects.
- Easier experiment tracking and model monitoring.

This approach effectively bypassed the storage restrictions of local hardware and temporary Colab environments.

---

## Checkpointing

Training checkpoints are saved periodically:

```python
checkpoint_manager.save()
```

Stored items include:

- Generator weights
- Discriminator weights
- Optimizer states

Benefits:

- Resume interrupted training.
- Prevent loss of progress.
- Compare model performance across training runs.

---

## Generated Images

Every 10 epochs, the generator produces a fixed set of samples using a constant random seed:

```python
seed = tf.random.normal([16, latent_dim])
```

Using a fixed seed allows visual comparison of image quality throughout training.

### Early Training

Generated samples resemble random noise.

### Mid Training

Basic digit structures begin to emerge.

### Late Training

Images become increasingly recognizable as handwritten digits.

The generated image grids are automatically saved to Google Drive, creating a visual timeline of the model's learning progress.

Example progression:

- Epoch 10 → Mostly noise
- Epoch 100 → Rough digit outlines
- Epoch 250 → Recognizable digits
- Epoch 500 → High-quality handwritten digit samples

Screenshots of these generated outputs can be included in the repository to demonstrate training progression and final model performance.

---

## Hyperparameters

| Parameter | Value |
|------------|--------|
| Batch Size | 256 |
| Latent Dimension | 100 |
| Epochs | 500 |
| Generator Learning Rate | 0.0002 |
| Discriminator Learning Rate | 0.0002 |
| Adam Beta 1 | 0.5 |

---

## Results

After training, the GAN successfully learns the MNIST distribution and generates realistic handwritten digits from random noise vectors.

Notable achievements include:

- Stable adversarial training.
- Efficient graph execution using `@tf.function`.
- Automated checkpoint recovery.
- Automated image generation and storage.
- Google Drive integration for persistent experiment tracking.
- Progressive improvement of generated digit quality over 500 training epochs.

---

## Future Improvements

Potential enhancements include:

- Wasserstein GAN (WGAN)
- Gradient Penalty (WGAN-GP)
- Spectral Normalization
- Mixed Precision Training
- TensorBoard Monitoring
- Larger convolutional architectures
- Training on more complex datasets such as Fashion-MNIST or CIFAR-10

---

## Technologies Used

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Google Colab
- Google Drive API Integration
- tqdm

---
