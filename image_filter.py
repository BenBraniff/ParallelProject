import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import kagglehub
import os
import time

# CUDA kernel to process the image (convert to grayscale)
kernel_code = """
__global__ void process_image_kernel(unsigned char *d_image, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;

        // Convert to grayscale by averaging the RGB values
        unsigned char gray_value = (d_image[idx] + d_image[idx + 1] + d_image[idx + 2]) / 3;
        d_image[idx] = gray_value;     // Red channel (after grayscale)
        d_image[idx + 1] = gray_value; // Green channel (after grayscale)
        d_image[idx + 2] = gray_value; // Blue channel (after grayscale)
    }
}
"""

def convert_image_to_matrix_cuda(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)  # Read the image in color (BGR)
    if image is None:
        print(f"Error loading image at {image_path}!")
        return None

    # Get image dimensions
    height, width, channels = image.shape
    image_size = width * height * channels

    # Flatten the image into a 1D array
    image_data = image.flatten().astype(np.uint8)

    # Allocate memory on the device (GPU)
    d_image = cuda.mem_alloc(image_data.nbytes)

    # Copy the image data from host to device
    cuda.memcpy_htod(d_image, image_data)

    # Compile the kernel
    mod = SourceModule(kernel_code)
    process_image_kernel = mod.get_function("process_image_kernel")

    # Define block and grid dimensions for kernel launch
    block_size = (16, 16, 1)
    grid_size = ((width + block_size[0] - 1) // block_size[0], 
                 (height + block_size[1] - 1) // block_size[1])

    # Launch the kernel to process the image (convert to grayscale)
    process_image_kernel(d_image, np.int32(width), np.int32(height), np.int32(channels), 
                         block=block_size, grid=grid_size)

    # Copy the result back to host memory
    cuda.memcpy_dtoh(image_data, d_image)

    # Reshape the processed data into the original image dimensions
    processed_image = image_data.reshape((height, width, channels))

    # Save the processed image (grayscale)
    output_path = image_path.replace(".jpg", "_processed.jpg")
    cv2.imwrite(output_path, processed_image)
    return output_path

def process_batch(batch_images):
    start_time = time.time()

    processed_images = []
    for image_path in batch_images:
        processed_image = convert_image_to_matrix_cuda(image_path)
        if processed_image:
            processed_images.append(processed_image)

    end_time = time.time()
    print(f"Processed {len(processed_images)} images in {end_time - start_time:.2f} seconds.")
    return processed_images

def main():
    # Download the dataset using KaggleHub
    dataset_path = kagglehub.dataset_download("kmader/food41")
    
    # Generate a list of image paths to process (example for "apple_pie" folder)
    image_folder = os.path.join(dataset_path, "images", "apple_pie")
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]
    
    # Process images in batches of 50
    batch_size = 50
    for i in range(0, len(image_paths), batch_size):
        batch_images = image_paths[i:i + batch_size]
        process_batch(batch_images)

if __name__ == "__main__":
    main()
