import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import kagglehub
import os
import time
import matplotlib.pyplot as plt


# Ben

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

def convert_image_to_matrix_cuda(image_path, block_size = (16, 16, 1)):
    #                                         ^^^
    # Made block_size a parameter so we can control it from the main function -Ben

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
    # made the grid size a parameter in function
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

#                                 \/ added this in parameter -Ben
def process_batch(batch_images, block_size = (16, 16, 1)):
    start_time = time.time()

    processed_images = []
    # process images in the batch linearly (or non-parallel way) using a for loop
    for image_path in batch_images:
        #                                                           \/ and here Ben
        processed_image = convert_image_to_matrix_cuda(image_path, block_size)
        if processed_image:
            processed_images.append(processed_image)

    end_time = time.time()
    print(f"Processed batch {len(processed_images)} images in {end_time - start_time:.2f} seconds.")
    return processed_images


def main():
    # Download the dataset using KaggleHub
    dataset_path = kagglehub.dataset_download("kmader/food41")
    
    # Generate a list of image paths to process (example for "apple_pie" folder)
    image_folder = os.path.join(dataset_path, "images", "apple_pie")
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".jpeg"))] # could end in .jpeg too -Ben
    
    # old code
    '''
    # This block_size is where we can control the thread count -Ben
    block_size = (16, 16, 1)

    # Process images in batches of 50
    batch_size = 50
    total_start_time = time.time()
    for i in range(0, len(image_paths), batch_size):
        batch_images = image_paths[i:i + batch_size]
        process_batch(batch_images, block_size=block_size)
        #                            ^ added this -Ben
    total_end_time = time.time()
    print(f"Total processing time: {total_end_time - total_start_time:.2f} seconds.")
    '''

    # new code testing mulitple block sizes
    '''
    This block_sizes is where we can control the thread count -Ben
    You can experiment with (8, 8, 1), (16, 16, 1), (32, 32, 1), (64, 1, 1), etc.
    (blockDim.x, blockDim.y, blockDim.z)
    blockDim.x: Number of threads in the x-direction (width of the block).
    blockDim.y: Number of threads in the y-direction (height of the block).
    blockDim.z: Number of threads in the z-direction (depth of the block, typically 1 unless doing 3D computations).
    
    More to test:
    (4, 4, 1): Small block, 16 threads per block. Great for small images.
    (8, 4, 1): 32 threads per block. A good balance for some cases.
    (32, 8, 1): 256 threads per block. Larger block sizes may be more efficient.
    (64, 4, 1): 256 threads per block, but fewer threads in the y-direction.
    (64, 16, 1): A large block size of 1024 threads per block. Useful for very large images or high-performance GPUs.
    
    total_threads = grid_dim.x * grid_dim.y * block_dim.x * block_dim.y
    '''
    block_sizes = [(8, 8, 1), (16, 16, 1), (32, 32, 1), (64, 1, 1), (4, 4, 1), (32, 8, 1)]
    times = []

    for block_size in block_sizes:
        start_time = time.time()
        # Process images in batches of 50 (for each block size)
        batch_size = 50
        for i in range(0, len(image_paths), batch_size):
            batch_images = image_paths[i:i + batch_size]
            process_batch(batch_images, block_size=block_size)
        
        end_time = time.time()
        times.append(end_time - start_time)

    # Plotting the resulting times for differing block sizes
    block_labels = [f"{b[0]}x{b[1]}" for b in block_sizes]
    plt.plot(block_labels, times, marker='o')
    plt.xlabel('Block Size (threads per block)')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time vs Block Size')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

# I can't run this because of my mac doesn't have an nvidia GPU :( -Ben