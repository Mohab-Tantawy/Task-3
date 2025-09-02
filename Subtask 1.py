# Setup Commands: (inside VSCode terminal)
## (one-time) python -m venv .venv
## (Windows: every re-open) ./.venv/Scripts/activate.bat
## (Other systems: every re-open) ./.venv/Scripts/activate
## (one-time) pip install matplotlib opencv-python numpy
import numpy as np
import matplotlib.pyplot as plt
import cv2

def convolve(image, kernal):

    # Start by flipping the kernel horizontally then vertically (related to the mathematical proof of convolution)
    if (len(kernal.shape)!=2):
        raise ValueError ("kernal must be 2D")
        exit()
    elif (kernal.shape[0] % 2 == 0 or kernal.shape[1] % 2 == 0):
        raise ValueError("kernal must be odd")
        exit()
    kernal_flipH = np.fliplr(kernal)
    kernal_flipped = np.flipud(kernal_flipH)
    # Pad the image
    kernal_hight, kernal_width = kernal_flipped.shape
    pad_top = kernal_hight // 2
    pad_bottom = kernal_hight // 2
    pad_left = kernal_width // 2
    pad_right = kernal_width // 2

    #edge case handling
    if kernal_hight % 2 == 0:
        pad_bottom = max (0, pad_bottom - 1)
    if kernal_width % 2 == 0:
        pad_right = max (0, pad_right - 1)
    
    padded_image = np.pad(image , ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')

    hight, width = image.shape
    output = np.zeros((hight, width), dtype=np.float32)

    #convulution operation
    for y in range (hight):
        for x in range (width):
            region = padded_image[y:y+kernal_hight, x:x+kernal_width]
            output[y, x] = np.sum(region * kernal_flipped)
    return output
# Take notice that OpenCV handles the image as a numpy array when opening it
def gaussianFilter(image , kernel_size = 5 , sigma = 1.0):
    if kernel_size % 2 == 0:
        raise ValueError("size must be odd")
        exit()
    ax = np.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
    xx ,yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    kernel  = kernel / np.sum(kernel)
    return convolve(image , kernel)

def medianFilter(image , kernel_size = 5):
    if kernel_size % 2 == 0:
        raise ValueError("size must be odd")
        exit()
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='edge')
    output = np.zeros_like(image, dtype=np.float32)
    
    # Apply median filter
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded_image[y:y+kernel_size, x:x+kernel_size]
            output[y, x] = np.median(region)
    
    return output

    

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
img_float = img.astype(np.float32)

# Original image
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')
# Box filter (blur)
axes[0, 1].imshow(convolve(img_float, np.ones((5, 5)) / 25), cmap='gray')
axes[0, 1].set_title('Box Filter')
axes[0, 1].axis('off')
# Horizontal Sobel filter (detects vertical edges)
axes[0, 2].imshow(convolve(img_float, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])), cmap='gray')
axes[0, 2].set_title('Horizontal Sobel Filter')
axes[0, 2].axis('off')
# Vertical Sobel filter (detects horizontal edges)
axes[1, 0].imshow(convolve(img_float, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])), cmap='gray')
axes[1, 0].set_title('Vertical Sobel Filter')
axes[1, 0].axis('off')
# Gaussian filter
gaussian_filtered = gaussianFilter(img_float, kernel_size=5, sigma=1.5)
axes[1  , 1].imshow(gaussian_filtered, cmap='gray')
axes[1, 1].set_title('Gaussian Filter (5x5, σ=1.5)')
axes[1, 1].axis('off')
# Median filter
median_filtered = medianFilter(img_float, kernel_size=5)
axes[1, 2].imshow(median_filtered, cmap='gray')
axes[1, 2].set_title('Median Filter (5x5)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()