import numpy as np
import matplotlib.pyplot as plt
import cv2
# Take notice that OpenCV handles the image as a numpy array when opening it 
img = cv2.imread('shapes.jpg')
out = img.copy()
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #hsv

# Make a mask for each color (red, blue, black)
# blue mask
blue_mask = cv2.inRange(hsv_image, np.array([100, 50, 50]), np.array([130, 255, 255]))
#red mask 
red_mask = cv2.inRange(hsv_image, np.array([0, 50, 50]), np.array([10, 255, 255]))
#black mask
black_mask = cv2.inRange(hsv_image, np.array([0, 0, 0]), np.array([180, 255, 30]))

# Change all pixels that fit within the blue mask to black
out[blue_mask > 0] = [0, 0, 0]
# Change all pixels that fit within the red mask to blue
out[red_mask > 0] = [255, 0, 0]
# Change all pixels that fit within the black mask to red
out[black_mask > 0] = [0, 0, 255]

# Display the original and processed images side by side
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(out)
axes[1].set_title('Processed Image')
axes[1].axis('off')

plt.show()