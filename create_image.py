import numpy as np
import cv2

# Create a 64x64 image with 3 color channels (BGR)
image = np.zeros((64, 64, 3), dtype=np.uint8)

# Fill the image with some colors
# For example, a blue square
image[10:30, 10:30] = [255, 0, 0]  # Blue
# A green square
image[10:30, 34:54] = [0, 255, 0]  # Green
# A red square
image[34:54, 10:30] = [0, 0, 255]  # Red
# A yellow square
image[34:54, 34:54] = [0, 255, 255] # Yellow

# Save the image
cv2.imwrite("sample_image.png", image)

print("Image created successfully: sample_image.png")
