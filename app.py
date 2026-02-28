import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Read Image (Grayscale)
# -----------------------------
image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found!")
    exit()

# -----------------------------
# 2️⃣ Sobel Edge Detection
# -----------------------------
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
gradient_magnitude = cv2.normalize(
    gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX
)
sobel_edges = np.uint8(gradient_magnitude)

# -----------------------------
# 3️⃣ Morphological Edge Detection
# -----------------------------
kernel = np.ones((3, 3), np.uint8)

eroded = cv2.erode(image, kernel, iterations=1)
morph_edges = cv2.subtract(image, eroded)

# -----------------------------
# 4️⃣ Display Results
# -----------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Sobel Edges")
plt.imshow(sobel_edges, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Eroded Image")
plt.imshow(eroded, cmap='gray')
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("Morphological Edges")
plt.imshow(morph_edges, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
