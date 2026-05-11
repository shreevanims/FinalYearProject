import cv2
import sys

if len(sys.argv) != 2:
    print("Usage: python3 load.py <size>")
    exit()

size = int(sys.argv[1])

# Load image
img = cv2.imread("sample.jpg")

if img is None:
    print("Error: Could not load image")
    exit()

# Resize
img = cv2.resize(img, (size, size))

# Normalize (0–1)
img = img.astype('float32') / 255.0

# 🔥 Convert HWC → CHW (VERY IMPORTANT)
img = img.transpose(2, 0, 1)

# Save as binary
img.tofile("image.bin")

print(f"Image prepared: {size}x{size} (CHW format)")