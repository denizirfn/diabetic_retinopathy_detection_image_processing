import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

# Define file paths
image_path = "A.Segmentation1/IDRiD_06.jpg"  # Replace with the path to your input image
save_path = "sonuc_exa.jpg"   # Replace with the directory to save results

# Create save_path if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load the input image
img = cv.imread(image_path, -1)
if img is None:
    raise ValueError(f"Input image not found: {image_path}")


# Resize the image
def imgResize(img):
    h, w = img.shape[:2]
    perc = 500 / w
    w1, h1 = 500, int(h * perc)
    img_rs = cv.resize(img, (w1, h1))
    return img_rs

img_resized = imgResize(img)

# Convert to grayscale
img_gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

# K-Means clustering for segmentation
def kmeansClust(img, k, attempts, max_iter, acc, use='OD'):
    if use == 'OD':
        img_rsp = img.reshape((-1, 1))
    else:
        img_rsp = img.reshape((-1, 3))

    img_rsp = img_rsp.astype('float32')
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iter, acc)
    _, labels, centers = cv.kmeans(img_rsp, k, None, criteria, attempts, cv.KMEANS_RANDOM_CENTERS)
    centers = centers.astype('uint8')

    labels = labels.flatten()
    seg_img = centers[labels.flatten()]
    seg_img = seg_img.reshape(img.shape)
    return seg_img

img_kmeans = kmeansClust(img_gray, 6, 10, 400, 0.99)

# Visualize K-Means segmentation result
plt.imshow(img_kmeans, cmap="gray")
plt.title("K-Means segmentasyonu")
plt.axis("off")
plt.show()

# Create a circular template
template = np.ones((95, 95), dtype="uint8") * 0
template = cv.circle(template, (47, 47), 46, 255, -1)

# Template matching
method = cv.TM_CCOEFF_NORMED
temp_mat = cv.matchTemplate(img_kmeans, template, method)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(temp_mat)
x, y = max_loc[0] + 45, max_loc[1] + 45

# Draw the optic disc mask
img_mark = img_gray.copy()
img_mark = cv.circle(img_mark, (x, y), 40, 0, -1)

# Perform exudates detection
_, img_gc, _ = cv.split(img_resized)
clus_seg = kmeansClust(img_resized, 6, 5, 20, 0.69, use='EX')
clus_seg_gray = cv.cvtColor(clus_seg, cv.COLOR_BGR2GRAY)


# Thresholding for large exudates
_, kthm = cv.threshold(clus_seg_gray, np.max(clus_seg_gray) - 1, 255, cv.THRESH_BINARY)


# Edge detection for small exudates
def cannyEdges(img, th1, th2):
    edges = cv.Canny(img, th1, th2)
    return edges

edges = cannyEdges(img_gc, 70, 120)


# Combine masks for final exudates detection
img_clean = kthm.copy()
img_final = cv.bitwise_or(kthm, img_clean)
img_final[img_mark == 0] = 0


# Save the final result
final_path = os.path.join(save_path, "final_exudates_mask.jpg")
cv.imwrite(final_path, img_final)
print(f"Final mask saved at:{final_path}")
################################################################
# Exudate Density Hesaplama
def calculate_exudate_density(image):

    exudate_count = cv2.countNonZero(image)  # Beyaz (eksüda) piksel sayısı
    total_pixels = image.size  # Toplam piksel sayısı
    exudate_density = exudate_count / total_pixels  # Yoğunluk hesaplama
    return exudate_density

# img_final üzerinde hesaplama yapalım
exudate_density = calculate_exudate_density(img_final)

# Sonucu yazdır
print(f"Exudate Density (Eksüda Yoğunluğu): {exudate_density:.4f}")