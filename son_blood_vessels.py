import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize


# Function to Resize the Images
def resize(img):
    ratio = min([1152 / img.shape[0], 1500 / img.shape[1]])
    return cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interpolation=cv2.INTER_CUBIC)

# Function for Adaptive Histogram Equalization
def clahe_equalized(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    return cl1


# Specify input folder and image number
input_folder_train = "A.Segmentation1/IDRiD_02.jpg"
img_number = 1  # Change this to process a specific image

if img_number < 10:
    img_file = os.path.join(input_folder_train)
else:
    img_file = os.path.join(input_folder_train, f"IDRiD_{str(img_number)}.jpg")

img = cv2.imread(img_file)

if img is not None:
    # Process fundus image
    img2 = resize(img)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img2)
    img_enhanced = clahe_equalized(g)

    # Pipeline 1
    img_medf = cv2.medianBlur(img_enhanced, 131)
    img_sub = cv2.subtract(img_medf, img_enhanced)
    img_subf = cv2.blur(img_sub, (5, 5))
    _, img_darkf = cv2.threshold(img_subf, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_darkl = cv2.morphologyEx(img_darkf, cv2.MORPH_OPEN, kernel)

    # Pipeline 2
    img_medf1 = cv2.medianBlur(img_enhanced, 131)
    img_sub1 = cv2.subtract(img_medf1, img_enhanced)
    img_subf1 = cv2.blur(img_sub1, (5, 5))
    _, img_darkf1 = cv2.threshold(img_subf1, 10, 255, cv2.THRESH_BINARY)
    img_darkl1 = cv2.morphologyEx(img_darkf1, cv2.MORPH_OPEN, kernel)

    # Combine results
    img_both = cv2.bitwise_or(img_darkl, img_darkl1)

    # Apply circular mask
    #circular_mask = create_circular_mask(img_both)
    #img_both = cv2.bitwise_and(img_both, img_both, mask=circular_mask)

    # Noise cleaning
    kernel = np.ones((3, 3), np.uint8)
    img_both = cv2.morphologyEx(img_both, cv2.MORPH_OPEN, kernel)
    img_both = cv2.morphologyEx(img_both, cv2.MORPH_CLOSE, kernel)

    # Visualization

    plt.figure(figsize=(8, 6))
    plt.title("orijinal Görüntü")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("Yeniden Boyutlandırılmış Görüntü")
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Görselleştirme adımları (her biri ayrı açılır)
    plt.figure(figsize=(8, 6))
    plt.title("Gri Görüntü")
    plt.imshow(gray, cmap='gray')
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("Kırmızı Kanal")
    plt.imshow(r, cmap="gray")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("Mavi Kanal")
    plt.imshow(b, cmap="gray")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("Yeşil Kanal")
    plt.imshow(g, cmap="gray")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("Yeşil Kanala Uygulanan Clahe Sonucu")
    plt.imshow(img_enhanced, cmap='gray')
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("Pipeline 1 Sonucu")
    plt.imshow(img_darkl, cmap="gray")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("Pipeline 2 Sonucu")
    plt.imshow(img_darkl1, cmap="gray")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("Sonuç Görüntüsü")
    plt.imshow(img_both, cmap="gray")
    plt.axis("off")
    plt.show()


else:
    print("Input image not found.")
####################################
# Skeletonize işlemini hazırlama
def skeletonize_image(image):
    """
    Gri tonlamalı bir görüntüyü skeletonize eder.
    """
    binary_image = image > 0  # İkili görüntüye çevir
    skeleton = skeletonize(binary_image)  # Skeletonize işlemi
    return skeleton.astype(np.uint8) * 255  # Çıktıyı 0 ve 255 değerleriyle çarp

# Damar uzunluğunu hesaplama
def calculate_vessel_length(image):
    """
    Skeletonize edilmiş bir görüntüde damar uzunluğunu hesaplar.
    """
    return np.sum(image > 0)  # Beyaz piksel sayısı toplamı

# Damar uzunluğu işlemi
if img is not None:
    # Daha önceki işlemlerden sonra "img_both" görüntüsü işlenmiş durumdadır.
    skeleton = skeletonize_image(img_both)  # Skeletonize işlemi
    vessel_length = calculate_vessel_length(skeleton)  # Damar uzunluğu hesaplama

    print(f"Damar Uzunluğu: {vessel_length} piksel")

    # Sonuç Görüntüleri
    plt.figure(figsize=(8, 6))
    plt.title("Skeletonized Görüntü")
    plt.imshow(skeleton, cmap="gray")
    plt.axis("off")
    plt.show()

else:
    print("Input image not found.")
from skimage.morphology import skeletonize

# Damar Tortuosities Hesaplama
def calculate_vessel_tortuosity(vessel_image):
    # Skeletonize işlemi
    vessel_skeleton = skeletonize(vessel_image // 255)  # Binary görüntü olmalı
    vessel_length = np.sum(vessel_skeleton)

    # Kontur hesaplama
    contours, _ = cv2.findContours(vessel_skeleton.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)

    # Tortuosity hesaplama
    if vessel_length > 0:
        tortuosity = perimeter / vessel_length
    else:
        tortuosity = 0

    return tortuosity, vessel_length

# img_both görüntüsü için tortuosity ve uzunluk hesaplama
tortuosity, vessel_length = calculate_vessel_tortuosity(img_both)

# Sonuçları yazdır
print(f"Damar Uzunluğu: {vessel_length}")
print(f"Damar Kıvrımlılığı: {tortuosity:.4f}")
