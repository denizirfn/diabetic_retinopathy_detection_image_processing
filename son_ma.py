import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_gamma(image, gamma=1.0):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def extract_ma(image):
    # 1. Split channels
    r, g, b = cv2.split(image)
    comp = 255 - g

    plt.figure(figsize=(8, 6))
    plt.title("Yeşil Kanal Görüntüsü")
    plt.imshow(g, cmap='gray')
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("Yeşil Kanal Tamamlayıcısı")
    plt.imshow(comp, cmap='gray')
    plt.axis("off")
    plt.show()


    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    histe = clahe.apply(comp)
    plt.figure(figsize=(8, 6))
    plt.title("CLAHE Uygulanmış Görüntü")
    plt.imshow(histe, cmap='gray')
    plt.axis("off")
    plt.show()


    adjustImage = adjust_gamma(histe, gamma=3)
    comp = 255 - adjustImage
    plt.figure(figsize=(8, 6))
    plt.title("Gamma Ayarlanmış Görüntü (gamma=3)")
    plt.imshow(comp, cmap='gray')
    plt.axis("off")
    plt.show()


    J = adjust_gamma(comp, gamma=4)
    J = 255 - J
    J = adjust_gamma(J, gamma=4)
    plt.figure(figsize=(8, 6))
    plt.title("Gamma Ayarlanmış Görüntü (gamma=4)")
    plt.imshow(J, cmap='gray')
    plt.axis("off")
    plt.show()


    K = np.ones((11, 11), np.float32)
    L = cv2.filter2D(J, -1, K)
    plt.figure(figsize=(8, 6))
    plt.title("Filtre Uygulanmış Görüntü")
    plt.imshow(L, cmap='gray')
    plt.axis("off")
    plt.show()


    ret3, thresh2 = cv2.threshold(L, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    plt.figure(figsize=(8, 6))
    plt.title("Eşiklenmiş Görüntü")
    plt.imshow(thresh2, cmap='gray')
    plt.axis("off")
    plt.show()


    kernel2 = np.ones((9, 9), np.uint8)
    tophat = cv2.morphologyEx(thresh2, cv2.MORPH_TOPHAT, kernel2)
    plt.figure(figsize=(8, 6))
    plt.title("Top-Hat Uygulanmış Görsel")
    plt.imshow(tophat, cmap='gray')
    plt.axis("off")
    plt.show()


    kernel3 = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, kernel3)
    plt.figure(figsize=(8, 6))
    plt.title("Açma İşlemi Uygulanan Görüntü")
    plt.imshow(opening, cmap='gray')
    plt.axis("off")
    plt.show()

    return opening

# Test the function
# Load an example image
image_path = "normal/IDRiD_01.jpg"  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

plt.figure()
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")
plt.show()

# Process the image
result = extract_ma(image)
##################################33
def Ratio_MA(image):
    num_white_pixels = cv2.countNonZero(image)
    total_pixels = image.size
    ratio = num_white_pixels / total_pixels
    return ratio

# Calculate the ratio
ma_ratio = Ratio_MA(result)
print(f"Microaneurysm Ratio: {ma_ratio:.4f}")