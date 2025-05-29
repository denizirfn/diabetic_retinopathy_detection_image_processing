import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
 # Görselleştirme için kullanılan backend

# OPTİK DİSK SEGMENTASYONU

def imgResize(img):
    h = img.shape[0]
    w = img.shape[1]
    perc = 500 / w
    w1 = 500
    h1 = int(h * perc)
    img_rs = cv2.resize(img, (w1, h1))
    return img_rs

def kmeansclust(img, k, attempts, max_iter, acc, use='OD'):
    if use == 'OD':
        img_rsp = img.reshape((-1, 1))
    else:
        img_rsp = img.reshape((-1, 3))

    img_rsp = img_rsp.astype('float32')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, acc)
    _, labels, (centers) = cv2.kmeans(img_rsp, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype('uint8')

    labels = labels.flatten()
    seg_img = centers[labels.flatten()]
    seg_img = seg_img.reshape(img.shape)
    return seg_img, labels, centers

def optic_dick(img, temp):
    img_rs = imgResize(img)
    img_grey = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)

    img_k, labels, centers = kmeansclust(img_grey, 4, 10, 400, 0.99)

    metd = cv2.TM_CCOEFF_NORMED
    temp_mat = cv2.matchTemplate(img_k, temp, metd)

    _, _, _, max_loc = cv2.minMaxLoc(temp_mat)
    x = max_loc[0] + 45
    y = max_loc[1] + 45
    temp_mat = img_grey.copy()
    img_mark = cv2.circle(temp_mat, (x, y), 40, 0, -1)

    brightest_cluster_idx = np.argmax(centers)
    mask = (labels.reshape(img_grey.shape) == brightest_cluster_idx).astype(np.uint8) * 255

    circular_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.circle(circular_mask, (x, y), 500, 255, thickness=-1)

    final_mask = cv2.bitwise_and(mask, circular_mask)

    return img_rs, img_grey, img_k, final_mask, (x, y), img_mark

def refine_mask_with_shape_factor(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("Maskede kontur bulunamadı!")

    largest_contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    shape_factor = (4 * np.pi * area) / (perimeter**2)
    print(f"Shape Factor: {shape_factor:.4f}")

    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius *0.99)

    refined_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.circle(refined_mask, center, radius, 255, -1)

    return refined_mask, shape_factor

def invert_and_combine(image, mask):
    resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    inverted_mask = cv2.bitwise_not(resized_mask)
    combined_image = cv2.bitwise_and(image, image, mask=inverted_mask)
    return combined_image, inverted_mask

# Giriş görüntüsü ve şablon
template_path = "template.jpeg"
image_path = "A.Segmentation1/IDRiD_06.jpg"

image = cv2.imread(image_path, -1)
template = cv2.imread(template_path, -1)

if image is None or template is None:
    print("Görüntü veya şablon yüklenemedi!")
else:
    resized_image, gray_image, segmented_image, final_mask, center, img_mark = optic_dick(image, template)

    plt.figure(figsize=(8, 6))
    plt.title("orijinal Görüntü")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Görselleştirme adımları (her biri ayrı açılır)
    plt.figure(figsize=(8, 6))
    plt.title("yeniden boyutlandırılmış Görüntü")
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("Gri Tonlama Görüntü")
    plt.imshow(gray_image, cmap="gray")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("K-Means Segmentasyonu")
    plt.imshow(segmented_image, cmap="gray")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("Şablon Görüntüsü")
    plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


    plt.figure(figsize=(8, 6))
    plt.title("İlk Segmentasyon Maskesi")
    plt.imshow(final_mask, cmap="gray")
    plt.axis("off")
    plt.show()

    refined_mask, shape_factor = refine_mask_with_shape_factor(final_mask)

    plt.figure(figsize=(8, 6))
    plt.title(f"Düzeltilmiş Maske\n(Shape Factor: {shape_factor:.4f})")
    plt.imshow(refined_mask, cmap="gray")
    plt.axis("off")
    plt.show()

    combined_image, inverted_mask = invert_and_combine(resized_image, refined_mask)

    plt.figure(figsize=(8, 6))
    plt.title("Terslenmiş Maske")
    plt.imshow(inverted_mask, cmap="gray")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("Maske uygulanan görüntü")
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

################
def area_optic_disc(image):

    white_pixels = (image == 255).sum()
    return white_pixels

areas_dict = {}  # Beyaz piksel sayısını saklamak için sözlük

# Refined mask üzerindeki beyaz pikselleri hesapla
white_pixel_count = area_optic_disc(refined_mask)
areas_dict["A.Segmentation1/IDRiD_06.jpg"] = white_pixel_count  # Görsel adıyla birlikte sakla

# Hesaplanan alanı yazdır
print(f"Optik disk alanı (beyaz pikseller): {white_pixel_count}")
