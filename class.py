import os
import cv2
import numpy as np
import shutil
from sklearn.cluster import KMeans

def resize_images(image_folder, output_folder):
    for i in range(3): 
        class_folder = os.path.join(output_folder, f'class_{i}')
        os.makedirs(class_folder, exist_ok=True)
    images = []
    for i, filename in enumerate(sorted(os.listdir(image_folder))):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.JPG') or filename.endswith('.PNG') or filename.endswith('.JPEG'):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img = cv2.resize(img, (200, 200))  
            images.append(img.flatten()) 

            resized_img_path = os.path.join(output_folder, filename)
            cv2.imwrite(resized_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            flipped_img = cv2.flip(img, 1)  
            flipped_img_path = os.path.join(class_folder, f'flipped_{filename}')
            cv2.imwrite(flipped_img_path, cv2.cvtColor(flipped_img, cv2.COLOR_RGB2BGR))

    data = np.array(images)

    kmeans = KMeans(n_clusters=3) 
    kmeans.fit(data)

    labels = kmeans.labels_

    for i, label in enumerate(labels):
        img_path = os.path.join(output_folder, sorted(os.listdir(image_folder))[i])  # 使用原图像的名称
        class_folder = os.path.join(output_folder, f'class_{label}')

        shutil.move(img_path, class_folder)


image_folder_car = 'C:/Users/mnb35/Downloads/PYTHON/SIGN/CAR'
output_folder_car = 'C:/Users/mnb35/Downloads/PYTHON/SIGN/CAR_CLASSIFIED'
resize_images(image_folder_car, output_folder_car)

image_folder_test = 'C:/Users/mnb35/Downloads/PYTHON/SIGN/TEST'
output_folder_test = 'C:/Users/mnb35/Downloads/PYTHON/SIGN/TEST_CLASSIFIED'
resize_images(image_folder_test, output_folder_test)


#It defines a function resize_images that takes an image folder path and an output folder path as input.
#It creates subfolders in the output folder for different classes (in this case, 3 classes).
#It loads image data from the input folder, resizes the images to a specified size (200x200), and flattens them into 1D vectors.
#It saves the resized images and their horizontally flipped versions to the output folder.
#It converts the image data into a NumPy array.
#It creates a K-Means clustering model and fits the data.
#It assigns cluster labels to the images.
#It moves the image files to their respective class folders based on the assigned labels.