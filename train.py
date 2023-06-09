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
    filenames = []
    for i, filename in enumerate(sorted(os.listdir(image_folder))):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.JPG') or filename.endswith('.PNG') or filename.endswith('.JPEG'):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (200, 200))
            cv2.imwrite(os.path.join(output_folder, f'{filename[:-4]}_original.png'), img)

            # Save and append the 90 degree rotated image
            img_rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(output_folder, f'{filename[:-4]}_rotated_90.png'), img_rotated_90)
            images.append(img_rotated_90.flatten())

            # Save and append the 180 degree rotated image
            img_rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
            cv2.imwrite(os.path.join(output_folder, f'{filename[:-4]}_rotated_180.png'), img_rotated_180)
            images.append(img_rotated_180.flatten())

            # Save and append the 270 degree rotated image
            img_rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(output_folder, f'{filename[:-4]}_rotated_270.png'), img_rotated_270)
            images.append(img_rotated_270.flatten())

            filenames.append([f'{filename[:-4]}_original.png', f'{filename[:-4]}_flipped.png', f'{filename[:-4]}_rotated_90.png', f'{filename[:-4]}_rotated_180.png', f'{filename[:-4]}_rotated_270.png'])

    data = np.array(images)

    kmeans = KMeans(n_clusters=3) 
    kmeans.fit(data)

    labels = kmeans.labels_

    for i in range(0, len(labels), 5):
        label = labels[i]
        class_folder = os.path.join(output_folder, f'class_{label}')
        for j in range(5):
            filename = filenames[i // 5][j]
            src_path = os.path.join(output_folder, filename)
            dst_path = os.path.join(class_folder, filename)
            if not os.path.exists(src_path) or os.path.exists(dst_path):
                continue
            shutil.move(src_path, dst_path)


image_folder_car = 'C:/Users/mnb35/Downloads/PYTHON/SIGN/CAR'
output_folder_car = 'C:/Users/mnb35/Downloads/PYTHON/SIGN/CAR_CLASSIFIED'
resize_images(image_folder_car, output_folder_car)

image_folder_test = 'C:/Users/mnb35/Downloads/PYTHON/SIGN/TEST'
output_folder_test = 'C:/Users/mnb35/Downloads/PYTHON/SIGN/TEST_CLASSIFIED'
resize_images(image_folder_test, output_folder_test)
