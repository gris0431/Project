import pandas as pd
import os

os.chdir('photo')
output_folder = "photo_network"
directory_data = output_folder
labels = []
image_paths = []
for subdir, _, files in os.walk(directory_data):
    for file in files:
        img_path = os.path.join(subdir, file)
        label = os.path.basename(subdir)
        labels.append(label)
        image_paths.append(img_path)

# Создаем DataFrame с метками, путями к изображениям, путем к папке и именем каждого фото
data = {'image_path': image_paths, 'label': labels}
df = pd.DataFrame(data)

# Сохраняем DataFrame в CSV-файл
df.to_csv('labels.csv', index=False)