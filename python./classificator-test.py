from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
import os
from keras.preprocessing import image

# Указываем разрешение для изображений к единому формату
image_width, image_height = 250, 250

# Указываем путь к исходной папке с данными
directory_data = 'dir/data'

# Функция для загрузки изображений и меток из директории
def load_images_from_directory(directory):
    images = []
    labels = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            img_path = os.path.join(subdir, file)
            img = image.load_img(img_path, target_size=(image_width, image_height))
            img_array = image.img_to_array(img)
            img_array /= 255.0
            images.append(img_array)
            label = os.path.basename(subdir)
            labels.append(label)
    return np.array(images), np.array(labels)

# Загрузка данных
X, y = load_images_from_directory(directory_data)

# Разделение данных на выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразуем метки в категориальный формат
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_enc = to_categorical(label_encoder.fit_transform(y_train))
y_test_enc = to_categorical(label_encoder.transform(y_test))

# Настройка параметров модели
if K.image_data_format() != 'channels_first':
    input_shape = (image_width, image_height, 3)
else:
    input_shape = (3, image_width, image_height)

pattern = Sequential() # Создание модели

# Первый слой нейросети
pattern.add(Conv2D(32, (3, 3), input_shape=input_shape))
pattern.add(Activation('relu'))
pattern.add(MaxPooling2D(pool_size=(2, 2)))

# Второй слой нейросети
pattern.add(Conv2D(32, (3, 3)))
pattern.add(Activation('relu'))
pattern.add(MaxPooling2D(pool_size=(2, 2)))

# Третий слой нейросети
pattern.add(Conv2D(64, (3, 3)))
pattern.add(Activation('relu'))
pattern.add(MaxPooling2D(pool_size=(2, 2)))

#Активация, свертка, объединение, исключение
pattern.add(Flatten())
pattern.add(Dense(64))
pattern.add(Activation('relu'))
pattern.add(Dropout(0.5))
pattern.add(Dense(len(np.unique(y)))) # число классов
pattern.add(Activation('softmax'))

#Компилируем модель с выбранными параметрами. Укажем метрику для оценки.
pattern.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

# Задаём параметры аугментации
train_datagen = ImageDataGenerator(
    rescale=1. / 255, # коэффициент масштабирования
    shear_range=0.2, # Интенсивность сдвига
    zoom_range=0.2, # Диапазон случайного увеличения
    horizontal_flip=True) # Произвольный поворот по горизонтали

# Предобработка обучающей выборки
train_datagen.fit(X_train)

# Обучение модели
pattern.fit(
    train_datagen.flow(X_train, y_train_enc, batch_size=25),
    steps_per_epoch=len(X_train) // 25,
    epochs=8,
    validation_data=(X_test, y_test_enc),
    validation_steps=len(X_test) // 25)

# Сохранение весов модели
pattern.save_weights('first_model_weights.h5')
pattern.save('first_model.h5')

# Загрузка и использование модели для предсказания
pattern.load_weights('first_model_weights.h5')
prediction = pattern.predict(X_test)
