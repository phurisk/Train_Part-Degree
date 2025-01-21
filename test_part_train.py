import pandas as pd

import tensorflow as tf

import PIL
from keras import models, layers
from tensorflow.keras import optimizers, callbacks, models, layers
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import os
from keras.callbacks import Callback
from PIL import Image, ImageFile
from IPython.display import Image

from efficientnet.keras import EfficientNetB5 as Net
from efficientnet.keras import center_crop_and_resize, preprocess_input

from torch.utils.data import DataLoader, Dataset



ImageFile.LOAD_TRUNCATED_IMAGES = True




print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Available GPUs:", tf.config.list_physical_devices('GPU'))



dataframe = pd.read_csv (r'/home/phu/Desktop/Data_test_server/output.csv')

train = dataframe[dataframe['split']== 'Train']
test = dataframe[dataframe['split']== 'Valid']

print(train.shape)
print(test.shape)


DATA_PATH = "/home/phu/Desktop/Data_test_server"
os.chdir(DATA_PATH)
train_dir = os.path.join(DATA_PATH, 'Train')
print(train_dir)
test_dir = os.path.join(DATA_PATH, 'Valid')
print(test_dir)


conv_base = Net(weights='imagenet')

batch_size = 20
width = 456
height = 456
epochs = 100

input_shape = (height, width, 3)

conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape)


# create new model with a new classification layer
x = conv_base.output
global_average_layer = layers.GlobalAveragePooling2D(name = 'head_pooling')(x)
dropout_layer_1 = layers.Dropout(0.50,name = 'head_dropout')(global_average_layer)
prediction_layer = layers.Dense(15, activation='softmax',name = 'prediction_layer')(dropout_layer_1)

model = models.Model(inputs= conv_base.input, outputs=prediction_layer)
model.summary()


#showing before&after freezing
print('This is the number of trainable layers '
      'before freezing the conv base:', len(model.trainable_weights))
#conv_base.trainable = False  # freeze to keep convolutional base's weight
for layer in conv_base.layers:
    layer.trainable = False
print('This is the number of trainable layers '
      'after freezing the conv base:', len(model.trainable_weights))  #freez แล้วจะเหลือ max pool and dense


# Train ด้วย ImageDataGenerator ของ Keras ซึ่งจะเพิ่มข้อมูลเสริมระหว่างการฝึกเพื่อลดโอกาสเกิด overfitting
#overfitting เกิดจากข้อมูลที่ซับซ้อนกันเกินไป
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255, #โมเดลส่วนใหญ่ต้องใช้ RGB ในช่วง 0–1
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_dataframe(
        dataframe = train,
        directory = train_dir,
        x_col = 'img_path',
        y_col = 'class_part',
        target_size=(height, width),
        batch_size=batch_size)

test_generator = test_datagen.flow_from_dataframe(
        dataframe = test,
        directory = test_dir,
        x_col = 'img_path',
        y_col = 'class_part',
        target_size=(height, width),
        batch_size=batch_size)


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=2e-5),
              metrics=['acc'])

epochs = 25

# สร้าง dataset แบบตัวอย่าง
class SampleDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return idx

# สร้าง DataLoader และกำหนด num_workers
train_dataset = SampleDataset()
train_loader = DataLoader(train_dataset, batch_size=20, num_workers=2)

# เช็คจำนวน num_workers
print(f'Number of workers: {train_loader.num_workers}')



history = model.fit(
      train_generator,
      steps_per_epoch= len(train)//batch_size,
      epochs=epochs,
      validation_data=test_generator,
      validation_steps= len(test) //batch_size,
      workers=2,
      use_multiprocessing=True)




acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_x = range(len(acc))

plt.plot(epochs_x, acc, 'r', label='Training acc')
plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs_x, loss, 'r', label='Training loss')
plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()



os.makedirs('/home/phu/Desktop/Data_test_server/model_h5_part', exist_ok=True)
model.save('/home/phu/Desktop/Data_test_server/model_h5_part/B5_part_1-18_100epoch.h5')