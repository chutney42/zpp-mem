import os
import keras
from keras.datasets import cifar100
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    print(X_train.shape)
    print(X_test.shape)
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train -= 128
    X_test -= 128
    X_train /= 255
    X_test /= 255

    number_of_classes = 100

    Y_train = np_utils.to_categorical(y_train, number_of_classes)
    Y_test = np_utils.to_categorical(y_test, number_of_classes)

    m = Sequential()
    m.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32,32,3)))
    m.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    m.add(MaxPooling2D((2, 2), strides=(2, 2)))

    m.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    m.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    m.add(MaxPooling2D((2, 2), strides=(2, 2)))

    m.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    m.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    m.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    m.add(MaxPooling2D((2, 2), strides=(2, 2)))

    m.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    m.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    m.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    m.add(MaxPooling2D((2, 2), strides=(2, 2)))

    m.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    m.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    m.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    m.add(MaxPooling2D((2, 2), strides=(2, 2)))

    m.add(Flatten())
    m.add(Dense(4096, activation='relu'))
    m.add(BatchNormalization(axis=-1))
    m.add(Dense(4096, activation='relu'))
    m.add(BatchNormalization(axis=-1))

    m.add(Dense(100, activation='softmax'))

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./demo', histogram_freq=0, write_graph=True, write_images=True)


    m.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.005, momentum=0.9), metrics=['accuracy'])
    m.summary()
    gen = ImageDataGenerator()
    test_gen = ImageDataGenerator()
    train_generator = gen.flow(X_train, Y_train, batch_size=256)
    test_generator = test_gen.flow(X_test, Y_test, batch_size=256)
    m.fit_generator(train_generator, steps_per_epoch=50000 // 256, epochs=20, validation_data=test_generator,
                    validation_steps=10000 // 256, callbacks=[tbCallBack])
