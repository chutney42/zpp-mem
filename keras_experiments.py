from keras.datasets import cifar10
import os

from keras.datasets import cifar10
from keras.layers import Conv2D
from keras.layers import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print(X_train.shape)
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train/=255
    X_test/=255

    number_of_classes = 10

    Y_train = np_utils.to_categorical(y_train, number_of_classes)
    Y_test = np_utils.to_categorical(y_test, number_of_classes)

    model = Sequential()
    model.add(Conv2D(10, (3, 3), input_shape=(32,32,3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Flatten())
    # Fully connected layer
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                             height_shift_range=0.08, zoom_range=0.08)
    test_gen = ImageDataGenerator()
    train_generator = gen.flow(X_train, Y_train, batch_size=64)
    test_generator = test_gen.flow(X_test, Y_test, batch_size=64)
    model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=5,
                        validation_data=test_generator, validation_steps=10000//64)
