# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SmallerVGGNet:
        @staticmethod
        def build(width, height, depth, classes):
                # initialize the model along with the input shape to be
                # "channels last" and the channels dimension itself
                model = Sequential()
                inputShape = (height, width, depth)
                chanDim = -1

                # if we are using "channels first", update the input shape
                # and channels dimension
                if K.image_data_format() == "channels_first":
                        inputShape = (depth, height, width)
                        chanDim = 1

                        # CONV => RELU => POOL
                model.add(Conv2D(3, (11, 11), activation='relu', input_shape=inputShape))
                model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

                model.add(Conv2D(96, (5, 5), activation='relu'))
                model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
                model.add(Dropout(0.5))
                model.add(Conv2D(192, (3, 3)))
                model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
                model.add(Dropout(0.5))
                model.add(Conv2D(192, (3, 3)))

                model.add(Flatten())
                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                # return the constructed network architecture
                return model