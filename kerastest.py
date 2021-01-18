# 3. Import libraries and modules
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.preprocessing import LabelEncoder
import loaddataset
import os
np.random.seed(123)  # for reproducibility

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 4. Load pre-shuffled MNIST data into train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#(X_train, y_train), (X_test, y_test) = loaddataset.ligatures_dataset()
print(X_train.shape)
# 5. Preprocess input data
res = 28
X_train = X_train.reshape(X_train.shape[0], res, res, 1)
X_test = X_test.reshape(X_test.shape[0], res, res, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

encoder = LabelEncoder()
Y_train = encoder.fit_transform(y_train)
Y_test = encoder.fit_transform(y_test)


# 6. Preprocess class labels
Y_train = np_utils.to_categorical(Y_train, 48)
Y_test = np_utils.to_categorical(Y_test, 48)

# 7. Define model architecture
model = Sequential()

#model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1, res, res)))
model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(res, res, 1)))
#model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dense(48, activation='softmax'))

# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 9. Fit model on training data
model.fit(X_train, Y_train, batch_size=48, epochs=10, verbose=1)

# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=1)

print(score)