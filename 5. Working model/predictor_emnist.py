import pandas as pd
import numpy as np
np.random.seed(1337) # for reproducibility

from keras import backend as K
from keras.models import Sequential, load_model
from keras.utils import np_utils
import mkdocs

# input image dimensions
img_rows, img_cols = 28, 28

batch_size = 128 # Number of images used in each optimization step
nb_classes = 62 # One class per digit/lowercase letter/uppercase letter
nb_epoch = 2

test  = pd.read_csv("test.csv").values
print('test shape:', test.shape)

if K.image_data_format() == 'channels_first':
    X_test = test[:, 1:].reshape(test.shape[0], 1, img_rows, img_cols)
else:
    X_test = test[:, 1:].reshape(test.shape[0], img_rows, img_cols, 1)

y_test = test[:, 0]

X_test = X_test.astype('float32')
X_test /= 255

Y_test = np_utils.to_categorical(y_test, nb_classes)

print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

model=load_model('emnist-cnn.h5')
# Predict the label for X_test
yPred = model.predict_classes(X_test)

# Save prediction in file for Kaggle submission
np.savetxt("result.csv", np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
