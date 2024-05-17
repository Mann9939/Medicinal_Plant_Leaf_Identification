import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras_preprocessing import image
import PIL

model = load_model('mymodel.h5')
test_image = image.load_img(r"C:\Users\manis\Downloads\Augmented Images (Version 02)\Augmented Images (Version 02)\Augmented Curry Leaf\Curry Leaf Aug (981).jpg", target_size = (256,256))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = model.predict(test_image)
print(result)