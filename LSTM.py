import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("model.h5")

def LSTM_inference(ls):
    lm_list = np.array(ls)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    if results[0][0] > 0.5:
        return "SWING HEAD"
    else:
        return "SWING HAND"
    return label

