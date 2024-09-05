from keras.src.utils.summary_utils import weight_memory_size
import tensorflow
import tensorflow.python.keras as keras
from keras import Model, Sequential
from keras.api.layers import Dense, Input, InputLayer

seq = Sequential([InputLayer((12,)),Dense(12)])
seq.summary()
for weight in seq.weights:
    print(weight)
    
