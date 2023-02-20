# File only generates new routes based on the trained model
# This is what should be ran on services

from __future__ import print_function
import IPython
import sys
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import os
import keras
import tensorflow as tf
import keras.backend as K
import re
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Activation, Input, LSTM, Reshape, Lambda, RepeatVector, Masking
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
#%matplotlib inline
import matplotlib.image as mpimg
import matplotlib.cbook as cbook
import re 
import matplotlib.pyplot as plt
import PIL
plt.rcParams["figure.figsize"] = (30,10)
from model_helper import *
from DeepRouteSetHelper import *

from pathlib import Path

cwd = Path()

#TODO: n_values???
n_values = 278

# number of dimensions for the hidden state of each LSTM cell.
n_a = 64 

reshapor = Reshape((1, n_values))                  
LSTM_cell = LSTM(n_a, return_state = True)        
densor = Dense(n_values, activation='softmax')

x_initializer = np.zeros((1, 1, n_values))
x_initializer = np.random.rand(1, 1, n_values) / 100
a_initializer = np.random.rand(1, n_a) * 150
c_initializer = np.random.rand(1, n_a) / 2

## ensemble to a StringList
benchmark_handString_seq_path = cwd.parent / '../preprocessing' / 'benchmark_handString_seq_X'
with open(benchmark_handString_seq_path, 'rb') as f:
    benchmark_handString_seq = pickle.load(f)
handStringList = []
for key in benchmark_handString_seq.keys():
    handStringList.append(benchmark_handString_seq[key])

with open(cwd.parent / "../raw_data" / "holdIx_to_holdStr", 'rb') as f:
    holdIx_to_holdStr = pickle.load(f)  

def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, n_values), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    
    Returns:
    results -- numpy-array of shape (Ty, n_values), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    # Step 1: Use your inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
    indices =  np.argmax(pred, axis = 2)
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (Ty, n_values)
    results =  to_categorical(indices, num_classes = np.shape(x_initializer)[2])
    
    return results, indices

def deepRouteSet(LSTM_cell, densor, n_values = n_values, n_a = 64, Ty = 12):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, number of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    def one_hot(x):
        x = K.argmax(x)
        x = tf.one_hot(x, n_values) 
        x = RepeatVector(1)(x)
        return x
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    ### START CODE HERE ###
    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
    outputs = []
    
    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        # Step 2.A: Perform one step of LSTM_cell (≈1 line)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)

        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, n_values) (≈1 line)
        outputs.append(out)
        
        # Step 2.D: 
        # Select the next value according to "out",
        # Set "x" to be the one-hot representation of the selected value
        # See instructions above.
        x = Lambda(one_hot)(out)
        
        
    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs = [x0, a0, c0], outputs = outputs)
    
    ### END CODE HERE ###
    
    return inference_model

inference_model = deepRouteSet(LSTM_cell, densor, n_values = n_values, n_a = 64, Ty = 12)

# load model weight
inference_model.load_weights(cwd.parent / "../model/DeepRouteSetMedium_v1.h5")

# Gen 40 routes - 4.3
passCount = 0
passGeneratedHandSequenceList = []
for i in range(40):
    x_initializer = np.zeros((1, 1, n_values))
    x_initializer = np.random.rand(1, 1, n_values) / 100
    a_initializer = np.random.rand(1, n_a) * 150
    c_initializer = np.random.rand(1, n_a) /2
    
    results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
    passCheck, outputListInString, outputListInIx = sanityCheckAndOutput(
        indices, holdIx_to_holdStr, handStringList, printError = True)
    if passCheck: 
        print(outputListInString)
        #plotAProblem(outputListInString)
        passCount = passCount + 1
        passGeneratedHandSequenceList.append(outputListInString)
print (passCount, "passCheck out of 400")        




#4.4
# Feed in the hold feature.csv files
left_hold_feature_path = cwd.parent / '../raw_data' / 'HoldFeature2016LeftHand.csv'
right_hold_feature_path = cwd.parent / '../raw_data' / 'HoldFeature2016RightHand.csv'

LeftHandfeatures = pd.read_csv(left_hold_feature_path, dtype=str)
RightHandfeatures = pd.read_csv(right_hold_feature_path, dtype=str)
# convert features from pd dataframe to dictionary of left and right hand
RightHandfeature_dict = {}
LeftHandfeature_dict = {}
for index in RightHandfeatures.index:
    LeftHandfeature_item = LeftHandfeatures.loc[index]
    LeftHandfeature_dict[(int(LeftHandfeature_item['X_coord']), int(LeftHandfeature_item['Y_coord']))] = np.array(
        list(LeftHandfeature_item['Difficulties'])).astype(int)
    RightHandfeature_item = RightHandfeatures.loc[index]
    RightHandfeature_dict[(int(RightHandfeature_item['X_coord']), int(RightHandfeature_item['Y_coord']))] = np.array(
        list(RightHandfeature_item['Difficulties'])).astype(int)


#4.5
save_path = cwd / 'MediumProblemOfDeepRouteSet_v1'
dim22Vec, listOfSavedSequence = moveGeneratorForAllGeneratedProblem(passGeneratedHandSequenceList, save_path, "HardDeepRouteSet_v1_id", print_result = False)
save_pickle(listOfSavedSequence, cwd / '../MediumProblemSequenceOfDeepRouteSet_v1')





### Now doing eval
cwd = os.getcwd()
parent_wd = cwd.replace('\\', '/').replace('/model', '')
raw_path = parent_wd + '/out/MediumProblemOfDeepRouteSet_v1'

with open(raw_path, 'rb') as f:
    raw_gen_set = pickle.load(f)

test_set = convert_generated_data_into_test_set(raw_gen_set, parent_wd + '/preprocessing/test_set_medium_gen_v1')

X_test = test_set['X']


np.random.seed(0)
tf.random.set_seed(0)
inputs = Input(shape = (12, 22))
mask = Masking(mask_value = 0.).compute_mask(inputs)
lstm0 = LSTM(20, activation='tanh', input_shape=(12, 22), kernel_initializer='glorot_normal', return_sequences = 'True')(
    inputs, mask = mask)
dense1 = Dense(100, activation='relu', kernel_initializer='glorot_normal')(lstm0)
dense2 = Dense(80, activation='relu', kernel_initializer='glorot_normal')(dense1)
dense3 = Dense(75, activation='relu', kernel_initializer='glorot_normal')(dense2)
dense4 = Dense(50, activation='relu', kernel_initializer='glorot_normal')(dense3)
dense5 = Dense(20, activation='relu', kernel_initializer='glorot_normal')(dense4)
dense6 = Dense(10, activation='relu', kernel_initializer='glorot_normal')(dense5)
flat = Flatten()(dense6)
softmax2 = Dense(10, activation='softmax', name = 'softmax2')(flat)
lstm1 = LSTM(20, activation='tanh', kernel_initializer='glorot_normal', return_sequences = True)(dense6)
lstm2 = LSTM(20, activation='tanh', kernel_initializer='glorot_normal')(lstm1)
dense7 = Dense(15, activation='relu', kernel_initializer='glorot_normal', name = 'dense7')(lstm2)
dense8 = Dense(15, activation='relu', kernel_initializer='glorot_normal', name = 'dense8')(dense7)
softmax3 = Dense(10, activation='softmax', name = 'softmax2')(dense8)

def custom_loss(layer):
    def loss(y_true,y_pred):
        loss1 = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss2 = K.sparse_categorical_crossentropy(y_true, layer)
        return K.mean(loss1 + loss2, axis=-1)
    return loss

GradeNet = Model(inputs=[inputs], outputs=[softmax3])
GradeNet.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', #loss=custom_loss(softmax2),
                metrics=['sparse_categorical_accuracy'])

# load model weight
GradeNet.load_weights(parent_wd + '/model/GradeNet.h5')
test_Y_gen = GradeNet.predict(X_test).argmax(axis = 1)

seq_raw_path = parent_wd + '/out/MediumProblemSequenceOfDeepRouteSet_v1'
with open(seq_raw_path, 'rb') as f:
    gen_seq_raw = pickle.load(f)

for i, route in enumerate(gen_seq_raw):
    print('Key = ' + test_set['keys'][i])
    print('V grade = ' + convert_num_to_V_grade(test_Y_gen[i]))
    #plotAProblem(route, 
    #             title = 'Key = ' + test_set['keys'][i] + '/ V grade = ' + convert_num_to_V_grade(test_Y_gen[i]), key = test_set['keys'][i],
    #             save = True, show = False)
    print('Route: ', route)