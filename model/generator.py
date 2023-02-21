import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from json import dumps as json_dumps
import argparse
from os import environ

# Turn off tensorflow warnings
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Input,
    LSTM,
    Reshape,
    Lambda,
    RepeatVector,
    Masking,
)
from tensorflow.keras.utils import to_categorical

from model_helper import *
from DeepRouteSetHelper import *


def deepRouteSet(LSTM_cell, densor, n_values, n_a, Ty=12):
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
    a0 = Input(shape=(n_a,), name="a0")
    c0 = Input(shape=(n_a,), name="c0")
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
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)

    return inference_model


def predict_and_sample(
    inference_model,
    x_initializer,
    a_initializer,
    c_initializer,
):
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
    pred = inference_model.predict(
        [x_initializer, a_initializer, c_initializer], verbose=(1 if VERBOSE else 0)
    )
    # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(pred, axis=2)
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (Ty, n_values)
    results = to_categorical(indices, num_classes=np.shape(x_initializer)[2])

    return results, indices


# TODO: should be in separate file, currently using old output setup so compatible with saved weights
def grade_net():
    """Build GradeNet model and compile it."""
    np.random.seed(0)
    tf.random.set_seed(0)
    inputs = Input(shape=(12, 22))
    mask = Masking(mask_value=0.0).compute_mask(inputs)
    lstm0 = LSTM(
        20,
        activation="tanh",
        input_shape=(12, 22),
        kernel_initializer="glorot_normal",
        return_sequences="True",
    )(inputs, mask=mask)
    dense1 = Dense(100, activation="relu", kernel_initializer="glorot_normal")(lstm0)
    dense2 = Dense(80, activation="relu", kernel_initializer="glorot_normal")(dense1)
    dense3 = Dense(75, activation="relu", kernel_initializer="glorot_normal")(dense2)
    dense4 = Dense(50, activation="relu", kernel_initializer="glorot_normal")(dense3)
    dense5 = Dense(20, activation="relu", kernel_initializer="glorot_normal")(dense4)
    dense6 = Dense(10, activation="relu", kernel_initializer="glorot_normal")(dense5)
    flat = Flatten()(dense6)
    softmax2 = Dense(10, activation="softmax", name="softmax2")(flat)
    lstm1 = LSTM(
        20, activation="tanh", kernel_initializer="glorot_normal", return_sequences=True
    )(dense6)
    lstm2 = LSTM(20, activation="tanh", kernel_initializer="glorot_normal")(lstm1)
    dense7 = Dense(15, activation="relu", kernel_initializer="glorot_normal")(lstm2)
    dense8 = Dense(15, activation="relu", kernel_initializer="glorot_normal")(dense7)
    softmax3 = Dense(10, activation="softmax", name="softmax2")(dense8)

    GradeNet = Model(inputs=[inputs], outputs=[softmax3])

    return GradeNet


# TODO: this is quick and dirty, make this into separate method
def move_list_to_vectors(move_sequence):
    """Converts a list of moves to a vector of 22-dim moves."""
    move_sequence_vectors = np.zeros((22, len(move_sequence)))
    for move_idx, move_features in enumerate(move_sequence):
        move_sequence_vectors[0:2, move_idx] = move_features["TargetHoldString"]
        move_sequence_vectors[2, move_idx] = move_features[
            "TargetHoldHand"
        ]  # only express once
        move_sequence_vectors[3, move_idx] = move_features["TargetHoldScore"]
        move_sequence_vectors[4:6, move_idx] = move_features["RemainingHoldString"]
        move_sequence_vectors[6, move_idx] = move_features["RemainingHoldScore"]
        move_sequence_vectors[7:9, move_idx] = move_features["dxdyRtoT"]
        move_sequence_vectors[9:11, move_idx] = move_features["MovingHoldString"]
        move_sequence_vectors[11, move_idx] = move_features["MovingHoldScore"]
        move_sequence_vectors[12:14, move_idx] = move_features["dxdyMtoT"]
        move_sequence_vectors[14:21, move_idx] = move_features["FootPlacement"]
        move_sequence_vectors[21, move_idx] = move_features["MoveSuccessRate"]

    return move_sequence_vectors


def stringToCoordiante(coord_str: str):
    """Convert coordinate string (e.g. "J5") to integer (9,4)"""
    coord_str = coord_str.upper()
    return (ord(coord_str[0]) - ord("A"), int(coord_str[1:]) - 1)


if __name__ == "__main__":
    # set constants
    cwd = Path().cwd()

    # TODO: WE SHOULD NOT BE COMPILING MODEL HERE (slow, load pre-compiled)
    # think this is number of holds?
    n_values = 278
    # num dimensions of LSTM hidden state
    n_a = 64
    # layers
    reshapor = Reshape((1, n_values))
    LSTM_cell = LSTM(n_a, return_state=True)
    densor = Dense(n_values, activation="softmax")

    # load magic handString list
    benchmark_handString_seq_path = (
        cwd.parent / "preprocessing" / "benchmark_handString_seq_X"
    )
    with open(benchmark_handString_seq_path, "rb") as f:
        benchmark_handString_seq = pickle.load(f)
    # TODO: this looks INCREDIBLY redundant, just converting values into list
    handStringList = []
    for key in benchmark_handString_seq.keys():
        handStringList.append(benchmark_handString_seq[key])

    # map of hold indicies to hold strings
    with open(cwd.parent / "raw_data" / "holdIx_to_holdStr", "rb") as f:
        holdIx_to_holdStr = pickle.load(f)

    # load feature dictionaries for grading
    left_hold_feature_path = cwd.parent / "raw_data" / "HoldFeature2016LeftHand.csv"
    right_hold_feature_path = cwd.parent / "raw_data" / "HoldFeature2016RightHand.csv"

    LeftHandfeatures = pd.read_csv(left_hold_feature_path, dtype=str)
    RightHandfeatures = pd.read_csv(right_hold_feature_path, dtype=str)
    RightHandfeature_dict = {}
    LeftHandfeature_dict = {}
    for index in RightHandfeatures.index:
        LeftHandfeature_item = LeftHandfeatures.loc[index]
        LeftHandfeature_dict[
            (int(LeftHandfeature_item["X_coord"]), int(LeftHandfeature_item["Y_coord"]))
        ] = np.array(list(LeftHandfeature_item["Difficulties"])).astype(int)
        RightHandfeature_item = RightHandfeatures.loc[index]
        RightHandfeature_dict[
            (
                int(RightHandfeature_item["X_coord"]),
                int(RightHandfeature_item["Y_coord"]),
            )
        ] = np.array(list(RightHandfeature_item["Difficulties"])).astype(int)

    # create inference model
    inference_model = deepRouteSet(LSTM_cell, densor, n_values=n_values, n_a=64, Ty=12)

    # load model weights
    inference_model.load_weights(cwd / "DeepRouteSetMedium_v1.h5")

    # create grading model
    grade_model = grade_net()

    # load model weights
    grade_model.load_weights(cwd / "GradeNet.h5")

    # ===================== Begin generation and grading ======================

    # repeatedly generate routes until we get a valid one with requried grade
    # TODO: grade prediction takes a long time, do in batches
    generated_valid_grade = False
    while not generated_valid_grade:
        x_initializer = np.random.rand(1, 1, n_values) / 100
        a_initializer = np.random.rand(1, n_a) * 150
        c_initializer = np.random.rand(1, n_a) / 2

        results, indices = predict_and_sample(
            inference_model, x_initializer, a_initializer, c_initializer
        )
        passCheck, outputListInString, outputListInIx = sanityCheckAndOutput(
            indices,
            holdIx_to_holdStr,
            handStringList,
            printError=VERBOSE,
        )

        # if sanity check fails, generate another route
        if not passCheck:
            if VERBOSE:
                print(f"\n{' Sanity check failed ':=^80}")
                print(outputListInString)
            continue

        if VERBOSE:
            print(f"\n{' Sanity check passed ':=^80}")
            print(outputListInString)

        # convert route into moves to pass to generator
        move_sequence = moveGeneratorFromStrList(outputListInString, string_mode=False)
        num_moves = len(move_sequence)

        # convert moves into the required 22dim vector
        sequence_vectors = move_list_to_vectors(move_sequence)

        # we need to transpose the sequence vectors and pad
        seq_data_pad = np.zeros((1, 12, 22))
        tmax = np.array([num_moves])
        seq_data_pad[0, :num_moves, :] = sequence_vectors.T
        input_set = {"X": seq_data_pad, "tmax": tmax}

        # apply standardisation to sequence vectors based on that used for training
        X = normalization(input_set)["X"]

        # grade the move sequence
        grade_prob = grade_model.predict(X, verbose=(1 if VERBOSE else 0))
        grade_pred = grade_prob.argmax() + 4
        if VERBOSE:
            print(
                f"Predicted grade: V{grade_prob.argmax() + 4} ({grade_prob.max():.2f})"
            )

        holds = []

        for hold_str in outputListInString:
            x, y = stringToCoordiante(hold_str[:-3])
            holds.append({"x": x, "y": y})

        # dump JSON string of holds
        print(json_dumps(holds))

        # TODO: check grade
        # if the grade matches the desired grade then we are done
        generated_valid_grade = True
