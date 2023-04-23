import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from json import dumps as json_dumps
import argparse
from os import environ
import random

# Turn off tensorflow warnings
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow.keras.backend as K
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from DeepRouteSet_helper import sanityCheckAndOutput, moveGeneratorFromStrList
from model_helper import normalization


# TODO: move to DeepRouteSetHelper
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


# TODO: this should be imported from preprocessing.preprocessing_helper
def str_to_coordinate(coord_str: str):
    """Convert coordinate string (e.g. "J5") to integer (9,4)"""
    coord_str = coord_str.upper()
    return (ord(coord_str[0]) - ord("A"), int(coord_str[1:]) - 1)

def load_wall(wall: str):
    file = open(wall, 'r')
    return [[float(i) for i in x.strip('\n').split(',')] for x in file.readlines()]


if __name__ == "__main__":
    # get input arguments and check for grade
    parser = argparse.ArgumentParser(
        description='Generate a MoonBoard route of a given grade.')
    parser.add_argument(
        'grade', nargs=1, type=int,
        help='a grade between 0 and 13 (Hueco V-scale)')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='verbose output')
    parser.add_argument(
        '-w', '--wall',
        help='wall file to map outputs to'
    )
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    VERBOSE = args.verbose
    grade = args.grade
    wall = load_wall(args.wall)

    # set constants
    cwd = Path(__file__).parent

    # think this is number of holds?
    n_values = 278
    # num dimensions of LSTM hidden state
    n_a = 64

    # load magic handString list
    benchmark_handString_seq_path = (
        cwd.parent / "preprocessing" / "benchmark_handString_seq_X"
    )
    with open(benchmark_handString_seq_path, "rb") as f:
        benchmark_handString_seq = pickle.load(f)
    # TODO: this looks INCREDIBLY redundant, just converting values into list
    # Removing handStringList for now
    # handStringList = []
    # for key in benchmark_handString_seq.keys():
    #     handStringList.append(benchmark_handString_seq[key])

    # map of hold indicies to hold strings
    with open(cwd.parent / "raw_data" / "holdIx_to_holdStr", "rb") as f:
        holdIx_to_holdStr = pickle.load(f)

    # load feature dictionaries for generating move features
    left_hold_feature_path = cwd.parent / "raw_data" / "hold_features_2016_LH.csv"
    right_hold_feature_path = cwd.parent / "raw_data" / "hold_features_2016_RH.csv"

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

    # TODO: disable WARNING:tensorflow:No training configuration found in save file, so the model was *not* compiled. Compile it manually.
    # load inference model
    inference_model = load_model(cwd / "DeepRouteSet")

    # create grading model
    grade_model = load_model(cwd / "GradeNet")

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
            # [],     # Hand string list, ignoring for now
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
                f"Predicted grade: V{grade_pred} ({grade_prob.max():.2f})"
            )

        # TODO: check grade, for testing will return first route
        # if the grade matches the desired grade then we are done
        # generated_valid_grade = grade_pred == grade 
        generated_valid_grade = True
    
    holds = []

    for hold_str in outputListInString:
        x, y = str_to_coordinate(hold_str[:-3])
        # magic numbers for normalisation. only for moonboard though so this needs changing to be proper
        #TODO: fix magic numbers
        x = (90 + 52 * x) / 665
        y = (1020 - 52 * y) / 1023
        
        wall.sort(key = lambda p: (p[0] - x)**2 + (p[1] - y)**2)
        sample = list(filter(lambda p: (p[0] - x)**2 + (p[1] - y)**2 < (52/665)**2, wall))
        if len(sample) == 0:
            print("COULD NOT FIND HOLD AT LOCATION", x, y)
            hold = [x, y]
        hold = random.choice(sample)
        holds.append({"x": hold[0], "y": hold[1]})


    # dump JSON string of holds
    print(json_dumps(holds))
