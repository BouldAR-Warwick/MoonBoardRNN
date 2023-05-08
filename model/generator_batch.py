import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json
import argparse
from os import environ
import random

# Turn off tensorflow warnings
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# these are requried for the lambda function within DeepRouteSet
import tensorflow as tf
from tensorflow.keras.backend import argmax
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
    file = open(wall, "r")
    return [[float(i) for i in x.strip("\n").split(",")] for x in file.readlines()]


def one_off_accuracy(y_true, y_pred):
    """Computes the accuracy of a grade prediction including +/-1 errors

    Args:
        y_true: true grades
        y_pred: predicted grades
    """
    return tf.reduce_mean(tf.cast(abs(y_true - tf.math.round(y_pred)) <= 1, tf.float32))


if __name__ == "__main__":
    # get input arguments and check for grade
    parser = argparse.ArgumentParser(
        description="Generate a MoonBoard route of a given grade."
    )
    parser.add_argument(
        "outfile", nargs=1, type=str, help="file to pickle dump the routes in"
    )
    parser.add_argument(
        "num_routes", nargs=1, type=int, help="number of routes to generate"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-w", "--wall", help="wall file to map outputs to")
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    VERBOSE = args.verbose
    outfile = args.outfile[0]
    num_routes = args.num_routes[0]

    if VERBOSE: print(f"Generating {num_routes} routes")
    if VERBOSE: print(f"Saving routes to {outfile}")
    if VERBOSE: print(f"Loading wall from {args.wall}...")
    wall = load_wall(args.wall)

    # set constants
    cwd = Path(__file__).parent

    # number of MoonBoard holds (inc left/right hand)
    n_values = 278
    # num dimensions of LSTM hidden state
    n_a = 64

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
    grade_model = load_model(cwd / "GradeNet", custom_objects={"one_off_accuracy": one_off_accuracy})

    # ===================== Begin generation ======================
    routes = []
    for i in range(num_routes) :
        if VERBOSE: print(f"Generating route {i+1} of {num_routes}")
        passed = False
        while (not passed):
            x_initializer = np.random.rand(1, 1, n_values) / 100
            a_initializer = np.random.rand(1, n_a) * 150
            c_initializer = np.random.rand(1, n_a) / 2

            results, indices = predict_and_sample(
                inference_model, x_initializer, a_initializer, c_initializer
            )
            passed, outputListInString, outputListInIx = sanityCheckAndOutput(
                indices,
                holdIx_to_holdStr,
                # [],     # Hand string list, ignoring for now
                printError=VERBOSE,
            )

            print(f"Generated route {outputListInString}")
            # if sanity check fails, generate another route
            if not passed:
                if VERBOSE:
                    print(f"{' Sanity check failed ':=^80}\n")
                continue

            if VERBOSE:
                print(f"{' Sanity check passed ':=^80}\n")

        routes.append(outputListInString)

    # prepare routes for grading
    if VERBOSE: print("Preparing routes for grading...")
    # we need to transpose the sequence vectors and pad
    seq_data_pad = np.zeros((num_routes, 12, 22))
    tmax = np.array([0]*num_routes)
    for i in range(len(routes)):
        # convert route into moves to pass to generator
        move_sequence = moveGeneratorFromStrList(routes[i], string_mode=False)
        num_moves = len(move_sequence)
        tmax[i] = num_moves

        # convert moves into the required 22dim vector
        sequence_vectors = move_list_to_vectors(move_sequence)
        seq_data_pad[i, :num_moves, :] = sequence_vectors.T

        # map route to holds on the wall
        holds = []
        for hold_str in outputListInString:
            x, y = str_to_coordinate(hold_str[:-3])
            # magic numbers for normalisation. only for moonboard though so this needs changing to be proper
            # TODO: fix magic numbers
            x = (90 + 52 * x) / 665
            y = (1020 - 52 * y) / 1023

            wall.sort(key = lambda p: (p[0] - x)**2 + (p[1] - y)**2)
            sample = list(filter(lambda p: (p[0] - x)**2 + (p[1] - y)**2 < (52/665)**2, wall))
            if len(sample) == 0:
                if VERBOSE: print("COULD NOT FIND HOLD AT LOCATION", x, y)
                hold = [x, y]
            else:
                hold = random.choice(sample)
            holds.append({"x": hold[0], "y": hold[1]})
        routes[i] = holds

    input_set = {"X": seq_data_pad, "tmax": tmax}
    print(tmax)
    # for each of the routes predict the grade and store in dict
    graded_routes = {}
    # apply standardisation to sequence vectors based on that used for training
    X = normalization(input_set)["X"]
    # grade the move sequences
    raw_grades = grade_model.predict(X, verbose=(1 if VERBOSE else 0))
    if VERBOSE: print("Raw grades:", raw_grades)
    # TODO: should we actually be adding 4 here? need to double check
    grades = np.round(raw_grades.reshape(num_routes)) + 4

    # create dict of routes by grade
    for i in range(len(routes)):
        grade = int(grades[i])
        if VERBOSE: print(f"Route {i+1} of {len(routes)} graded as {grade}")
        if grade not in graded_routes:
            graded_routes[grade] = []
        graded_routes[grade].append(routes[i])

    # dump JSON string of holds
    with open(outfile, "w") as f:
        json.dump(graded_routes, f, indent=4)
