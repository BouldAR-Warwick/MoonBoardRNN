import numpy as np
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import matplotlib.cbook as cbook
import re
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow import make_ndarray


def save_pickle(data, file_name):
    """
    Saves data as pickle format
    """
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def plot_confusion_matrix(Y_true, Y_predict, title=None):
    """
    Plot the confusion matrix.
    """
    labels = ["V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13"]
    conf_matrix = metrics.confusion_matrix(Y_true, Y_predict)
    df_cm = pd.DataFrame(
        (conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True)),
        index=[i for i in labels],
        columns=[i for i in labels],
    )
    plt.figure(dpi=150, figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
    plt.xlabel("predicted grade")
    plt.ylabel("actual grade")
    if title:
        plt.title(title)
    plt.show()


def plot_history(history, model_name, losses=["loss"], metrics=["accuracy"]):
    """
    Plot the training history of the model
    """
    # Plot training & validation accuracy values
    if metrics:
        history_metrics = {}
        for metric in metrics:
            history_metrics["Train " + metric] = history.history[metric]
            history_metrics["Validation " + metric] = history.history["val_" + metric]
    history_losses = {}
    for loss in losses:
        history_losses["Train " + loss] = history.history[loss]
        history_losses["Validation " + loss] = history.history["val_" + loss]

    _, axes = plt.subplots(nrows=1, ncols=(2 if metrics else 1), figsize=(10, 5))
    ax1 = axes[0] if metrics else axes
    for key in history_losses:
        ax1.plot(history_losses[key], label=key)
    ax1.set_title("Loss of " + model_name)
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    if metrics:
        for key in history_metrics:
            axes[1].plot(history_metrics[key], label=key)
        axes[1].set_title("Accuracy of " + model_name)
        axes[1].set_ylabel("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

    plt.tight_layout()


def plot_tb_history(log_dir, model_name, metric=None):
    # fetch events from logs
    train_ea = EventAccumulator(str(next(log_dir.glob('train/*'))))
    train_ea.Reload()
    val_ea = EventAccumulator(str(next(log_dir.glob('validation/*'))))
    val_ea.Reload()

    # load the loss and accuracy tensors
    train_loss = train_ea.Tensors('epoch_loss')
    val_loss = val_ea.Tensors('epoch_loss')
    if metric:
        train_acc = train_ea.Tensors(metric)
        val_acc = val_ea.Tensors(metric)

    train_loss = [t.step for t in train_loss], [make_ndarray(t.tensor_proto).item() for t in train_loss]
    val_loss = [t.step for t in val_loss], [make_ndarray(t.tensor_proto).item() for t in val_loss]
    if metric:
        train_acc = [t.step for t in train_acc], [make_ndarray(t.tensor_proto).item() for t in train_acc]
        val_acc = [t.step for t in val_acc], [make_ndarray(t.tensor_proto).item() for t in val_acc]

    # plot
    _, axes = plt.subplots(nrows=1, ncols=2)
    ax1 = axes[0] if metric else axes
    ax1.plot(*train_loss)
    ax1.plot(*val_loss)
    ax1.set_title("Loss of " + model_name)
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend(["Train", "Val"], loc="upper left")
    
    if metric:
        axes[1].plot(*train_acc)
        axes[1].plot(*val_acc)
        axes[1].set_title("Accuracy of " + model_name)
        axes[1].set_ylabel("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend(["Train", "Val"], loc="upper left")

    plt.tight_layout()


def compute_accuracy(y_true, y_predict):
    """
    Compute the accuracy of the model
    - complete_accurate: the model output match the expected output exactly
    - roughly_accurate: the model output is only different from the expected output by less than or equal to 1 grade.
    """
    complete_accurate = np.sum(y_true == y_predict) / len(y_true)
    roughly_accurate = np.sum(np.abs(y_true - y_predict) <= 1) / len(y_true)
    return (complete_accurate, roughly_accurate)


def plotAProblem(stringList, title=None, key=None):
    """ Draw a moonboard problem on the layout"""
    parent = Path().cwd().parent
    image_file = cbook.get_sample_data(
        parent / "raw_data" / "moonboard2016Background.jpg"
    )
    img = plt.imread(image_file)
    x = []
    y = []
    for hold in stringList:
        # Using re.findall()
        # Splitting text and number in string
        res = [re.findall(r"(\w+?)(\d+)", hold.split("-")[0])[0]]

        alphabateList = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        ixInXAxis = alphabateList.index(res[0][0])

        x = x + [(90 + 52 * ixInXAxis)]  # * img.shape[0] / 1024]
        y = y + [(1020 - 52 * int(res[0][1]))]  # * img.shape[1] / 1024]

    # Create a figure. Equal aspect so circles look circular
    plt.rcParams["figure.figsize"] = (10, 20)
    fig, ax = plt.subplots(1)
    ax.set_aspect("equal")

    # Show the image
    ax.imshow(img)

    # Now, loop through coord arrays, and create a circle at each x,y pair
    count = 0
    for xx, yy in zip(x, y):
        if yy == 84:
            circ = plt.Circle((xx, yy), 30, color="r", fill=False, linewidth=2)
        elif count < 2:
            circ = plt.Circle((xx, yy), 30, color="g", fill=False, linewidth=2)
        else:
            circ = plt.Circle((xx, yy), 30, color="b", fill=False, linewidth=2)
        ax.add_patch(circ)
        count = count + 1

    # Show the image
    if title:
        plt.title(title)
        plt.savefig(key + ".jpg", dpi=200)
    plt.show()


def convert_num_to_V_grade(num):
    dic = {
        0: "V4",
        1: "V5",
        2: "V6",
        3: "V7",
        4: "V8",
        5: "V9",
        6: "V10",
        7: "V11",
        8: "V12",
        9: "V13",
    }
    return dic[num]


def normalization(input_set):
    # TODO: WHAT THE FUCK ARE THESE MAGIC NUMBERS
    mu_x = 5.0428571
    sig_x = 3.079590
    mu_y = 9.8428571
    sig_y = 4.078289957
    mu_hand = 4.2428571
    sig_hand = 2.115829552
    mu_diff = 12.118308
    sig_diff = 11.495348196

    mu_vec = np.array(
        [
            mu_x,
            mu_y,
            0,
            mu_hand,
            mu_x,
            mu_y,
            mu_hand,
            0,
            0,
            mu_x,
            mu_y,
            mu_hand,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            mu_diff,
        ]
    )
    sig_vec = np.array(
        [
            sig_x,
            sig_y,
            1,
            sig_hand,
            sig_x,
            sig_y,
            sig_hand,
            sig_x,
            sig_y,
            sig_x,
            sig_y,
            sig_hand,
            sig_x,
            sig_y,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            sig_diff,
        ]
    )

    mask = np.zeros_like(input_set["X"])
    for i in range(len(mask)):
        mask[i, 0 : int(input_set["tmax"][i]), :] = 1

    X_normalized = np.copy(input_set["X"])
    X_normalized -= mu_vec
    X_normalized /= sig_vec
    X_normalized *= mask

    output_set = input_set
    output_set["X"] = X_normalized
    return output_set


def prepare_generated_data_for_grading(generated_problems, save_path):
    n_sample = len(generated_problems)
    X_seq_data_merge = np.zeros((n_sample, 12, 22))
    keys_seq_merge = []
    tmax_seq_merge = np.zeros(n_sample)

    i = 0
    for key, value in generated_problems.items():
        X_data = value.T
        X_seq_data_merge[i, 0 : X_data.shape[0], :] = X_data
        keys_seq_merge.append(key)
        tmax_seq_merge[i] = X_data.shape[0]
        i = i + 1

    test_set_gen = {
        "X": X_seq_data_merge,
        "keys": keys_seq_merge,
        "tmax": tmax_seq_merge,
    }

    test_set_gen_normalized = normalization(test_set_gen)
    save_pickle(test_set_gen_normalized, save_path)
    return test_set_gen_normalized
