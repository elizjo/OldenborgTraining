import pathlib
import platform
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
from time import sleep
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools

from fastai.callback.wandb import WandbCallback
from fastai.vision.learner import load_learner
from fastai.vision.utils import get_image_files
from fastai.interpret import ClassificationInterpretation

import wandb
# from utils import y_from_filename  # noqa: F401 (needed for fastai load_learner)


@contextmanager
def set_posix_windows():
    # NOTE: This is a workaround for a bug in fastai/pathlib on Windows
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


def parse_args():
    arg_parser = ArgumentParser("Track performance of trained networks.")

    # Wandb configuration
    arg_parser.add_argument("wandb_name", help="Name of run inference results.")
    arg_parser.add_argument("wandb_project", help="Wandb project name.")
    arg_parser.add_argument("wandb_notes", help="Wandb run description.")
    arg_parser.add_argument("wandb_model", help="Path to the model to evaluate.")
    arg_parser.add_argument("dataset_name", help="WandB Dataset for inference images")

    return arg_parser.parse_args()

def y_from_filename(x, filename):
    filename_stem = Path(filename).stem
    return filename_stem.split('_')[1]

# cm is confusion matrix in numpy representation
def plot_confusion_matrix(interp, cmap:str="Blues", plot_txt:bool=True, norm_dec:int=2, normalize = False,  **kwargs):
    "Plot the confusion matrix, with `title` and using `cmap`."
        # This function is mainly copied from the sklearn docs
    cm = interp.confusion_matrix()
    vocab = interp.vocab

    if normalize:
        title = 'Normalized confusion matrix'
        filename = 'myfig-norm'
    else:
        title = 'Confusion matrix, without normalization'
        filename = 'myfig'
            
    if normalize: 
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(**kwargs)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    plt.title(title)

    tick_marks = np.arange(len(vocab))
    plt.xticks(tick_marks, vocab, rotation=90)
    plt.yticks(tick_marks, vocab, rotation=0)

    if plot_txt:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
            plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center", color="white"
                        if cm[i, j] > thresh else "black")

    ax = fig.gca()
    ax.set_ylim(len(vocab)-.5,-.5)

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.grid(False)
    
    plt.savefig(filename)


def main():
    args = parse_args()

    wandb_entity = "arcslaboratory"
    wandb_project = args.wandb_project
    wandb_name = args.wandb_name
    wandb_notes = args.wandb_notes
    wandb_model = args.wandb_model

    run = wandb.init(
        job_type="validation",
        entity=wandb_entity,
        name=wandb_name,
        project=wandb_project,
        notes=wandb_notes,
    )

    if run is None:
        raise Exception("wandb.init() failed")


    # Changed from wand_name to dataset_name
    artifact = run.use_artifact(f"{args.dataset_name}:latest")
    data_dir = artifact.download()

    # Download the fastai learner
    artifact = run.use_artifact(f"{wandb_model}.pkl:latest", type="model")
    model_dir = artifact.download()
    
    print("Model directory contents:", os.listdir(model_dir))
    model_filename = Path(model_dir) / f"{wandb_model}.pkl"


    # Load the learner and its model
    # TODO: this doesn't load the "best" model, but the last one
    # We should probably also download the weights and load them manually
    if platform.system() == "Windows":
        with set_posix_windows():
            model = load_learner(model_filename)
    else:
        model = load_learner(model_filename)

    # TODO: temporary fix? (we might remove callback on training side)
    model.remove_cb(WandbCallback)

    # Check if directory with inference images is empty

    image_filenames: list = get_image_files(data_dir)
    assert len(image_filenames) > 0

    dl_test = model.dls.test_dl(image_filenames, with_labels=True)
    # model.predict(image_filenames)

    interp = ClassificationInterpretation.from_learner(model, dl = dl_test)

    cm = plot_confusion_matrix(interp = interp)
    cm_norm = plot_confusion_matrix(interp = interp, normalize = True)
    # raise SystemExit

    num_correct = 0

    for filename in image_filenames:
        
        # obtain full path to image to feed into model
        # image_filename = os.path.join(local_images, filename)

        filename = str(filename)

        # obtain correct label
        if 'forward' in filename:
            correct_action = 'forward'
        elif 'left' in filename:
            correct_action = 'left'
        else:
            correct_action = 'right'

        # Predict correct action
        action_to_take, action_index, action_probs = model.predict(filename)
        action_prob = action_probs[action_index]

        print(f"Moving {action_to_take} with probabilities {action_prob:.2f}")

        # calculate if prediction matches label
        if correct_action == action_to_take:
            num_correct+=1

    accuracy = num_correct / len(image_filenames)

    run.log({"confusion matrix": wandb.Image("myfig.png")})
    run.log({"normalized, confusion matrix": wandb.Image("myfig-norm.png")})

    run.log({"accuracy": accuracy})

if __name__ == "__main__":
    main()
