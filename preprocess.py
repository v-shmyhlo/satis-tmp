import glob
import os
from tqdm import tqdm
import click
from PIL import Image
import cv2
import pandas as pd


@click.command()
@click.option("--data-dir", type=click.Path(), default="./satis-cv-ai-exercise-data")
def main(data_dir):
    preprocess_subset(data_dir, "train")
    preprocess_subset(data_dir, "val")


def preprocess_subset(data_dir, subset):
    data = pd.read_csv(os.path.join(data_dir, f"{subset}_action_classes.csv"))
    for _, row in tqdm(data.iterrows(), desc="subset"):
        preprocess_video(row, data_dir, subset)


def preprocess_video(row, data_dir, subset):
    frame_files = glob.glob(
        os.path.join(data_dir, subset, row["participant_id"], row["video_id"], "*.jpg")
    )
    frame_files = [
        file
        for file in frame_files
        if row["start_frame"] <= file_to_frame_number(file) <= row["stop_frame"]
    ]
    frame_files = sorted(frame_files)

    frames_to_video(
        frame_files,
        os.path.join(
            data_dir,
            f"{subset}_videos",
            row["participant_id"] + "_" + row["video_id"] + ".mp4",
        ),
    )


def file_to_frame_number(file):
    file = os.path.basename(file)
    file, _ = os.path.splitext(file)
    frame = int(file[6:])
    return frame


def frames_to_video(frame_files, video_file):
    os.makedirs(os.path.dirname(video_file), exist_ok=True)
    first_frame = Image.open(frame_files[0])
    video = cv2.VideoWriter(
        video_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        60,
        (first_frame.width, first_frame.height),
    )
    for frame_file in frame_files:
        video.write(cv2.imread(frame_file))
    video.release()


if __name__ == "__main__":
    main()
