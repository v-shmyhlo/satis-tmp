import imp
import tempfile
from operator import mod
from contextlib import contextmanager
import torch
import requests
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
import csv
from io import StringIO
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms._transforms_video import NormalizeVideo
from transforms import SpatialCrop, TemporalCrop, DepthNorm
import cv2
import glob
from PIL import Image


class ActionRecognizer:
    def __init__(self, device="cpu") -> None:
        self.device = device

        response = requests.get(
            "https://dl.fbaipublicfiles.com/omnivore/epic_action_classes.csv"
        )
        reader = csv.reader(StringIO(response.text))
        self.epic_id_to_action = {
            idx: " ".join(rows) for idx, rows in enumerate(reader)
        }

        model_name = "omnivore_swinB_epic"
        model = torch.hub.load("facebookresearch/omnivore:main", model=model_name)

        # Set to eval mode and move to desired device
        model = model.to(device)
        model = model.eval()
        self.model = model

        num_frames = 32
        sampling_rate = 2
        frames_per_second = 30

        clip_duration = (num_frames * sampling_rate) / frames_per_second

        self.video_transform = ApplyTransformToKey(
            key="video",
            transform=T.Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    T.Lambda(lambda x: x / 255.0),
                    ShortSideScale(size=224),
                    NormalizeVideo(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    TemporalCrop(frames_per_clip=32, stride=40),
                    SpatialCrop(crop_size=224, num_crops=3),
                ]
            ),
        )

    def predict(self, file):
        # Initialize an EncodedVideo helper class
        video = EncodedVideo.from_path(file)

        # Load the desired clip
        video_data = video.get_clip(start_sec=0.0, end_sec=2.0)

        # Apply a transform to normalize the video input
        video_data = self.video_transform(video_data)

        # Move the inputs to the desired device
        video_inputs = video_data["video"]

        # Take the first clip
        # The model expects inputs of shape: B x C x T x H x W
        video_input = video_inputs[0][None, ...]

        # Pass the input clip through the model
        with torch.no_grad():
            prediction = self.model(video_input.to(self.device), input_type="video")

            # Get the predicted classes
            pred_classes = prediction.topk(k=5).indices

        # Map the predicted classes to the label names
        pred_class_names = [self.epic_id_to_action[int(i)] for i in pred_classes[0]]
        print("Top 5 predicted actions: %s" % ", ".join(pred_class_names))


class Video:
    def __init__(self, files) -> None:
        self.files = files

    @contextmanager
    def write_tmp(self):
        with tempfile.NamedTemporaryFile("wb", suffix=".mp4") as f:
            frame = Image.open(self.files[0])
            video = cv2.VideoWriter(
                f.name,
                cv2.VideoWriter_fourcc(*"mp4v"),
                24,
                (frame.width, frame.height),
            )
            for file in self.files:
                video.write(cv2.imread(file))
            video.release()

            yield f.name


def main():
    ar = ActionRecognizer()

    data = [glob.glob("./satis-cv-ai-exercise-data/train/P01/P01_05/*.jpg")]
    data = [sorted(x) for x in data]
    data = [Video(x) for x in data]

    for video in data:
        with video.write_tmp() as filename:
            x = ar.predict(filename)


if __name__ == "__main__":
    main()
