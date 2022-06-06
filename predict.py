import os

import click
import pandas as pd
import torch
import torchvision.transforms as T
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms._transforms_video import NormalizeVideo
from tqdm import tqdm

from transforms import SpatialCrop, TemporalCrop


class ActionRecognizer:
    def __init__(self, device="cpu") -> None:
        self.device = device

        # Initialize class mapping
        classes = pd.read_csv(
            "https://dl.fbaipublicfiles.com/omnivore/epic_action_classes.csv",
            names=["verb", "noun"],
        )
        classes = classes["verb"] + " " + classes["noun"]
        classes = {c: i for i, c in enumerate(classes)}
        self.classes = classes

        # Initialize model
        model = torch.hub.load(
            "facebookresearch/omnivore:main", model="omnivore_swinB_epic"
        )
        model = model.to(device)
        model = model.eval()
        self.model = model

        # Initialize preprocessing
        self.video_transform = ApplyTransformToKey(
            key="video",
            transform=T.Compose(
                [
                    UniformTemporalSubsample(32),
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

        # The model expects inputs of shape: B x C x T x H x W
        video_input = video_inputs[0][None, ...]

        # Pass the input clip through the model
        with torch.no_grad():
            pred = self.model(video_input.to(self.device), input_type="video")
            pred = (
                pred[0, [self.classes["take bag"], self.classes["open bag"]]]
                .data.cpu()
                .numpy()
                .tolist()
            )
            return pred


@click.command()
@click.option("--data-dir", type=click.Path(), default="./satis-cv-ai-exercise-data")
@click.option("--subset", type=str, default="val")
def main(data_dir, subset):
    data = pd.read_csv(os.path.join(data_dir, f"{subset}_action_classes.csv"))
    ar = ActionRecognizer()

    scores = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        video_file = os.path.join(
            data_dir,
            f"{subset}_videos",
            row["participant_id"] + "_" + row["video_id"] + ".mp4",
        )
        score = ar.predict(video_file)
        scores.append(score)

    data["take_bag_score"], data["open_bag_score"] = zip(*scores)
    data.to_csv(os.path.join(data_dir, f"{subset}_predictions.csv"), index=False)


if __name__ == "__main__":
    main()
