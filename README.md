# Satis test task

## Approach

1. Initially I thought "Action Recognition" task is to find subsequence of frames in original video with target action, but after reading a bit about the dataset on https://epic-kitchens.github.io/2022 I found out that this is simply a video classification task, which is a bit easier.
2. So I went to [Action Recognition on EPIC-KITCHENS-100](https://paperswithcode.com/sota/action-recognition-on-epic-kitchens-100) page on [paperswithcode.com](https://paperswithcode.com) and just picked the first repo which had both trained model the straightforward inference setup - [Omnivore](https://github.com/facebookresearch/omnivore).
3. Then I checked that all is working, made the preprocessing code to convert frames to videos, made the inference code and finally evaluation code.
4. The model takes video as input and outputs scores for 3806 classes, I'm using only `open bag`, and `take bag` classes (also it's possible that more sophisticated postprocessing will perform better).
5. I didn't consider any other possible requirements (such as inference cost and speed) and focused only on solving the task as specified in exercise doc.

## Instructions

I'm using [Poetry](https://python-poetry.org) for package management, everything is tested on MacOS.

### Download data

Download https://bit.ly/3miRzjy, unzip, and put in project dir under `satis-cv-ai-exercise-data` dir

### Install dependencies

```python
poetry install
```

### Preprocess data

```python
poetry run python preprocess.py
```

### Run inference

```python
poetry run python predict.py --subset train
```

```python
poetry run python predict.py --subset val
```

### Compute metrics

```python
poetry run python evaluate.py --preds-file ./satis-cv-ai-exercise-data/train_predictions.csv
```

```python
poetry run python evaluate.py --preds-file ./satis-cv-ai-exercise-data/val_predictions.csv
```

#### Val metrics

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| take bag     | 1.00      | 0.67   | 0.80     | 3       |
| open bag     | 0.75      | 1.00   | 0.86     | 3       |
| accuracy     |           |        | 0.83     | 6       |
| macro avg    | 0.88      | 0.83   | 0.83     | 6       |
| weighted avg | 0.88      | 0.83   | 0.83     | 6       |

## Next steps

-   Error analysis
-   Compute per-participant metric
-   Parallelize preprocessing for large datasets
-   Use cheaper/faster model
-   Use original EPIC-KITCHENS evaluation code
