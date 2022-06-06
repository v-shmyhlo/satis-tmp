# Satis test task

## Approach

1. Initially I thought "Action Recognition" task is to find subsequence of frames in original video with target action, but after reading a bit about the dataset on https://epic-kitchens.github.io/2022 I found out that this is simply a video classification task, which is a bit easier.
2. So I went to ["Action Recognition on EPIC-KITCHENS-100"](https://paperswithcode.com/sota/action-recognition-on-epic-kitchens-100) section on [paperswithcode.com](https://paperswithcode.com) and just picked the first repo which had both trained model the straightforward inference setup - [Omnivore](https://github.com/facebookresearch/omnivore).
3. Then I checked that all is working, made the preprocessing code to convert frames to videos, made the inference code and finally evaluation code.
4. I didn't consider any other possible requirements (such as inference cost and speed) and focused only on solving the task as specified in exercise doc.

## Instructions

I'm using [Poetry](https://python-poetry.org) as package manager, everything is tested on MacOS.

### Install dependencies

```python
poetry install
```

### Download data

```

```

### Preprocess data

```python
poetry run python preprocess.py
```

### Run inference

```python
poetry run python predict.py
```

### Compute metrics

```python
poetry run python evaluate.py
```

## Next steps

-   Error analysis
-   Compute per-participant metric
-   Parallelize preprocessing for large datasets
