import click
import pandas as pd
from sklearn.metrics import classification_report


@click.command()
@click.option("--preds-file", type=click.Path(), required=True)
def main(preds_file):
    preds = pd.read_csv(preds_file)
    preds["action_class_pred"] = preds.apply(
        lambda row: "take bag"
        if row["take_bag_score"] > row["open_bag_score"]
        else "open bag",
        axis=1,
    )

    classes = ["take bag", "open bag"]
    y_true = preds["action_class"].apply(lambda c: classes.index(c.strip()))
    y_pred = preds["action_class_pred"].apply(lambda c: classes.index(c.strip()))
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)


if __name__ == "__main__":
    main()
