# %%
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import zarr


# this function reads the csv file for the scatterplot function
# returns two lists of both time frames and f1 scores
def read_csv_file(file_path):
    time_frames = []
    f1_scores = []

    with open(file_path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            time_frames.append(int(row["time_frame"]))
            f1_scores.append(float(row["f1_score"]))

    return time_frames, f1_scores


# this function plots the F1 Score versus the Time Frame for all videos
# plots
def plot_scatter(directories):
    for directory in directories:
        csv_files = list(Path(directory).rglob("new_evaluation.csv"))

        for csv_file in csv_files:
            time_frames, f1_scores = read_csv_file(csv_file)
            plt.scatter(time_frames, f1_scores)

    plt.xlabel("Time Frame")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Time Frame for All Videos")
    plt.grid(True)
    plt.show()


# this function plots the f1 score frequency from the 100 videos
# it returns a list of f1 scores for each of the videos
def plot_hist():
    total_FN = 0
    total_FP = 0
    total_TP = 0
    f1 = []

    for file_num in range(1, 101):
        base_path = Path(
            f"/nrs/funke/data/mhat/synthetic_data/test1/{file_num}/data.zarr"
        )

        csv_file = base_path.parent / "new_evaluation.csv"
        with open(csv_file) as f:
            datareader = csv.reader(f)
            next(datareader)
            for r in datareader:
                total_TP += int(r[1])
                total_FP += int(r[2])
                total_FN += int(r[3])

        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        f1.append(
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # find a better way to save the different f1 scores
    plt.hist(f1)
    plt.xlabel("F1 Score")
    plt.ylabel("Frquency")
    plt.title("Calculated F1 Plot for All 100 Videos")
    plt.show()

    return f1


# this function shows the correlation between the Cell Length and F1 score per video
# plots
def plot_length_f1():
    f1 = plot_hist()
    cell_length = []

    for file_num in range(1, 101):
        base_path = Path(
            f"/nrs/funke/data/mhat/synthetic_data/test1/{file_num}/data.zarr"
        )
        zarr_group = zarr.open_group(base_path, mode="r")
        print(zarr_group.attrs)
        length = zarr_group.attrs["simulation"]["cell_width"]
        cell_length.append(length)

    plt.scatter(f1, cell_length)
    plt.xlabel("F1 Score")
    plt.ylabel("Cell Width")
    plt.title("Cell Width vs. F1 Score Graph")
    plt.show()


# this function gets the total values for the TP, FP, FN and uses it to calculate precision, recall, F1
# returns a list of a dictionary of the different values
def total_info():
    total_TP = 0
    total_FP = 0
    total_FN = 0
    results = []

    for file_num in range(1, 101):
        base_path = Path(
            f"/nrs/funke/data/mhat/synthetic_data/test1/{file_num}/data.zarr"
        )

        csv_file = base_path.parent / "new_evaluation.csv"
        with open(csv_file) as f:
            datareader = csv.reader(f)
            next(datareader)
            for r in datareader:
                total_TP += int(r[1])
                total_FP += int(r[2])
                total_FN += int(r[3])

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    results = [
        {
            "Total_TP": total_TP,
            "Total_FP": total_FP,
            "Total_FN": total_FN,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }
    ]

    return results


# %%
if __name__ == "__main__":
    base_path = Path("/nrs/funke/data/mhat/synthetic_data/test1")
    num_directories = 100
    directories = [base_path / str(i) for i in range(1, num_directories + 1)]
    plot_scatter(directories)

    # plot_hist()

    # plot_length_f1()

    # results = total_info()
    # print(results)

    # f1 = plot_hist()
    # for score in f1:
    #     if score < 0.9:
    #         print(f1.index(score))
    # value = f1[0]
    # print(value)


# %%
