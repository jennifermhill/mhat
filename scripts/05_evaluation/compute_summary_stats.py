import csv
import json
import os
from pathlib import Path


# Segmentation (mask_eval.csv) calculations
def segmentation_eval_info(dt):
    total_TP = 0
    total_FP = 0
    total_FN = 0
    results = []

    for file_num in range(1, 101):
        base_path = Path(f"/nrs/funke/data/mhat/synthetic_data/test1/{file_num}/{dt}")

        csv_file = base_path / "mask_eval.csv"
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

    results = {
        "Segmentation Evaluation": {
            "Total_TP": total_TP,
            "Total_FP": total_FP,
            "Total_FN": total_FN,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }
    }

    return results


# track division (track_metrics) calculations
def division_info(dt):
    total_TP = 0
    total_FP = 0
    total_FN = 0
    for file_num in range(1, 101):
        base_path = Path(f"/nrs/funke/data/mhat/synthetic_data/test1/{file_num}/{dt}")
        json_path = base_path / "track_metrics.json"

        with open(json_path) as f:
            data = json.load(f)

        # for entry in data:
        #     results = entry["results"]
        #     if isinstance(results, dict):
        #         for i in range(2):
        #             frame_buffer_key = f"Frame Buffer {i}"
        #             if frame_buffer_key in results:
        #                 frame_buffer_results = results[frame_buffer_key]
        #                 if isinstance(frame_buffer_results, dict):
        #                     total_TP += frame_buffer_results.get("True Positive Divisions", 0)
        #                     total_FP += frame_buffer_results.get("False Positive Divisions", 0)
        #                     total_FN += frame_buffer_results.get("False Negative Divisions", 0)

        for entry in data:
            if "results" in entry and "Frame Buffer 2" in entry["results"]:
                results = entry["results"]["Frame Buffer 2"]
                total_TP += results.get("True Positive Divisions", 0)
                total_FP += results.get("False Positive Divisions", 0)
                total_FN += results.get("False Negative Divisions", 0)

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    division_results = {
        "Division Evalution": {
            "Total_Division_TP": total_TP,
            "Total_Division_FP": total_FP,
            "Total_Division_FN": total_FN,
            "Division_Precision": precision,
            "Division_Recall": recall,
            "Division_F1_Score": f1,
        }
    }

    return division_results


def average_TRA(dt):
    total_TRA = 0
    count = 0
    for file_num in range(1, 101):
        base_path = Path(f"/nrs/funke/data/mhat/synthetic_data/test1/{file_num}/{dt}")
        json_path = base_path / "track_metrics.json"

        with open(json_path) as f:
            data = json.load(f)

            for entry in data:
                if "results" in entry and "TRA" in entry["results"]:
                    tra_score = entry["results"]["TRA"]
                    total_TRA += tra_score
                    count += 1

    average = total_TRA / count if count > 0 else 0
    score = {"Average TRA": average}
    return score


if __name__ in "__main__":
    dt = "2024-08-09_15-40-36"
    # making_dir = os.mkdir(f"/nrs/funke/data/mhat/synthetic_data/test1/results")
    make_dir = os.mkdir(f"/nrs/funke/data/mhat/synthetic_data/test1/results/{dt}")
    json_path = f"/nrs/funke/data/mhat/synthetic_data/test1/results/{dt}/evaluation_metrics.json"

    # try:
    #     os.makedirs(json_path)
    # except FileExistsError:
    #     pass

    results = segmentation_eval_info(dt)
    division_results = division_info(dt)
    average = average_TRA(dt)

    data = [results, division_results, average]
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
