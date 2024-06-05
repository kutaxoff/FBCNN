import re
import glob
import json
import matplotlib.pyplot as plt
from io import BytesIO
import xlsxwriter
import os.path

RESULTS_PATH = 'comparison_results'

if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)



def parse_logs(logs):
    # Regular expressions to match the required lines
    quality_factor_re = re.compile(
        r"--------------- quality factor: (\d+) ---------------"
    )
    metrics_re = re.compile(
        r"Average PSNR/SSIM/PSNRB/Hamming Score - (.+?) -: ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)\."
    )
    qf_accuracy_re = re.compile(r"Average QF prediction accuracy - (.+?): ([\d.]+)%")

    # Initialize the result dictionary
    result = {
        "psnr": {},
        "ssim": {},
        "psnrb": {},
        "hamming_score": {},
        "qf_pred_accuracy": {},
    }

    # Variables to keep track of current quality factor and task name
    current_qf = None
    task_name = None

    # Process each line in the logs
    for line in logs.split("\n"):
        # Check for quality factor line
        qf_match = quality_factor_re.search(line)
        if qf_match:
            current_qf = qf_match.group(1)

        # Check for average metrics line
        metrics_match = metrics_re.search(line)
        if metrics_match and current_qf is not None:
            task_name = metrics_match.group(1)
            psnr, ssim, psnrb, hamming_score = metrics_match.groups()[1:]
            result["psnr"][current_qf] = float(psnr)
            result["ssim"][current_qf] = float(ssim)
            result["psnrb"][current_qf] = float(psnrb)
            result["hamming_score"][current_qf] = (
                float(hamming_score) / 100
            )  # Convert to fraction

        # Check for quality factor prediction accuracy line
        qf_accuracy_match = qf_accuracy_re.search(line)
        if qf_accuracy_match and current_qf is not None:
            task_name = qf_accuracy_match.group(1)
            qf_pred_accuracy = qf_accuracy_match.group(2)
            result["qf_pred_accuracy"][current_qf] = (
                float(qf_pred_accuracy) / 100
            )  # Convert to fraction

    return result, task_name


def parse_multiple_logs(file_patterns):
    all_results = []

    # Iterate over each file pattern
    for pattern in file_patterns:
        # Get the list of files matching the pattern
        log_files = glob.glob(pattern)

        for log_file in log_files:
            with open(log_file, "r") as file:
                logs = file.read()
                parsed_data, task_name = parse_logs(logs)
                all_results.append({"task_name": task_name, "data": parsed_data})

    return all_results


def plot_metrics(parsed_results):
    metrics = ["psnr", "ssim", "psnrb", "hamming_score", "qf_pred_accuracy"]
    labels = ["PSNR", "SSIM", "PSNRB", "Hamming Score", "QF Prediction Accuracy"]
    graphs = {}

    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 6))
        for result in parsed_results:
            task_name = result["task_name"]
            data = result["data"][metric]
            quality_factors = sorted(data.keys(), key=int)
            values = [data[qf] for qf in quality_factors]

            plt.plot(quality_factors, values, label=task_name)

        plt.xlabel("Quality Factor")
        plt.ylabel(labels[i])
        plt.title(f"Comparison of {labels[i]}")
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(RESULTS_PATH, f"{metric}_comparison.png"))

        plt.show()

    return graphs


def save_data_to_excel(parsed_results, filename=os.path.join(RESULTS_PATH, "parsed_results.xlsx")):
    # Create an Excel workbook and add a worksheet
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()

    # Define the metrics
    metrics = ["psnr", "ssim", "psnrb", "hamming_score", "qf_pred_accuracy"]
    labels = ["PSNR", "SSIM", "PSNRB", "Hamming Score", "QF Prediction Accuracy"]

    # Write headers
    headers = ["Quality Factor"] + [
        f"{metric.upper()} - {result['task_name']}"
        for result in parsed_results
        for metric in metrics
    ]
    worksheet.write_row(0, 0, headers)

    # Prepare data for writing and plotting
    data = {}
    for metric in metrics:
        data[metric] = {}
        for result in parsed_results:
            task_name = result["task_name"]
            metric_data = result["data"][metric]
            for qf, value in metric_data.items():
                if qf not in data[metric]:
                    data[metric][qf] = []
                data[metric][qf].append(value)

    # Write data to worksheet
    row = 1
    for qf in sorted(data[metrics[0]].keys(), key=int):
        worksheet.write(row, 0, int(qf))
        col = 1
        for metric in metrics:
            for value in data[metric][qf]:
                worksheet.write(row, col, value)
                col += 1
        row += 1

    # Add charts for each metric
    for i, metric in enumerate(metrics):
        chart = workbook.add_chart({"type": "line"})
        col = 1 + i * len(parsed_results)
        for j, result in enumerate(parsed_results):
            task_name = result["task_name"]
            chart.add_series(
                {
                    "name": headers[col + j],
                    "categories": [worksheet.name, 1, 0, row - 1, 0],
                    "values": [worksheet.name, 1, col + j, row - 1, col + j],
                }
            )
        chart.set_title(
            {"name": f"Comparison of {labels[i]}"}
        )
        chart.set_x_axis({"name": "Quality Factor"})
        chart.set_y_axis({"name": labels[i]})
        chart.set_legend({"position": "bottom"})
        worksheet.insert_chart(row + 1 + i * 15, 0, chart)

    workbook.close()


# Define file patterns to search for log files
file_patterns = ["test_results/*.log"]
# Parse the log files
parsed_results = parse_multiple_logs(file_patterns)

# Save the parsed results to a JSON file
with open(os.path.join(RESULTS_PATH, "parsed_results.json"), "w") as json_file:
    json.dump(parsed_results, json_file, indent=4)

# Print the parsed results for verification
print(json.dumps(parsed_results, indent=4))

# Plot metrics and save plots to a dictionary
graphs = plot_metrics(parsed_results)

# Save data and charts to Excel file
save_data_to_excel(parsed_results)
