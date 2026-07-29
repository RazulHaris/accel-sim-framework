import os
import csv
import re

# Directory containing rodinia output files
INPUT_DIR = "./"

# Output CSV
OUTPUT_FILE = "warp_stalls.csv"


def parse_warp_occupancy(file_path, kernel_name):

    results = []

    with open(file_path, 'r') as f:
        text = f.read()

    # find all Warp Occupancy sections
    matches = re.findall(r"Warp Occupancy Distribution:(.*?)(?:\n\n|\Z)", text, re.DOTALL)

    for i, section in enumerate(matches):

        pairs = re.findall(r"(\S+):(\d+)", section)

        data = {}
        for k, v in pairs:
            data[k] = int(v)

        stall = data.get("Stall", 0)
        idle = data.get("W0_Idle", 0)
        scoreboard = data.get("W0_Scoreboard", 0)

        ready = 0
        for k in data:
            if re.match(r"W\d+", k) and k not in ["W0_Idle", "W0_Scoreboard"]:
                ready += data[k]

        total = stall + idle + scoreboard + ready

        if total == 0:
            continue

        results.append({
            "Kernel": f"{kernel_name}_kernel{i}",
            "Stall": stall,
            "Idle": idle,
            "Scoreboard": scoreboard,
            "Ready": ready,
            "TotalCycles": total,
            "Stall%": round(100 * stall / total, 3),
            "Idle%": round(100 * idle / total, 3),
            "Scoreboard%": round(100 * scoreboard / total, 3),
            "Ready%": round(100 * ready / total, 3)
        })

    return results


# Store all parsed data
results = []

# Walk through rodinia directory
for root, dirs, files in os.walk(INPUT_DIR):

    for file in files:

        if file.startswith("rodinia") and file.endswith(".txt"):

            path = os.path.join(root, file)

            kernel_name = file.replace(".txt", "")

            parsed = parse_warp_occupancy(path, kernel_name)

            results.extend(parsed)


# Write CSV
with open(OUTPUT_FILE, "w", newline="") as csvfile:

    fieldnames = [
        "Kernel",
        "Stall",
        "Idle",
        "Scoreboard",
        "Ready",
        "TotalCycles",
        "Stall%",
        "Idle%",
        "Scoreboard%",
        "Ready%"
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for r in results:
        writer.writerow(r)


print("CSV generated:", OUTPUT_FILE)
