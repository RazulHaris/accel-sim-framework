import os
import re
import csv

FOLDER = "."
OUTPUT = "metrics.csv"

patterns = {
    # traffic
    "core_r": re.compile(r"traffic_breakdown_coretomem\[GLOBAL_ACC_R\]\s*=\s*(\d+)\s*\{([^}]*)\}"),
    "core_w": re.compile(r"traffic_breakdown_coretomem\[GLOBAL_ACC_W\]\s*=\s*(\d+)\s*\{([^}]*)\}"),
    "mem_r": re.compile(r"traffic_breakdown_memtocore\[GLOBAL_ACC_R\]\s*=\s*(\d+)\s*\{([^}]*)\}"),
    "mem_w": re.compile(r"traffic_breakdown_memtocore\[GLOBAL_ACC_W\]\s*=\s*(\d+)\s*\{([^}]*)\}"),

    "cycle": re.compile(r"gpu_tot_sim_cycle\s*=\s*(\d+)"),

    # READ
    "hit_r": re.compile(r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\]\s*=\s*(\d+)"),
    "hit_res_r": re.compile(r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT_RESERVED\]\s*=\s*(\d+)"),
    "miss_r": re.compile(r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[MISS\]\s*=\s*(\d+)"),
    "res_fail_r": re.compile(r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[RESERVATION_FAIL\]\s*=\s*(\d+)"),

    # WRITE
    "hit_w": re.compile(r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_W\]\[HIT\]\s*=\s*(\d+)"),
    "hit_res_w": re.compile(r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_W\]\[HIT_RESERVED\]\s*=\s*(\d+)"),
    "miss_w": re.compile(r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_W\]\[MISS\]\s*=\s*(\d+)"),
    "res_fail_w": re.compile(r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_W\]\[RESERVATION_FAIL\]\s*=\s*(\d+)")
}

FLIT_SIZE = 40
EFFECTIVE_LINKS = 32   # realistic active parallelism
PEAK_BYTES_PER_CYCLE = FLIT_SIZE * EFFECTIVE_LINKS  # 1280


def extract_last(pattern, text):
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def compute_bytes(total, breakdown_str):
    if not breakdown_str:
        return int(total)

    total_bytes = 0
    for entry in breakdown_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        size, count = entry.split(":")
        total_bytes += int(size) * int(count)

    return total_bytes


def get_int(pattern, text):
    val = extract_last(pattern, text)
    return int(val) if val else 0


def parse_file(path):
    with open(path, "r", errors="ignore") as f:
        text = f.read()

    # ---- traffic ----
    core_r = extract_last(patterns["core_r"], text)
    core_w = extract_last(patterns["core_w"], text)
    mem_r = extract_last(patterns["mem_r"], text)
    mem_w = extract_last(patterns["mem_w"], text)

    core_r_bytes = compute_bytes(*core_r) if core_r else 0
    core_w_bytes = compute_bytes(*core_w) if core_w else 0
    mem_r_bytes = compute_bytes(*mem_r) if mem_r else 0
    mem_w_bytes = compute_bytes(*mem_w) if mem_w else 0

    total_bytes = core_r_bytes + core_w_bytes + mem_r_bytes + mem_w_bytes

    # ---- cycles ----
    cycles = get_int(patterns["cycle"], text)

    # ---- reservation fail (READ + WRITE) ----
    hit = get_int(patterns["hit_r"], text) + get_int(patterns["hit_w"], text)
    miss = get_int(patterns["miss_r"], text) + get_int(patterns["miss_w"], text)
    hit_res = get_int(patterns["hit_res_r"], text) + get_int(patterns["hit_res_w"], text)
    res_fail = get_int(patterns["res_fail_r"], text) + get_int(patterns["res_fail_w"], text)

    total_access = hit + miss + hit_res + res_fail
    rf = res_fail / total_access if total_access else 0

    # ---- bandwidth ----
    bw_util = (
        total_bytes /
        (cycles * PEAK_BYTES_PER_CYCLE)
        if cycles else 0
    )

    return {
        "total_bytes": total_bytes,
        "gpu_tot_sim_cycle": cycles,
        "reservation_fail_rate": rf,
        "bandwidth_util": bw_util,
        "HIT_total": hit,
        "MISS_total": miss,
        "HIT_RESERVED_total": hit_res,
        "RESERVATION_FAIL_total": res_fail
    }


rows = []

for file in os.listdir(FOLDER):
    if file.endswith(".txt"):
        metrics = parse_file(file)
        row = {"benchmark": file.replace(".txt", ""), **metrics}
        rows.append(row)

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("[✔] metrics.csv generated")