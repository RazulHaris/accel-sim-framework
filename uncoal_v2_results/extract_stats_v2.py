#!/usr/bin/env python3

import os
import re
import csv
import argparse

PROCESS_RE = re.compile(r"Processing kernel")
KERNEL_ID_RE = re.compile(r"-kernel id\s*=\s*(\d+)")
LOAD_RE = re.compile(r"gpgpu_n_l1_load_accesses \s*=\s*(\d+)")
COAL_ACCESS_RE = re.compile(r"gpgpu_n_coal_l1_accesses\s*=\s*(\d+)")
UNCOAL_ACCESS_RE = re.compile(r"gpgpu_n_uncoal_l1_accesses\s*=\s*(\d+)")
UNCOAL_MISS_RE = re.compile(r"gpgpu_n_uncoal_l1_miss\s*=\s*(\d+)")
COAL_MISS_RE = re.compile(r"gpgpu_n_coal_l1_miss\s*=\s*(\d+)")

def get_benchmark_name(filepath, root):
    """
    Build a unique benchmark name from the directory structure.

    Examples
    --------
    backprop-rodinia-3.1/65536/A100-SASS/file.o52
        -> backprop-rodinia-3.1_65536

    bfs-rodinia-3.1/__data_graph4096_txt/A100-SASS/file.o53
        -> bfs-rodinia-3.1__data_graph4096_txt

    hotspot-rodinia-3.1/512_2_2___data_temp_512___data_power_512_output_out/A100-SASS/file.o54
        -> hotspot-rodinia-3.1_512_2_2___data_temp_512___data_power_512_output_out
    """

    rel = os.path.relpath(filepath, root)
    parts = rel.split(os.sep)

    # parts = [benchmark, input_case, A100-SASS, output_file]

    # Keep the config arm (A100-SASS vs A100-SASS-ORACLE_PREF) in the key,
    # otherwise the two arms of an oracle run collide into one benchmark name.
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}|{parts[2]}"

    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"

    return parts[0]


def build_row(benchmark, kernel_id, load_access, coal_access, uncoal_access,
             coal_miss, uncoal_miss):
    load_access = load_access if load_access is not None else 0
    coal_access = coal_access if coal_access is not None else 0
    uncoal_access = uncoal_access if uncoal_access is not None else 0
    coal_miss = coal_miss if coal_miss is not None else 0
    uncoal_miss = uncoal_miss if uncoal_miss is not None else 0

    load_miss = coal_miss + uncoal_miss
    coal_miss_rate = coal_miss / coal_access if coal_access else 0.0
    uncoal_miss_rate = uncoal_miss / uncoal_access if uncoal_access else 0.0

    return [
        benchmark,
        kernel_id,
        load_access,
        coal_access,
        uncoal_access,
        load_miss,
        coal_miss,
        uncoal_miss,
        coal_miss_rate,
        uncoal_miss_rate,
    ]


def parse_output_file(filename, benchmark):
    rows = []

    kernel_id = None
    load_access = None
    coal_access = None
    uncoal_access = None
    coal_miss = None
    uncoal_miss = None

    with open(filename, "r", errors="ignore") as f:
        for line in f:

            # New kernel begins
            if PROCESS_RE.search(line):
                if kernel_id is not None:
                    rows.append(build_row(
                        benchmark, kernel_id, load_access, coal_access,
                        uncoal_access, coal_miss, uncoal_miss
                    ))

                kernel_id = None
                load_access = None
                coal_access = None
                uncoal_access = None
                coal_miss = None
                uncoal_miss = None
                continue

            m = KERNEL_ID_RE.search(line)
            if m:
                kernel_id = int(m.group(1))
                continue

            m = LOAD_RE.search(line)
            if m:
                load_access = int(m.group(1))
                continue

            m = COAL_ACCESS_RE.search(line)
            if m:
                coal_access = int(m.group(1))
                continue

            m = UNCOAL_ACCESS_RE.search(line)
            if m:
                uncoal_access = int(m.group(1))
                continue

            m = COAL_MISS_RE.search(line)
            if m:
                coal_miss = int(m.group(1))
                continue

            m = UNCOAL_MISS_RE.search(line)
            if m:
                uncoal_miss = int(m.group(1))
                continue

    # Save the last kernel
    if kernel_id is not None:
        rows.append(build_row(
            benchmark, kernel_id, load_access, coal_access,
            uncoal_access, coal_miss, uncoal_miss
        ))

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract coalesced/uncoalesced L1D load access and miss statistics, and their miss rates, from all benchmarks."
    )

    parser.add_argument(
        "root",
        help="sim_run directory (e.g. sim_run_12.8)"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="results.csv"
    )

    args = parser.parse_args()

    all_rows = []

    for dirpath, _, files in os.walk(args.root):
        for f in files:

            # Only process simulator output files.  The torque/slurm launcher
            # produced "<jobname>.oNN"; run_uncoal_v2.sh runs justrun.sh
            # directly, which tees to "gpgpu-sim-out_<date>.txt" (job.log is a
            # byte-identical copy, so matching only one avoids double counting).
            # The ".txt" suffix matters: an aborted attempt is renamed to
            # "gpgpu-sim-out_<date>.txt.FAILED_RUN" and is left in the job
            # directory, so a bare startswith() would parse the failed run too
            # and emit a spurious all-zero record beside the good one.
            if not (re.search(r"\.o\d+$", f)
                    or (f.startswith("gpgpu-sim-out_") and f.endswith(".txt"))):
                continue

            filepath = os.path.join(dirpath, f)

            benchmark = get_benchmark_name(filepath, args.root)

            all_rows.extend(
                parse_output_file(filepath, benchmark)
            )

    all_rows.sort(key=lambda x: (x[0], x[1]))

    with open(args.output, "w", newline="") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "benchmark",
            "kernel_id",
            "load_access",
            "coal_access",
            "uncoal_access",
            "load_miss",
            "coal_miss",
            "uncoal_miss",
            "coal_miss_rate",
            "uncoal_miss_rate",
        ])

        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} kernel records to {args.output}")


if __name__ == "__main__":
    main()