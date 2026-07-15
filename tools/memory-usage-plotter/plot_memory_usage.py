#!/usr/bin/env python3
#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import re
import sys
import matplotlib.pyplot as plt


def parse_args():
    try:
        in_file = sys.argv[1]
        out_file = sys.argv[2]
    except Exception:
        print(f"Usage: {sys.argv[0]} INPUT_LOG OUTPUT_PNG")
        exit(1)
    return in_file, out_file


def parse_log_file(file_path):
    parts_data = {}

    with open(file_path, 'r') as file:
        is_static_allocation_pass = False
        for line in file:
            if re.search(r'Start Pass StaticAllocation', line):
                is_static_allocation_pass = True

            if not is_static_allocation_pass:
                continue

            part_match = re.search(r'\b(part\d+|main):', line)
            if not part_match:
                continue

            part_name = part_match.group(1)
            if part_name not in parts_data:
                parts_data[part_name] = {
                    'max_allocated': [],
                    'fragmentation': [],
                    'used': [],
                    'free': [],
                    'local_timestamp': 0,
                }

            part = parts_data[part_name]
            ts = part['local_timestamp']

            max_allocated_match = re.search(r'Max allocated size (\d+)', line)
            if max_allocated_match:
                part['max_allocated'].append((ts, int(max_allocated_match.group(1))))
                continue

            fragmentation_match = re.search(r'Increased allocation size to (\d+) B due to fragmentation!', line)
            if fragmentation_match:
                part['fragmentation'].append((ts, int(fragmentation_match.group(1))))
                continue

            used_memory_match = re.search(r'DDR used memory\s+(\d+)', line)
            if used_memory_match:
                part['used'].append((ts, int(used_memory_match.group(1))))
                continue

            free_memory_match = re.search(r'DDR free memory\s+(\d+)', line)
            if free_memory_match:
                part['free'].append((ts, int(free_memory_match.group(1))))
                part['local_timestamp'] += 1
                continue

    return parts_data


def part_sort_key(part_name):
    """Sort logs from functions by part number, with main placed at the end."""
    if part_name == 'main':
        return (1, 0)
    match = re.match(r'part(\d+)', part_name)
    if match:
        return (0, int(match.group(1)))
    return (0, 0)


def sort_and_flatten_parts(parts_data):
    """Sort parts by number and flatten per-part data into a single timeline."""
    sorted_part_names = sorted(parts_data.keys(), key=part_sort_key)

    max_allocated_size_over_time = []
    fragmentation_points = []
    used_memory_over_time = []
    free_memory_over_time = []
    part_boundaries = []

    global_offset = 0
    for part_name in sorted_part_names:
        part = parts_data[part_name]
        if not part['max_allocated'] and not part['free']:
            continue
        part_boundaries.append((global_offset, part_name))
        for ts, val in part['max_allocated']:
            max_allocated_size_over_time.append((ts + global_offset, val))
        for ts, val in part['fragmentation']:
            fragmentation_points.append((ts + global_offset, val))
        for ts, val in part['used']:
            used_memory_over_time.append((ts + global_offset, val))
        for ts, val in part['free']:
            free_memory_over_time.append((ts + global_offset, val))
        global_offset += part['local_timestamp']

    return max_allocated_size_over_time, fragmentation_points, used_memory_over_time, free_memory_over_time, part_boundaries


def get_scale_and_unit(max_allocated_size):
    byte = 1
    kilobyte = byte * 1024
    megabyte = kilobyte * 1024
    gigabyte = megabyte * 1024

    return ((gigabyte, "GB") if max_allocated_size >= gigabyte else
            (megabyte, "MB") if max_allocated_size >= megabyte else
            (kilobyte, "KB") if max_allocated_size >= kilobyte else
            (byte, "B"))


def convert_bytes_to_readable_size(sizes_in_bytes, divider):
    return [(timestamp, size / divider) for timestamp, size in sizes_in_bytes]


def scale_all_series(scale, max_allocated_sizes, max_allocated_sizes_increases, fragmentation_max_allocated_sizes, used_memories, free_memories):
    return (
        convert_bytes_to_readable_size(max_allocated_sizes, scale),
        convert_bytes_to_readable_size(max_allocated_sizes_increases, scale),
        convert_bytes_to_readable_size(fragmentation_max_allocated_sizes, scale),
        convert_bytes_to_readable_size(used_memories, scale),
        convert_bytes_to_readable_size(free_memories, scale),
    )


def configure_plot(unit):
    plt.xlabel('Allocation Step [async op]')
    plt.xlim(xmin=0)
    plt.ylabel(f'Memory Size [{unit}]')
    plt.ylim(ymin=0)
    plt.title('Memory Usage Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.yticks()
    plt.tight_layout()
    plt.ticklabel_format(style='plain')


def draw_part_boundaries(part_boundaries, y_max, total_timestamps, min_label_span_fraction=0.0025):
    """Draw vertical dotted lines and labels to separate parts on the graph.

    Labels are omitted for parts whose timestamp span is smaller than
    min_label_span_fraction of the total timestamps to improve readability.
    """
    min_span = min_label_span_fraction * total_timestamps
    spans = [
        next_ts - cur_ts
        for (cur_ts, _), (next_ts, _) in zip(part_boundaries, part_boundaries[1:])
    ] + [total_timestamps - part_boundaries[-1][0]]

    for (timestamp, part_name), span in zip(part_boundaries, spans):
        plt.axvline(x=timestamp, color='gray', linestyle=':', linewidth=0.5, zorder=0)
        if span >= min_span:
            plt.text(timestamp + 0.5, y_max * 0.98, part_name, rotation=90,
                     fontsize=11, color='gray', verticalalignment='top')


def draw_series(max_allocated_sizes, max_allocated_sizes_increases, fragmentation_max_allocated_sizes,
                used_memories, free_memories, part_boundaries):
    plt.figure(figsize=(40, 6))
    if max_allocated_sizes_increases:
        plt.scatter(*zip(*max_allocated_sizes_increases), label='Max Allocated Size\nIncreased',
                    s=5, marker='o', color='magenta', zorder=2)
    if fragmentation_max_allocated_sizes:
        plt.scatter(*zip(*fragmentation_max_allocated_sizes),
                    label='Max Allocated Size\nIncreased Due To\nFragmentation', s=5, marker='^', color='red', zorder=3)
    plt.plot(*zip(*max_allocated_sizes), label='Max Allocated Size', color='blue', zorder=1)
    final_t, final_size = max_allocated_sizes[-1]
    plt.annotate(f'{final_size:.2f}', xy=(final_t, final_size), xytext=(5, 5),
                 textcoords='offset points', color='blue', fontsize=8)
    plt.plot(*zip(*used_memories), label='Used Memory', color='red', zorder=0)
    plt.plot(*zip(*free_memories), label='Free Memory', color='green', zorder=0)
    y_max = max(size for _, size in max_allocated_sizes)
    total_timestamps = max_allocated_sizes[-1][0] + 1
    draw_part_boundaries(part_boundaries, y_max, total_timestamps)


def plot_memory_usage(max_allocated_sizes, max_allocated_sizes_increases, fragmentation_max_allocated_sizes,
                      used_memories, free_memories, part_boundaries):
    scale, unit = get_scale_and_unit(max_allocated_sizes[-1][1])
    max_allocated_sizes, max_allocated_sizes_increases, fragmentation_max_allocated_sizes, used_memories, free_memories = scale_all_series(
        scale, max_allocated_sizes, max_allocated_sizes_increases, fragmentation_max_allocated_sizes, used_memories, free_memories)
    draw_series(max_allocated_sizes, max_allocated_sizes_increases,
                fragmentation_max_allocated_sizes, used_memories, free_memories, part_boundaries)
    configure_plot(unit)


def save_plot(out_file_path):
    plt.savefig(out_file_path)


if __name__ == "__main__":
    in_file, out_file = parse_args()

    parts_data = parse_log_file(in_file)
    max_allocated_size_over_time, fragmentation_points, used_memory_over_time, free_memory_over_time, part_boundaries = sort_and_flatten_parts(
        parts_data)

    # Convert to non-decreasing max allocated size over time for better visualization
    running_max = 0
    max_allocated_size_over_time = [(t, running_max := max(running_max, s)) for t, s in max_allocated_size_over_time]

    # Get the points where max allocated size increased
    running_max = 0
    max_allocated_size_increase_points = [
        (t, s) for t, s in max_allocated_size_over_time if s > running_max and (running_max := s)]

    # Get only the points where fragmentation caused an increase in max allocated size
    fragmentation_points = [(t, s) for t, s in fragmentation_points if (t, s) in max_allocated_size_increase_points]

    plot_memory_usage(max_allocated_size_over_time, max_allocated_size_increase_points,
                      fragmentation_points, used_memory_over_time, free_memory_over_time, part_boundaries)

    save_plot(out_file)
