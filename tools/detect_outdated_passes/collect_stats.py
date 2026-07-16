#!/usr/bin/env python3
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import argparse


def _collect_log_files(folders: list[str]) -> list[str]:
    paths = []
    for folder in folders:
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if os.path.isfile(path):
                paths.append(path)
    return paths


def _read_compilation_log(path: str) -> list[tuple[str, bool]]:
    with open(path) as f:
        lines = [line.split() for line in f if line.strip()]
    return [(name, change == "CHANGED") for name, change in lines]


def _calc_stats(models: list[list[tuple[str, bool]]]) -> list[tuple[str, int]]:
    passes = {name: 0 for model in models for name, _ in model}

    for model in models:
        used_passes = set(name for name, change in model if change)
        for name in used_passes:
            passes[name] += 1

    return sorted(passes.items(), key=lambda kv: (kv[1], kv[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='append', default=[], dest='dirs')
    args = parser.parse_args()

    paths = _collect_log_files(args.dirs)
    models = [_read_compilation_log(path) for path in paths]
    passes = _calc_stats(models)

    for pass_name, usage_freq in passes:
        print(f'{pass_name}\t{usage_freq}/{len(models)}')

    print(f"\nUnused passes:\t{sum(1 for _, freq in passes if freq == 0)}/{len(passes)}")


if __name__ == "__main__":
    main()
