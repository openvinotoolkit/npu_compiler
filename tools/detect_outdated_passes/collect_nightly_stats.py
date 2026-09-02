#!/usr/bin/env python3
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import defaultdict
from typing import Callable, Iterable
import argparse
import json
import os
import os.path

Run = list[tuple[str, bool]]
# A configuration is identified by (platform, model_set, variant),
# e.g. ('LNL', 'pub', 'CID_ORT') for a Windows_LNL_CID_ORT folder in the pub set.
Config = tuple[str, str, str]
# For each pass name: how many runs used it and the set of configs where it was used.
Usage = tuple[dict[str, tuple[int, set[Config]]], int]


def _parse_run(output: list[str]) -> Run:
    run: Run = []
    marker = '[pass-usage-observer]'
    for row in output:
        if marker not in row:
            continue
        parts = row.split(marker, 1)[1].strip().rsplit(None, 1)
        if len(parts) != 2:
            continue
        name, status = parts
        run.append((name, status == 'CHANGED'))
    if not run:
        print(f'    [warn] no pass-usage-observer markers found in {len(output)} output lines')
    return run


def _parse_run_report(path: str) -> list[Run]:
    print(f'  [parse] {path}')
    with open(path) as f:
        body = json.load(f)

    result: list[Run] = []
    for key, values in body.items():
        compile_result = values.get('Compile') or values.get('Compile_ORT', {})
        run = _parse_run(compile_result.get('output', []))
        if run:
            result.append(run)
        else:
            print(f'    [skip] no usable run for model "{key}" (keys: {list(values.keys())})')
    print(f'  [parse] -> {len(result)} model(s) with pass data')
    return result


def _is_compile_dir(name: str) -> bool:
    # Match 'Run-...-Compile' and 'Run-...-Compile_ORT', but not 'Run-...-CompileTime'.
    last = name.split('-')[-1]
    return last == 'Compile' or last.startswith('Compile_')


def _find_run_report(root: str) -> list[Run]:
    found_files = [
        os.path.join(subdir, file)
        for subdir, _, files in os.walk(root)
        for file in files
        if file == 'run-report.json' and _is_compile_dir(os.path.basename(subdir))
    ]
    print(f'  [scan] {root}: found {len(found_files)} Compile run-report.json file(s)')
    return [run for path in found_files for run in _parse_run_report(path)]


def _list_reports(dirs: list[tuple[str, str]]) -> dict[Config, list[Run]]:
    result: dict[Config, list[Run]] = {}

    for model_set, root in dirs:
        print(f'[list] scanning "{root}" for model set "{model_set}"')
        platform_dirs = [
            os.path.join(dirpath, name)
            for dirpath, dirnames, _ in os.walk(root)
            for name in dirnames
            if name.startswith('Ubuntu_') or name.startswith('Windows_')
        ]
        print(f'[list] found {len(platform_dirs)} platform dir(s)')
        for path in platform_dirs:
            name = os.path.basename(path)
            parts = name.split('_')
            platform = parts[1]
            variant = '_'.join(parts[2:])
            config: Config = (platform, model_set, variant)
            print(f'[list] processing platform={platform}, set={model_set}, variant={variant}, dir={name}')
            result.setdefault(config, []).extend(_find_run_report(path))

    return result


def calc_usage(runs: list[Run], config: Config) -> Usage:
    passes: dict[str, tuple[int, set[Config]]] = {pass_name: (0, set()) for run in runs for pass_name, _ in run}
    for run in runs:
        used_passes = set(name for name, change in run if change)
        for name in used_passes:
            hits, configs = passes[name]
            passes[name] = (hits + 1, configs | {config})
    return passes, len(runs)


def merge_usages(usages: Iterable[Usage]) -> Usage:
    total = 0
    union: dict[str, tuple[int, set[Config]]] = {}
    for stats, count in usages:
        total += count
        for pass_name, (hits, configs) in stats.items():
            u_hits, u_configs = union.get(pass_name, (0, set()))
            union[pass_name] = (u_hits + hits, u_configs | configs)
    return union, total


def print_usage(f: str, usage: Usage, fmt_config: Callable[[Config], str], skip_threshold: float):
    with open(f, mode='w') as out:
        out.write('Passes, Number of uses, Comments, Category\n')
        stats, total = usage
        for pass_name, (hits, configs) in sorted(stats.items(), key=lambda kv: (kv[1][0], kv[0])):
            if total > 0 and hits / total > skip_threshold:
                labels = 'SKIP_CATEGORIES'
            else:
                labels = ''.join(f'{label};' for label in sorted(fmt_config(c) for c in configs))
            out.write(f'{pass_name},{hits}/{total},,{labels}\n')

        unused = sum(int(hits == 0) for hits, _ in stats.values())
        out.write(f'Unused passes:,{unused}/{len(stats)},,\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs=2, action='append', metavar=('SET', 'DIR'),
                        help='A model set name (e.g. pub) and a path to the directory of log folders', required=True)
    parser.add_argument('-o', required=True, metavar='OUTPUT', help='Output directory')
    parser.add_argument('-t', '--skip-threshold', type=float, default=5.0, metavar='PERCENT',
                        help='If a pass is used in more than PERCENT%% of runs, print SKIP_CATEGORIES '
                             'instead of the list of configurations (default: 5)')
    args = parser.parse_args()

    skip_threshold = args.skip_threshold / 100.0

    os.makedirs(args.o, exist_ok=True)
    reports = _list_reports(args.d)

    usage_by_config = {config: calc_usage(runs, config) for config, runs in reports.items()}

    # summary.csv: config shown as platform/set/variant, e.g. LNL/pub/CID_ORT
    print_usage(os.path.join(args.o, 'summary.csv'),
                merge_usages(usage_by_config.values()),
                lambda c: f'{c[0]}/{c[1]}/{c[2]}', skip_threshold)

    # <platform>.csv: config shown as set/variant, e.g. pub/CID_ORT
    by_platform: dict[str, list[Usage]] = defaultdict(list)
    for config, usage in usage_by_config.items():
        by_platform[config[0]].append(usage)
    for platform, platform_usages in by_platform.items():
        print_usage(os.path.join(args.o, f'{platform}.csv'),
                    merge_usages(platform_usages),
                    lambda c: f'{c[1]}/{c[2]}', skip_threshold)

    # <set>_<platform>.csv: config shown as variant only, e.g. CID_ORT
    by_set_platform: dict[tuple[str, str], list[Usage]] = defaultdict(list)
    for config, usage in usage_by_config.items():
        by_set_platform[(config[1], config[0])].append(usage)
    for (model_set, platform), sp_usages in by_set_platform.items():
        print_usage(os.path.join(args.o, f'{model_set}_{platform}.csv'),
                    merge_usages(sp_usages),
                    lambda c: c[2], skip_threshold)


if __name__ == "__main__":
    main()
