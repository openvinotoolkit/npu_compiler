#!/usr/bin/env python3

#
# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import logging
import argparse
from pydantic import BaseModel, computed_field
from enum import Enum
import os
import json
import subprocess
import shutil
import time
import shlex
from datetime import datetime
import threading


timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20260313_121300


def folder_retention(*, folder: str, threshold_in_bytes: int):
    # folder retention: if total size of folder exceeds threshold, delete files until size is under threshold
    os.makedirs(folder, exist_ok=True)

    total_size = sum(
        os.path.getsize(os.path.join(folder, file))
        for file in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, file))
    )
    if total_size > threshold_in_bytes:
        # files sorted in alphabetical order (timestamp used in names => older files will be removed first)
        files = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        )
        for file in files:
            if total_size <= threshold_in_bytes:
                break
            file_size = os.path.getsize(file)
            os.remove(file)
            total_size -= file_size


def create_log_file() -> str:
    logs_folder = os.path.join(get_npu_repo_dir(), "scripts/build/logs")
    os.makedirs(logs_folder, exist_ok=True)
    folder_retention(
        folder=logs_folder,
        threshold_in_bytes=30 * 1024 * 1024,  # 30 MB
    )

    # create new log file
    return os.path.join(logs_folder, f"{timestamp_str}_build.log")


def get_npu_repo_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logger = logging.getLogger(__name__)
logger.handlers.clear()  # remove any stale handlers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(create_log_file(), mode="w"),
        logging.StreamHandler(),
    ],
)


class BuildType(Enum):
    Release = "Release"
    RelWithDebInfo = "RelWithDebInfo"
    Debug = "Debug"


class OVRetrieveFlow(BaseModel):
    ov_branch: str = "master"
    ov_commit: str
    ov_repo_dir: str
    force_update: bool = False
    force_update_if_does_not_match: bool = False


class OVBuildFlow(BaseModel):
    ov_repo_dir: str
    ov_cmake_preset: str
    build_type: BuildType
    force_rebuild: bool = False
    parallel_build_num_processes: int

    @computed_field
    def ov_build_dir(self) -> str:
        return os.path.join(self.ov_repo_dir, f"build-x86_64/{self.build_type.value}")


def human_time(seconds: float) -> str:
    hours, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if mins:
        parts.append(f"{mins}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


class BuildTimeStat(BaseModel):
    export_build_metrics: bool = True
    create_project_ov_sec: float = 0
    build_time_ov_sec: float = 0
    create_project_npu_sec: float = 0
    build_time_npu_deps_flatbuffers_sec: float = 0
    build_time_npu_deps_vpucostmodel_sec: float = 0
    build_time_npu_deps_llvm_sec: float = 0
    build_time_npu_deps_elf_sec: float = 0
    build_time_npu_sec: float = 0

    @computed_field
    def total_time(self) -> float:
        return sum(
            getattr(self, name)
            for name, info in self.model_fields.items()
            if info.annotation == float
        )

    @computed_field
    def create_project_ov_human(self) -> str:
        return human_time(self.create_project_ov_sec)

    @computed_field
    def build_time_ov_human(self) -> str:
        return human_time(self.build_time_ov_sec)

    @computed_field
    def create_project_npu_human(self) -> str:
        return human_time(self.create_project_npu_sec)

    @computed_field
    def build_time_npu_deps_flatbuffers_sec_human(self) -> str:
        return human_time(self.build_time_npu_deps_flatbuffers_sec)

    @computed_field
    def build_time_npu_deps_vpucostmodel_sec_human(self) -> str:
        return human_time(self.build_time_npu_deps_vpucostmodel_sec)

    @computed_field
    def build_time_npu_deps_llvm_sec_human(self) -> str:
        return human_time(self.build_time_npu_deps_llvm_sec)

    @computed_field
    def build_time_npu_deps_elf_sec_human(self) -> str:
        return human_time(self.build_time_npu_deps_elf_sec)

    @computed_field
    def build_time_npu_human(self) -> str:
        return human_time(self.build_time_npu_sec)

    @computed_field
    def total_time_human(self) -> str:
        return human_time(self.total_time)


class NPUBuildFlow(BaseModel):
    ov_repo_dir: str
    build_type: BuildType
    npu_cmake_preset: str
    force_project_create: bool = False
    fail_if_ov_does_not_match: bool = True
    parallel_build_num_processes: int
    npu_commit: str | None = None
    npu_git_diff_exists: bool | None = None
    enable_profiling: bool = False

    @computed_field
    def npu_build_dir(self) -> str:
        return os.path.join(get_npu_repo_dir(), f"build-x86_64/{self.build_type.value}")

    @computed_field
    def ov_build_dir(self) -> str:
        return os.path.join(self.ov_repo_dir, f"build-x86_64/{self.build_type.value}")

    @computed_field
    def ninja_log_file(self) -> str:
        return os.path.join(
            get_npu_repo_dir(), f"build-x86_64/{self.build_type.value}/.ninja_log"
        )

    @computed_field
    def profiling_file(self) -> str | None:
        if self.enable_profiling is True:
            return os.path.join(
                get_npu_repo_dir(),
                f"scripts/build/profiling/{timestamp_str}_profiling.json",
            )
        else:
            return None

    @computed_field
    def build_clang_capture_file(self) -> str | None:
        if self.enable_profiling is True:
            return os.path.join(
                get_npu_repo_dir(),
                f"scripts/build/profiling/{timestamp_str}_build_clang_capture_file.bin",
            )
        else:
            return None

    @computed_field
    def build_analysis_file(self) -> str | None:
        if self.enable_profiling is True:
            return os.path.join(
                get_npu_repo_dir(),
                f"scripts/build/profiling/{timestamp_str}_build_analysis.txt",
            )
        else:
            return None


class FullFlow(BaseModel):
    build_time_stat: BuildTimeStat | None = None
    ov_retrieve_flow: OVRetrieveFlow | None = None
    ov_build_flow: OVBuildFlow | None = None
    npu_build_flow: NPUBuildFlow | None = None


def system_call(
    cmd,
    *,
    cwd: None | str = None,
    extra_env: dict[str, str] | None = None,
    output_file: str | None = None,
    cmd_could_fail: bool = False,
) -> float:
    logger.info(f"running cmd='{cmd}' ({cwd=}, {extra_env=}) ...")
    env = {
        **os.environ,
        **(extra_env or {}),
        "GIT_TERMINAL_PROMPT": "0",  # prevent hang waiting for credentials
        "PYTHONUNBUFFERED": "1",  # unbuffer Python side
    }

    # Inject --progress for git commands that support it
    progress_cmds = ("clone", "fetch", "pull", "push")
    cmd_parts = shlex.split(cmd)
    if (
        cmd_parts[0] == "git"
        and len(cmd_parts) >= 2
        and cmd_parts[1] in progress_cmds
        and "--progress" not in cmd_parts
    ):
        cmd_parts.insert(2, "--progress")

    output_file_object = open(output_file, "w") if output_file is not None else None

    def pump_stream(stream, handler):
        try:
            for line in iter(stream.readline, ""):
                handler(line)
        finally:
            stream.close()

    def handle_info_line(line):
        if output_file_object is not None:
            output_file_object.write(line)
            output_file_object.flush()
        else:
            logger.info(line.rstrip())

    def handle_error_line(line):
        logger.error(line.rstrip())
        if output_file_object is not None:
            output_file_object.write(line)
            output_file_object.flush()

    try:
        start_time = time.perf_counter()
        with subprocess.Popen(
            cmd_parts,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        ) as process:
            stdout_thread = threading.Thread(
                target=pump_stream,
                args=(process.stdout, handle_info_line),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=pump_stream,
                args=(process.stderr, handle_error_line),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            stdout_thread.join()
            stderr_thread.join()
            process.wait()
            elapsed_time = time.perf_counter() - start_time
            if process.returncode != 0:
                if cmd_could_fail:
                    logger.warning(
                        f"Command '{cmd}' failed with exit code {process.returncode}, but it's allowed to fail, so continuing..."
                    )
                else:
                    logger.error(
                        f"Command '{cmd}' failed with exit code {process.returncode}"
                    )
                    raise RuntimeError(f"Command '{cmd}' failed {process.returncode}")
    except Exception as e:
        if cmd_could_fail:
            logger.warning(
                f"Command '{cmd}' failed with exception: {e}, but it's allowed to fail, so continuing..."
            )
        else:
            logger.error(f"Command '{cmd}' failed with exception: {e}")
            raise
    finally:
        if output_file_object is not None:
            output_file_object.close()

    return elapsed_time


def get_ov_commit_from_npu_repo() -> str:
    config_file = os.path.join(get_npu_repo_dir(), "validation/openvino_config.json")
    with open(config_file) as file:
        data = json.load(file)
        commit = data["openvinotoolkit"]
    return commit


def get_repo_commit(repo_path) -> str | None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=repo_path,
    ).stdout.strip()
    return commit if commit != "" else None


def get_repo_diff_exists(repo_path) -> bool:
    # Returns exit code 0 (true) if clean, 1 (false) if changes exist
    result = subprocess.run(
        ["git", "diff", "--quiet"],
        capture_output=True,
        text=True,
        cwd=repo_path,
    )
    return result.returncode != 0


def update_repo(*, repo_dir: str, branch: str, commit: str):
    system_call("git reset --hard", cwd=repo_dir)
    system_call("git clean -f -d -x", cwd=repo_dir)
    system_call("git fetch --all", cwd=repo_dir)
    system_call(
        f"git reset --hard origin/{branch}",
        cwd=repo_dir,
    )
    system_call(
        f"git reset --hard {commit}",
        cwd=repo_dir,
    )
    system_call("git submodule sync --recursive", cwd=repo_dir)
    system_call(
        "git submodule update --init --recursive --force",
        cwd=repo_dir,
    )
    system_call("git clean -f -d -x", cwd=repo_dir)
    system_call("git status", cwd=repo_dir)


def create_ov_cmake_preset(npu_repo_dir: str, ov_repo_dir: str):
    ov_preset_file = os.path.join(ov_repo_dir, "CMakeUserPresets.json")
    npu_preset_file = os.path.join(npu_repo_dir, "CMakeUserPresets.json")
    if not os.path.exists(npu_preset_file):
        with open(npu_preset_file, "w") as f:
            f.write('{\n    "version": 4,\n    "include": ["CMakePresets.json"]\n}\n')
            logger.info(f"User preset file '{npu_preset_file}' created successfully")

    if os.path.exists(ov_preset_file):
        logger.info(f"Preset file '{ov_preset_file}' already exists, skipping creation")
        return
    with open(ov_preset_file, "w") as f:
        f.write(f'{{\n    "version": 4,\n    "include": ["{npu_preset_file}"]\n}}\n')
        logger.info(f"Preset file '{ov_preset_file}' created successfully")


def update_openvino_repo(ov_retrieve_flow: OVRetrieveFlow):
    ov_local_commit = get_repo_commit(ov_retrieve_flow.ov_repo_dir)
    if ov_local_commit is None:
        system_call(
            f"git clone https://github.com/openvinotoolkit/openvino.git {ov_retrieve_flow.ov_repo_dir}"
        )
        ov_local_commit = get_repo_commit(ov_retrieve_flow.ov_repo_dir)
    if ov_retrieve_flow.force_update is True:
        update_repo(
            repo_dir=ov_retrieve_flow.ov_repo_dir,
            branch=ov_retrieve_flow.ov_branch,
            commit=ov_retrieve_flow.ov_commit,
        )
        logger.info(f"openvino updated to the commit: {ov_retrieve_flow.ov_commit}")
        return
    if ov_local_commit != ov_retrieve_flow.ov_commit:
        logger.warning(
            f"Local OpenVINO commit {ov_local_commit} DOES NOT match the specified commit {ov_retrieve_flow.ov_commit}"
        )
        if ov_retrieve_flow.force_update_if_does_not_match is True:
            logger.info("Force updating OpenVINO to match the specified commit")
            update_repo(
                repo_dir=ov_retrieve_flow.ov_repo_dir,
                branch=ov_retrieve_flow.ov_branch,
                commit=ov_retrieve_flow.ov_commit,
            )
            logger.info(f"openvino updated to the commit: {ov_retrieve_flow.ov_commit}")
        else:
            raise RuntimeError(
                f"Local OpenVINO commit {ov_local_commit} DOES NOT match the specified commit {ov_retrieve_flow.ov_commit}"
            )
    else:
        logger.info(
            f"Local OpenVINO commit {ov_local_commit} matches the specified commit {ov_retrieve_flow.ov_commit}, skipping update"
        )


def build_openvino(*, ov_build_flow: OVBuildFlow, build_time_stat: BuildTimeStat):
    os.makedirs(ov_build_flow.ov_build_dir, exist_ok=True)

    # skip build if build directory is not empty
    if (
        len(os.listdir(ov_build_flow.ov_build_dir)) != 0
        and ov_build_flow.force_rebuild is False
    ):
        logger.info(
            f"OpenVINO build directory {ov_build_flow.ov_build_dir} is not empty, skipping build"
        )
        return

    # create project and build
    build_time_stat.create_project_ov_sec = system_call(
        f"cmake -S {ov_build_flow.ov_repo_dir} --preset {ov_build_flow.ov_cmake_preset}",
        cwd=ov_build_flow.ov_build_dir,
    )
    build_time_stat.build_time_ov_sec = system_call(
        f"cmake --build --preset {ov_build_flow.ov_cmake_preset} --parallel {ov_build_flow.parallel_build_num_processes}",
        cwd=ov_build_flow.ov_repo_dir,
    )


def build_npu(*, npu_build_flow: NPUBuildFlow, build_time_stat: BuildTimeStat):
    # fill metadata about NPU repo state
    npu_build_flow.npu_commit = get_repo_commit(get_npu_repo_dir())
    npu_build_flow.npu_git_diff_exists = get_repo_diff_exists(get_npu_repo_dir())

    # check ov commit matched before build
    if npu_build_flow.fail_if_ov_does_not_match is True:
        if get_repo_commit(npu_build_flow.ov_repo_dir) != get_ov_commit_from_npu_repo():
            logger.error(
                "Local OpenVINO commit does not match the specified commit in the repo"
            )
            raise RuntimeError(
                "Local OpenVINO commit does not match the specified commit in the repo"
            )

    # create project
    os.makedirs(npu_build_flow.npu_build_dir, exist_ok=True)
    if npu_build_flow.force_project_create is True:
        shutil.rmtree(npu_build_flow.npu_build_dir, ignore_errors=True)
        logger.info(
            f"NPU build directory {npu_build_flow.npu_build_dir} is cleaned from previous builds"
        )
        os.makedirs(npu_build_flow.npu_build_dir, exist_ok=True)
    if len(os.listdir(npu_build_flow.npu_build_dir)) != 0:
        logger.info(
            f"NPU build directory {npu_build_flow.npu_build_dir} is not empty, skipping project creation"
        )
    else:
        os.chdir(get_npu_repo_dir())
        build_time_stat.create_project_npu_sec = system_call(
            f"cmake -S {get_npu_repo_dir()} --preset {npu_build_flow.npu_cmake_preset}",
            cwd=npu_build_flow.npu_build_dir,
            extra_env={"OPENVINO_HOME": npu_build_flow.ov_repo_dir},
        )
        logger.info(
            f"NPU project is created from preset '{npu_build_flow.npu_cmake_preset}' in the directory {npu_build_flow.npu_build_dir}"
        )

    build_time_stat.build_time_npu_deps_llvm_sec = system_call(
        f"ninja -j{npu_build_flow.parallel_build_num_processes} thirdparty/llvm-project/llvm/all",
        cwd=npu_build_flow.npu_build_dir,
        extra_env={
            "CCACHE_IGNOREOPTIONS": "-ftime-trace"
        },  # -ftime-trace is not supported by llvm, so ignoring it to prevent ccache misses
    )
    build_time_stat.build_time_npu_deps_elf_sec = system_call(
        f"ninja -j{npu_build_flow.parallel_build_num_processes} thirdparty/elf/all",
        cwd=npu_build_flow.npu_build_dir,
    )
    build_time_stat.build_time_npu_deps_flatbuffers_sec = system_call(
        f"ninja -j{npu_build_flow.parallel_build_num_processes} _deps/flatbuffers-build/all",
        cwd=npu_build_flow.npu_build_dir,
    )
    build_time_stat.build_time_npu_deps_vpucostmodel_sec = system_call(
        f"ninja -j{npu_build_flow.parallel_build_num_processes} thirdparty/vpucostmodel/all",
        cwd=npu_build_flow.npu_build_dir,
    )

    # build project
    if npu_build_flow.enable_profiling:
        system_call(
            f"ClangBuildAnalyzer --start {npu_build_flow.npu_build_dir}",
            cwd=npu_build_flow.npu_build_dir,
            cmd_could_fail=True,
        )
    build_time_stat.build_time_npu_sec = system_call(
        f"cmake --build --preset {npu_build_flow.npu_cmake_preset} --parallel {npu_build_flow.parallel_build_num_processes}",
        cwd=get_npu_repo_dir(),
    )
    if npu_build_flow.enable_profiling:
        system_call(
            f"ClangBuildAnalyzer --stop {npu_build_flow.npu_build_dir} {npu_build_flow.build_clang_capture_file}",
            cwd=npu_build_flow.npu_build_dir,
            cmd_could_fail=True,
        )

    # prepare profiling artifacts
    if npu_build_flow.enable_profiling:
        # profiling retention
        profiling_folder = os.path.join(get_npu_repo_dir(), "scripts/build/profiling")
        os.makedirs(profiling_folder, exist_ok=True)
        folder_retention(
            folder=profiling_folder,
            threshold_in_bytes=3 * 1024 * 1024 * 1024,  # 3 GB
        )

        # prepare ninjatracing profiling artifacts
        if not os.path.exists(npu_build_flow.ninja_log_file):
            logger.warning(
                f"Ninja log file {npu_build_flow.ninja_log_file} does not exist, cannot export profiling data"
            )
        else:
            system_call(
                f"ninjatracing -e {npu_build_flow.ninja_log_file}",
                output_file=npu_build_flow.profiling_file,
            )
            logger.info(
                f"Build profiling data is exported to {npu_build_flow.profiling_file}"
            )

        # prepare ClangBuildAnalyzer artifacts
        system_call(
            f"ClangBuildAnalyzer --analyze {npu_build_flow.build_clang_capture_file}",
            output_file=npu_build_flow.build_analysis_file,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            cmd_could_fail=True,
        )
        logger.info(f"Build analysis saved to {npu_build_flow.build_analysis_file}")


def run_full_flow(full_flow: FullFlow):
    # retrieve OV if needed
    if full_flow.ov_retrieve_flow is not None:
        logger.info("Starting OpenVINO retrieve flow")
        os.makedirs(full_flow.ov_retrieve_flow.ov_repo_dir, exist_ok=True)
        update_openvino_repo(full_flow.ov_retrieve_flow)
        create_ov_cmake_preset(
            npu_repo_dir=get_npu_repo_dir(),
            ov_repo_dir=full_flow.ov_retrieve_flow.ov_repo_dir,
        )
        logger.info("Finished OpenVINO retrieve flow")

    # build OV if needed
    if full_flow.ov_build_flow is not None:
        logger.info("Starting OpenVINO build flow")
        build_openvino(
            ov_build_flow=full_flow.ov_build_flow,
            build_time_stat=full_flow.build_time_stat,
        )
        logger.info("Finished OpenVINO build flow")

    # build NPU
    if full_flow.npu_build_flow is not None:
        logger.info("Starting NPU build flow")
        build_npu(
            npu_build_flow=full_flow.npu_build_flow,
            build_time_stat=full_flow.build_time_stat,
        )
        logger.info("Finished NPU build flow")

    # export build metrics if needed
    if (
        full_flow.build_time_stat is not None
        and full_flow.build_time_stat.export_build_metrics
    ):
        metrics_file = os.path.join(
            get_npu_repo_dir(), "scripts/build/build_metrics.json"
        )
        metrics_dump = full_flow.model_dump_json(indent=4)
        logger.info(f"Build info:\n{metrics_dump}")
        with open(metrics_file, "w") as f:
            f.write(metrics_dump)
        logger.info(f"Build metrics are exported to {metrics_file}")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ov_commit",
        type=str,
        help="OpenVINO commit hash, if not specified then commit from the NPU repo is used",
    )
    parser.add_argument(
        "--ov_repo_dir",
        type=str,
        required=True,
        help="Path to the local OpenVINO repository",
    )
    parser.add_argument(
        "--force_ov_update_if_does_not_match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force update OpenVINO only if local OV commit does not match the specified commit in NPU repo",
    )
    parser.add_argument(
        "--force_ov_update",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force update OpenVINO even if local OV commit matches the specified commit",
    )
    parser.add_argument(
        "--build_ov",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set to true if OpenVINO build is required",
    )
    parser.add_argument(
        "--force_ov_rebuild",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force rebuild OpenVINO even if some files already exist in build directory",
    )
    parser.add_argument(
        "--build_type",
        type=str,
        choices=[e.value for e in BuildType],
        required=True,
        help="Build type for OpenVINO and NPU",
    )
    parser.add_argument(
        "--ov_cmake_preset",
        type=str,
        required=False,
        help="OpenVINO Cmake preset",
    )
    parser.add_argument(
        "--npu_cmake_preset",
        type=str,
        required=True,
        help="NPU Cmake preset",
    )
    parser.add_argument(
        "--force_project_create",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force NPU project recreating even if some files already exist in the build directory",
    )
    parser.add_argument(
        "--fail_if_ov_does_not_match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail NPU build if local OV commit does not match the specified commit in the repo",
    )
    parser.add_argument(
        "--export_build_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export build metrics to .json file",
    )
    parser.add_argument(
        "--parallel_build_num_processes",
        type=int,
        default=24,
        help="Number of number of concurrent processes to use while building (e.g. for ninja -j or cmake --build --parallel)",
    )
    parser.add_argument(
        "--profiling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable build time profiling and prepare artifacts for analysis",
    )

    args = parser.parse_args()

    # run flow
    full_flow = FullFlow(
        build_time_stat=BuildTimeStat(export_build_metrics=args.export_build_metrics),
        ov_retrieve_flow=OVRetrieveFlow(
            ov_commit=args.ov_commit
            if args.ov_commit
            else get_ov_commit_from_npu_repo(),
            ov_repo_dir=args.ov_repo_dir,
            force_update=args.force_ov_update,
            force_update_if_does_not_match=args.force_ov_update_if_does_not_match,
        ),
        npu_build_flow=NPUBuildFlow(
            ov_repo_dir=args.ov_repo_dir,
            build_type=args.build_type,
            npu_cmake_preset=args.npu_cmake_preset,
            force_project_create=args.force_project_create,
            fail_if_ov_does_not_match=args.fail_if_ov_does_not_match,
            parallel_build_num_processes=args.parallel_build_num_processes,
            enable_profiling=args.profiling,
        ),
    )
    if args.build_ov:
        if args.ov_cmake_preset is None:
            raise ValueError(
                "Argument --ov_cmake_preset is required when --build_ov is set to true"
            )
        full_flow.ov_build_flow = OVBuildFlow(
            ov_repo_dir=args.ov_repo_dir,
            ov_cmake_preset=args.ov_cmake_preset,
            build_type=args.build_type,
            force_rebuild=args.force_ov_rebuild,
            parallel_build_num_processes=args.parallel_build_num_processes,
        )
    logger.info(
        f"Starting full build flow with parameters:\n{full_flow.model_dump_json(indent=4)}"
    )
    run_full_flow(full_flow)
