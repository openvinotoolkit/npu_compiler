#!/bin/bash
#
# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if [ $# -gt 0 ]
then
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]
    then
        USAGE="
Script that runs all lit-tests on each platform.

Usage: ./run_all_lit_tests.sh [PATH_TESTS] [PATH_LIT_TOOL]

PATH_TESTS    - optional, the path to the root directory containing all lit-tests to be run; default value '[PATH_SCRIPT]/lit-tests/NPU'
PATH_LIT_TOOL - optional, path to the lit tool lit.py; default value '[PATH_SCRIPT]/lit-tool/lit.py'
"
        echo "$USAGE"
        exit 0
    fi
fi

PATH_SCRIPT="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 || exit ; pwd -P )"
PATH_TESTS="$PATH_SCRIPT/NPU"
PATH_LIT_TOOL="$PATH_SCRIPT/lit-tool/lit.py"

if [ $# -gt 0 ]
then
    PATH_TESTS="$1"
    if [ $# -gt 1 ]
    then
        PATH_LIT_TOOL="$2"
    fi
fi

echo "PATH_TESTS=$PATH_TESTS"
echo "PATH_LIT_TOOL=$PATH_LIT_TOOL"

CMD_NPU3720_TESTS="python3 $PATH_LIT_TOOL --param platform=NPU3720 $PATH_TESTS/NPU"
CMD_NPU4000_TESTS="python3 $PATH_LIT_TOOL --param platform=NPU4000 $PATH_TESTS/NPU"
CMD_NPU5010_TESTS="python3 $PATH_LIT_TOOL --param platform=NPU5010 $PATH_TESTS/NPU"
CMD_NPU5020_TESTS="python3 $PATH_LIT_TOOL --param platform=NPU5020 $PATH_TESTS/NPU"

EXIT_CODE=0

echo ""
echo "Executing tests on NPU3720 platform: $CMD_NPU3720_TESTS"
eval "$CMD_NPU3720_TESTS"; EXIT_CODE=$(($EXIT_CODE + $?))
echo ""
echo "Executing tests on NPU4000 platform: $CMD_NPU4000_TESTS"
eval "$CMD_NPU4000_TESTS"; EXIT_CODE=$(($EXIT_CODE + $?))
echo ""
echo "Executing tests on NPU5010 platform: $CMD_NPU5010_TESTS"
eval "$CMD_NPU5010_TESTS"; EXIT_CODE=$(($EXIT_CODE + $?))
echo ""
echo "Executing tests on NPU5020 platform: $CMD_NPU5020_TESTS"
eval "$CMD_NPU5020_TESTS"; EXIT_CODE=$(($EXIT_CODE + $?))
echo ""
if [ $EXIT_CODE -ne 0 ]
then
    echo "FAILURES identified"
    exit 1
else
    echo "All tests PASSED"
fi
