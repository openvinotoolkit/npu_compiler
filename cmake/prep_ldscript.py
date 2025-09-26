#
# Copyright (C) 2025 Intel Corporation.
# SPDX-License-Identifier: Apache-2.0
#
import os
import sys


def usage():
    print(f"Usage: python {sys.argv[0]} FILENAME [FILENAME...] OUTPUT_FILENAME")
    sys.exit(1)


def prep_ldscript_header(filename_ld, filename_h, varname):
    with open(filename_ld, "r") as infile:
        text = infile.read()
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    vardef_begin = f'static const char* {varname} = R"ldscript(\n'
    vardef_end = '\n)ldscript";\n'

    result = "#pragma once\n\n" + vardef_begin + "\n".join(lines) + vardef_end

    with open(filename_h, "w") as outfile:
        outfile.write(result)
    print(f" Generated header file as {filename_h}")


def main():
    if len(sys.argv) < 3:
        usage()
    input_ld = sys.argv[1]
    output_h = sys.argv[2]
    varname = "SHAVE_LD_SCRIPT"

    prep_ldscript_header(input_ld, output_h, varname)


if __name__ == "__main__":
    main()
