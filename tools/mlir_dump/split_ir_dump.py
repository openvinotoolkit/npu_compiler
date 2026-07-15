#!/usr/bin/env python
#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from contextlib import contextmanager


@contextmanager
def import_from(rel_path):  # special infrastructure to support local importing
    """Add module import relative path to sys.path"""
    import sys
    import os
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(cur_dir, rel_path))
    yield
    sys.path.pop(0)


with import_from('.'):
    import lib.dump_utils as dump_utils
    from lib.ir_dump_splitter import IrDumpSplitter

import os
import sys
import argparse
from pathlib import Path
import textwrap


def parse_arguments(args):
    parser = argparse.ArgumentParser(prog=__file__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''
        Splits IR dump into individual files.
        The general file naming scheme is: <INPUTFILENAME>_<DUMPKIND>_<PASSNAME>_<PASSCOUNTER>.mlir, where:
        * INPUTFILENAME - is the input file name
                        (allows quick immediate rejection by input IR dump)
        * DUMPKIND      - "before" or "after"
                        (if input IR dump has both, 2 files are generated)
        * PASSNAME      - pass name written in upper camel case style (e.g. "InitResources")
        * PASSCOUNTER   - an index that specifies invocation number of the pass
                        (some passes can be called multiple times e.g. Canonicalizer)
    '''))
    parser.add_argument('input_file', help='Input file with IR inside separated by dump lines')
    parser.add_argument('output_dir', help='Directory to put the split files into')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Make the tool more verbose (Note: you can also use VERBOSE=1 environment variable)')
    return parser.parse_args(args)


def main():
    args = parse_arguments(sys.argv[1:])
    # Verbosity could be set either as a CLI arg or as an environment variable
    # (VERBOSE=1 is a "standardized" thing)
    is_verbose = args.verbose or (os.getenv('VERBOSE', '0') == '1')

    input_path = Path(args.input_file).resolve()
    if not input_path.is_file():
        print('Input path:', str(input_path), 'does not point to a valid location')
        return 1
    if is_verbose:
        print('Input file:', str(input_path))

    output_path = Path(args.output_dir).resolve()
    if is_verbose:
        print('Output directory:', str(output_path))
    os.makedirs(str(output_path), exist_ok=True)

    splitter = IrDumpSplitter(input_path, output_path, is_verbose)
    # IR dump splitter being verbose is enough
    dump_utils.traverse_ir_dump_path(input_path, splitter, is_verbose=False)

    return 0


if __name__ == '__main__':
    sys.exit(main())
