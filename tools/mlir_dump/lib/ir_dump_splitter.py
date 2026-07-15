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
    import dump_utils

from pathlib import Path
import os
import collections


class IrDumpSplitter(dump_utils.IrDumpReader):
    """
    IR dump reader that splits the IR dump into individual IR dumps.

    For instance, "IR Dump Before X" and "IR Dump After X" encountered in one
    dump file would be split into two dump files. The class also takes care of
    potential "duplicate" passes (such as Canonicalizer that can run multiple
    times throughout the whole program).
    """

    def __init__(self, input_file: Path, output_dir: Path, is_verbose=False):
        self._input_file_path = input_file
        self._output_dir_path = output_dir
        self._is_verbose = is_verbose
        self._curr_file = open(os.devnull, 'w')  # by default write to /dev/null

        # some passes can be run multiple times e.g. Canonicalizer
        self._pass_duplicate_counters = collections.defaultdict(lambda: 0, {})

    def __del__(self):
        # this is an extra guardrail that the current file would be eventually
        # closed, but Python has no real guarantee when
        if self._is_verbose:
            print('Wrote {} bytes to the current file'.format(self._curr_file.tell()))
        self._curr_file.close()

    def _make_unique_file_path(self, order, pass_name):
        """
        Constructs a unique file path for the individual dump.
        """
        input_path = self._input_file_path
        output_dir_path = self._output_dir_path

        # Note: keep *separate* counters for before_Canonicalizer and
        # after_Canonicalizer
        pass_string = '{}_{}'.format(order, pass_name)
        counter = self._pass_duplicate_counters[pass_string]
        self._pass_duplicate_counters[pass_string] += 1

        # unique file path has to be aware of the duplicates
        file_suffix = '_{}_{:04d}.mlir'.format(pass_string, int(counter))
        return output_dir_path / (input_path.stem + file_suffix)

    def set_ir_dump_file(self, _):  # unused
        pass

    def on_new_pass_line_read(self, line, order, pass_name):
        """
        Sets up a new file to which the new pass information would be written.
        """
        if self._is_verbose:
            print('Wrote {} bytes to the current file'.format(self._curr_file.tell()))
        self._curr_file.close()

        order = order.lower()
        new_file_path = self._make_unique_file_path(order, pass_name)
        if self._is_verbose:
            print('Creating new output file:', new_file_path)

        self._curr_file = open(str(new_file_path), 'w')
        self._curr_file.write(line)

    def on_regular_line_read(self, line):
        """
        Writes the line to the currently open dump file.
        """
        self._curr_file.write(line)


#
# Tests:
#
if __name__ == '__main__':
    import unittest
    import tempfile

    class TestIrDumpSplitter(unittest.TestCase):
        def test_not_suitable_file(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfile_path = os.path.join(tmpdir, 'tmpfile.smth')
                tmpdir = Path(tmpdir)
                with open(tmpfile_path, 'w+') as tmpfile:
                    content = '''
Some preceding text.
// -----// IR Dumps Before InitResources (init-resources) //----- // <- wrong "pattern"
Content #1.
Continues...
xxx// -----// IR Dump After InitResources (init-resources) //----- // <- wrong "pattern"
Content #2.
func.func @main() {
    return
}
'''
                    tmpfile.write(content)
                    tmpfile.seek(0, os.SEEK_SET)
                    splitter = IrDumpSplitter(Path(tmpfile_path), tmpdir)
                    dump_utils.traverse_ir_dump(tmpfile, splitter)

                files = os.listdir(tmpdir)
                self.assertEqual(len(files), 1)
                self.assertEqual(files[0], 'tmpfile.smth')

        def _check_split_file(self, file_path, expected_content):
            file_path = Path(file_path).resolve()
            self.assertTrue(file_path.is_file())
            with open(str(file_path), 'r') as currfile:
                curr_content = currfile.read()
                self.assertEqual(curr_content, expected_content)

        def test_all(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfile_path = os.path.join(tmpdir, 'tmpfile.mlir')
                tmpdir = Path(tmpdir)
                with open(tmpfile_path, 'w+') as tmpfile:
                    content = '''
Some preceding text.
This is going to be completely ignored. In reality, this never happens because
    the IR dump file is well-formed.
// -----// IR Dump Before InitResources (init-resources) //----- //
Content #1.
Continues...
// -----// IR Dump After InitResources (init-resources) //----- //
Content #2.
func.func @main() {
    return
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
Before canonicalizer 1.
// -----// IR Dump After Canonicalizer (canonicalize) //----- //
After canonicalizer 1.

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
Before canonicalizer 2.
// -----// IR Dump After Canonicalizer (canonicalize) //----- //
After canonicalizer 2.
'''
                    tmpfile.write(content)
                    tmpfile.seek(0, os.SEEK_SET)

                    splitter = IrDumpSplitter(Path(tmpfile_path), tmpdir)
                    dump_utils.traverse_ir_dump(tmpfile, splitter)
                    del splitter  # ensure the last file is also closed

                self.assertEqual(len(os.listdir(tmpdir)), 1 + 6)  # tmpfile + 6 individual dumps

                common_file_start = tmpdir / 'tmpfile'
                self._check_split_file('{}_before_InitResources_0000.mlir'.format(common_file_start),
                                       '// -----// IR Dump Before InitResources (init-resources) //----- //\nContent #1.\nContinues...\n')

                self._check_split_file('{}_after_InitResources_0000.mlir'.format(common_file_start),
                                       '// -----// IR Dump After InitResources (init-resources) //----- //\nContent #2.\nfunc.func @main() {\n    return\n}\n\n')

                self._check_split_file('{}_before_Canonicalizer_0000.mlir'.format(common_file_start),
                                       '// -----// IR Dump Before Canonicalizer (canonicalize) //----- //\nBefore canonicalizer 1.\n')
                self._check_split_file('{}_after_Canonicalizer_0000.mlir'.format(common_file_start),
                                       '// -----// IR Dump After Canonicalizer (canonicalize) //----- //\nAfter canonicalizer 1.\n\n')

                self._check_split_file('{}_before_Canonicalizer_0001.mlir'.format(common_file_start),
                                       '// -----// IR Dump Before Canonicalizer (canonicalize) //----- //\nBefore canonicalizer 2.\n')
                self._check_split_file('{}_after_Canonicalizer_0001.mlir'.format(common_file_start),
                                       '// -----// IR Dump After Canonicalizer (canonicalize) //----- //\nAfter canonicalizer 2.\n')

    unittest.main()
