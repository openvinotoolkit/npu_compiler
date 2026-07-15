#!/usr/bin/env python
#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
import io
from pathlib import Path


class IrDumpReader(abc.ABC):
    """
    An abstract class that represents an MLIR IR dump "reader". Thus far it
    mainly represents a collection of callbacks that are called when a
    particular line of MLIR IR dump is encountered.

    Users are free to implement their own custom behaviour reader by relying on
    this API. See traverse_ir_dump(), traverse_ir_dump_path() for examples of
    usages of this API.
    """

    IR_DUMP_PASS_LINE_IDENTIFIER = '// -----// IR Dump '

    @abc.abstractmethod
    def set_ir_dump_file(self, file: io.IOBase):
        """
        An API that passes a file-like object to the instance of this class.
        Allows implementors to perform non-trivial file operations directly when
        required.
        """
        pass

    @abc.abstractmethod
    def on_new_pass_line_read(self, line: str, order: str, pass_name: str):
        """
        A callback that is invoked when a special line that signifies "new pass"
        is encountered. Example:
            // -----// IR Dump After OptimizeCopies (optimize-copies) //----- //

        Parameters
        ----------
        line
            The whole line that is being currently read.
        order
            The IR printing order of the special line: Before / After.
        pass_name
            The name of the pass.
        """
        pass

    @abc.abstractmethod
    def on_regular_line_read(self, line: str):
        """
        A callback that is invoked when a "regular" line (i.e. not special) is
        encountered. Example:
            func.func @main(...)

        Parameters
        ----------
        line
            The whole line that is being currently read.
        """
        pass


def extract_character_word(string: str, offset: int):
    """
    Returns a "word" that is located between offset and the next whitespace
    character. Additionally, returns position of the found whitespace to allow
    composability.
    """
    next_space = string.find(' ', offset)
    return string[offset:next_space], next_space


def __parse_ir_dump_line(line: str):
    order, order_end = extract_character_word(line, len(IrDumpReader.IR_DUMP_PASS_LINE_IDENTIFIER))
    order_end += 1  # Note: order_end is set to ' ' and pass name is right after that
    pass_name, _ = extract_character_word(line, order_end)
    return order, pass_name


def traverse_ir_dump(file: io.IOBase, reader: IrDumpReader, is_verbose=False):
    """Reads IR dump file using a specified reader."""
    if is_verbose:
        print('Reading file:', file)
    reader.set_ir_dump_file(file)

    line = file.readline()
    while line:
        # parse current line
        if line.startswith(IrDumpReader.IR_DUMP_PASS_LINE_IDENTIFIER):
            if is_verbose:
                print('Found IR dump pass line:', line.strip())
            order, pass_name = __parse_ir_dump_line(line)
            reader.on_new_pass_line_read(line, order, pass_name)
        else:
            reader.on_regular_line_read(line)

        line = file.readline()

    return 0


def traverse_ir_dump_path(path: str, reader: IrDumpReader, is_verbose=False):
    """
    Reads IR dump file by the given path using a specified reader.

    Note: This is effectively a helper wrapper around traverse_ir_dump() except
    that the file object exists *only* within this function call.
    """
    file_path = Path(path).resolve()
    if not file_path.is_file():
        print('Specified file path {} does not point to a valid file'.format(str(file_path)))
        return 1

    with open(str(file_path), 'r') as ir_dump_file:
        traverse_ir_dump(ir_dump_file, reader, is_verbose)


#
# Tests:
#
if __name__ == '__main__':
    import os
    import unittest
    import tempfile

    class DummyReader(IrDumpReader):
        def __init__(self):
            self.file = None
            self.special_lines = []
            self.regular_lines = []

        def set_ir_dump_file(self, file):
            self.file = file
            return

        def on_new_pass_line_read(self, line, order, pass_name):
            self.special_lines.append((line, order, pass_name))

        def on_regular_line_read(self, line):
            self.regular_lines.append(line)

    class TestReading(unittest.TestCase):
        def test_empty_file(self):
            with tempfile.NamedTemporaryFile('w+') as tmpfile:
                reader = DummyReader()
                traverse_ir_dump(tmpfile, reader)

                self.assertEqual(reader.file, tmpfile)
                self.assertEqual(len(reader.special_lines), 0)
                self.assertEqual(len(reader.regular_lines), 0)

        def test_with_content(self):
            with tempfile.NamedTemporaryFile('w+') as tmpfile:
                content = '''// -----// IR Dump Before InitResources (init-resources) //----- //
Non-IR dump line does not matter here.
// -----// IR Dump After InitResources (init-resources) //----- //
func.func @main() {}
'''
                tmpfile.write(content)
                tmpfile.seek(0, os.SEEK_SET)  # reset the position to read what was written

                reader = DummyReader()
                traverse_ir_dump(tmpfile, reader)

                self.assertEqual(reader.file, tmpfile)
                self.assertEqual(len(reader.special_lines), 2)
                self.assertEqual(
                    reader.special_lines[0], ('// -----// IR Dump Before InitResources (init-resources) //----- //\n', 'Before', 'InitResources'))
                self.assertEqual(
                    reader.special_lines[1], ('// -----// IR Dump After InitResources (init-resources) //----- //\n', 'After', 'InitResources'))
                self.assertEqual(len(reader.regular_lines), 2)
                self.assertEqual(reader.regular_lines[0], 'Non-IR dump line does not matter here.\n')
                self.assertEqual(reader.regular_lines[1], 'func.func @main() {}\n')

        def test_file_opening(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfile_path = os.path.join(tmpdir, 'tmpfile')
                tmpfile = open(tmpfile_path, 'w')
                content = '''
// -----// IR Dump Before InitResources (init-resources) //----- //
Non-IR dump line does not matter here.
'''.strip()
                tmpfile.write(content)
                tmpfile.close()  # note: it is closed but not removed!

                reader = DummyReader()
                traverse_ir_dump_path(str(tmpfile_path), reader)

                self.assertEqual(len(reader.special_lines), 1)
                self.assertEqual(
                    reader.special_lines[0], ('// -----// IR Dump Before InitResources (init-resources) //----- //\n', 'Before', 'InitResources'))
                self.assertEqual(len(reader.regular_lines), 1)
                self.assertEqual(reader.regular_lines[0], 'Non-IR dump line does not matter here.')

    unittest.main()
