# edit xml tool

## Summary

The tool cuts the network to the specified layer and creates new IR.

The used weights section after the cut is copied to an equivalently named .bin file (unless this file already exists and has the expected size).

## Prerequsites

The tool works with IR v10 and IR v11.

It works with python library lxml. The library needs installing the following way:
pip3 install lxml

## Usage


The tool has the following command line arguments:

* `-m <path to the IR file with name and extension>` - path to IR file with name and extension
* `-l <name of the layer>` - name of the layer which is supposed to be the output
* `-tw` - trims the .bin to the smallest contiguous section still in use after the network is cut (optional)
* `--overwrite` - overwrites the existing .bin file even if it has the expected size (optional)
* `-o <output path for IR cut>` - output path where the result of the tool execution is placed (optional)

> Note: in general, consult with the tool's help to better understand which
> options are available.

## Example

python3 edit_xml.py -m `name-of-original-file`.xml -l `name-of-the-layer` -tw --overwrite

### The output

The output file is expected to be the IR based on the original and ended by the specified layer.
`name-of-original-file`-cut-`name-of-the output-layer`.xml
