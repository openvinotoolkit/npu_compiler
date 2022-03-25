#!/usr/bin/python3
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import argparse
import numpy as np
import cv2
import ast
import re

# Changes the order of channels of an input tensor
#
# command:
#   python3 post_process.py --file <path to results file> --shape n,c,h,w --zmajor

def convert_output(file_path, shape, datatype=np.uint8, zmajor=True, output_file="./output_transposed.dat"):

    new_shape = [int(shape[0]),
                 int(shape[1]),
                 int(shape[2]),
                 int(shape[3])]

    arr = np.fromfile(file_path, dtype=datatype)
    data = np.reshape(arr, (new_shape[2], new_shape[3], new_shape[1]))
    if (zmajor):
        data = data.transpose([2, 0, 1])
    else:
        data = data.transpose([0, 1, 2])

    fp = open(output_file, "wb")
    fp.write ((data.flatten()).astype(datatype).data)
    fp.close


def main():
    parser = argparse.ArgumentParser(description='Convert result file to format suitable for KMB.')
    parser.add_argument('--file', type=str, required=True, help='path to output-0.bin')
    parser.add_argument('--dtype', type=str)
    parser.add_argument('--shape', type=str)
    parser.add_argument('--zmajor', action='store_true')
    parser.add_argument('--output', type=str, default="./output_transposed.dat")

    args = parser.parse_args()

    image_shape = args.shape.split(',')

    datatype = np.uint8
    if args.dtype is not None:
        if args.dtype == "FP16":
            datatype = np.float16
        elif args.dtype == "FP32":
            datatype = np.float32
        elif args.dtype == "I32":
            datatype = np.int32
        elif args.dtype == "U32":
            datatype = np.uint32
        else:
            datatype = np.uint8

    convert_output(args.file, image_shape, datatype, zmajor=True, output_file=args.output)
    
if __name__ == "__main__":
    main()
