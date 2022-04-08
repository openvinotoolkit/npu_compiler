#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
import numpy as np
import os
import sys
import subprocess

def main():
    nroot = os.getenv("NZOO_ROOT", "../../../../../../migNetworkZoo")
    resnet = os.path.join(nroot, "tools/kmb_test_generation/resnet50.py")
    onnx_path = os.path.join(nroot, "public/Resnet/ONNX/unknown/resnet50.onnx")

    cmd = "python3 " + resnet + " -c 3 -y 224 -x 224 --quantize --onnx_model "+ onnx_path
    cmd = cmd.split()
    code = subprocess.run(cmd)

    outfile = os.path.join(nroot, "internal/unit_tests/CompilerTestsKmb/layers/resnet50_weights/quantized_model.tflite")

    print("\nFILE:"+outfile, end="")

if __name__=='__main__':
    main()
