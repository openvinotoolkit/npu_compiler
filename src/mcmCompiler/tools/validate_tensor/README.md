# Validator

Utility to test a network.
1) It will run the classification_sample_async in CPU mode with the provided input xml and input image
2) It will then run the classification_sample_async in KMB mode to compile a blob.
3) It creates an binary representation of the input image (input-0.bin) with the required transposing and reshaping for NCE.
4) It deploys the blob and the input-0.bin to InferenceManagerDemo to run on the EVM
4) It validates the results of InferenceManagerDemo against the CPU plugin results

## Prerequisite:

pip3 install opencv-python

Environmental variables
- OPENVINO_HOME path to the openvino repo

- VPUIP_HOME path to the vpuip_2 repo

## Build

Validator can only be built as part of the main build, so needs to be built from ./build dir under mcmCompiler root directory

## Usage

There are 2 modes of use:

1) Normal operation

  a) Auto generate input

This generates an input based on the input layer shape of the supplied model

command: `./validate -m <path to XML> -k <ip address of EVM>`

  b) Supply image input

Provide a path to a .bmp, .png, etc, image file. If not the correct size, it will crop the image to the right size

command: `./validate -m <path to XML> -k <ip address of EVM> -i <path to image>`

  c) Supply flattened file binary input

Provide a path to an already flattened binary input, .dat or .bin. eg, output of Fathom. This must be the correct size as assumes it is already transposed to CH or Z major.

command: `./validate -m <path to XML> -k <ip address of EVM> -i <path to .bin or .dat file>`



2) Validate mode only

command: `./validate --mode validate -b <path_to_blob> -a <path_to_kmb_results> -e <path_to_expected_results>`
  
  - KMB results - must be raw binary of quantized uint8 (eg, output of Inference Manager Demo)
  
  - Expected results - output of reference function or CPU Plugin, must be raw binary of fp32
  
  - Tolerance - percent value (e.g. 1 for 1%) of allowed error, applied per sample - per each sample the validation criteria is:
  abs(actual - expected) < abs(tolerance * 0.01 * expected)

