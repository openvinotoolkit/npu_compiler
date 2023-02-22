//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" %s | vpux-translate --export-VPUIP -o %t
// RUN: flatc --raw-binary --json %vpuip_schema_file% -- %t
// RUN: FileCheck %s --input-file %basename_t.json
// RUN: rm %basename_t.json
//
// This file generates a blob with nv12_to_rgb activation shave
// demonstrate that the runtime cannot handle this.  It's also a lit test to help
// check for regressions in the VPUIP dialect.
//

module @Test {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        IE.DataInfo "Parameter_143" : tensor<1x360x320x1xf16>
    }
    outputsInfo : {
        IE.DataInfo "NV12toRGB_144" : tensor<1x240x320x3xf16>
    }

// Sub-module, which holds SW kernel declarations and optional implementations.
// Used to group those declarations for faster access.
module @VPU.SW {
    // The declaration should match C++ params structure in decomposed form.
    // `memref` will be translated to `MemRefData`, while raw scalars will be translated as is.
    func private @builtin_YuvToRgb(%input : memref<*xf16>, %output : memref<*xf16>, %rgbFormat : i64)
        attributes {
            VPU.kernel_code = "single_shave_convert_color_nv12_to_rgb_single_plane.cpp",
            VPU.kernel_entry = "single_shave_convert_color_nv12_to_rgb_single_plane"
        }
}

func @main(%0: memref<1x360x320x1xf16>, %1: memref<1x240x320x3xf16>) -> memref<1x240x320x3xf16> {

    %in_tile0_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x360x320x1xf16, [@CMX_NN, 0]>
    %out_tile0_cmx = VPURT.DeclareBuffer "CMX_NN" [0] <460800> -> memref<1x240x320x3xf16, [@CMX_NN, 0]>

    %b0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %b1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier

    VPURT.Task updates(%b0 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%0 : memref<1x360x320x1xf16>) outputs(%in_tile0_cmx : memref<1x360x320x1xf16, [@CMX_NN, 0]>) -> memref<1x360x320x1xf16, [@CMX_NN, 0]>
    }

    // Genetic Kernel information for the scheduler.
    VPURT.Task waits(%b0  : !VPURT.Barrier) updates(%b1  : !VPURT.Barrier) {
        VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
                    @VPU.SW::@builtin_YuvToRgb            // The reference to the Kernel function.
                    inputs(%in_tile0_cmx as %arg0: memref<1x360x320x1xf16, [@CMX_NN, 0]>)     // Inputs/outputs buffers for generic operation interface
                    outputs(%out_tile0_cmx as %arg1: memref<1x240x320x3xf16, [@CMX_NN, 0]>)   // and their mapping to inner region.
                    on tile 0                           // The tile index to execute on.

        -> memref<1x240x320x3xf16, [@CMX_NN, 0]> {

                // The arguments mapping, the order must match the kernel parameter structure.
                VPUIP.SW.Kernel.run {attrs = [0]}(%arg0, %arg1)
                    : memref<1x360x320x1xf16, [@CMX_NN, 0]>
                    , memref<1x240x320x3xf16, [@CMX_NN, 0]>
        }
    }

    VPURT.Task waits(%b1 : !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%out_tile0_cmx : memref<1x240x320x3xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x240x320x3xf16>) -> memref<1x240x320x3xf16>
    }
    return %1: memref<1x240x320x3xf16>

}


}

// CHECK:   identifier: "Test"

// CHECK:   net_input: [
// CHECK:     {
// CHECK:       name: "Parameter_143",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           360,
// CHECK:           320,
// CHECK:           1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:          2.0,
// CHECK:          230400.0,
// CHECK:          640.0,
// CHECK:          2.0,
// CHECK:          2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ],

// CHECK:   net_output: [
// CHECK:     {
// CHECK:       name: "NV12toRGB_144",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           240,
// CHECK:           320,
// CHECK:           3
// CHECK:       ],
// CHECK:       strides: [
// CHECK:           2.0,
// CHECK:           460800.0,
// CHECK:           1920.0
// CHECK:           6.0,
// CHECK:           2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ],

// CHECK:   task_count: 5,

// CHECK:   options: [
// CHECK:   ],

// CHECK:   in_tensor_desc: [
// CHECK:     {
// CHECK:       name: "Parameter_143",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           360,
// CHECK:           320,
// CHECK:           1
// CHECK:       ],
// CHECK:       strides: [
// CHECK:          2.0,
// CHECK:          230400.0,
// CHECK:          640.0,
// CHECK:          2.0,
// CHECK:          2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableInput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ],

// CHECK:   out_tensor_desc: [
// CHECK:     {
// CHECK:       name: "NV12toRGB_144",
// CHECK:       dimensions: [
// CHECK:           1,
// CHECK:           240,
// CHECK:           320,
// CHECK:           3
// CHECK:       ],
// CHECK:       strides: [
// CHECK:           2.0,
// CHECK:           460800.0,
// CHECK:           1920.0,
// CHECK:           6.0,
// CHECK:           2.0
// CHECK:       ],
// CHECK:       data: {
// CHECK:         data_index: 0
// CHECK:       },
// CHECK:       locale: "ProgrammableOutput",
// CHECK:       data_dtype: "FP16"
// CHECK:     }
// CHECK:   ]
