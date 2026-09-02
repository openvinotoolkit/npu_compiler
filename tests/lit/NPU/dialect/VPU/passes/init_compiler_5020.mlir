//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% compilation-mode=ReferenceSW" %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU5020

module @test attributes {} {

// CHECK: module @test attributes {config.compilationMode = #config.compilation_mode<ReferenceSW>, config.elf_version = #config.version<{{[^>]*}}>, config.platform = #config.platform<NPU5020>, config.revisionID = #config.revision_id<REVISION_NONE>}

// CHECK-DAG:    {{  }}config.PipelineOptions @Options {
// CHECK-DAG:    {{    }}config.Option @config.BarrierMaxSlotCount : 256
// CHECK-DAG:    {{    }}config.Option @config.AutoPaddingODU : false
// CHECK-DAG:    {{    }}config.Option @config.AutoPaddingIDU : false
// CHECK-DAG:    {{    }}config.Option @config.FragmentationAvoidRatioPipeliningLargeWeights : 3.200000e-01 : f32
// CHECK-DAG:    {{  }}}

// CHECK-DAG:    {{  }}config.ExecutorResource 2 of @DMA_NN
// CHECK-DAG:    {{  }}config.ExecutorResource 1 of @M2I
// CHECK-DAG:    {{  }}config.Resources 1 of @NCE at 2.100000e+03 MHz {
// CHECK-DAG:    {{    }}config.ExecutorResource 2 of @SHAVE_ACT
// CHECK-DAG:    {{    }}config.ExecutorResource 1 of @DPU
// CHECK-DAG:    {{    }}config.MemoryResource 2096128 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
// CHECK-DAG:   {{  }}config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}

}
