//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% compilation-mode=ReferenceSW workload-management-enable=true" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU40XX

// CHECK: module @test attributes {VPU.arch = #VPU.arch_kind<NPU40XX>, VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>, VPU.revisionID = #VPU.revision_id<REVISION_NONE>}
module @test {

// CHECK-DAG:    {{  }}IE.PipelineOptions @Options {
// CHECK-DAG:    {{    }}IE.Option @VPU.BarrierMaxVariantSum : 256
// CHECK-DAG:    {{    }}IE.Option @VPU.BarrierMaxVariantCount : 256
// CHECK-DAG:    {{    }}IE.Option @VPU.AutoPaddingODU : false
// CHECK-DAG:    {{    }}IE.Option @VPU.AutoPaddingIDU : false
// CHECK-DAG:    {{    }}IE.Option @VPU.MaxKernelSize : 11
// CHECK-DAG:    {{  }}}

// CHECK-DAG:    {{  }}IE.ExecutorResource 2 of @DMA_NN
// CHECK-DAG:    {{  }}IE.ExecutorResource 1 of @M2I
// CHECK-DAG:    {{  }}IE.TileResource 6 of @NCE at 1.850000e+03 MHz {
// CHECK-DAG:    {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
// CHECK-DAG:    {{    }}IE.ExecutorResource 1 of @DPU
// CHECK-DAG:    {{    }}IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
// CHECK-DAG:    {{    }}IE.MemoryResource 1474560 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
// CHECK-DAG:    {{  }}IE.MemoryResource 4194304000 bytes of @DDR {VPU.bandwidth = 64 : i64, VPU.derateFactor = 6.000000e-01 : f64}

}
