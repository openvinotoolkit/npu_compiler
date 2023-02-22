//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --wrap-into-async-regions %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @LinearGraph
func @LinearGraph(%arg0: memref<1x1x1x100xf16>, %arg1: memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16> {
    %0 = memref.alloc() :  memref<1x1x1x100xf16>
    %1 = VPUIP.ReLUUPA inputs(%arg0 : memref<1x1x1x100xf16>) outputs(%0 : memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16>
    %2 = VPUIP.Copy inputs(%1 : memref<1x1x1x100xf16>) outputs(%arg1 : memref<1x1x1x100xf16>) -> memref<1x1x1x100xf16>
    return %2 : memref<1x1x1x100xf16>

    // CHECK:       [[VAR0:%.+]] = memref.alloc()

    // CHECK:       [[TOKEN1:%.+]], [[FUTURE1:%.+]] = async.execute -> !async.value<memref<1x1x1x100xf16>>
    // CHECK-SAME:          VPUIP.executor = @SHAVE_UPA
    // CHECK:           [[INNER_VAR1:%.+]] = VPUIP.ReLUUPA inputs(%arg0 : memref<1x1x1x100xf16>) outputs([[VAR0]] : memref<1x1x1x100xf16>)
    // CHECK:           async.yield [[INNER_VAR1]]

    // CHECK:       [[VAR1:%.+]] = async.await [[FUTURE1]]

    // CHECK:       [[TOKEN2:%.+]], [[FUTURE2:%.+]] = async.execute -> !async.value<memref<1x1x1x100xf16>>
    // CHECK-SAME:          VPUIP.executor = @DMA_NN
    // CHECK:           [[INNER_VAR2:%.+]] = VPUIP.Copy inputs([[VAR1]] : memref<1x1x1x100xf16>) outputs(%arg1 : memref<1x1x1x100xf16>)
    // CHECK:           async.yield [[INNER_VAR2]]

    // CHECK:       [[VAR2:%.+]] = async.await [[FUTURE2]]

    // CHECK:       return [[VAR2]]
}

// -----

// CHECK-LABEL: @ConcatView
func @ConcatView(%arg0: memref<50x1x1xf16>, %arg1: memref<100x1x1xf16>) -> memref<100x1x1xf16> {
    %0 = VPUIP.SubView %arg1 [0, 0 ,0] [50, 1, 1] : memref<100x1x1xf16> to memref<50x1x1xf16>
    %1 = VPUIP.ReLUUPA inputs(%arg0 : memref<50x1x1xf16>) outputs(%0 : memref<50x1x1xf16>) -> memref<50x1x1xf16>

    %2 = VPUIP.SubView %arg1 [50, 0, 0] [50, 1, 1] : memref<100x1x1xf16> to memref<50x1x1xf16>
    %3 = VPUIP.Copy inputs(%arg0 : memref<50x1x1xf16>) outputs(%2 : memref<50x1x1xf16>) -> memref<50x1x1xf16>

    %4 = VPUIP.ConcatView inputs(%1, %3 : memref<50x1x1xf16>, memref<50x1x1xf16>) outputs(%arg1 : memref<100x1x1xf16>) -> memref<100x1x1xf16>
    return %4 : memref<100x1x1xf16>

    // CHECK:       [[VAR0:%.+]] = VPUIP.SubView %arg1 [0, 0, 0] [50, 1, 1]
    // CHECK:       [[TOKEN1:%.+]], [[FUTURE1:%.+]] = async.execute -> !async.value<memref<50x1x1xf16>>
    // CHECK-SAME:          VPUIP.executor = @SHAVE_UPA
    // CHECK:           [[INNER_VAR1:%.+]] = VPUIP.ReLUUPA inputs(%arg0 : memref<50x1x1xf16>) outputs([[VAR0]] : memref<50x1x1xf16>)
    // CHECK:           async.yield [[INNER_VAR1]]
    // CHECK:       [[VAR1:%.+]] = async.await [[FUTURE1]]

    // CHECK:       [[VAR2:%.+]] = VPUIP.SubView %arg1 [50, 0, 0] [50, 1, 1]
    // CHECK:       [[TOKEN3:%.+]], [[FUTURE3:%.+]] = async.execute -> !async.value<memref<50x1x1xf16>>
    // CHECK-SAME:          VPUIP.executor = @DMA_NN
    // CHECK:           [[INNER_VAR3:%.+]] = VPUIP.Copy inputs(%arg0 : memref<50x1x1xf16>) outputs([[VAR2]] : memref<50x1x1xf16>)
    // CHECK:           async.yield [[INNER_VAR3]]
    // CHECK:       [[VAR3:%.+]] = async.await [[FUTURE3]]

    // CHECK:       [[VAR4:%.+]] = VPUIP.ConcatView inputs([[VAR1]], [[VAR3]] : memref<50x1x1xf16>, memref<50x1x1xf16>) outputs(%arg1 : memref<100x1x1xf16>)

    // CHECK:       return [[VAR4]] : memref<100x1x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEClusterTiling
func @NCEClusterTiling(%arg0: memref<1x32x16x16xf16, #NHWC, @DDR>) -> memref<1x32x16x16xf16, #NHWC, @CMX_NN> {
    %0 = memref.alloc() : memref<1x32x16x16xf16, #NHWC, @CMX_NN>
    %1 = VPUIP.NCEClusterTiling inputs(%arg0 as %arg2: memref<1x32x16x16xf16, #NHWC, @DDR>) outputs(%0 as %arg3: memref<1x32x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x32x16x16xf16, #NHWC, @CMX_NN> {
      %2 = VPUIP.Copy inputs(%arg2 : memref<1x32x16x16xf16, #NHWC, @DDR>) outputs(%arg3 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x32x16x16xf16, #NHWC, @CMX_NN>
    }
    return %1 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       [[BUF0:%.+]] = memref.alloc()

    // CHECK:       [[T1:%.+]], [[R1:%.+]] = async.execute -> !async.value
    // CHECK-SAME:          VPUIP.executor = @DMA_NN
    // CHECK:           [[R2:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:          inputs(%arg0 as [[ARG0:%.+]]: memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:          outputs([[BUF0]] as [[ARG1:%.+]]: memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:               VPUIP.Copy
    // CHECK-SAME:              inputs([[ARG0]] : memref<1x32x16x16xf16, #NHWC, @DDR>)
    // CHECK-SAME:              outputs([[ARG1]] : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK:           async.yield [[R2]]

    // CHECK:       [[R2:%.+]] = async.await [[R1]]

    // CHECK:       return [[R2]]
}
