//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --platform=%platform% --split-input-file --default-hw-mode="allow-custom-values=true" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#transposeWeights = affine_map<(d0, d1, d2) -> (d1, d0, d2)>

// CHECK-LABEL: @GroupQuantCorrection
module @GroupQuantCorrection {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "activation" : tensor<1x1x512xf16>
        DataInfo "weights" : tensor<8x96x64xui2>
        DataInfo "scale" : tensor<8x96x1xf16>
        DataInfo "zeroPoint" : tensor<8x96x1xui2>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x96xf16>
    }

    // CHECK:       func.func nested @builtin_GroupQuantCorrection(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xui2, @CMX_NN>, memref<*xf16, @CMX_NN>)
    // CHECK:       func.func @main(
    func.func @main(%activation: tensor<1x1x512xf16>, %weights: tensor<8x96x64xui2>,
                    %scale: tensor<8x96x1xf16>, %zeroPoint: tensor<8x96x1xui2>)
            -> tensor<1x1x96xf16> {
        %weightsF16 = IE.Convert(%weights) {dstElemType = f16} : tensor<8x96x64xui2> -> tensor<8x96x64xf16>
        %zeroPointF16 = IE.Convert(%zeroPoint) {dstElemType = f16} : tensor<8x96x1xui2> -> tensor<8x96x1xf16>
        %weightsSub = IE.Subtract(%weightsF16, %zeroPointF16) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<8x96x64xf16>, tensor<8x96x1xf16> -> tensor<8x96x64xf16>
        %weightsDequant = IE.Multiply(%weightsSub, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<8x96x64xf16>, tensor<8x96x1xf16> -> tensor<8x96x64xf16>
        %weightsTranspose = IE.Transpose(%weightsDequant) {order_value = #transposeWeights}
            : tensor<8x96x64xf16> -> tensor<96x8x64xf16>
        %weightsReshape = IE.AffineReshape(%weightsTranspose) {
            dim_mapping = [[0], [1], [1]], shape_value = [96, 512]
        } : tensor<96x8x64xf16> -> tensor<96x512xf16>
        %matmul = IE.MatMul(%activation, %weightsReshape) {transpose_b}
            : tensor<1x1x512xf16>, tensor<96x512xf16> -> tensor<1x1x96xf16>
        return %matmul : tensor<1x1x96xf16>

        // The decomposed quantized MatMul is lowered to NCE convolution tasks.
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>

        // Correction buffers retain ui2 zero points and are tiled across NPU tiles.
        // CHECK:       [[GQC_TILE_0:%.+]] = VPUIP.SW.Kernel
        // CHECK-SAME:      @VPU.SW::@builtin_GroupQuantCorrection
        // CHECK-SAME:      inputs({{.*}}: memref<{{[12]}}x96x1x1xf16, [@CMX_NN, 0]>, {{.*}}: memref<{{[12]}}x1x1x64xf16, [@CMX_NN, 0]>, {{.*}}: memref<{{[12]}}x96x1x1xf16, [@CMX_NN, 0]>, {{.*}}: memref<{{[12]}}x96x1x1xui2, [@CMX_NN, 0]>)
        // CHECK-SAME:      outputs({{.*}}: memref<{{[12]}}x96x1x1xf16, [@CMX_NN, 0]>){{.*}}on tile 0
        // CHECK:           VPUIP.SW.Kernel.run
        // CHECK:       [[GQC_TILE_1:%.+]] = VPUIP.SW.Kernel
        // CHECK-SAME:      @VPU.SW::@builtin_GroupQuantCorrection
        // CHECK-SAME:      outputs({{.*}}: memref<{{[12]}}x96x1x1xf16, [@CMX_NN, 1]>){{.*}}on tile 1
        // CHECK:           VPUIP.SW.Kernel.run
    }
}
