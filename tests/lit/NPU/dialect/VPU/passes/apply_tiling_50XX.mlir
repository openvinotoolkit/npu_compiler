//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --apply-tiling --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEConvWithUnpaddedOutputChannels
module @NCEConvWithUnpaddedOutputChannels {
    config.Resources 3 of @NCE at 6.000000e+02 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    // CHECK: ([[INPUT:%.+]]: tensor<1x16x512x512xf16, {order = #NHWC}>, [[WEIGHTS:%.+]]: tensor<3x16x1x1xf16, {order = #NHWC}>, [[WEIGHTS_TABLE:%.+]]: tensor<16x1x1x4xsi32>)
    func.func @main(%input: tensor<1x16x512x512xf16, {order = #NHWC}>, %weights: tensor<3x16x1x1xf16, {order = #NHWC}>, %weights_table: tensor<16x1x1x4xsi32>) -> tensor<1x3x512x512xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%input, %weights, %weights_table) {
            input_padding = [0, 13, 0, 0],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [3, 16, 1, 1],
            strides = [1, 1],
            tilingStrategy = [1, 1, 3, 1]
        } : tensor<1x16x512x512xf16, {order = #NHWC}>, tensor<3x16x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x3x512x512xf16, {order = #NHWC}>
        return %0 : tensor<1x3x512x512xf16, {order = #NHWC}>
    }

    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 171, 512]
    // CHECK:       [[CONV0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      -> tensor<1x3x171x512xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 171, 0] [1, 16, 171, 512]
    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      -> tensor<1x3x171x512xf16, {order = #NHWC}>
    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 342, 0] [1, 16, 170, 512]
    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      -> tensor<1x3x170x512xf16, {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV0]], [[CONV1]], [[CONV2]])
    // CHECK-SAME:      -> tensor<1x3x512x512xf16, {order = #NHWC}>
    // CHECK:       return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEDepthConvWithUnpaddedOutputChannels
module @NCEDepthConvWithUnpaddedOutputChannels {
    config.Resources 3 of @NCE at 6.000000e+02 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    // CHECK: ([[INPUT:%.+]]: tensor<1x16x512x512xf16, {order = #NHWC}>, [[WEIGHTS:%.+]]: tensor<3x1x4x8xf16, {order = #NHWC}>, [[WEIGHTS_TABLE:%.+]]: tensor<16x1x1x4xsi32>)
    func.func @main(%input: tensor<1x16x512x512xf16, {order = #NHWC}>, %weights: tensor<3x1x4x8xf16, {order = #NHWC}>, %weights_table: tensor<16x1x1x4xsi32>) -> tensor<1x3x509x505xf16, {order = #NHWC}> {
        %0 = VPU.NCE.DepthConvolution(%input, %weights, %weights_table) {
            input_padding = [0, 13, 0, 0],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [3, 1, 4, 8],
            strides = [1, 1],
            tilingStrategy = [1, 1, 3, 1]
        } -> tensor<1x3x509x505xf16, {order = #NHWC}>
        return %0 : tensor<1x3x509x505xf16, {order = #NHWC}>
    }

    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 173, 512]
    // CHECK:       [[DEPTH_CONV0:%.+]] = VPU.NCE.DepthConvolution([[SLICE0]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      -> tensor<1x3x170x505xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 170, 0] [1, 16, 173, 512]
    // CHECK:       [[DEPTH_CONV1:%.+]] = VPU.NCE.DepthConvolution([[SLICE1]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      -> tensor<1x3x170x505xf16, {order = #NHWC}>
    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 340, 0] [1, 16, 172, 512]
    // CHECK:       [[DEPTH_CONV2:%.+]] = VPU.NCE.DepthConvolution([[SLICE2]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      -> tensor<1x3x169x505xf16, {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[DEPTH_CONV0]], [[DEPTH_CONV1]], [[DEPTH_CONV2]])
    // CHECK-SAME:      -> tensor<1x3x509x505xf16, {order = #NHWC}>
    // CHECK:       return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEMaxPoolWithUnpaddedOutputChannels
module @NCEMaxPoolWithUnpaddedOutputChannels {
    config.Resources 3 of @NCE at 6.000000e+02 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    // CHECK: ([[INPUT:%.+]]: tensor<1x16x512x512xf16, {order = #NHWC}>)
    func.func @main(%arg0: tensor<1x16x512x512xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf16, {order = #NHWC}> {
        %0 = VPU.NCE.MaxPool(%arg0) {
            input_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            tilingStrategy = [1, 1, 3, 1]
        } -> tensor<1x3x512x512xf16, {order = #NHWC}>
        return %0 : tensor<1x3x512x512xf16, {order = #NHWC}>
    }

    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 171, 512]
    // CHECK:       [[MAXPOOL0:%.+]] = VPU.NCE.MaxPool([[SLICE0]])
    // CHECK-SAME:      -> tensor<1x3x171x512xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 171, 0] [1, 16, 171, 512]
    // CHECK:       [[MAXPOOL1:%.+]] = VPU.NCE.MaxPool([[SLICE1]])
    // CHECK-SAME:      -> tensor<1x3x171x512xf16, {order = #NHWC}>
    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 342, 0] [1, 16, 170, 512]
    // CHECK:       [[MAXPOOL2:%.+]] = VPU.NCE.MaxPool([[SLICE2]])
    // CHECK-SAME:      -> tensor<1x3x170x512xf16, {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[MAXPOOL0]], [[MAXPOOL1]], [[MAXPOOL2]])
    // CHECK-SAME:      -> tensor<1x3x512x512xf16, {order = #NHWC}>
    // CHECK:       return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEAvgPoolWithUnpaddedOutputChannels
module @NCEAvgPoolWithUnpaddedOutputChannels {
    config.Resources 3 of @NCE at 6.000000e+02 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    // CHECK: ([[INPUT:%.+]]: tensor<1x16x512x512xf16, {order = #NHWC}>)
    func.func @main(%arg0: tensor<1x16x512x512xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf16, {order = #NHWC}> {
        %0 = VPU.NCE.AveragePool(%arg0) {
            input_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            tilingStrategy = [1, 1, 3, 1]
        } -> tensor<1x3x512x512xf16, {order = #NHWC}>
        return %0 : tensor<1x3x512x512xf16, {order = #NHWC}>
    }

    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 171, 512]
    // CHECK:       [[AVGPOOL0:%.+]] = VPU.NCE.AveragePool([[SLICE0]])
    // CHECK-SAME:      -> tensor<1x3x171x512xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 171, 0] [1, 16, 171, 512]
    // CHECK:       [[AVGPOOL1:%.+]] = VPU.NCE.AveragePool([[SLICE1]])
    // CHECK-SAME:      -> tensor<1x3x171x512xf16, {order = #NHWC}>
    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 342, 0] [1, 16, 170, 512]
    // CHECK:       [[AVGPOOL2:%.+]] = VPU.NCE.AveragePool([[SLICE2]])
    // CHECK-SAME:      -> tensor<1x3x170x512xf16, {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[AVGPOOL0]], [[AVGPOOL1]], [[AVGPOOL2]])
    // CHECK-SAME:      -> tensor<1x3x512x512xf16, {order = #NHWC}>
    // CHECK:       return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEEltwiseWithUnpaddedOutputChannels
module @NCEEltwiseWithUnpaddedOutputChannels {
    config.Resources 3 of @NCE at 6.000000e+02 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    // CHECK: ([[INPUT0:%.+]]: tensor<1x16x512x512xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x16x512x512xf16, {order = #NHWC}>)
    func.func @main(%arg0: tensor<1x16x512x512xf16, {order = #NHWC}>, %arg1: tensor<1x16x512x512xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
            input_padding = [0, 13, 0, 0],
            op_type = #VPU.eltwise_type<SUBTRACT>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            ppe = #VPU.PPEStub<>,
            tilingStrategy = [1, 1, 2, 1]
        } -> tensor<1x3x512x512xf16, {order = #NHWC}>
        return %0 : tensor<1x3x512x512xf16, {order = #NHWC}>
    }

    // CHECK:       [[SLICE0_INPUT0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 16, 256, 512]
    // CHECK:       [[SLICE0_INPUT1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 16, 256, 512]
    // CHECK:       [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[SLICE0_INPUT0]], [[SLICE0_INPUT1]])
    // CHECK-SAME:      -> tensor<1x3x256x512xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1_INPUT0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 256, 0] [1, 16, 256, 512]
    // CHECK:       [[SLICE1_INPUT1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 256, 0] [1, 16, 256, 512]
    // CHECK:       [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[SLICE1_INPUT0]], [[SLICE1_INPUT1]])
    // CHECK-SAME:      -> tensor<1x3x256x512xf16, {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[ELTWISE0]], [[ELTWISE1]])
    // CHECK-SAME:      -> tensor<1x3x512x512xf16, {order = #NHWC}>
    // CHECK:       return [[CONCAT]]
}

// -----

// CHECK-LABEL: @FlashSDPA8kSeqLen
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<1x1x8192x32xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<1x1x128x32xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<1x1x64x128xf16>,
// CHECK-SAME: [[ATTENTION_MASK:%[^, ]+]]: tensor<1x1x8192x128xf16>,
// CHECK-SAME: [[SCALE:%[^, ]+]]: tensor<1x1x1x1xf16>
func.func @FlashSDPA8kSeqLen(%arg0: tensor<1x1x8192x32xf16>, %arg1: tensor<1x1x128x32xf16>, %arg2: tensor<1x1x64x128xf16>, %arg3: tensor<1x1x8192x128xf16>, %arg4: tensor<1x1x1x1xf16>)
                                  -> tensor<1x1x8192x64xf16> {
    %cst = const.Declare tensor<1x1x64x4xsi32> = dense<0> : tensor<1x1x64x4xsi32>
    %cst_0 = const.Declare tensor<1x1x128x4xsi32> = dense<0> : tensor<1x1x128x4xsi32>

    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    %cst_2 = const.Declare tensor<1x1x8192x128xf16> = dense<0.000000e+00> : tensor<1x1x8192x128xf16>
    %cst_3 = const.Declare tensor<1x1x8192x1xf16> = dense<0.000000e+00> : tensor<1x1x8192x1xf16>
    %cst_4 = const.Declare tensor<1x1x8192x1xf16> = dense<0xFC00> : tensor<1x1x8192x1xf16>
    %cst_5 = const.Declare tensor<1x1x8192x64xf16> = dense<0.000000e+00> : tensor<1x1x8192x64xf16>

    %result_running_output, %result_running_max, %result_running_sum, %result_query =
        VPU.FlashSDPA(%arg0, %arg1, %arg2, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3, %arg4) {
                is_head = true, is_tail = true, kv_num_blocks = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>, source_seq_len_pad_size = 0 : i64, tilingStrategy = [1, 1, 2, 1]
            } : tensor<1x1x8192x32xf16>, tensor<1x1x128x32xf16>, tensor<1x1x64x128xf16>,
                tensor<1x1x8192x128xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x128x4xsi32>,
                tensor<1x1x64x4xsi32>, tensor<1x1x8192x64xf16>, tensor<1x1x8192x1xf16>,
                tensor<1x1x8192x1xf16>, tensor<1x1x8192x128xf16>, tensor<1x1x1x1xf16>
            -> tensor<1x1x8192x64xf16>, tensor<1x1x8192x1xf16>, tensor<1x1x8192x1xf16>, tensor<1x1x8192x32xf16>

    return %result_running_output : tensor<1x1x8192x64xf16>

    // CHECK-DAG:       [[IN_SUM0:%.+]] = const.Declare tensor<1x1x4096x1xf16> = dense<0.000000e+00> : tensor<1x1x8192x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 4096, 1]>]
    // CHECK-DAG:       [[IN_MAX0:%.+]] = const.Declare tensor<1x1x4096x1xf16> = dense<0xFC00> : tensor<1x1x8192x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 4096, 1]>]
    // CHECK-DAG:       [[IN_OUT0:%.+]] = const.Declare tensor<1x1x4096x64xf16> = dense<0.000000e+00> : tensor<1x1x8192x64xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 4096, 64]>]
    // CHECK-DAG:       [[IN_AUX0:%.+]] = const.Declare tensor<1x1x4096x128xf16> = dense<0.000000e+00> : tensor<1x1x8192x128xf16>, [#const.SubView<[0, 0, 0, 0], [1, 1, 4096, 128]>]

    // CHECK-DAG:       [[IN_SUM1:%.+]] = const.Declare tensor<1x1x4096x1xf16> = dense<0.000000e+00> : tensor<1x1x8192x1xf16>, [#const.SubView<[0, 0, 4096, 0], [1, 1, 4096, 1]>]
    // CHECK-DAG:       [[IN_MAX1:%.+]] = const.Declare tensor<1x1x4096x1xf16> = dense<0xFC00> : tensor<1x1x8192x1xf16>, [#const.SubView<[0, 0, 4096, 0], [1, 1, 4096, 1]>]
    // CHECK-DAG:       [[IN_OUT1:%.+]] = const.Declare tensor<1x1x4096x64xf16> = dense<0.000000e+00> : tensor<1x1x8192x64xf16>, [#const.SubView<[0, 0, 4096, 0], [1, 1, 4096, 64]>]
    // CHECK-DAG:       [[IN_AUX1:%.+]] = const.Declare tensor<1x1x4096x128xf16> = dense<0.000000e+00> : tensor<1x1x8192x128xf16>, [#const.SubView<[0, 0, 4096, 0], [1, 1, 4096, 128]>]

    // CHECK-DAG:       [[DPU_DESCRIPTORS_BUF:%.+]] = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    // CHECK-DAG:       [[WEIGHTS_TABLE_0:%.+]] = const.Declare tensor<1x1x128x4xsi32> = dense
    // CHECK-DAG:       [[WEIGHTS_TABLE_1:%.+]] = const.Declare tensor<1x1x64x4xsi32> = dense

    // CHECK:           [[QUERY0:%.+]] = VPU.Slice [[QUERY]] [0, 0, 0, 0] [1, 1, 4096, 32] : tensor<1x1x8192x32xf16> to tensor<1x1x4096x32xf16>
    // CHECK:           [[ATTENTION_MASK0:%.+]] = VPU.Slice [[ATTENTION_MASK]] [0, 0, 0, 0] [1, 1, 4096, 128] : tensor<1x1x8192x128xf16> to tensor<1x1x4096x128xf16>
    // CHECK:           [[RES_OUT0:%[^, ]+]], [[RES_MAX0:%[^, ]+]], [[RES_SUM0:%[^, ]+]], [[RES_QUERY0:%[^, ]+]] =
    // CHECK-SAME:              VPU.FlashSDPA([[QUERY0]], [[KEY]], [[VALUE]], [[IN_AUX0]],
    // CHECK-SAME:                            [[DPU_DESCRIPTORS_BUF]], [[WEIGHTS_TABLE_0]], [[WEIGHTS_TABLE_1]],
    // CHECK-SAME:                            [[IN_OUT0]], [[IN_MAX0]], [[IN_SUM0]], [[ATTENTION_MASK0]], [[SCALE]]) {
    // CHECK-SAME:                      is_head = true, is_tail = true, kv_num_blocks = 1 : i64
    // CHECK-SAME:                      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:                      source_seq_len_pad_size = 0 : i64
    // CHECK-SAME:                  -> tensor<1x1x4096x64xf16>, tensor<1x1x4096x1xf16>, tensor<1x1x4096x1xf16>, tensor<1x1x4096x32xf16>

    // CHECK:           [[QUERY1:%.+]] = VPU.Slice [[QUERY]] [0, 0, 4096, 0] [1, 1, 4096, 32] : tensor<1x1x8192x32xf16> to tensor<1x1x4096x32xf16>
    // CHECK:           [[ATTENTION_MASK1:%.+]] = VPU.Slice [[ATTENTION_MASK]] [0, 0, 4096, 0] [1, 1, 4096, 128] : tensor<1x1x8192x128xf16> to tensor<1x1x4096x128xf16>
    // CHECK:           [[RES_OUT1:%[^, ]+]], [[RES_MAX1:%[^, ]+]], [[RES_SUM1:%[^, ]+]], [[RES_QUERY1:%[^, ]+]] =
    // CHECK-SAME:              VPU.FlashSDPA([[QUERY1]], [[KEY]], [[VALUE]], [[IN_AUX1]],
    // CHECK-SAME:                            [[DPU_DESCRIPTORS_BUF]], [[WEIGHTS_TABLE_0]], [[WEIGHTS_TABLE_1]],
    // CHECK-SAME:                            [[IN_OUT1]], [[IN_MAX1]], [[IN_SUM1]], [[ATTENTION_MASK1]], [[SCALE]])
    // CHECK-SAME:                      is_head = true, is_tail = true, kv_num_blocks = 1 : i64
    // CHECK-SAME:                      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:                      source_seq_len_pad_size = 0 : i64
    // CHECK-SAME:                  -> tensor<1x1x4096x64xf16>, tensor<1x1x4096x1xf16>, tensor<1x1x4096x1xf16>, tensor<1x1x4096x32xf16>

    // CHECK:           [[CONCAT:%.+]] = VPU.Concat([[RES_OUT0]], [[RES_OUT1]])

    // CHECK:           return [[CONCAT]]
}

// -----

// CHECK-LABEL: @FlashSDPATargetSeqLenEqual1
// CHECK-SAME: [[QUERY:%[^, ]+]]: tensor<1x24x1x64xf16>,
// CHECK-SAME: [[KEY:%[^, ]+]]: tensor<1x24x1024x64xf16>,
// CHECK-SAME: [[VALUE:%[^, ]+]]: tensor<1x24x64x1024xf16>,
// CHECK-SAME: [[SCALE:%[^, ]+]]: tensor<1x1x1x1xf16>
func.func @FlashSDPATargetSeqLenEqual1(%arg0: tensor<1x24x1x64xf16>, %arg1: tensor<1x24x1024x64xf16>, %arg2: tensor<1x24x64x1024xf16>, %arg3: tensor<1x1x1x1xf16>) -> tensor<1x24x1x64xf16> {
    // Weights tables have actual data. Replaced with 0-es to reduce LIT test
    %cst = const.Declare tensor<1x1x64x4xsi32> = dense<0> : tensor<1x1x64x4xsi32>
    %cst_0 = const.Declare tensor<1x1x1024x4xsi32> = dense<0> : tensor<1x1x1024x4xsi32>

    %cst_1 = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    %cst_2 = const.Declare tensor<1x24x1x1024xf16> = dense<0.000000e+00> : tensor<1x24x1x1024xf16>
    %cst_3 = const.Declare tensor<1x24x1x1xf16> = dense<0.000000e+00> : tensor<1x24x1x1xf16>
    %cst_4 = const.Declare tensor<1x24x1x1xf16> = dense<0xFC00> : tensor<1x24x1x1xf16>
    %cst_5 = const.Declare tensor<1x24x1x64xf16> = dense<0.000000e+00> : tensor<1x24x1x64xf16>

    %result_running_output, %result_running_max, %result_running_sum, %result_query =
        VPU.FlashSDPA(%arg0, %arg1, %arg2, %cst_2, %cst_1, %cst_0, %cst, %cst_5, %cst_4, %cst_3, %arg3) {
                is_head = true,
                is_tail = true,
                kv_num_blocks = 1 : i64,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
                operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1>,
                source_seq_len_pad_size = 0 : i64,
                tilingStrategy = [1, 2, 1, 1]
            } : tensor<1x24x1x64xf16>, tensor<1x24x1024x64xf16>, tensor<1x24x64x1024xf16>,
                tensor<1x24x1x1024xf16>, tensor<1x1x2x256xsi32>, tensor<1x1x1024x4xsi32>,
                tensor<1x1x64x4xsi32>, tensor<1x24x1x64xf16>, tensor<1x24x1x1xf16>,
                tensor<1x24x1x1xf16>, tensor<1x1x1x1xf16>
            -> tensor<1x24x1x64xf16>, tensor<1x24x1x1xf16>, tensor<1x24x1x1xf16>, tensor<1x24x1x64xf16>

    return %result_running_output : tensor<1x24x1x64xf16>

    // CHECK-DAG:   [[IN_SUM0:%.+]] = const.Declare tensor<1x12x1x1xf16> = dense<0.000000e+00> : tensor<1x24x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 12, 1, 1]>]
    // CHECK-DAG:   [[IN_MAX0:%.+]] = const.Declare tensor<1x12x1x1xf16> = dense<0xFC00> : tensor<1x24x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 12, 1, 1]>]
    // CHECK-DAG:   [[IN_OUT0:%.+]] = const.Declare tensor<1x12x1x64xf16> = dense<0.000000e+00> : tensor<1x24x1x64xf16>, [#const.SubView<[0, 0, 0, 0], [1, 12, 1, 64]>]
    // CHECK-DAG:   [[IN_AUX0:%.+]] = const.Declare tensor<1x12x1x1024xf16> = dense<0.000000e+00> : tensor<1x24x1x1024xf16>, [#const.SubView<[0, 0, 0, 0], [1, 12, 1, 1024]>]

    // CHECK-DAG:   [[IN_SUM1:%.+]] = const.Declare tensor<1x12x1x1xf16> = dense<0.000000e+00> : tensor<1x24x1x1xf16>, [#const.SubView<[0, 12, 0, 0], [1, 12, 1, 1]>]
    // CHECK-DAG:   [[IN_MAX1:%.+]] = const.Declare tensor<1x12x1x1xf16> = dense<0xFC00> : tensor<1x24x1x1xf16>, [#const.SubView<[0, 12, 0, 0], [1, 12, 1, 1]>]
    // CHECK-DAG:   [[IN_OUT1:%.+]] = const.Declare tensor<1x12x1x64xf16> = dense<0.000000e+00> : tensor<1x24x1x64xf16>, [#const.SubView<[0, 12, 0, 0], [1, 12, 1, 64]>]
    // CHECK-DAG:   [[IN_AUX1:%.+]] = const.Declare tensor<1x12x1x1024xf16> = dense<0.000000e+00> : tensor<1x24x1x1024xf16>, [#const.SubView<[0, 12, 0, 0], [1, 12, 1, 1024]>]

    // CHECK-DAG:   [[DPU_DESCRIPTORS_BUF:%.+]] = const.Declare tensor<1x1x2x256xsi32> = dense<0> : tensor<1x1x2x256xsi32>
    // CHECK-DAG:   [[WEIGHTS_TABLE_0:%.+]] = const.Declare tensor<1x1x1024x4xsi32> = dense
    // CHECK-DAG:   [[WEIGHTS_TABLE_1:%.+]] = const.Declare tensor<1x1x64x4xsi32> = dense

    // CHECK-DAG:   [[QUERY0:%.+]] = VPU.Slice [[QUERY]] [0, 0, 0, 0] [1, 12, 1, 64] : tensor<1x24x1x64xf16> to tensor<1x12x1x64xf16>
    // CHECK-DAG:   [[KEY0:%.+]] = VPU.Slice [[KEY]] [0, 0, 0, 0] [1, 12, 1024, 64] : tensor<1x24x1024x64xf16> to tensor<1x12x1024x64xf16>
    // CHECK-DAG:   [[VALUE0:%.+]] = VPU.Slice [[VALUE]] [0, 0, 0, 0] [1, 12, 64, 1024] : tensor<1x24x64x1024xf16> to tensor<1x12x64x1024xf16>

    // CHECK:       [[RES_OUT0:%[^, ]+]], [[RES_MAX0:%[^, ]+]], [[RES_SUM0:%[^, ]+]], [[RES_QUERY0:%[^, ]+]] =
    // CHECK-SAME:          VPU.FlashSDPA([[QUERY0]], [[KEY0]], [[VALUE0]], [[IN_AUX0]],
    // CHECK-SAME:                        [[DPU_DESCRIPTORS_BUF]], [[WEIGHTS_TABLE_0]], [[WEIGHTS_TABLE_1]],
    // CHECK-SAME:                        [[IN_OUT0]], [[IN_MAX0]], [[IN_SUM0]], [[SCALE]]) {
    // CHECK-SAME:              is_head = true, is_tail = true, kv_num_blocks = 1 : i64,
    // CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:              source_seq_len_pad_size = 0 : i64

    // CHECK-DAG:   [[QUERY1:%.+]] = VPU.Slice [[QUERY]] [0, 12, 0, 0] [1, 12, 1, 64] : tensor<1x24x1x64xf16> to tensor<1x12x1x64xf16>
    // CHECK-DAG:   [[KEY1:%.+]] = VPU.Slice [[KEY]] [0, 12, 0, 0] [1, 12, 1024, 64] : tensor<1x24x1024x64xf16> to tensor<1x12x1024x64xf16>
    // CHECK-DAG:   [[VALUE1:%.+]] = VPU.Slice [[VALUE]] [0, 12, 0, 0] [1, 12, 64, 1024] : tensor<1x24x64x1024xf16> to tensor<1x12x64x1024xf16>

    // CHECK:       [[RES_OUT1:%[^, ]+]], [[RES_MAX1:%[^, ]+]], [[RES_SUM1:%[^, ]+]], [[RES_QUERY1:%[^, ]+]] =
    // CHECK-SAME:          VPU.FlashSDPA([[QUERY1]], [[KEY1]], [[VALUE1]], [[IN_AUX1]],
    // CHECK-SAME:                        [[DPU_DESCRIPTORS_BUF]], [[WEIGHTS_TABLE_0]], [[WEIGHTS_TABLE_1]],
    // CHECK-SAME:                        [[IN_OUT1]], [[IN_MAX1]], [[IN_SUM1]], [[SCALE]]) {
    // CHECK-SAME:              is_head = true, is_tail = true, kv_num_blocks = 1 : i64,
    // CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:              source_seq_len_pad_size = 0 : i64

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[RES_OUT0]], [[RES_OUT1]])

    // CHECK:       return [[CONCAT]] : tensor<1x24x1x64xf16>

}
