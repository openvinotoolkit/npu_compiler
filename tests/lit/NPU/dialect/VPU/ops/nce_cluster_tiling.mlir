//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ParsePrintClusterTiling(%arg0: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %weights = const.Declare tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<10> : tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>

    %0 = VPU.NCE.ClusterTiling (
            %arg0 as %arg1: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %weights as %arg2: tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %wt as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
                -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [64, 32, 3, 3],
                strides = [1, 1]
            } -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
      VPU.Yield %1
    }

    return %0 : tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK-DAG:        [[CST0:%.*]] = const.Declare tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-DAG:        [[CST:%.*]] = const.Declare tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:                   %arg0 as %arg1: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                   [[CST]] as %arg2: tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:                   [[CST0]] as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:                   -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    //CHECK:                [[VAL1:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    //CHECK-SAME:                            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:                            ppe = #VPU.PPEStub<>,
    //CHECK-SAME:                            strides = [1, 1]
    //CHECK-SAME:                } -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:                VPU.Yield [[VAL1]]
    //CHECK:            }

    //CHECK:    return [[VAL0]] : tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4
}>

!WeightsDistributed = !VPU.DistributedTensor<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WeightsTableDistributed = !VPU.DistributedTensor<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Input_DDR = tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!Weights_DDR = tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsTable_DDR = tensor<64x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
!Output_DDR = tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>

!InputStub_CMX = tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsStub_CMX = tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTableStub_CMX = tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

func.func @ParsePrintDistributedTensor(%arg0: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<10> : tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>

    %input_cmx = VPU.NCE.ClusterTiling(%arg0 as %arg1: !Input_DDR) -> !InputDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Input_DDR -> !InputStub_CMX
        VPU.Yield %0
    }

    %weights_cmx = VPU.NCE.ClusterTiling(%weights as %arg1: !Weights_DDR) -> !WeightsDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Weights_DDR -> !WeightsStub_CMX
        VPU.Yield %0
    }

    %wt_cmx = VPU.NCE.ClusterTiling(%wt as %arg1: !WeightsTable_DDR) -> !WeightsTableDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !WeightsTable_DDR -> !WeightsTableStub_CMX
        VPU.Yield %0
    }

    %output_cmx = VPU.NCE.ClusterTiling (
              %input_cmx as %arg1: !InputStub_CMX,
              %weights_cmx as %arg2: !WeightsStub_CMX,
              %wt_cmx as %arg3: !WeightsTableStub_CMX)
              -> !OutputDistributed {
        %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                  ppe = #VPU.PPEStub<>,
                  rawFilterShape = [64, 32, 3, 3],
                  strides = [1, 1]
              } -> !OutputStub_CMX
        VPU.Yield %0
    }

    %output = VPU.NCE.ClusterTiling(%output_cmx as %arg1: !OutputStub_CMX) -> !Output_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !OutputStub_CMX -> !Output_DDR
        VPU.Yield %0
    }

    return %output: !Output_DDR

    //CHECK-DAG:        [[WEIGHTS_TABLE:%.*]] = const.Declare tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK-DAG:        [[WEIGHTS:%.*]] = const.Declare tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]

    //CHECK:        [[INPUT_CMX:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>) -> !VPU.DistributedTensor<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}> {
    //CHECK:            [[RES0:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}> -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES0]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS]] as %arg1: tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}>) -> !VPU.DistributedTensor<64x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES1:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}> -> tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES1]]
    //CHECK:        }

    //CHECK:        [[WEIGHTS_TABLE_CMX:%.*]] = VPU.NCE.ClusterTiling ([[WEIGHTS_TABLE]] as %arg1: tensor<64x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>) -> !VPU.DistributedTensor<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    //CHECK:            [[RES2:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<64x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}> -> tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    //CHECK:            VPU.Yield [[RES2]]
    //CHECK:        }

    //CHECK:        [[OUT_CMX:%.*]] = VPU.NCE.ClusterTiling (
    //CHECK-SAME:             [[INPUT_CMX]] as %arg1: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_CMX]] as %arg2: tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    //CHECK-SAME:             [[WEIGHTS_TABLE_CMX]] as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    //CHECK-SAME:             -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    //CHECK:                [[RES4:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    //CHECK-SAME:                            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:                            ppe = #VPU.PPEStub<>,
    //CHECK-SAME:                            strides = [1, 1]
    //CHECK-SAME:             } -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES4]]
    //CHECK:        }

    //CHECK:        [[OUT:%.*]] = VPU.NCE.ClusterTiling ([[OUT_CMX]] as %arg1: tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}> {
    //CHECK:            [[RES5:%.*]] = VPU.Copy(%arg1) {out_mem_space = @DDR} : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    //CHECK:            VPU.Yield [[RES5]]
    //CHECK:        }

    //CHECK:        return [[OUT]] : tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!TensorDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Tensor_DDR = tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!TensorStub_CMX = tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

func.func @EraseSOHCopySequence(%arg0: !TensorDistributed) -> !TensorDistributed {
    %spilled_ddr = VPU.NCE.ClusterTiling(%arg0 as %arg1: !TensorStub_CMX) -> !Tensor_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !TensorStub_CMX -> !Tensor_DDR
        VPU.Yield %0
    }

    %output = VPU.NCE.ClusterTiling(%spilled_ddr as %arg1: !Tensor_DDR) -> !TensorDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Tensor_DDR -> !TensorStub_CMX
        VPU.Yield %0
    }

    return %output: !TensorDistributed
    // CHECK: return %arg0 : !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!TensorDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Tensor_DDR = tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!TensorStub_CMX = tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

func.func @EraseCopySequenceDontAlterCMXConsumers(%arg0: !TensorDistributed) -> (!TensorDistributed, !Tensor_DDR) {
    %spilled_ddr = VPU.NCE.ClusterTiling(%arg0 as %arg1: !TensorStub_CMX) -> !Tensor_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !TensorStub_CMX -> !Tensor_DDR
        VPU.Yield %0
    }

    %spilled_ddr2 = VPU.NCE.ClusterTiling(%arg0 as %arg1: !TensorStub_CMX) -> !Tensor_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !TensorStub_CMX -> !Tensor_DDR
        VPU.Yield %0
    }

    %output = VPU.NCE.ClusterTiling(%spilled_ddr as %arg1: !Tensor_DDR) -> !TensorDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Tensor_DDR -> !TensorStub_CMX
        VPU.Yield %0
    }

    return %output, %spilled_ddr2: !TensorDistributed, !Tensor_DDR

    // CHECK:    [[DDR_2:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}> {
    // CHECK:    VPU.Copy(%arg1) {out_mem_space = @DDR} : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:    return %arg0, [[DDR_2]] : !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>,
    // CHECK-SAME:  tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!TensorDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Tensor_DDR = tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!TensorStub_CMX = tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

func.func @EraseCopySequenceMultipleDDRConsumersSameMode(%arg0: !TensorDistributed) -> (!TensorDistributed, !TensorDistributed) {
    %spilled_ddr = VPU.NCE.ClusterTiling(%arg0 as %arg1: !TensorStub_CMX) -> !Tensor_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !TensorStub_CMX -> !Tensor_DDR
        VPU.Yield %0
    }

    %output = VPU.NCE.ClusterTiling(%spilled_ddr as %arg1: !Tensor_DDR) -> !TensorDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Tensor_DDR -> !TensorStub_CMX
        VPU.Yield %0
    }

    %output2 = VPU.NCE.ClusterTiling(%spilled_ddr as %arg1: !Tensor_DDR) -> !TensorDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Tensor_DDR -> !TensorStub_CMX
        VPU.Yield %0
    }

    return %output, %output2: !TensorDistributed, !TensorDistributed

    // CHECK:   return %arg0, %arg0 : !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>,
    // CHECK-SAME:  !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!TensorSegmented = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!TensorDuplicated = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!Tensor_DDR = tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!TensorStub_CMX = tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

func.func @EraseCopySequenceMultipleDDRConsumersDiffMode(%arg0: !TensorSegmented) -> (!TensorSegmented, !TensorDuplicated) {
    %spilled_ddr = VPU.NCE.ClusterTiling(%arg0 as %arg1: !TensorStub_CMX) -> !Tensor_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !TensorStub_CMX -> !Tensor_DDR
        VPU.Yield %0
    }

    %output = VPU.NCE.ClusterTiling(%spilled_ddr as %arg1: !Tensor_DDR) -> !TensorSegmented {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Tensor_DDR -> !TensorStub_CMX
        VPU.Yield %0
    }

    %output2 = VPU.NCE.ClusterTiling(%spilled_ddr as %arg1: !Tensor_DDR) -> !TensorDuplicated {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Tensor_DDR -> !TensorStub_CMX
        VPU.Yield %0
    }

    return %output, %output2: !TensorSegmented, !TensorDuplicated

    // CHECK:   [[DDR:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}> {
    // CHECK:   VPU.Copy(%arg1) {out_mem_space = @DDR} : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:   [[CMX:%.*]] = VPU.NCE.ClusterTiling (%0 as %arg1: tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>) -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}> {
    // CHECK:   VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}> -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:   return %arg0, [[CMX]] : !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>, !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!CopyOutTensorDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!CopyInTensorDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!Tensor_DDR = tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!TensorStub_CMX = tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

func.func @EraseSOKCopySequence(%arg0: !CopyOutTensorDistributed) -> !CopyInTensorDistributed {
    %spilled_ddr = VPU.NCE.ClusterTiling(%arg0 as %arg1: !TensorStub_CMX) -> !Tensor_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !TensorStub_CMX -> !Tensor_DDR
        VPU.Yield %0
    }

    %output = VPU.NCE.ClusterTiling(%spilled_ddr as %arg1: !Tensor_DDR) -> !CopyInTensorDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Tensor_DDR -> !TensorStub_CMX
        VPU.Yield %0
    }

    return %output: !CopyInTensorDistributed

    // CHECK:    [[CAST:%.*]] = VPU.DistributedCast(%arg0 : !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>) ->
    // CHECK-SAME:  !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:    return [[CAST]] : !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!CopyOutTensorDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!CopyInTensorDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4
}>

!Tensor_DDR = tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!TensorStub_CMX = tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

func.func @DontEraseSOKtoSOHCopySequence(%arg0: !CopyOutTensorDistributed) -> !CopyInTensorDistributed {
    %spilled_ddr = VPU.NCE.ClusterTiling(%arg0 as %arg1: !TensorStub_CMX) -> !Tensor_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !TensorStub_CMX -> !Tensor_DDR
        VPU.Yield %0
    }

    %output = VPU.NCE.ClusterTiling(%spilled_ddr as %arg1: !Tensor_DDR) -> !CopyInTensorDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Tensor_DDR -> !TensorStub_CMX
        VPU.Yield %0
    }

    return %output: !CopyInTensorDistributed

    // CHECK:  [[TENSOR_DDR:%.*]] = VPU.NCE.ClusterTiling (%arg0 as %arg1: tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}> {
    // CHECK:  VPU.Copy
    // CHECK:  [[TENSOR_CMX:%.*]] = VPU.NCE.ClusterTiling ([[TENSOR_DDR]] as %arg1: tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>) -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}> {
    // CHECK:  VPU.Copy
    // CHECK:  return [[TENSOR_CMX]] : !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    num_clusters = 4,
    strides = [1, 1]
}>

!WeightsFirstDistributed = !VPU.DistributedTensor<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WeightsTableFirstDistributed = !VPU.DistributedTensor<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!IntermediateDistributed = !VPU.DistributedTensor<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!WeightsSecondDistributed = !VPU.DistributedTensor<
    16x64x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WeightsTableSecondDistributed = !VPU.DistributedTensor<
    16x1x1x16xui8, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x16x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Input_DDR = tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsFirst_DDR = tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsTableFirst_DDR = tensor<64x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
!Intermediate_DDR = tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsSecond_DDR = tensor<16x64x1x1xf16, {mem_space = @DDR, order = #NHWC}>
!WeightsTableSecond_DDR = tensor<16x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}>
!Output_DDR = tensor<1x16x16x16xf16, {mem_space = @DDR, order = #NHWC}>

!InputStub_CMX = tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsFirstStub_CMX = tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTableFirstStub_CMX = tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
!IntermediateStub_CMX = tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsSecondStub_CMX = tensor<16x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
!WeightsTableSecondStub_CMX = tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

func.func @CanonicalizeTwoConvs(%arg0: !Input_DDR) -> !Output_DDR {
    %weights_first = const.Declare tensor<64x32x3x3xf16, {mem_space = @DDR, order = #NHWC}> = dense<1.000000e+00>
                   : tensor<64x32x3x3xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    %wt_first = const.Declare tensor<64x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}> = dense<10>
                   : tensor<64x1x1x4xsi32, {mem_space = @DDR}>
    %weights_second = const.Declare tensor<16x64x1x1xf16, {mem_space = @DDR, order = #NHWC}> = dense<1.000000e+00>
                   : tensor<16x64x1x1xf16, {mem_space = @DDR}>, [#const.Reorder<#NHWC>]
    %wt_second = const.Declare tensor<16x1x1x4xsi32, {mem_space = @DDR, order = #NCHW}> = dense<10>
                   : tensor<16x1x1x4xsi32, {mem_space = @DDR}>

    // First Convolution operation

    %input_cmx = VPU.NCE.ClusterTiling(%arg0 as %arg1: !Input_DDR) -> !InputDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Input_DDR -> !InputStub_CMX
        VPU.Yield %0
    }

    %weights_first_cmx = VPU.NCE.ClusterTiling(%weights_first as %arg1: !WeightsFirst_DDR) -> !WeightsFirstDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !WeightsFirst_DDR -> !WeightsFirstStub_CMX
        VPU.Yield %0
    }

    %wt_first_cmx = VPU.NCE.ClusterTiling(%wt_first as %arg1: !WeightsTableFirst_DDR) -> !WeightsTableFirstDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !WeightsTableFirst_DDR -> !WeightsTableFirstStub_CMX
        VPU.Yield %0
    }

    %intermediate_output_cmx = VPU.NCE.ClusterTiling (
              %input_cmx as %arg1: !InputStub_CMX,
              %weights_first_cmx as %arg2: !WeightsFirstStub_CMX,
              %wt_first_cmx as %arg3: !WeightsTableFirstStub_CMX) -> !IntermediateDistributed {
        %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                  ppe = #VPU.PPEStub<>,
                  rawFilterShape = [64, 32, 3, 3],
                  strides = [1, 1]
              } -> !IntermediateStub_CMX
        VPU.Yield %0
    }

    // Output is stored into DDR and loaded back again. It's expected to be optimized out by the canonicalizer

    %intermediate_ddr = VPU.NCE.ClusterTiling(%intermediate_output_cmx as %arg1: !IntermediateStub_CMX) -> !Intermediate_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !IntermediateStub_CMX -> !Intermediate_DDR
        VPU.Yield %0
    }

    %intermediate_input_cmx = VPU.NCE.ClusterTiling(%intermediate_ddr as %arg1: !Intermediate_DDR) -> !IntermediateDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !Intermediate_DDR -> !IntermediateStub_CMX
        VPU.Yield %0
    }

    // Second Convolution operation

    %weights_second_cmx = VPU.NCE.ClusterTiling(%weights_second as %arg1: !WeightsSecond_DDR) -> !WeightsSecondDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !WeightsSecond_DDR -> !WeightsSecondStub_CMX
        VPU.Yield %0
    }

    %wt_second_cmx = VPU.NCE.ClusterTiling(%wt_second as %arg1: !WeightsTableSecond_DDR) -> !WeightsTableSecondDistributed {
        %0 = VPU.Copy(%arg1) { out_mem_space = @CMX_NN } : !WeightsTableSecond_DDR -> !WeightsTableSecondStub_CMX
        VPU.Yield %0
    }

    %output_cmx = VPU.NCE.ClusterTiling (
              %intermediate_input_cmx as %arg1: !IntermediateStub_CMX,
              %weights_second_cmx as %arg2: !WeightsSecondStub_CMX,
              %wt_second_cmx as %arg3: !WeightsTableSecondStub_CMX) -> !OutputDistributed {
        %0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                  ppe = #VPU.PPEStub<>,
                  rawFilterShape = [16, 64, 1, 1],
                  strides = [1, 1]
              } -> !OutputStub_CMX
        VPU.Yield %0
    }

    %output = VPU.NCE.ClusterTiling(%output_cmx as %arg1: !OutputStub_CMX) -> !Output_DDR {
        %0 = VPU.Copy(%arg1) { out_mem_space = @DDR } : !OutputStub_CMX -> !Output_DDR
        VPU.Yield %0
    }

    return %output: !Output_DDR

    // CHECK:    [[INTERMEDIATE:%.*]] = VPU.NCE.ClusterTiling ([[INPUT:%.*]] as %arg1: tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                    [[WEIGHTS_1:%.*]] as %arg2: tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                    [[WT_1:%.*]] as %arg3: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:             -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    // CHECK:        [[TEMP_1:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    // CHECK-SAME:       -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK-NOT: [[SPILLED:%.*]] = VPU.NCE.ClusterTiling ([[TEMP_OUTPUT:%.*]] as %arg1: tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK-NOT:     [[WHATEVER_DDR:%.*]] = VPU.Copy(%arg1) {out_mem_space = @DDR} : tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK-NOT: [[FILLED:%.*]] = VPU.NCE.ClusterTiling ([[TEMP_INPUT:%.*]] as %arg1: tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK-NOT:     [[WHATEVER_CMX:%.*]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}> -> tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:    [[OUTPUT:%.*]] = VPU.NCE.ClusterTiling ([[INTERMEDIATE]] as %arg1: tensor<1x64x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                    [[WEIGHTS_1:%.*]] as %arg2: tensor<16x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
    // CHECK-SAME:                    [[WT_1:%.*]] as %arg3: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK-SAME:            ) -> !VPU.DistributedTensor<1x16x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}> {
    // CHECK:        [[TEMP_2:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    // CHECK-SAME:       -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!SparseTypeCMX = !VPU.SparseTensor<data=tensor<1x64x52x52xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x64x52x52xi1, {mem_space = @CMX_NN, order = #NHWC}>>
!SparseTypeDDR = !VPU.SparseTensor<data=tensor<1x64x52x52xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x52x52xi1, {order = #NHWC}>>

!InputDistributedType = !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x52x52xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
                                               sparsity_map=!VPU.DistributedTensor<1x64x52x52xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

!OutputDistributedType = !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x52x52xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
                                                sparsity_map=!VPU.DistributedTensor<1x64x52x52xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>

func.func @OptimizeCMXDDRCMXCopiesSparseType(%input: !InputDistributedType) -> !OutputDistributedType {
    %0 = VPU.NCE.ClusterTiling (%input as %arg1: !SparseTypeCMX) -> !SparseTypeDDR {
        %3 = VPU.Copy(%arg1) : !SparseTypeCMX -> !SparseTypeDDR
        VPU.Yield %3
    }
    %1 = VPU.NCE.ClusterTiling (%0 as %arg1: !SparseTypeDDR) -> !OutputDistributedType {
        %3 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : !SparseTypeDDR -> !SparseTypeCMX
        VPU.Yield %3
    }
    return %1: !OutputDistributedType


    // CHECK:       [[DISTRIBUTED_CAST:%.*]] = VPU.DistributedCast
    // CHECK-SAME:    (%arg0 : !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x52x52xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                               sparsity_map=!VPU.DistributedTensor<1x64x52x52xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>)
    // CHECK-SAME:    -> !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x52x52xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:                         sparsity_map=!VPU.DistributedTensor<1x64x52x52xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
    // CHECK:       return [[DISTRIBUTED_CAST]] :
    // CHECK-SAME:      !VPU.SparseTensor<data=!VPU.DistributedTensor<1x64x52x52xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
    // CHECK-SAME:      sparsity_map=!VPU.DistributedTensor<1x64x52x52xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>>
}
