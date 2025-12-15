//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --correct-NCE-workloads %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ConvLargeSparseOutput
// CHECK-SAME:    ([[INPUT_DDR:%.+]]: tensor<1x64x40x40xf16, {order = #NHWC}>)
func.func @ConvLargeSparseOutput(%input_ddr: tensor<1x64x40x40xf16, {order = #NHWC}>) -> !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {order = #NHWC}>> {
    %cst_weights = const.Declare tensor<384x64x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<384x64x4x4xf16>, [#const.Reorder<#NHWC>]
    %input = VPU.Copy(%input_ddr) {out_mem_space = @CMX_NN} : tensor<1x64x40x40xf16, {order = #NHWC}> -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %weights = VPU.Copy(%cst_weights) {out_mem_space = @CMX_NN} : tensor<384x64x4x4xf16, {order = #NHWC}> -> tensor<384x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %conv_out = VPU.NCE.Convolution(%input, %weights) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [384, 64, 4, 4],
            strides = [1, 1]
        } : tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<384x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 384, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_4x16>
            }

    %output = VPU.Copy(%conv_out) : !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>>
        ->  !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {order = #NHWC}>>

    return %output : !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {order = #NHWC}>>

    // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<384x64x4x4xf16, {order = #NHWC}>
    // CHECK:       [[INPUT:%.+]] = VPU.Copy([[INPUT_DDR]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x64x40x40xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[WEIGHTS:%.+]] = VPU.Copy([[CST_WEIGHTS]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<384x64x4x4xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[CONV_OUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {mem_space = @CMX_NN, order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {mem_space = @CMX_NN, order = #NHWC}>> {
    // CHECK:               VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 384, 40, 80] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_4x16>
    // CHECK:           }

    // CHECK:       [[OUTPUT:%.+]] = VPU.Copy([[CONV_OUT]])
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {order = #NHWC}>>

    // CHECK:       return [[OUTPUT]] : !VPU.SparseTensor<data=tensor<1x384x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x384x37x37xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType0 = !quant.uniform<u8:f16, 1.000000e+00>

!Input_CMX = !VPU.DistributedTensor<
    1x3x224x224xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments
}>

!Output_CMX = !VPU.DistributedTensor<
    1x4x224x224x!qElemType0, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [7, 7],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2: i64,
    uniform_distributed_segments
}>

!Input_DDR = tensor<1x3x224x224xf16, {order = #NCHW}>
!InputStub_CMX = tensor<1x3x224x224xf16, {mem_space = @CMX_NN, order = #NCHW}>
!OutputStub_CMX = tensor<1x4x224x224x!qElemType0, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK-LABEL: @NCEPermuteClustered
func.func @NCEPermuteClustered(%arg0: !Input_DDR) -> !Output_CMX {
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : !Input_DDR -> !Input_CMX

    %output = VPU.NCE.Permute(%0) {
            dstElemType = !qElemType0, dstOrder = #NHWC,
            expandedChannels = 4 : i64, minimumHardwareExecutionCost = 4294967195 : i64,
            ppe = #VPU.PPEStub<>
        } -> !Output_CMX {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 4, 112, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 112, 0] outSizes [1, 4, 112, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    return %output : !Output_CMX

    // CHECK:       VPU.NCE.Permute
    // CHECK-SAME:      dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {
    // CHECK-SAME:           mode = "OVERLAPPED",
    // CHECK-SAME:           num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:           kernel = [7, 7],
    // CHECK-SAME:           pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:           strides = [1, 1],
    // CHECK-SAME:           num_clusters = 2 : i64,
    // CHECK-SAME:           uniform_distributed_segments}>

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 0] outSizes [1, 3, 112, 224]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 0

    // CHECK:       VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 112, 0] outSizes [1, 3, 112, 224]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      cluster_id = 1
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 0.92406077665441178>

!InputType = !VPU.DistributedTensor<
    1x3x224x224xf16, #NCHW, @CMX_NN, {
        mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
        compute_shapes = [[1, 3, 75, 224], [1, 3, 75, 224], [1, 3, 74, 224]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 75, 0], [0, 0, 150, 0]],
        memory_shapes = [[1, 3, 75, 224], [1, 3, 75, 224], [1, 3, 74, 224]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 75, 0], [0, 0, 150, 0]]
    }>

!OutputType = !VPU.DistributedTensor<
    1x4x224x224x!qElemType, #NHWC, @CMX_NN, {
        mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
        compute_shapes = [[1, 4, 75, 224], [1, 4, 75, 224], [1, 4, 74, 224]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 75, 0], [0, 0, 150, 0]],
        memory_shapes = [[1, 4, 77, 224], [1, 4, 76, 224], [1, 4, 74, 224]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 75, 0], [0, 0, 150, 0]]
    }>

module @NCEPermuteODUAutopadRemoveChannelPadding {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

    func.func @main(%input: !InputType) -> !OutputType {
        %output = VPU.NCE.Permute(%input) {
                dstElemType = !qElemType,
                dstOrder = #NHWC,
                expandedChannels = 4 : i64,
                ppe = #VPU.PPEFp<
                    mode = <NOOP>,
                    clamp_low = 0.000000e+00 : f64,
                    clamp_high = 2.550000e+02 : f64,
                    scale = 1.0821799012187578 : f64,
                    prelu_alpha = [1.000000e+00],
                    bias = 0.000000e+00 : f64,
                    adder = 0.000000e+00 : f64
                >} -> !OutputType {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 4, 75, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 75, 0] outSizes [1, 4, 75, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_8x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 150, 0] outSizes [1, 4, 74, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_8x16> attributes {cluster_id = 2 : i64}
        }

        return %output : !OutputType

        // CHECK:       VPU.NCE.Permute

        // CHECK-NEXT:  VPU.DPU.Workload
        // CHECK-SAME:      outOffsets [0, 0, 0, 0] outSizes [1, 3, 75, 224]
        // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        // CHECK-SAME:      cluster_id = 0

        // CHECK-NEXT:  VPU.DPU.Workload
        // CHECK-SAME:      outOffsets [0, 0, 75, 0] outSizes [1, 3, 75, 224]
        // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        // CHECK-SAME:      cluster_id = 1

        // CHECK-NEXT:  VPU.DPU.Workload
        // CHECK-SAME:      outOffsets [0, 0, 150, 0] outSizes [1, 3, 74, 224]
        // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        // CHECK-SAME:      cluster_id = 2
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!SparseInputType = !VPU.SparseTensor<
    data=tensor<1x64x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
    sparsity_map=tensor<1x64x4x4xi1, {mem_space = [@CMX_NN, 0], order = #NHWC}>
>

// CHECK-LABEL: @DepthConvSparseInputWithoutL1aOpt
func.func @DepthConvSparseInputWithoutL1aOpt(
    %ACT: !SparseInputType,
    %FILT: tensor<64x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
) -> tensor<1x64x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> {
    // CHECK:   [[ACT:%.+]]: !VPU.SparseTensor<{{.*}}>, [[FILT:%.+]]: tensor<64x16x1x1xf16{{.*}}>
    %DWCONV = VPU.NCE.DepthConvolution(%ACT, %FILT) {
        minimumHardwareExecutionCost = 790 : i64,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<
            mode = <NOOP>,
            clamp_low = -2147483648 : i64,
            clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64,
            fp_prelu_alpha = 1.000000e+00 : f64
        >,
        rawFilterShape = [64, 1, 3, 3],
        strides = [1, 1]
    } -> tensor<1x64x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> {
        VPU.DPU.Workload
            outOffsets [0, 0, 0, 0]
            outSizes [1, 64, 16, 16]
            <left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
            <CUBOID_16x16>
    }
    // CHECK:   [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[ACT]], [[FILT]])
    // CHECK-NEXT:  VPU.DPU.Workload
    // CHECK-SAME:      outOffsets [0, 0, 0, 0]
    // CHECK-SAME:      outSizes [1, 64, 16, 16]

    return %DWCONV : tensor<1x64x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:   return [[DWCONV]] : tensor<1x64x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Input_CMX = !VPU.SparseTensor<
    data = !VPU.DistributedTensor<
        1x512x14x14x!quant.uniform<u8:f16, 0.0062617507635378371>, #NHWC, @CMX_NN, {
        mode = "DUPLICATED",
        num_clusters = 3 : i64,
        uniform_distributed_segments}>,
    sparsity_map = !VPU.DistributedTensor<
        1x512x14x14xi1, #NHWC, @CMX_NN, {
        mode = "DUPLICATED",
        num_clusters = 3 : i64,
        uniform_distributed_segments}>
>

!Weights_CMX = !VPU.SparseTensor<
    data = !VPU.DistributedTensor<
        256x512x3x3x!quant.uniform<u8:f16, 0.00092766667697943892>, #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [3, 1, 1, 1],
        num_clusters = 3 : i64,
        alignment = [16, 1, 1, 1],
        compute_shapes = [[96, 512, 3, 3], [96, 512, 3, 3], [64, 512, 3, 3]],
        compute_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [192, 0, 0, 0]],
        memory_shapes = [[96, 512, 3, 3], [96, 512, 3, 3], [64, 512, 3, 3]],
        memory_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [192, 0, 0, 0]]}>,
    sparsity_map = !VPU.DistributedTensor<
        256x1x1x4608xi1, #NCHW, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [3, 1, 1, 1],
        num_clusters = 3 : i64,
        alignment = [16, 1, 1, 1],
        compute_shapes = [[96, 1, 1, 4608], [96, 1, 1, 4608], [64, 1, 1, 4608]],
        compute_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [192, 0, 0, 0]],
        memory_shapes = [[96, 1, 1, 4608], [96, 1, 1, 4608], [64, 1, 1, 4608]],
        memory_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [192, 0, 0, 0]]}>,
        is_weights,
        #VPU.SparsityCompression<axis = 0 : i64, numElems = dense_resource<__elided__> : tensor<256xi64>, alignment = 16 : i64>
>

!WeightsTable_CMX = !VPU.DistributedTensor<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3 : i64,
    alignment = [16, 1, 1, 1],
    compute_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [64, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [192, 0, 0, 0]],
    memory_shapes = [[96, 1, 1, 4], [96, 1, 1, 4], [64, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [96, 0, 0, 0], [192, 0, 0, 0]]}
>

!Output_CMX = !VPU.SparseTensor<
    data = !VPU.DistributedTensor<
        1x256x7x7x!quant.uniform<u8:f16, 0.0048209779402788944>, #NHWC, @CMX_NN, {
        mode = "DUPLICATED|SEGMENTED",
        num_tiles = [1, 3, 1, 1],
        num_clusters = 3 : i64,
        alignment = [1, 16, 1, 1],
        compute_shapes = [[1, 96, 7, 7], [1, 96, 7, 7], [1, 64, 7, 7]],
        compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0]],
        memory_shapes = [[1, 256, 7, 7], [1, 256, 7, 7], [1, 256, 7, 7]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
    sparsity_map = !VPU.DistributedTensor<
        1x256x7x7xi1, #NHWC, @CMX_NN, {
        mode = "DUPLICATED|SEGMENTED",
        num_tiles = [1, 3, 1, 1],
        num_clusters = 3 : i64,
        alignment = [1, 16, 1, 1],
        compute_shapes = [[1, 96, 7, 7], [1, 96, 7, 7], [1, 64, 7, 7]],
        compute_offsets = [[0, 0, 0, 0], [0, 96, 0, 0], [0, 192, 0, 0]],
        memory_shapes = [[1, 256, 7, 7], [1, 256, 7, 7], [1, 256, 7, 7]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
>

// CHECK-LABEL: @SOKConvSparseOutput
func.func @SOKConvSparseOutput(%arg0: !Input_CMX, %arg1: !Weights_CMX) -> !Output_CMX {
    %2 = VPU.NCE.Convolution(%arg0, %arg1) {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = 0.000000e+00 : f64, clamp_high = 2.550000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
            rawFilterShape = [256, 512, 3, 3], strides = [2, 2]} : !Input_CMX, !Weights_CMX -> !Output_CMX {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 96, 7, 7] <left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64> <CUBOID_4x16> attributes {cluster_id = 0 : i64}
                VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 96, 7, 7] <left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64> <CUBOID_4x16> attributes {cluster_id = 1 : i64}
                VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 64, 7, 7] <left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64> <CUBOID_8x16> attributes {cluster_id = 2 : i64}
        }

    return %2 : !Output_CMX

    // CHECK:       VPU.NCE.Convolution
    // CHECK:          VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 96, 7, 7] <left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64> <CUBOID_4x16> attributes {cluster_id = 0 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 96, 0, 0] outSizes [1, 96, 7, 7] <left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64> <CUBOID_4x16> attributes {cluster_id = 1 : i64}
    // CHECK-NEXT:     VPU.DPU.Workload outOffsets [0, 192, 0, 0] outSizes [1, 64, 7, 7] <left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64> <CUBOID_8x16> attributes {cluster_id = 2 : i64}
}
