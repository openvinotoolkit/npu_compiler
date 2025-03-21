//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Convolution
module @Convolution {
    // CHECK-DAG:  {{  }}IE.ExecutorResource 2 of @DMA_NN
    // CHECK-DAG:  {{  }}IE.TileResource {activity_factor = {{[0-9]+.[0-9]+}} : f64} 2 of @NCE at 1.300000e+03 MHz
    // CHECK-DAG:  {{    }}IE.ExecutorResource 1 of @DPU
    // CHECK-DAG:  {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
    // CHECK-DAG:  {{    }}IE.ExecutorResource 1 of @SHAVE_NN

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[ARG0:%.+]]: memref<1x3x62x62xf16, @DDR>,
    // CHECK-SAME:      [[ARG1:%.+]]: memref<1x48x60x60xf16, @DDR>) -> memref<1x48x60x60xf16, @DDR>
    func.func @main(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %1 = IE.Convolution(%arg, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        return %1 : tensor<1x48x60x60xf32>

        // CHECK-DAG:   const.Declare memref<1x1x1x2688xf16>

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<ELTWISE>
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<ELTWISE>

        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
        // CHECK-SAME:      [[input_0:%.*]] : memref<1x16x32x62xf16, #NHWC, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[weight_0:%.*]] : memref<48x16x3x3xf16, {order = #NHWC, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>}, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[weight_sm_0:%.*]] : memref<48x1x1x256xi1, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[weight_table_0:%.*]] : memref<48x1x1x4xsi32, [@CMX_NN, 0]>)
        // CHECK-SAME:      [[parent_input_0:%.*]] : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>)
        // CHECK-SAME:      [[parent_output_0:%.*]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME:      [[output_0:%.*]] : memref<1x48x30x60xf16, [@CMX_NN, 0]>)
        // CHECK:               DPUTask

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
        // CHECK-SAME:      [[input_1:%.*]] : memref<1x16x30x62xf16, #NHWC, [@CMX_NN, 1]>)
        // CHECK-SAME:      [[weight_1:%.*]] : memref<48x16x3x3xf16, {order = #NHWC, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>}, [@CMX_NN, 1]>)
        // CHECK-SAME:      [[weight_sm_1:%.*]] : memref<48x1x1x256xi1, [@CMX_NN, 1]>
        // CHECK-SAME:      [[weight_table_1:%.*]] : memref<48x1x1x4xsi32, [@CMX_NN, 1]>)
        // CHECK-SAME:      [[parent_input_1:%.*]] : !VPUIP.DistributedBuffer<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>)
        // CHECK-SAME:      [[parent_output_1:%.*]] : !VPUIP.DistributedBuffer<1x48x60x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME:      [[output_1:%.*]] : memref<1x48x30x60xf16, [@CMX_NN, 1]>)
        // CHECK:               DPUTask
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType2 = !quant.uniform<u8:f16, 0.047852594712201289:128>

// CHECK-LABEL: @ScaleShiftSubgraph
module @ScaleShiftSubgraph {
    // CHECK-DAG:  {{  }}IE.ExecutorResource 2 of @DMA_NN
    // CHECK-DAG:  {{  }}IE.TileResource {activity_factor = {{[0-9]+.[0-9]+}} : f64} 2 of @NCE at 1.300000e+03 MHz
    // CHECK-DAG:  {{    }}IE.ExecutorResource 1 of @DPU
    // CHECK-DAG:  {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
    // CHECK-DAG:  {{    }}IE.ExecutorResource 1 of @SHAVE_NN

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x256x20x20xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x512x20x20xf32>
    }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[ARG0:%.+]]: memref<1x256x20x20xf32, @DDR>,
    // CHECK-SAME:      [[ARG1:%.+]]: memref<1x512x20x20xf32, @DDR>) -> memref<1x512x20x20xf32, @DDR>
    func.func @main(%arg: tensor<1x256x20x20xf32>) -> tensor<1x512x20x20xf32> {

        %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-2.05229545> : tensor<1x1x1x1xf32>
        %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<2.0362618> : tensor<1x1x1x1xf32>
        %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<-2.05229545> : tensor<1x1x1x1xf32>
        %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<2.0362618> : tensor<1x1x1x1xf32>
        %0 = IE.FakeQuantize(%arg, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x256x20x20xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x256x20x20xf32>

        %1 = IE.Concat(%0, %0) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x256x20x20xf32>, tensor<1x256x20x20xf32> -> tensor<1x512x20x20xf32>

        %cst_4 = const.Declare tensor<1x512x1x1xf32> = dense<1.0> : tensor<1x512x1x1xf32>
        %2 = IE.Multiply(%1, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x20x20xf32>, tensor<1x512x1x1xf32> -> tensor<1x512x20x20xf32>

        %cst_5 = const.Declare tensor<1x512x1x1xf32> = dense<1.0> : tensor<1x512x1x1xf32>
        %3 = IE.Add(%2, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x20x20xf32>, tensor<1x512x1x1xf32> -> tensor<1x512x20x20xf32>

        %cst_6 = const.Declare tensor<1xf32> = dense<1.000000e-01> : tensor<1xf32>
        %4 = IE.PRelu(%3, %cst_6) : tensor<1x512x20x20xf32>, tensor<1xf32> -> tensor<1x512x20x20xf32>

        %cst_7 = const.Declare tensor<1x1x1x1xf32> = dense<-6.12513208> : tensor<1x1x1x1xf32>
        %cst_8 = const.Declare tensor<1x1x1x1xf32> = dense<6.07727957> : tensor<1x1x1x1xf32>
        %cst_9 = const.Declare tensor<1x1x1x1xf32> = dense<-6.12513208> : tensor<1x1x1x1xf32>
        %cst_10 = const.Declare tensor<1x1x1x1xf32> = dense<6.07727957> : tensor<1x1x1x1xf32>
        %5 = IE.FakeQuantize(%4, %cst_7, %cst_8, %cst_9, %cst_10) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x512x20x20xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x512x20x20xf32>

        return %5 : tensor<1x512x20x20xf32>

        // CHECK-DAG:   const.Declare memref<1x1x1x12288xf16>

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<DWCONV>}
        // CHECK-SAME:      input([[input_0:%.*]] : memref<1x512x10x20xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
        // CHECK-SAME:      weights([[weight_0:%.*]] : memref<512x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
        // CHECK-SAME:      weight_table([[weight_table_0:%.*]] : memref<512x1x1x4xsi32, [@CMX_NN, 0]>)
        // CHECK-SAME:      parent_input([[parent_input_0:%.*]] : !VPUIP.DistributedBuffer<1x512x20x20xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME:      parent_output([[parent_output_0:%.*]] : !VPUIP.DistributedBuffer<1x512x20x20x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME:      outputs([[output_0:%.*]] : memref<1x512x10x20x!qElemType2, #NHWC, [@CMX_NN, 0]>) -> memref<1x512x10x20x!qElemType2, #NHWC, [@CMX_NN, 0]> variants : {
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:       } PPE : {
        // CHECK:               PPETask {ppe = #VPU.PPEInt<mode = <LPRELU>

        // CHECK:       VPURT.Task waits([[barrier_0:%.*]] : !VPURT.Barrier) updates([[barrier_1:%.*]] : !VPURT.Barrier)
        // CHECK:       VPUIP.NCEClusterTask {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<DWCONV>}
        // CHECK-SAME:      input([[input_1:%.*]] : memref<1x512x10x20xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 1]>)
        // CHECK-SAME:      weights([[weight_1:%.*]] : memref<512x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
        // CHECK-SAME:      weight_table([[weight_table_1:%.*]] : memref<512x1x1x4xsi32, [@CMX_NN, 1]>)
        // CHECK-SAME:      parent_input([[parent_input_0:%.*]] : !VPUIP.DistributedBuffer<1x512x20x20xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME:      parent_output([[parent_output_0:%.*]] : !VPUIP.DistributedBuffer<1x512x20x20x!qElemType2, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
        // CHECK-SAME:      outputs([[output_1:%.*]] : memref<1x512x10x20x!qElemType2, #NHWC, [@CMX_NN, 1]>) -> memref<1x512x10x20x!qElemType2, #NHWC, [@CMX_NN, 1]> variants : {
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:               DPUTask
        // CHECK:       } PPE : {
        // CHECK:               PPETask {ppe = #VPU.PPEInt<mode = <LPRELU>
  }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DynamicReshape
module @DynamicReshape {
    // CHECK-DAG:  {{  }}VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration
    // CHECK-DAG:  {{  }}module @VPU.SW {
    // CHECK-DAG:  {{    }}func.func private @builtin_DynamicReshape
    // CHECK-DAG:  {{    }}func.func private @builtin_Convert
    // CHECK-DAG:  {{    }}func.func private @runtime
    // CHECK-DAG:  {{  }}IE.ExecutorResource 2 of @DMA_NN
    // CHECK-DAG:  {{  }}IE.TileResource {activity_factor = {{.+}} : f64} 2 of @NCE at 1.300000e+03 MHz
    // CHECK-DAG:  {{    }}builtin.module @ReservedMemory {
    // CHECK-DAG:  {{      }}module @SWKernelPrefetchingReservedMemory {
    // CHECK-DAG:  {{        }}IE.MemoryResource {{[0-9]+}} bytes of @CMX_NN offset {{[0-9]+}}
    // CHECK-DAG:  {{    }}IE.MemoryResource {{[0-9]+}} bytes of @CMX_NN_FragmentationAware
    // CHECK-DAG:  {{    }}IE.MemoryResource {{[0-9]+}} bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = {{.+}} : f64}
    // CHECK-DAG:  {{    }}IE.ExecutorResource 1 of @DPU
    // CHECK-DAG:  {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
    // CHECK-DAG:  {{    }}IE.ExecutorResource 1 of @SHAVE_NN

    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "Parameter_57" : tensor<1x3x10x16xf32>
    } outputsInfo : {
        DataInfo "Reshape_65" : tensor<1x1x2x240xf32>
    }
    // CHECK:   IE.CNNNetwork
    // CHECK-SAME:  inputsInfo
    // CHECK:       DataInfo "Parameter_57" : tensor<1x3x10x16xf32>
    // CHECK:       DataInfo "vpux_ie_shape_Parameter_57" : tensor<4xsi32>
    // CHECK:       outputsInfo
    // CHECK:       DataInfo "Reshape_65" : tensor<1x1x2x240xf32>
    // CHECK:       DataInfo "vpux_ie_shape_Reshape_65" : tensor<4xsi32>

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[ARG0:%.+]]: memref<1x3x10x16xf32, @DDR>,
    // CHECK-SAME:      [[SHAPE0:%.+]]: memref<4xsi32, @DDR>,
    // CHECK-SAME:      [[ARG1:%.+]]: memref<1x1x2x240xf32, @DDR>,
    // CHECK-SAME:      [[SHAPE0:%.+]]: memref<4xsi32, @DDR>
    // CHECK-SAME:      -> (memref<1x1x2x240xf32, @DDR>, memref<4xsi32, @DDR>)
    func.func @main(%arg0: tensor<1x3x?x?xf32, {bounds = [1, 3, 10, 16], order = #NCHW}>) -> tensor<1x1x2x?xf32, {bounds = [1, 1, 2, 240], order = #NCHW}> {

        %cst = const.Declare tensor<4xsi64> = dense<[1, 1, 2, -1]> : tensor<4xsi64>
        %0 = IE.DynamicReshape(%arg0, %cst) {output_bounds = [1, 1, 2, 240], output_shape = [1, 1, 2, -9223372036854775808]} : tensor<1x3x?x?xf32, {bounds = [1, 3, 10, 16], order = #NCHW}>, tensor<4xsi64> -> tensor<1x1x2x?xf32, {bounds = [1, 1, 2, 240], order = #NCHW}>
        return %0 : tensor<1x1x2x?xf32, {bounds = [1, 1, 2, 240], order = #NCHW}>

        // CHECK:       VPURT.Task waits([[barrier_0:%.+]] : !VPURT.Barrier) updates([[barrier_1:%.+]] : !VPURT.Barrier) {
        // CHECK:       [[convert:%.+]], [[convert_shape:%.+]] = VPUIP.SW.Kernel
        // CHECK-SAME:          dynamicInputShapesMap = array<i32: 0>
        // CHECK-SAME:          dynamicOutputShapesMap = array<i32: 0>
        // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 1, 0>
        // CHECK-SAME:      @VPU.SW::@builtin_Convert
        // CHECK-SAME:          inputs({{%.+}} as {{%.+}}: memref<1x3x10x16xf32, [@CMX_NN, 0]>)
        // CHECK-SAME:          dynamicInputShapes({{%[0-9]+}} : memref<4xsi32, [@CMX_NN, 0]>)
        // CHECK-SAME:          outputs({{%.+}} as {{%.+}}: memref<1x3x10x16xf16, [@CMX_NN, 0]>)
        // CHECK-SAME:          dynamicOutputShapes({{%.+}} : memref<4xsi32, [@CMX_NN, 0]>)

        // CHECK:       VPURT.Task waits([[barrier_1]] : !VPURT.Barrier) updates([[barrier_2:%.+]] : !VPURT.Barrier) {
        // CHECK:       [[reshape:%.+]], [[reshape_shape:%.+]] = VPUIP.SW.Kernel
        // CHECK-SAME:          dynamicInputShapesMap = array<i32: 0, -1>
        // CHECK-SAME:          dynamicOutputShapesMap = array<i32: 0>
        // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 1, 0>
        // CHECK-SAME:      @VPU.SW::@builtin_DynamicReshape
        // CHECK-SAME:          inputs({{%[0-9]+}} as {{%arg[0-9]+}}: memref<1x3x10x16xf16, [@CMX_NN, 0]>, {{%[0-9]+}} as {{%arg[0-9]+}}: memref<4xsi32, [@CMX_NN, 0]>)
        // CHECK-SAME:          dynamicInputShapes({{%[0-9]+}} : memref<4xsi32, [@CMX_NN, 0]>)
        // CHECK-SAME:          outputs({{%[0-9]+}} as {{%arg[0-9]+}}: memref<1x1x2x240xf16, [@CMX_NN, 0]>
        // CHECK-SAME:          dynamicOutputShapes({{%.+}} : memref<4xsi32, [@CMX_NN, 0]>)

        // CHECK:       VPURT.Task waits([[barrier_2:%.+]] : !VPURT.Barrier) updates([[barrier_3:%.+]] : !VPURT.Barrier) {
        // CHECK:       [[convert:%.+]], [[convert_shape:%.+]] = VPUIP.SW.Kernel
        // CHECK-SAME:          dynamicInputShapesMap = array<i32: 0>
        // CHECK-SAME:          dynamicOutputShapesMap = array<i32: 0>
        // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 1, 0>
        // CHECK-SAME:      @VPU.SW::@builtin_Convert
        // CHECK-SAME:          inputs({{%.+}} as {{%.+}}: memref<1x1x2x240xf16, [@CMX_NN, 0]>)
        // CHECK-SAME:          dynamicInputShapes({{%[0-9]+}} : memref<4xsi32, [@CMX_NN, 0]>)
        // CHECK-SAME:          outputs({{%.+}} as {{%.+}}: memref<1x1x2x240xf32, [@CMX_NN, 0]>)
        // CHECK-SAME:          dynamicOutputShapes({{%.+}} : memref<4xsi32, [@CMX_NN, 0]>)

        // CHECK:       return {{%.+}}, {{%.+}} : memref<1x1x2x240xf32, @DDR>, memref<4xsi32, @DDR>
    }
}

// -----

module @BatchedGroupConvWithBroadcast {
  // CHECK-DAG:  {{  }}IE.ExecutorResource 2 of @DMA_NN
  // CHECK-DAG:  {{  }}IE.TileResource {activity_factor = {{[0-9]+.[0-9]+}} : f64} 2 of @NCE at 1.300000e+03 MHz
  // CHECK-DAG:  {{    }}IE.ExecutorResource 1 of @DPU
  // CHECK-DAG:  {{    }}IE.ExecutorResource 2 of @SHAVE_ACT
  // CHECK-DAG:  {{    }}IE.ExecutorResource 1 of @SHAVE_NN

  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "Parameter_1" : tensor<4x1x2x2xf16>
    DataInfo "Parameter_2" : tensor<1x1x3x3xf16>
  } outputsInfo : {
    DataInfo "GroupConvolution_10" : tensor<4x1x2x2xf16>
  }

  // CHECK:       func.func @main(
  // CHECK-SAME:      [[ARG0:%.+]]: memref<4x1x2x2xf16, @DDR>,
  // CHECK-SAME:      [[ARG1:%.+]]: memref<1x1x3x3xf16, @DDR>,
  // CHECK-SAME:      -> memref<4x1x2x2xf16, @DDR>

  func.func @main(%arg0: tensor<4x1x2x2xf16>, %arg1: tensor<1x1x3x3xf16>) -> tensor<4x1x2x2xf16> {
    %0 = IE.GroupConvolution(%arg0, %arg1) {dilations = [1, 1], groups = 1 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
         } : tensor<4x1x2x2xf16>, tensor<1x1x3x3xf16> -> tensor<4x1x2x2xf16>
    return %0 : tensor<4x1x2x2xf16>

    // CHECK: VPURT.Task waits([[BAR_0:%.+]] : !VPURT.Barrier) updates([[BAR_1:%.+]] : !VPURT.Barrier) {
    // CHECK: VPUIP.NCEClusterTask {
    // CHECK:   kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK:   kernel_size = [3, 3], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>
    // CHECK-SAME: task_type = #VPUIP.nce_task_type<DWCONV>
    // CHECK:   input([[INPUT_0:%.+]] : memref<1x16x2x2xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:   weights([[WEIGHTS_0:%.+]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:   weight_table([[WT_0:%.+]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK:   parent_input([[PARENT_IN:%.+]] : !VPUIP.DistributedBuffer<1x16x2x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK:   parent_output([[PARENT_OUT:%.+]] : !VPUIP.DistributedBuffer<1x16x2x2xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK:   outputs([[OUT_0:%.+]] : memref<1x16x2x2xf16, [@CMX_NN, 0]>) -> memref<1x16x2x2xf16, [@CMX_NN, 0]> variants : {
    // CHECK:     DPUTask {cluster_id = 0 : i64
    // CHECK:   PPETask {ppe = #VPU.PPEInt<mode = <NOOP>

    // CHECK: VPURT.Task waits([[BAR_0:%.+]] : !VPURT.Barrier) updates([[BAR_1:%.+]] : !VPURT.Barrier) {
    // CHECK: VPUIP.NCEClusterTask {
    // CHECK:   kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK:   kernel_size = [3, 3], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>
    // CHECK-SAME: task_type = #VPUIP.nce_task_type<DWCONV>
    // CHECK:   input([[INPUT_1:%.+]] : memref<1x16x2x2xf16, #NHWC, [@CMX_NN, 1]>)
    // CHECK:   weights([[WEIGHTS_1:%.+]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    // CHECK:   weight_table([[WT_1:%.+]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    // CHECK:   parent_input([[PARENT_IN:%.+]] : !VPUIP.DistributedBuffer<1x16x2x2xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK:   parent_output([[PARENT_OUT:%.+]] : !VPUIP.DistributedBuffer<1x16x2x2xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK:   outputs([[OUT_1:%.+]] : memref<1x16x2x2xf16, [@CMX_NN, 1]>) -> memref<1x16x2x2xf16, [@CMX_NN, 1]> variants : {
    // CHECK:     DPUTask {cluster_id = 1 : i64
    // CHECK:   PPETask {ppe = #VPU.PPEInt<mode = <NOOP>

  }
}
