//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% workload-management-enable=false allow-custom-values=true" --insert-barrier-to-mark-end-of-descriptor-group %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8<0:254>:f16:0, {2.3954496608944391E-5:127,1.8652968519315946E-5:127,5.714536651851624E-6:127,9.2288640540415852E-6:127,5.2774985005536418E-5:127,1.0355251041922983E-5:127,4.7608623354453737E-5:127,4.2622483621432085E-5:127,6.3378041184793303E-5:127,9.0411328893946842E-6:127,3.7636343888410434E-5:127,9.1462623415969495E-6:127,3.0472522645484744E-5:127,1.3648806181992955E-6:127,8.4283783679872047E-5:127,5.5778683639886814E-5:127,1.6640490434301182E-5:127,5.4847537063238186E-5:127,9.0531476839320868E-5:127,1.0873389056348425E-5:127,8.4944597379429132E-5:127,5.5928868571604333E-5:127,7.9477865864911412E-5:127,1.5408973994217519E-5:127,2.7033287709153543E-5:127,7.3740801473302161E-6:127,4.5475997324064961E-5:127,4.5415923351377953E-5:127,1.4605484609528789E-5:127,8.4554116556963585E-6:127,1.9478985643762302E-5:127,4.6332051434854825E-6:127,9.6568911094365155E-6:127,1.3648806594488189E-4:127,4.8584825410617617E-6:127,1.0588037686085138E-5:127,9.6493818628506396E-6:127,1.0663130151943897E-5:127,3.8086898683562992E-5:127,4.142100416769193E-5:127,7.5805844284418062E-6:127,5.2684874046505907E-5:127,9.1462623415969495E-6:127,1.327634796382874E-5:127,6.1538275771253692E-6:127,2.9691561000553643E-5:127,5.6079053503321852E-5:127,1.4605484609528789E-5:127,2.8324878121924213E-5:127,6.1695969949557085E-5:127,5.0492174043430117E-5:127,2.3218590443528543E-5:127,1.5026002418337845E-5:127,4.4184406911294294E-5:127,4.0009265809547241E-5:127,2.0289984275036911E-5:127,2.0680465097502461E-5:127,5.5906340831846705E-6:127,1.6054769200602853E-5:127,1.0940972275621308E-5:127,2.3008331539124016E-5:127,3.4182090458907481E-5:127,9.4766691913754922E-6:127,8.9450145330954728E-5:127}>
!qElemType3 = !quant.uniform<u8:f16, 1.000000e+00:114>

module attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.PipelineOptions @Options {
    config.Option @config.MetadataMaxVariantCount : 12
    config.Option @config.MetadataMaxInvariantCount : 6
    config.Option @config.MetadataMaxKernelInvocationCount : 4
    config.Option @config.MetadataMaxKernelRangeCount : 4
  }
// CHECK-LABEL: @insertBarrierBetweenEvery3rdSetOfDPUtasks
func.func @insertBarrierBetweenEvery3rdSetOfDPUtasks(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x112x112xf16, @DDR>) -> memref<1x64x112x112xf16, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // buffers
    %3 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <CMX_NN> [1] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 1]>
    %22 = VPURT.DeclareBuffer <CMX_NN> [2] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 2]>
    %23 = VPURT.DeclareBuffer <CMX_NN> [3] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 3]>
    %25 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, [@CMX_NN, 0]>
    %29 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>
    %30 = VPURT.DeclareBuffer <CMX_NN> [1] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>
    %31 = VPURT.DeclareBuffer <CMX_NN> [2] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 2]>
    %32 = VPURT.DeclareBuffer <CMX_NN> [3] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 3]>
    %41 = VPURT.DeclareBuffer <CMX_NN> [0] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %42 = VPURT.DeclareBuffer <CMX_NN> [1] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>
    %43 = VPURT.DeclareBuffer <CMX_NN> [2] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 2]>
    %44 = VPURT.DeclareBuffer <CMX_NN> [3] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 3]>
    %45 = VPURT.DeclareBuffer <CMX_NN> [0] <75840> -> memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %46 = VPURT.DeclareBuffer <CMX_NN> [1] <75840> -> memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>
    %47 = VPURT.DeclareBuffer <CMX_NN> [2] <75840> -> memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 2]>
    %48 = VPURT.DeclareBuffer <CMX_NN> [3] <75840> -> memref<1x16x59x224x!qElemType3, #NHWC, [@CMX_NN, 3]>

    //  DPUx(y): x - DPUTask id, y - cluster id
    //
    //                   DMA
    //                    |
    //                    b0
    //          /                    \
    //  DPU1(0)...DPU7(0)    DPU8(1)...DPU14(1) (7 DPU tasks without barriers between them on two different FIFOs)
    //          \                    /
    //                    b1
    //                    |
    //                   DMA

    // DMA
    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%11 : memref<1x3x56x224xf16, [@CMX_NN, 0]>) -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    }

    // DPU1
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU2
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU3
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU4
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU5
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU6
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU7
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }


    // DPU8
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU9
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU10
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU11
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU12
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU13
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU14
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%25 : memref<1x64x28x112xf16, [@CMX_NN, 0]>) outputs(%7 : memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>) -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    }

    return %arg1 : memref<1x64x112x112xf16, @DDR>

    // execution groups: [DPU1, DPU2, DPU3][DPU4, DPU5, DPU6][DPU7]
    //              [DPU8, DPU9, DPU10][DPU11, DPU12, DPU13][DPU14]
    //                   DMA
    //                    |
    //                    b0
    //       /\          /    \          /\        \         \
    //   DPU1..DPU3(0)  /      |     DPU8..DPU10(1) |         |
    //       \/   |    /       |         \/    |    |         |
    //       |   b2   /        |         |     b3   |         |
    //       |    |  /\        |         |     |   /\         |
    //       |   DPU4..DPU6(0) |         |    DPU11..DPU13(1) |
    //       |       \/        |         |         \/         |  no barrier between DPU6 and DPU7 and
    //       |       |         |         |         |          |  between DPU13 and DPU14 because grand child
    //       |       |         /         |         |          /  group does not exist
    //       |       |  DPU7(0)          |         |   DPU14(1)
    //        \      \    |             /          /  /
    //                    b1
    //                    |
    //                    DMA

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT:            VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // DMA
    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA

    // DPU1
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 0
    // DPU8
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 1

    // DPU2
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 0
    // DPU9
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 1

    // DPU3
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]], [[BAR2]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 0
    // DPU10
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]], [[BAR3]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 1

    // DPU4
    // CHECK:       VPURT.Task waits([[BAR0]], [[BAR2]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 0
    // DPU11
    // CHECK:       VPURT.Task waits([[BAR0]], [[BAR3]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 1

    // DPU5
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 0
    // DPU12
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 1

    // DPU6
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 0
    // DPU13
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 1

    // DPU7
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 0
    // DPU14
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 1

    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA

    // CHECK:       return %arg1 : memref<1x64x112x112xf16, @DDR>
}
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8<0:254>:f16:0, {2.3954496608944391E-5:127,1.8652968519315946E-5:127,5.714536651851624E-6:127,9.2288640540415852E-6:127,5.2774985005536418E-5:127,1.0355251041922983E-5:127,4.7608623354453737E-5:127,4.2622483621432085E-5:127,6.3378041184793303E-5:127,9.0411328893946842E-6:127,3.7636343888410434E-5:127,9.1462623415969495E-6:127,3.0472522645484744E-5:127,1.3648806181992955E-6:127,8.4283783679872047E-5:127,5.5778683639886814E-5:127,1.6640490434301182E-5:127,5.4847537063238186E-5:127,9.0531476839320868E-5:127,1.0873389056348425E-5:127,8.4944597379429132E-5:127,5.5928868571604333E-5:127,7.9477865864911412E-5:127,1.5408973994217519E-5:127,2.7033287709153543E-5:127,7.3740801473302161E-6:127,4.5475997324064961E-5:127,4.5415923351377953E-5:127,1.4605484609528789E-5:127,8.4554116556963585E-6:127,1.9478985643762302E-5:127,4.6332051434854825E-6:127,9.6568911094365155E-6:127,1.3648806594488189E-4:127,4.8584825410617617E-6:127,1.0588037686085138E-5:127,9.6493818628506396E-6:127,1.0663130151943897E-5:127,3.8086898683562992E-5:127,4.142100416769193E-5:127,7.5805844284418062E-6:127,5.2684874046505907E-5:127,9.1462623415969495E-6:127,1.327634796382874E-5:127,6.1538275771253692E-6:127,2.9691561000553643E-5:127,5.6079053503321852E-5:127,1.4605484609528789E-5:127,2.8324878121924213E-5:127,6.1695969949557085E-5:127,5.0492174043430117E-5:127,2.3218590443528543E-5:127,1.5026002418337845E-5:127,4.4184406911294294E-5:127,4.0009265809547241E-5:127,2.0289984275036911E-5:127,2.0680465097502461E-5:127,5.5906340831846705E-6:127,1.6054769200602853E-5:127,1.0940972275621308E-5:127,2.3008331539124016E-5:127,3.4182090458907481E-5:127,9.4766691913754922E-6:127,8.9450145330954728E-5:127}>
!qElemType3 = !quant.uniform<u8:f16, 1.000000e+00:114>

module attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.PipelineOptions @Options {
    config.Option @config.MetadataMaxVariantCount : 12
    config.Option @config.MetadataMaxInvariantCount : 6
    config.Option @config.MetadataMaxKernelInvocationCount : 4
    config.Option @config.MetadataMaxKernelRangeCount : 4
  }

// CHECK-LABEL: @barBetweenEvery3rdSetOfDPUtasksWithNoBarDeps
func.func @barBetweenEvery3rdSetOfDPUtasksWithNoBarDeps(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x112x112xf16, @DDR>) -> memref<1x64x112x112xf16, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // buffers
    %3 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <CMX_NN> [1] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 1]>
    %22 = VPURT.DeclareBuffer <CMX_NN> [2] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 2]>
    %23 = VPURT.DeclareBuffer <CMX_NN> [3] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 3]>
    %25 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, [@CMX_NN, 0]>
    %29 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>
    %30 = VPURT.DeclareBuffer <CMX_NN> [1] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>
    %31 = VPURT.DeclareBuffer <CMX_NN> [2] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 2]>
    %32 = VPURT.DeclareBuffer <CMX_NN> [3] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 3]>
    %41 = VPURT.DeclareBuffer <CMX_NN> [0] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %42 = VPURT.DeclareBuffer <CMX_NN> [1] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>
    %43 = VPURT.DeclareBuffer <CMX_NN> [2] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 2]>
    %44 = VPURT.DeclareBuffer <CMX_NN> [3] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 3]>
    %45 = VPURT.DeclareBuffer <CMX_NN> [0] <75840> -> memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %46 = VPURT.DeclareBuffer <CMX_NN> [1] <75840> -> memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>
    %47 = VPURT.DeclareBuffer <CMX_NN> [2] <75840> -> memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 2]>
    %48 = VPURT.DeclareBuffer <CMX_NN> [3] <75840> -> memref<1x16x59x224x!qElemType3, #NHWC, [@CMX_NN, 3]>

    //  DPUx(y): x - DPUTask id, y - cluster id
    //
    //         DMA
    //          |
    //          b0
    //        /     \
    //  DPU1(0)    DPU8(1)
    //  DPU2(0)    DPU9(1)
    //  DPU3(0)    DPU10(1)
    //  DPU4(0)    DPU11(1)  (7 DPU tasks without barriers between them on two different FIFOs)
    //  DPU5(0)    DPU12(1)
    //  DPU6(0)    DPU13(1)
    //  DPU7(0)    DPU14(1)
    //       \       /
    //          b1
    //          |
    //         DMA

    // DMA
    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%11 : memref<1x3x56x224xf16, [@CMX_NN, 0]>) -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    }

    // DPU1
    VPURT.Task waits(%bar0: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU2
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU3
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU4
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU5
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU6
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU7
    VPURT.Task updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }


    // DPU8
    VPURT.Task waits(%bar0: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU9
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU10
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU11
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU12
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU13
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU14
    VPURT.Task updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%25 : memref<1x64x28x112xf16, [@CMX_NN, 0]>) outputs(%7 : memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>) -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    }

    return %arg1 : memref<1x64x112x112xf16, @DDR>

    // execution groups: [DPU1, DPU2, DPU3][DPU4, DPU5, DPU6][DPU7]
    //              [DPU8, DPU9, DPU10][DPU11, DPU12, DPU13][DPU14]
    //         DMA
    //          |
    //          b0
    //        /    \
    //  DPU1(0)    DPU8(1)
    //  DPU2(0)    DPU9(1)
    //  DPU3(0)    DPU10(1)
    //     |         |
    //     b1        b2
    //     |         |
    //  DPU4(0)    DPU11(1)
    //  DPU5(0)    DPU12(1)
    //  DPU6(0)    DPU13(1) barrier between DPU6 and DPU7 and between DPU13 and DPU14
    //     |         |      is not needed as the grand child execution group does not exist
    //  DPU7(0)    DPU14(1)
    //        \    /
    //          b3
    //          |
    //         DMA

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT:            VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // DMA
    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:         VPUIP.NNDMA

    // DPU1
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) {
    // CHECK:         DPUTask {cluster_id = 0
    // DPU8
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) {
    // CHECK:         DPUTask {cluster_id = 1

    // DPU2
    // CHECK:       VPURT.Task {
    // CHECK:         DPUTask {cluster_id = 0
    // DPU9
    // CHECK:       VPURT.Task {
    // CHECK:         DPUTask {cluster_id = 1

    // DPU3
    // CHECK:       VPURT.Task updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:         DPUTask {cluster_id = 0
    // DPU10
    // CHECK:       VPURT.Task updates([[BAR2]] : !VPURT.Barrier) {
    // CHECK:         DPUTask {cluster_id = 1

    // DPU4
    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    // CHECK:         DPUTask {cluster_id = 0
    // DPU11
    // CHECK:       VPURT.Task waits([[BAR2]] : !VPURT.Barrier) {
    // CHECK:         DPUTask {cluster_id = 1

    // DPU5
    // CHECK:       VPURT.Task {
    // CHECK:         DPUTask {cluster_id = 0
    // DPU12
    // CHECK:       VPURT.Task {
    // CHECK:         DPUTask {cluster_id = 1

    // DPU6
    // CHECK:       VPURT.Task {
    // CHECK:         DPUTask {cluster_id = 0
    // DPU13
    // CHECK:       VPURT.Task {
    // CHECK:         DPUTask {cluster_id = 1

    // DPU7
    // CHECK:       VPURT.Task updates([[BAR3]] : !VPURT.Barrier) {
    // CHECK:         DPUTask {cluster_id = 0
    // DPU14
    // CHECK:       VPURT.Task updates([[BAR3]] : !VPURT.Barrier) {
    // CHECK:         DPUTask {cluster_id = 1

    // CHECK:       VPURT.Task waits([[BAR3]] : !VPURT.Barrier) {
    // CHECK:         VPUIP.NNDMA

    // CHECK:       return %arg1 : memref<1x64x112x112xf16, @DDR>
}
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8<0:254>:f16:0, {2.3954496608944391E-5:127,1.8652968519315946E-5:127,5.714536651851624E-6:127,9.2288640540415852E-6:127,5.2774985005536418E-5:127,1.0355251041922983E-5:127,4.7608623354453737E-5:127,4.2622483621432085E-5:127,6.3378041184793303E-5:127,9.0411328893946842E-6:127,3.7636343888410434E-5:127,9.1462623415969495E-6:127,3.0472522645484744E-5:127,1.3648806181992955E-6:127,8.4283783679872047E-5:127,5.5778683639886814E-5:127,1.6640490434301182E-5:127,5.4847537063238186E-5:127,9.0531476839320868E-5:127,1.0873389056348425E-5:127,8.4944597379429132E-5:127,5.5928868571604333E-5:127,7.9477865864911412E-5:127,1.5408973994217519E-5:127,2.7033287709153543E-5:127,7.3740801473302161E-6:127,4.5475997324064961E-5:127,4.5415923351377953E-5:127,1.4605484609528789E-5:127,8.4554116556963585E-6:127,1.9478985643762302E-5:127,4.6332051434854825E-6:127,9.6568911094365155E-6:127,1.3648806594488189E-4:127,4.8584825410617617E-6:127,1.0588037686085138E-5:127,9.6493818628506396E-6:127,1.0663130151943897E-5:127,3.8086898683562992E-5:127,4.142100416769193E-5:127,7.5805844284418062E-6:127,5.2684874046505907E-5:127,9.1462623415969495E-6:127,1.327634796382874E-5:127,6.1538275771253692E-6:127,2.9691561000553643E-5:127,5.6079053503321852E-5:127,1.4605484609528789E-5:127,2.8324878121924213E-5:127,6.1695969949557085E-5:127,5.0492174043430117E-5:127,2.3218590443528543E-5:127,1.5026002418337845E-5:127,4.4184406911294294E-5:127,4.0009265809547241E-5:127,2.0289984275036911E-5:127,2.0680465097502461E-5:127,5.5906340831846705E-6:127,1.6054769200602853E-5:127,1.0940972275621308E-5:127,2.3008331539124016E-5:127,3.4182090458907481E-5:127,9.4766691913754922E-6:127,8.9450145330954728E-5:127}>
!qElemType3 = !quant.uniform<u8:f16, 1.000000e+00:114>

module attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.PipelineOptions @Options {
    config.Option @config.MetadataMaxVariantCount : 2
    config.Option @config.MetadataMaxInvariantCount : 2
    config.Option @config.MetadataMaxKernelInvocationCount : 4
    config.Option @config.MetadataMaxKernelRangeCount : 4
  }

// CHECK-LABEL: @insertBarrierBetweenConsecutiveDPUtasksWithSharedBarriers
func.func @insertBarrierBetweenConsecutiveDPUtasksWithSharedBarriers(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // buffers
    %3 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %25 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, [@CMX_NN, 0]>
    %29 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>
    %41 = VPURT.DeclareBuffer <CMX_NN> [0] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %45 = VPURT.DeclareBuffer <CMX_NN> [0] <75840> -> memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>

    //  SWx(y): x - SWTask id, y - cluster id
    //
    //         DMA
    //          |
    //          b0
    //         /|\
    // DPU0(0),DPU1(0),DPU2(0)
    //         \|/
    //          b1
    //          |
    //         DMA

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%11 : memref<1x3x56x224xf16, [@CMX_NN, 0]>) -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    }
    // DPU0
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU1
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU2
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%25 : memref<1x64x28x112xf16, [@CMX_NN, 0]>) outputs(%7 : memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>) -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    }

    return %arg1: memref<1x3x64x64xf16, @DDR>

    // execution groups: [DPU0][DPU1][DPU2]
    //
    //         DMA
    //          |
    //          b0
    //     /         \    \
    //   DPU0(0)     |     |
    //    |   \      |     |
    //    |   b2     |     |
    //    |    |    /      |
    //    |   DPU1(0)      |
    //    |    |           | no barrier between DPU1 and DPU2
    //    |    |           | because grand child group does not exist
    //    |    |          /
    //    |    |    DPU2(0)
    //     \    \      /
    //          b1
    //           |
    //          DMA

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT:            VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // DPU0
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]], [[BAR2]] : !VPURT.Barrier, !VPURT.Barrier)
    // CHECK:         DPUTask
    // DPU1
    // CHECK:       VPURT.Task waits([[BAR0]], [[BAR2]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask
    // DPU2
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask

    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA

    // CHECK:       return
}
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8<0:254>:f16:0, {2.3954496608944391E-5:127,1.8652968519315946E-5:127,5.714536651851624E-6:127,9.2288640540415852E-6:127,5.2774985005536418E-5:127,1.0355251041922983E-5:127,4.7608623354453737E-5:127,4.2622483621432085E-5:127,6.3378041184793303E-5:127,9.0411328893946842E-6:127,3.7636343888410434E-5:127,9.1462623415969495E-6:127,3.0472522645484744E-5:127,1.3648806181992955E-6:127,8.4283783679872047E-5:127,5.5778683639886814E-5:127,1.6640490434301182E-5:127,5.4847537063238186E-5:127,9.0531476839320868E-5:127,1.0873389056348425E-5:127,8.4944597379429132E-5:127,5.5928868571604333E-5:127,7.9477865864911412E-5:127,1.5408973994217519E-5:127,2.7033287709153543E-5:127,7.3740801473302161E-6:127,4.5475997324064961E-5:127,4.5415923351377953E-5:127,1.4605484609528789E-5:127,8.4554116556963585E-6:127,1.9478985643762302E-5:127,4.6332051434854825E-6:127,9.6568911094365155E-6:127,1.3648806594488189E-4:127,4.8584825410617617E-6:127,1.0588037686085138E-5:127,9.6493818628506396E-6:127,1.0663130151943897E-5:127,3.8086898683562992E-5:127,4.142100416769193E-5:127,7.5805844284418062E-6:127,5.2684874046505907E-5:127,9.1462623415969495E-6:127,1.327634796382874E-5:127,6.1538275771253692E-6:127,2.9691561000553643E-5:127,5.6079053503321852E-5:127,1.4605484609528789E-5:127,2.8324878121924213E-5:127,6.1695969949557085E-5:127,5.0492174043430117E-5:127,2.3218590443528543E-5:127,1.5026002418337845E-5:127,4.4184406911294294E-5:127,4.0009265809547241E-5:127,2.0289984275036911E-5:127,2.0680465097502461E-5:127,5.5906340831846705E-6:127,1.6054769200602853E-5:127,1.0940972275621308E-5:127,2.3008331539124016E-5:127,3.4182090458907481E-5:127,9.4766691913754922E-6:127,8.9450145330954728E-5:127}>
!qElemType3 = !quant.uniform<u8:f16, 1.000000e+00:114>

module attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
config.PipelineOptions @Options {
  config.Option @config.MetadataMaxVariantCount : 2
  config.Option @config.MetadataMaxInvariantCount : 2
  config.Option @config.MetadataMaxKernelInvocationCount : 4
  config.Option @config.MetadataMaxKernelRangeCount : 4
}

// CHECK-LABEL: @insertBarrierBetweenConsecutiveDPUtasks
func.func @insertBarrierBetweenConsecutiveDPUtasks(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // buffers
    %3 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %25 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, [@CMX_NN, 0]>
    %29 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>
    %41 = VPURT.DeclareBuffer <CMX_NN> [0] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %45 = VPURT.DeclareBuffer <CMX_NN> [0] <75840> -> memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>

    //  SWx(y): x - DPUTask id, y - cluster id
    //
    //       DMA
    //        |
    //        b0
    //        |
    //      DPU0(0)
    //        |
    //      DPU1(0)
    //        |
    //      DPU2(0)
    //        |
    //        b1
    //        |
    //       DMA

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%11 : memref<1x3x56x224xf16, [@CMX_NN, 0]>) -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    }
    // DPU0
    VPURT.Task waits(%bar0 : !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU1
    VPURT.Task {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU2
    VPURT.Task updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%25 : memref<1x64x28x112xf16, [@CMX_NN, 0]>) outputs(%7 : memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>) -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    }

    return %arg1: memref<1x3x64x64xf16, @DDR>

    // execution groups: [DPU0][DPU1][DPU2]
    //       DMA
    //        |
    //        b0
    //        |
    //      DPU0(0)
    //        |
    //        b1
    //        |
    //      DPU1(0)
    //        |
    //      DPU2(0)
    //        |
    //        b2
    //        |
    //       DMA

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT:            VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // DPU0
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask
    // DPU1
    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    // CHECK:         DPUTask
    // DPU2
    // CHECK:       VPURT.Task updates([[BAR2]] : !VPURT.Barrier) {
    // CHECK:         DPUTask

    // CHECK:       VPURT.Task waits([[BAR2]] : !VPURT.Barrier) {
    // CHECK:         VPUIP.NNDMA

    // CHECK:       return
}
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8<0:254>:f16:0, {2.3954496608944391E-5:127,1.8652968519315946E-5:127,5.714536651851624E-6:127,9.2288640540415852E-6:127,5.2774985005536418E-5:127,1.0355251041922983E-5:127,4.7608623354453737E-5:127,4.2622483621432085E-5:127,6.3378041184793303E-5:127,9.0411328893946842E-6:127,3.7636343888410434E-5:127,9.1462623415969495E-6:127,3.0472522645484744E-5:127,1.3648806181992955E-6:127,8.4283783679872047E-5:127,5.5778683639886814E-5:127,1.6640490434301182E-5:127,5.4847537063238186E-5:127,9.0531476839320868E-5:127,1.0873389056348425E-5:127,8.4944597379429132E-5:127,5.5928868571604333E-5:127,7.9477865864911412E-5:127,1.5408973994217519E-5:127,2.7033287709153543E-5:127,7.3740801473302161E-6:127,4.5475997324064961E-5:127,4.5415923351377953E-5:127,1.4605484609528789E-5:127,8.4554116556963585E-6:127,1.9478985643762302E-5:127,4.6332051434854825E-6:127,9.6568911094365155E-6:127,1.3648806594488189E-4:127,4.8584825410617617E-6:127,1.0588037686085138E-5:127,9.6493818628506396E-6:127,1.0663130151943897E-5:127,3.8086898683562992E-5:127,4.142100416769193E-5:127,7.5805844284418062E-6:127,5.2684874046505907E-5:127,9.1462623415969495E-6:127,1.327634796382874E-5:127,6.1538275771253692E-6:127,2.9691561000553643E-5:127,5.6079053503321852E-5:127,1.4605484609528789E-5:127,2.8324878121924213E-5:127,6.1695969949557085E-5:127,5.0492174043430117E-5:127,2.3218590443528543E-5:127,1.5026002418337845E-5:127,4.4184406911294294E-5:127,4.0009265809547241E-5:127,2.0289984275036911E-5:127,2.0680465097502461E-5:127,5.5906340831846705E-6:127,1.6054769200602853E-5:127,1.0940972275621308E-5:127,2.3008331539124016E-5:127,3.4182090458907481E-5:127,9.4766691913754922E-6:127,8.9450145330954728E-5:127}>
!qElemType3 = !quant.uniform<u8:f16, 1.000000e+00:114>

module attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
config.PipelineOptions @Options {
  config.Option @config.MetadataMaxVariantCount : 2
  config.Option @config.MetadataMaxInvariantCount : 2
  config.Option @config.MetadataMaxKernelInvocationCount : 2
  config.Option @config.MetadataMaxKernelRangeCount : 2
}

module @VPU.SW {
  func.func private @builtin_Subtract(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_sub.cpp", VPU.kernel_entry = "eltwise_sub"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @insertBarrierBetweenConsecutiveSWtasks
func.func @insertBarrierBetweenConsecutiveSWtasks(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x3x64x64xf16, @DDR>
    %buf2_0 = VPURT.DeclareBuffer <CMX_NN> [0] <196608> -> memref<1x32x1x1xf16, [@CMX_NN, 0]>

    //  SWx(y): x - SWTask id, y - cluster id
    //
    //       DMA
    //        |
    //        b0
    //        |
    //      SW0(0)
    //        |
    //      SW1(0)
    //        |
    //      SW2(0)
    //        |
    //        b1
    //        |
    //       DMA

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    // SW0
    VPURT.Task waits(%bar0 : !VPURT.Barrier) {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Subtract inputs(%buf2_0 as %arg4: memref<1x32x1x1xf16, [@CMX_NN, 0]>, %buf2_0 as %arg2: memref<1x32x1x1xf16, [@CMX_NN, 0]>) outputs(%buf2_0 as %arg3: memref<1x32x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x1x1xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg4, %arg2, %arg3) : memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>
      }
    }
    // SW1
    VPURT.Task {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Subtract inputs(%buf2_0 as %arg4: memref<1x32x1x1xf16, [@CMX_NN, 0]>, %buf2_0 as %arg2: memref<1x32x1x1xf16, [@CMX_NN, 0]>) outputs(%buf2_0 as %arg3: memref<1x32x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x1x1xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg4, %arg2, %arg3) : memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>
      }
    }
    // SW2
    VPURT.Task updates(%bar1: !VPURT.Barrier) {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Subtract inputs(%buf2_0 as %arg4: memref<1x32x1x1xf16, [@CMX_NN, 0]>, %buf2_0 as %arg2: memref<1x32x1x1xf16, [@CMX_NN, 0]>) outputs(%buf2_0 as %arg3: memref<1x32x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x1x1xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg4, %arg2, %arg3) : memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>
      }
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
   }

    return %arg1: memref<1x3x64x64xf16, @DDR>

    // execution groups: [SW0][SW1][SW2]
    //       DMA
    //        |
    //        b0
    //        |
    //      SW0(0)
    //        |
    //        b1
    //        |
    //      SW1(0)  no barrier at the end of the group
    //        |     because grand child group does not exist
    //      SW2(0)
    //        |
    //        b3
    //        |
    //       DMA

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT:            VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // SW0
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SW
    // SW1
    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    // CHECK:         VPUIP.SW
    // SW2
    // CHECK:       VPURT.Task updates([[BAR2]] : !VPURT.Barrier) {
    // CHECK:         VPUIP.SW

    // CHECK:       VPURT.Task waits([[BAR2]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA

    // CHECK:       return
}
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8<0:254>:f16:0, {2.3954496608944391E-5:127,1.8652968519315946E-5:127,5.714536651851624E-6:127,9.2288640540415852E-6:127,5.2774985005536418E-5:127,1.0355251041922983E-5:127,4.7608623354453737E-5:127,4.2622483621432085E-5:127,6.3378041184793303E-5:127,9.0411328893946842E-6:127,3.7636343888410434E-5:127,9.1462623415969495E-6:127,3.0472522645484744E-5:127,1.3648806181992955E-6:127,8.4283783679872047E-5:127,5.5778683639886814E-5:127,1.6640490434301182E-5:127,5.4847537063238186E-5:127,9.0531476839320868E-5:127,1.0873389056348425E-5:127,8.4944597379429132E-5:127,5.5928868571604333E-5:127,7.9477865864911412E-5:127,1.5408973994217519E-5:127,2.7033287709153543E-5:127,7.3740801473302161E-6:127,4.5475997324064961E-5:127,4.5415923351377953E-5:127,1.4605484609528789E-5:127,8.4554116556963585E-6:127,1.9478985643762302E-5:127,4.6332051434854825E-6:127,9.6568911094365155E-6:127,1.3648806594488189E-4:127,4.8584825410617617E-6:127,1.0588037686085138E-5:127,9.6493818628506396E-6:127,1.0663130151943897E-5:127,3.8086898683562992E-5:127,4.142100416769193E-5:127,7.5805844284418062E-6:127,5.2684874046505907E-5:127,9.1462623415969495E-6:127,1.327634796382874E-5:127,6.1538275771253692E-6:127,2.9691561000553643E-5:127,5.6079053503321852E-5:127,1.4605484609528789E-5:127,2.8324878121924213E-5:127,6.1695969949557085E-5:127,5.0492174043430117E-5:127,2.3218590443528543E-5:127,1.5026002418337845E-5:127,4.4184406911294294E-5:127,4.0009265809547241E-5:127,2.0289984275036911E-5:127,2.0680465097502461E-5:127,5.5906340831846705E-6:127,1.6054769200602853E-5:127,1.0940972275621308E-5:127,2.3008331539124016E-5:127,3.4182090458907481E-5:127,9.4766691913754922E-6:127,8.9450145330954728E-5:127}>
!qElemType3 = !quant.uniform<u8:f16, 1.000000e+00:114>

module attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
config.PipelineOptions @Options {
  config.Option @config.MetadataMaxVariantCount : 2
  config.Option @config.MetadataMaxInvariantCount : 2
  config.Option @config.MetadataMaxKernelInvocationCount : 2
  config.Option @config.MetadataMaxKernelRangeCount : 2
}

module @VPU.SW {
  func.func private @builtin_Subtract(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_sub.cpp", VPU.kernel_entry = "eltwise_sub"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @insertBarrierBetweenConsecutiveSWtasks2
func.func @insertBarrierBetweenConsecutiveSWtasks2(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x3x64x64xf16, @DDR>
    %buf2_0 = VPURT.DeclareBuffer <CMX_NN> [0] <196608> -> memref<1x32x1x1xf16, [@CMX_NN, 0]>

    //  SWx(y): x - SWTask id, y - cluster id
    //                DMA
    //        SW0(0)  |
    //        |      b0
    //        |     / |
    //        SW1(0)  |
    //        |     \/
    //        |     /\
    //        SW2(0)  |
    //              \ |
    //               b1
    //                |
    //                DMA

    // DMA
    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    // SW0
    VPURT.Task {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Subtract inputs(%buf2_0 as %arg4: memref<1x32x1x1xf16, [@CMX_NN, 0]>, %buf2_0 as %arg2: memref<1x32x1x1xf16, [@CMX_NN, 0]>) outputs(%buf2_0 as %arg3: memref<1x32x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x1x1xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg4, %arg2, %arg3) : memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>
      }
    }
    // SW1
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Subtract inputs(%buf2_0 as %arg4: memref<1x32x1x1xf16, [@CMX_NN, 0]>, %buf2_0 as %arg2: memref<1x32x1x1xf16, [@CMX_NN, 0]>) outputs(%buf2_0 as %arg3: memref<1x32x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x1x1xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg4, %arg2, %arg3) : memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>
      }
    }
    // SW2
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Subtract inputs(%buf2_0 as %arg4: memref<1x32x1x1xf16, [@CMX_NN, 0]>, %buf2_0 as %arg2: memref<1x32x1x1xf16, [@CMX_NN, 0]>) outputs(%buf2_0 as %arg3: memref<1x32x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x1x1xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg4, %arg2, %arg3) : memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>
      }
    }
    // DMA
    VPURT.Task waits(%bar1 : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    return %arg1: memref<1x3x64x64xf16, @DDR>

    // execution groups: [SW0][SW1][SW2]
    //        SW0(0) DMA
    //        |       |
    //        b1     b0
    //        |     / |
    //        SW1(0)  |
    //               \/  no barrier between SW1 and SW2
    //               /\  because grand child group does not exist
    //              / |
    //        SW2(0)  |
    //              \ |
    //               b2
    //                |
    //                DMA

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT:            VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    // CHECK:         VPUIP.NNDMA
    // SW0
    // CHECK:       VPURT.Task updates([[BAR1]] : !VPURT.Barrier) {
    // CHECK:         VPUIP.SW
    // SW1
    // CHECK:       VPURT.Task waits([[BAR0]], [[BAR1]] : !VPURT.Barrier, !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SW
    // SW2
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SW
    // CHECK:       VPURT.Task waits([[BAR2]] : !VPURT.Barrier) {
    // CHECK:         VPUIP.NNDMA

    // CHECK:       return
}
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8<0:254>:f16:0, {2.3954496608944391E-5:127,1.8652968519315946E-5:127,5.714536651851624E-6:127,9.2288640540415852E-6:127,5.2774985005536418E-5:127,1.0355251041922983E-5:127,4.7608623354453737E-5:127,4.2622483621432085E-5:127,6.3378041184793303E-5:127,9.0411328893946842E-6:127,3.7636343888410434E-5:127,9.1462623415969495E-6:127,3.0472522645484744E-5:127,1.3648806181992955E-6:127,8.4283783679872047E-5:127,5.5778683639886814E-5:127,1.6640490434301182E-5:127,5.4847537063238186E-5:127,9.0531476839320868E-5:127,1.0873389056348425E-5:127,8.4944597379429132E-5:127,5.5928868571604333E-5:127,7.9477865864911412E-5:127,1.5408973994217519E-5:127,2.7033287709153543E-5:127,7.3740801473302161E-6:127,4.5475997324064961E-5:127,4.5415923351377953E-5:127,1.4605484609528789E-5:127,8.4554116556963585E-6:127,1.9478985643762302E-5:127,4.6332051434854825E-6:127,9.6568911094365155E-6:127,1.3648806594488189E-4:127,4.8584825410617617E-6:127,1.0588037686085138E-5:127,9.6493818628506396E-6:127,1.0663130151943897E-5:127,3.8086898683562992E-5:127,4.142100416769193E-5:127,7.5805844284418062E-6:127,5.2684874046505907E-5:127,9.1462623415969495E-6:127,1.327634796382874E-5:127,6.1538275771253692E-6:127,2.9691561000553643E-5:127,5.6079053503321852E-5:127,1.4605484609528789E-5:127,2.8324878121924213E-5:127,6.1695969949557085E-5:127,5.0492174043430117E-5:127,2.3218590443528543E-5:127,1.5026002418337845E-5:127,4.4184406911294294E-5:127,4.0009265809547241E-5:127,2.0289984275036911E-5:127,2.0680465097502461E-5:127,5.5906340831846705E-6:127,1.6054769200602853E-5:127,1.0940972275621308E-5:127,2.3008331539124016E-5:127,3.4182090458907481E-5:127,9.4766691913754922E-6:127,8.9450145330954728E-5:127}>
!qElemType3 = !quant.uniform<u8:f16, 1.000000e+00:114>

module attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
config.PipelineOptions @Options {
  config.Option @config.MetadataMaxVariantCount : 2
  config.Option @config.MetadataMaxInvariantCount : 2
  config.Option @config.MetadataMaxKernelInvocationCount : 2
  config.Option @config.MetadataMaxKernelRangeCount : 2
}

module @VPU.SW {
  func.func private @builtin_Subtract(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>) attributes {VPU.kernel_code = "eltwise_sub.cpp", VPU.kernel_entry = "eltwise_sub"}
  func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @noInsertBarrierBetweenConsecutiveSWtasksIfPathExists
func.func @noInsertBarrierBetweenConsecutiveSWtasksIfPathExists(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %buf0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>
    %buf1 = VPURT.DeclareBuffer <DDR> <32> -> memref<1x3x64x64xf16, @DDR>
    %buf2_0 = VPURT.DeclareBuffer <CMX_NN> [0] <196608> -> memref<1x32x1x1xf16, [@CMX_NN, 0]>

    //  SWx(y): x - SWTask id, y - cluster id
    //
    //       DMA
    //        |
    //        b0
    //        |
    //      SW0(0)
    //        |
    //        b1
    //        |
    //       DMA
    //        |
    //        b2
    //        |
    //      SW1(0)
    //        |
    //      SW2(0)
    //        |
    //        b3
    //        |
    //       DMA

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    // SW0(0)
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Subtract inputs(%buf2_0 as %arg4: memref<1x32x1x1xf16, [@CMX_NN, 0]>, %buf2_0 as %arg2: memref<1x32x1x1xf16, [@CMX_NN, 0]>) outputs(%buf2_0 as %arg3: memref<1x32x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x1x1xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg4, %arg2, %arg3) : memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>
      }
    }
    // DMA
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    // SW1(0)
    VPURT.Task waits(%bar2 : !VPURT.Barrier) {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Subtract inputs(%buf2_0 as %arg4: memref<1x32x1x1xf16, [@CMX_NN, 0]>, %buf2_0 as %arg2: memref<1x32x1x1xf16, [@CMX_NN, 0]>) outputs(%buf2_0 as %arg3: memref<1x32x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x1x1xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg4, %arg2, %arg3) : memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>
      }
    }
    // SW2(0)
    VPURT.Task updates(%bar3: !VPURT.Barrier) {
      VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Subtract inputs(%buf2_0 as %arg4: memref<1x32x1x1xf16, [@CMX_NN, 0]>, %buf2_0 as %arg2: memref<1x32x1x1xf16, [@CMX_NN, 0]>) outputs(%buf2_0 as %arg3: memref<1x32x1x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x32x1x1xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg4, %arg2, %arg3) : memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>, memref<1x32x1x1xf16, [@CMX_NN, 0]>
      }
    }

    VPURT.Task waits(%bar3: !VPURT.Barrier) {
      VPUIP.NNDMA {port = 0 : i64} inputs(%buf0: memref<1x3x64x64xf16, @DDR>) outputs(%buf1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
   }

    return %arg1: memref<1x3x64x64xf16, @DDR>

    // execution groups: [SW0][SW1][SW2]
    // no barrier inserted between SW0(0) and SW1(0) because a barrier dependence is already present
    //       DMA
    //        |
    //        b0
    //        |
    //      SW0(0)
    //        |
    //        b1
    //        |
    //       DMA
    //        |
    //        b2
    //        |
    //      SW1(0)
    //        |
    //      SW2(0)
    //        |
    //        b3
    //        |
    //       DMA

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT:            VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // SW0
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         VPUIP.SW
    // DMA0
    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // SW1
    // CHECK:       VPURT.Task waits([[BAR2]] : !VPURT.Barrier) {
    // CHECK:         VPUIP.SW
    // SW2
    // CHECK:       VPURT.Task updates([[BAR3]] : !VPURT.Barrier) {
    // CHECK:         VPUIP.SW

    // CHECK:       VPURT.Task waits([[BAR3]] : !VPURT.Barrier) {
    // CHECK:         VPUIP.NNDMA

    // CHECK:       return
}
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8<0:254>:f16:0, {2.3954496608944391E-5:127,1.8652968519315946E-5:127,5.714536651851624E-6:127,9.2288640540415852E-6:127,5.2774985005536418E-5:127,1.0355251041922983E-5:127,4.7608623354453737E-5:127,4.2622483621432085E-5:127,6.3378041184793303E-5:127,9.0411328893946842E-6:127,3.7636343888410434E-5:127,9.1462623415969495E-6:127,3.0472522645484744E-5:127,1.3648806181992955E-6:127,8.4283783679872047E-5:127,5.5778683639886814E-5:127,1.6640490434301182E-5:127,5.4847537063238186E-5:127,9.0531476839320868E-5:127,1.0873389056348425E-5:127,8.4944597379429132E-5:127,5.5928868571604333E-5:127,7.9477865864911412E-5:127,1.5408973994217519E-5:127,2.7033287709153543E-5:127,7.3740801473302161E-6:127,4.5475997324064961E-5:127,4.5415923351377953E-5:127,1.4605484609528789E-5:127,8.4554116556963585E-6:127,1.9478985643762302E-5:127,4.6332051434854825E-6:127,9.6568911094365155E-6:127,1.3648806594488189E-4:127,4.8584825410617617E-6:127,1.0588037686085138E-5:127,9.6493818628506396E-6:127,1.0663130151943897E-5:127,3.8086898683562992E-5:127,4.142100416769193E-5:127,7.5805844284418062E-6:127,5.2684874046505907E-5:127,9.1462623415969495E-6:127,1.327634796382874E-5:127,6.1538275771253692E-6:127,2.9691561000553643E-5:127,5.6079053503321852E-5:127,1.4605484609528789E-5:127,2.8324878121924213E-5:127,6.1695969949557085E-5:127,5.0492174043430117E-5:127,2.3218590443528543E-5:127,1.5026002418337845E-5:127,4.4184406911294294E-5:127,4.0009265809547241E-5:127,2.0289984275036911E-5:127,2.0680465097502461E-5:127,5.5906340831846705E-6:127,1.6054769200602853E-5:127,1.0940972275621308E-5:127,2.3008331539124016E-5:127,3.4182090458907481E-5:127,9.4766691913754922E-6:127,8.9450145330954728E-5:127}>
!qElemType3 = !quant.uniform<u8:f16, 1.000000e+00:114>

module attributes {config.compilationMode = #config.compilation_mode<DefaultHW>} {
  config.PipelineOptions @Options {
    config.Option @config.MetadataMaxVariantCount : 18
    config.Option @config.MetadataMaxInvariantCount : 12
    config.Option @config.MetadataMaxKernelInvocationCount : 4
    config.Option @config.MetadataMaxKernelRangeCount : 4
  }

// CHECK-LABEL: @insertBarrierAtEndOfVariantLimit
func.func @insertBarrierAtEndOfVariantLimit(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x112x112xf16, @DDR>) -> memref<1x64x112x112xf16, @DDR> {
    // barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // buffers
    %3 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    %20 = VPURT.DeclareBuffer <CMX_NN> [0] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %21 = VPURT.DeclareBuffer <CMX_NN> [1] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 1]>
    %22 = VPURT.DeclareBuffer <CMX_NN> [2] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 2]>
    %23 = VPURT.DeclareBuffer <CMX_NN> [3] <140736> -> memref<64x1x1x4xsi32, [@CMX_NN, 3]>
    %25 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, [@CMX_NN, 0]>
    %29 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>
    %30 = VPURT.DeclareBuffer <CMX_NN> [1] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>
    %31 = VPURT.DeclareBuffer <CMX_NN> [2] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 2]>
    %32 = VPURT.DeclareBuffer <CMX_NN> [3] <141760> -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 3]>
    %41 = VPURT.DeclareBuffer <CMX_NN> [0] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %42 = VPURT.DeclareBuffer <CMX_NN> [1] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>
    %43 = VPURT.DeclareBuffer <CMX_NN> [2] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 2]>
    %44 = VPURT.DeclareBuffer <CMX_NN> [3] <130496> -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 3]>
    %45 = VPURT.DeclareBuffer <CMX_NN> [0] <75840> -> memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %46 = VPURT.DeclareBuffer <CMX_NN> [1] <75840> -> memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>
    %47 = VPURT.DeclareBuffer <CMX_NN> [2] <75840> -> memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 2]>
    %48 = VPURT.DeclareBuffer <CMX_NN> [3] <75840> -> memref<1x16x59x224x!qElemType3, #NHWC, [@CMX_NN, 3]>

    //  DPUx(y): x - DPUTask id, y - cluster id
    //
    //                   DMA
    //                    |
    //                    b0
    //          /                    \
    //  DPU1(0)...DPU7(0)    DPU8(1)...DPU14(1) (7 DPU tasks without barriers between them on two different FIFOs)
    //          \                    /
    //                    b1
    //                    |
    //                   DMA

    // DMA
    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%3 : memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%11 : memref<1x3x56x224xf16, [@CMX_NN, 0]>) -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    }

    // DPU1
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU2
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU3
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU4
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU5
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU6
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU7
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>) weight_table(%20 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, #NHWC, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }


    // DPU8
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU9
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU10
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU11
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU12
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU13
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    // DPU14
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 1]>) weight_table(%21 : memref<64x1x1x4xsi32, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, #NHWC, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, #NHWC, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%25 : memref<1x64x28x112xf16, [@CMX_NN, 0]>) outputs(%7 : memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>) -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    }

    return %arg1 : memref<1x64x112x112xf16, @DDR>

    // execution groups: [DPU1, DPU2, DPU3, DPU4, DPU5, DPU6][DPU7]
    //              [DPU8, DPU9, DPU10, DPU11, DPU12, DPU13][DPU14]
    //                   DMA
    //                    |
    //                    b0
    //       /\          /    \          /\        \         \
    //   DPU1..DPU3(0)  /      |     DPU8..DPU10(1) |         |
    //       \/        /       |         \/         |         |
    //       |        /        |         |          |         | 
    //       |       /\        |         |         /\         | 
    //       |   DPU4..DPU6(0) |         |    DPU11..DPU13(1) |
    //       |       \/        |         |         \/         | no barrier between DPU6 and DPU7 and between DPU13 and DPU14,
    //       |       |         |         |         |          | because grand child group does not exist
    //       |       |         /         |         |          /
    //       |       |  DPU7(0)          |         |   DPU14(1)
    //        \      \    |             /          /  /
    //                    b1
    //                    |
    //                    DMA

    // CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK-NOT:            VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // DMA
    // CHECK:       VPURT.Task updates([[BAR0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA

    // DPU1
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 0

    // DPU2
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 0

    // DPU3
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 0

    // DPU4
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 0

    // DPU5
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 0

    // DPU6
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 0

    // DPU7
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 0

    // DPU8
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 1
    // DPU9
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 1
    // DPU10
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 1
    // DPU11
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 1
    // DPU12
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 1
    // DPU13
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 1
    // DPU14
    // CHECK:       VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier)
    // CHECK:         DPUTask {cluster_id = 1

    // CHECK:       VPURT.Task waits([[BAR1]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA

    // CHECK:       return %arg1 : memref<1x64x112x112xf16, @DDR>
}
}
