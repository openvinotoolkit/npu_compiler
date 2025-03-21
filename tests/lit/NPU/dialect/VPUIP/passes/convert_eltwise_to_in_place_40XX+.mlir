//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-eltwise-to-in-place --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.014045423619887408:72>
!qElemType1 = !quant.uniform<u8<0:254>:f16:0, {0.0032472896763658899:127,0.0018755442748858233:127,0.0017921328544616699:127,0.0013221445984727754:127,0.0012499020090253334:127,0.0021332664283241812:127,0.0030072108497769816:127,0.0038371799499031128:127,0.0017091818447188131:127,0.0038851703715136669:127,0.002740082074338057:127,0.0015325846634511872:127,0.0016993074201223418:127,0.0026474710055223601:127,0.0024849516669596273:127,0.0015312655000236091:127,0.0014897457023305216:127,0.002086720128697673:127,0.0015050239685013538:127,0.0019807660673546979:127,0.0040369019733639214:127,0.0020443703245928905:127,0.003165307242100633:127,0.0010913987094023096:127,0.0020088358657566583:127,0.0023128066475935809:127,0.0026037143910025047:127,0.0013339638006030106:127,0.0013417860889059351:127,0.0036693640111938237:127,0.001358872206192317:127,0.0027316407425197089:127,0.0028161950937406286:127,0.0021381059030848226:127,0.0030168604663037878:127,0.0021637053940239855:127,0.0023203191794748381:127,0.0010942800307837059:127,0.0012016238894049576:127,0.0034935432156239909:127,0.0034669330270271602:127,0.0029516764513150915:127,0.0026073474583663339:127,0.0016469043774867622:127,0.0026913641944644956:127,0.0030008645977560931:127,0.0015427400981347392:127,0.0011106365778314785:127,0.0021462722087469627:127,0.0035388291351438508:127,0.003522398903613954:127,0.0040410879090076353:127,0.0024811355617102675:127,0.0032145826835331954:127,0.0025306071822098859:127,0.0010318539039356502:127,0.0032657457618262822:127,0.0026374313775009996:127,0.002608869019455797:127,0.0037320143594516543:127,0.0010950212168881273:127,0.0019200501714165756:127,0.0015697441701813945:127,0.0010532591286606676:127}>
!qElemType2 = !quant.uniform<u8:f16, 0.027439035153856463:128>
!qElemType3 = !quant.uniform<u8:f16, 0.01918038901160745:88>

!DistributedType = !VPUIP.DistributedBuffer<
    1x64x128x128x!qElemType3,
    #NHWC, @CMX_NN, {
        mode = "OVERLAPPED",
        num_tiles = [1, 1, 2, 1],
        kernel = [3, 3],
        pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1],
        num_clusters = 2 : i64,
        uniform_distributed_segments}>

!DistributedType2 = !VPUIP.DistributedBuffer<
    1x64x128x128x!qElemType2,
    #NHWC, @CMX_NN, {
        mode = "OVERLAPPED",
        num_tiles = [1, 1, 2, 1],
        kernel = [1, 1],
        pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1],
        num_clusters = 2 : i64,
        uniform_distributed_segments}>

// CHECK:    func @InplaceEltwiseUnequalTensorSize(%arg0: memref<1x64x128x128x!qElemType, #NHWC, @CMX_NN>, %arg1: memref<64x64x3x3x!qElemType1, #NHWC, @CMX_NN>, %arg2: memref<64x1x1x4xsi32, @CMX_NN>)
func.func @InplaceEltwiseUnequalTensorSize(%activation: memref<1x64x128x128x!qElemType, #NHWC, @CMX_NN>, %weights: memref<64x64x3x3x!qElemType1, #NHWC, @CMX_NN>,  %weights_table: memref<64x1x1x4xsi32, @CMX_NN>) -> !DistributedType2 {
    %conv_cmx_outbuf = VPURT.AllocDistributed -> !DistributedType
    %0 = VPUIP.NCEClusterTiling inputs(%activation as %arg2: memref<1x64x128x128x!qElemType, #NHWC, @CMX_NN>,
                                %weights as %arg3: memref<64x64x3x3x!qElemType1, #NHWC, @CMX_NN>,
                                %weights_table as %arg4: memref<64x1x1x4xsi32, @CMX_NN>)
                                outputs(%conv_cmx_outbuf as %arg5: memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>) -> !DistributedType {
      %235 = VPUIP.NCEClusterTask {
              kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
              kernel_size = [3, 3],
              kernel_strides = [1, 1],
              minimumHardwareExecutionCost = 4294967300 : i64,
              task_type = #VPUIP.nce_task_type<CONV>
          }
          input(%arg2 : memref<1x64x128x128x!qElemType, #NHWC, @CMX_NN>)
          weights(%arg3 : memref<64x64x3x3x!qElemType1, #NHWC, @CMX_NN>)
          weight_table(%arg4 : memref<64x1x1x4xsi32, @CMX_NN>)
          parent_input(%arg2 : memref<1x64x128x128x!qElemType, #NHWC, @CMX_NN>)
          parent_output(%arg5 : memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
          outputs(%arg5 : memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>) -> memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN> variants : {
              DPUTask {cluster_id = 0 : i64, inEnd = [127, 64, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 63, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>}
              DPUTask {cluster_id = 1 : i64, inEnd = [127, 64, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 63, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>}
          } PPE : {
            PPETask {ppe = #VPU.PPEStub<>}
          }
    }

    %ddr_buf = memref.alloc() : memref<1x64x128x128x!qElemType3, #NHWC>
    %1 = VPUIP.NCEClusterTiling inputs(%0 as %arg2: memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
                                outputs(%ddr_buf as %arg3: memref<1x64x128x128x!qElemType3, #NHWC>) -> memref<1x64x128x128x!qElemType3, #NHWC> {
      %235 = VPUIP.Copy
          inputs(%arg2 : memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
          outputs(%arg3 : memref<1x64x128x128x!qElemType3, #NHWC>) -> memref<1x64x128x128x!qElemType3, #NHWC>
    }

    %eltwise_cmx_inbuf = VPURT.AllocDistributed -> !DistributedType
    %2 = VPUIP.NCEClusterTiling inputs(%1 as %arg2: memref<1x64x128x128x!qElemType3, #NHWC>)
                                outputs(%eltwise_cmx_inbuf as %arg3: memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>) -> !DistributedType {
      %235 = VPUIP.Copy
          inputs(%arg2 : memref<1x64x128x128x!qElemType3, #NHWC>)
          outputs(%arg3 : memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>) -> memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>
    }

    %eltwise_cmx_outbuf = VPURT.AllocDistributed -> !DistributedType2
    %eltwise = VPUIP.NCEClusterTiling inputs(%2 as %arg2: memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>,
                                             %2 as %arg3: memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
                                      outputs(%eltwise_cmx_outbuf as %arg4: memref<1x64x128x128x!qElemType2, #NHWC, @CMX_NN>) -> !DistributedType2 {
      %235 = VPUIP.NCEClusterTask {
              is_inplace = true,
              minimumHardwareExecutionCost = 4294967300 : i64,
              task_type = #VPUIP.nce_task_type<ELTWISE>
          }
          input(%arg2 : memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
          weights(%arg3 : memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
          parent_input(%arg2 : memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
          parent_output(%arg4 : memref<1x64x128x128x!qElemType2, #NHWC, @CMX_NN>)
          outputs(%arg4 : memref<1x64x128x128x!qElemType2, #NHWC, @CMX_NN>) -> memref<1x64x128x128x!qElemType2, #NHWC, @CMX_NN> variants : {
              DPUTask {cluster_id = 0 : i64, inEnd = [127, 63, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 63, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
              DPUTask {cluster_id = 1 : i64, inEnd = [127, 64, 63], inStart = [0, 1, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 63, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
          } PPE : {
            PPETask {ppe = #VPU.PPEStub<>}
          }
    }

    return %eltwise : !DistributedType2

    // CHECK: [[CONV_CMX_OUTBUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:                    -> !VPUIP.DistributedBuffer<1x64x128x128x!qElemType3, #NHWC, @CMX_NN,
    // CHECK-SAME:                        {mode = "OVERLAPPED",
    // CHECK-SAME:                         num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:                         kernel = [3, 3],
    // CHECK-SAME:                         pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:                         strides = [1, 1],
    // CHECK-SAME:                         num_clusters = 2 : i64,
    // CHECK-SAME:                         uniform_distributed_segments}>

    // CHECK: [[CONV:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:                inputs(%arg0 as [[ARG3:%.+]]: memref<1x64x128x128x!qElemType, #NHWC, @CMX_NN>,
    // CHECK-SAME:                       %arg1 as [[ARG4:%.+]]: memref<64x64x3x3x!qElemType1, #NHWC, @CMX_NN>,
    // CHECK-SAME:                       %arg2 as [[ARG5:%.+]]: memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:                outputs([[CONV_CMX_OUTBUF]] as [[ARG6:%.+]]: memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
    // CHECK-SAME:                    -> !VPUIP.DistributedBuffer<1x64x128x128x!qElemType3, #NHWC, @CMX_NN,
    // CHECK-SAME:                        {mode = "OVERLAPPED",
    // CHECK-SAME:                         num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:                         kernel = [3, 3],
    // CHECK-SAME:                         pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:                         strides = [1, 1],
    // CHECK-SAME:                         num_clusters = 2 : i64,
    // CHECK-SAME:                         uniform_distributed_segments}> {
    // CHECK: [[INNER_CONV:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:                        {kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:                         kernel_size = [3, 3],
    // CHECK-SAME:                         kernel_strides = [1, 1],
    // CHECK-SAME:                         minimumHardwareExecutionCost = 4294967300 : i64,
    // CHECK-SAME:                         task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:                input([[ARG3]] : memref<1x64x128x128x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:                weights([[ARG4]] : memref<64x64x3x3x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:                weight_table([[ARG5]] : memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:                parent_input([[ARG3]] : memref<1x64x128x128x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:                parent_output([[ARG6]] : memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
    // CHECK-SAME:                outputs([[ARG6]] : memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>) -> memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN> variants : {
    // CHECK:                 DPUTask {cluster_id = 0 : i64, inEnd = [127, 64, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 63, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>}
    // CHECK:                 DPUTask {cluster_id = 1 : i64, inEnd = [127, 64, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [127, 63, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>}
    // CHECK:               } PPE : {
    // CHECK:                 PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:               }
    // CHECK:             }

    // CHECK: [[DDR_BUF:%.+]] = memref.alloc() : memref<1x64x128x128x!qElemType3, #NHWC>
    // CHECK: [[CONV_COPY_OUT:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:                inputs([[CONV]] as [[ARG3:%.+]]: memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
    // CHECK-SAME:                outputs([[DDR_BUF]] as [[ARG4:%.+]]: memref<1x64x128x128x!qElemType3, #NHWC>) -> memref<1x64x128x128x!qElemType3, #NHWC> {
    // CHECK: [[INNER_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:                inputs([[ARG3]] : memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
    // CHECK-SAME:                outputs([[ARG4]] : memref<1x64x128x128x!qElemType3, #NHWC>) -> memref<1x64x128x128x!qElemType3, #NHWC>
    // CHECK:             }

    // CHECK: [[ELTWISE_CMX_INBUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:                    -> !VPUIP.DistributedBuffer<1x64x128x128x!qElemType3, #NHWC, @CMX_NN,
    // CHECK-SAME:                        {mode = "OVERLAPPED",
    // CHECK-SAME:                         num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:                         kernel = [3, 3],
    // CHECK-SAME:                         pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:                         strides = [1, 1],
    // CHECK-SAME:                         num_clusters = 2 : i64,
    // CHECK-SAME:                         uniform_distributed_segments}>
    // CHECK: [[VIEW:%.+]] = VPUIP.ViewOp [[ELTWISE_CMX_INBUF]] : !VPUIP.DistributedBuffer<1x64x128x128x!qElemType3, #NHWC, @CMX_NN,
    // CHECK-SAME:                        {mode = "OVERLAPPED",
    // CHECK-SAME:                         num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:                         kernel = [3, 3],
    // CHECK-SAME:                         pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:                         strides = [1, 1],
    // CHECK-SAME:                         num_clusters = 2 : i64,
    // CHECK-SAME:                         uniform_distributed_segments}>
    // CHECK-SAME:                                             to !VPUIP.DistributedBuffer<1x64x128x128x!qElemType2, #NHWC, @CMX_NN,
    // CHECK-SAME:                        {mode = "OVERLAPPED",
    // CHECK-SAME:                         num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:                         kernel = [1, 1],
    // CHECK-SAME:                         pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                         strides = [1, 1],
    // CHECK-SAME:                         num_clusters = 2 : i64,
    // CHECK-SAME:                         uniform_distributed_segments}>

    // CHECK: [[ELTWISE_COPY_IN:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:                inputs([[CONV_COPY_OUT]] as [[ARG3:%.+]]: memref<1x64x128x128x!qElemType3, #NHWC>)
    // CHECK-SAME:                outputs([[ELTWISE_CMX_INBUF]] as [[ARG4:%.+]]: memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
    // CHECK-SAME:                    -> !VPUIP.DistributedBuffer<1x64x128x128x!qElemType3, #NHWC, @CMX_NN,
    // CHECK-SAME:                        {mode = "OVERLAPPED",
    // CHECK-SAME:                         num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:                         kernel = [3, 3],
    // CHECK-SAME:                         pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:                         strides = [1, 1],
    // CHECK-SAME:                         num_clusters = 2 : i64,
    // CHECK-SAME:                         uniform_distributed_segments}> {
    // CHECK: [[INNER_COPY2:%.+]] = VPUIP.Copy
    // CHECK-SAME:                inputs([[ARG3]] : memref<1x64x128x128x!qElemType3, #NHWC>)
    // CHECK-SAME:                outputs([[ARG4]] : memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>) -> memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>
    // CHECK:             }

    // CHECK: [[ELTWISE:%.+]] = VPUIP.NCEClusterTiling
    // CHECK-SAME:                inputs([[ELTWISE_COPY_IN]] as [[ARG3:%.+]]: memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>,
    // CHECK-SAME:                       [[ELTWISE_COPY_IN]] as [[ARG4:%.+]]: memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
    // CHECK-SAME:                outputs([[VIEW]] as [[ARG5:%.+]]: memref<1x64x128x128x!qElemType2, #NHWC, @CMX_NN>)
    // CHECK-SAME:                    -> !VPUIP.DistributedBuffer<1x64x128x128x!qElemType2, #NHWC, @CMX_NN,
    // CHECK-SAME:                        {mode = "OVERLAPPED",
    // CHECK-SAME:                         num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:                         kernel = [1, 1],
    // CHECK-SAME:                         pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                         strides = [1, 1],
    // CHECK-SAME:                         num_clusters = 2 : i64,
    // CHECK-SAME:                         uniform_distributed_segments}> {
    // CHECK: [[INNER_OP:%.+]] = VPUIP.NCEClusterTask {is_inplace = true, minimumHardwareExecutionCost = 4294967300 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:                input([[ARG3]] : memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
    // CHECK-SAME:                weights([[ARG4]] : memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
    // CHECK-SAME:                parent_input([[ARG3]] : memref<1x64x128x128x!qElemType3, #NHWC, @CMX_NN>)
    // CHECK-SAME:                parent_output([[ARG5]] : memref<1x64x128x128x!qElemType2, #NHWC, @CMX_NN>)
    // CHECK-SAME:                outputs([[ARG5]] : memref<1x64x128x128x!qElemType2, #NHWC, @CMX_NN>) -> memref<1x64x128x128x!qElemType2, #NHWC, @CMX_NN> variants : {
    // CHECK:                 DPUTask {cluster_id = 0 : i64, inEnd = [127, 63, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 63, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:                 DPUTask {cluster_id = 1 : i64, inEnd = [127, 64, 63], inStart = [0, 1, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 63, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:                 } PPE : {
    // CHECK:                   PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:                 }
    // CHECK:             }

    // CHECK: return [[ELTWISE]] : !VPUIP.DistributedBuffer<1x64x128x128x!qElemType2, #NHWC, @CMX_NN,
    // CHECK-SAME:                        {mode = "OVERLAPPED",
    // CHECK-SAME:                         num_tiles = [1, 1, 2, 1],
    // CHECK-SAME:                         kernel = [1, 1],
    // CHECK-SAME:                         pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:                         strides = [1, 1],
    // CHECK-SAME:                         num_clusters = 2 : i64,
    // CHECK-SAME:                         uniform_distributed_segments}>

  }
