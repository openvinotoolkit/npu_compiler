//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --insert-copy-for-eltwise-in-place-input %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0047491008160161037:146>

!qTypeCMX = memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>
!qTypeDDR = memref<1x32x103x512x!qElemType, #NHWC>
!qTypeConvCMX = memref<1x32x206x256x!qElemType, #NHWC, @CMX_NN>
!qTypeConvDDR = memref<1x32x206x256x!qElemType, #NHWC>

!DistributedType1 = !VPUIP.DistributedBuffer<
    1x32x103x512x!qElemType,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

!DistributedType2 = !VPUIP.DistributedBuffer<
    1x32x206x256x!qElemType,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

// CHECK-LABEL: InsertSpillingCopiesOnSecondInputWithCopyParent
func.func @InsertSpillingCopiesOnSecondInputWithCopyParent(
    %weights0 : memref<32x32x1x1xf16, #NHWC, @CMX_NN>)
      -> (!DistributedType1, !DistributedType2, !DistributedType1) {
    %in1 = memref.alloc() : !qTypeDDR
    %in2 = memref.alloc() : !qTypeDDR
    %eltwiseIn1CMXBuff = VPURT.AllocDistributed -> !DistributedType1
    %eltwiseIn1CMX = VPUIP.Copy inputs(%in1 : !qTypeDDR) outputs(%eltwiseIn1CMXBuff : !DistributedType1) -> !DistributedType1

    %eltwiseIn2CMXBuff = VPURT.AllocDistributed -> !DistributedType1
    %eltwiseIn2CMX = VPUIP.Copy inputs(%in2 : !qTypeDDR) outputs(%eltwiseIn2CMXBuff : !DistributedType1) -> !DistributedType1

    %eltwise = VPUIP.NCEClusterTask {
      is_inplace = true,
      minimumHardwareExecutionCost = 21125 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%eltwiseIn1CMX : !DistributedType1)
        weights(%eltwiseIn2CMX : !DistributedType1)
        parent_input(%eltwiseIn1CMX : !DistributedType1)
        parent_output(%eltwiseIn2CMXBuff : !DistributedType1)
        outputs(%eltwiseIn2CMXBuff : !DistributedType1) -> !DistributedType1
    variants : {
      DPUTask {
        cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
        outEnd = [511, 51, 31], outStart = [0, 0, 0],
        pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
      DPUTask {
        cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
        outEnd = [511, 51, 31], outStart = [0, 0, 0],
        pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
    } PPE : {
      PPETask {
        ppe = #VPU.PPEStub<>
        }
    }

    %conv0InCMXBuff = VPURT.AllocDistributed -> !DistributedType2
    %convInDDR = VPUIP.ShapeCast {shape = [1, 32, 206, 256]} inputs(%in1 : !qTypeDDR) -> !qTypeConvDDR
    %conv0InCMX = VPUIP.Copy inputs(%convInDDR : !qTypeConvDDR) outputs(%conv0InCMXBuff : !DistributedType2) -> !DistributedType2

    %convOutBuff0 = VPURT.AllocDistributed -> !DistributedType2
    %conv0 = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
      }
      input(%conv0InCMX : !DistributedType2)
      weights(%weights0 : memref<32x32x1x1xf16, #NHWC, @CMX_NN>)
      parent_input(%conv0InCMX : !DistributedType2)
      parent_output(%convOutBuff0 : !DistributedType2)
      outputs(%convOutBuff0 : !DistributedType2)
          -> !DistributedType2 variants : {
        DPUTask {
          cluster_id = 0 : i64,
          inEnd = [255, 102, 31], inStart = [0, 0, 0],
          mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
          outEnd = [255, 102, 31], outStart = [0, 0, 0],
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
        DPUTask {
          cluster_id = 1 : i64,
          inEnd = [255, 102, 31], inStart = [0, 102, 0],
          mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
          outEnd = [255, 102, 31], outStart = [0, 0, 0],
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
      } PPE : {
        PPETask {
          ppe = #VPU.PPEStub<>
        }
      }


    %convOutBuff1 = VPURT.AllocDistributed -> !DistributedType1
    %conv1 = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
      }
      input(%eltwiseIn2CMX : !DistributedType1)
      weights(%weights0 : memref<32x32x1x1xf16, #NHWC, @CMX_NN>)
      parent_input(%eltwiseIn2CMX : !DistributedType1)
      parent_output(%convOutBuff1 : !DistributedType1)
      outputs(%convOutBuff1 : !DistributedType1)
          -> !DistributedType1 variants : {
        DPUTask {
          cluster_id = 0 : i64,
          inEnd = [511, 51, 31], inStart = [0, 0, 0],
          mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
          outEnd = [511, 51, 31], outStart = [0, 0, 0],
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
        DPUTask {
          cluster_id = 1 : i64,
          inEnd = [511, 51, 31], inStart = [0, 52, 0],
          mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
          outEnd = [511, 51, 31], outStart = [0, 0, 0],
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
      } PPE : {
        PPETask {
          ppe = #VPU.PPEStub<>
        }
      }

    return %eltwise, %conv0, %conv1 : !DistributedType1, !DistributedType2, !DistributedType1

  // CHECK: [[DDR_BUF_0:%.+]] = memref.alloc
  // CHECK: [[DDR_BUF_1:%.+]] = memref.alloc

  // CHECK: [[ELTWISE_1_INPUT_BUF_1:%.+]] = VPURT.AllocDistributed
  // CHECK: [[ELTWISE_1_INPUT_1:%.+]] = VPUIP.Copy
  // CHECK-SAME: inputs([[DDR_BUF_0]] : memref<1x32x103x512x!qElemType, #NHWC>)
  // CHECK-SAME: outputs([[ELTWISE_1_INPUT_BUF_1]] : !VPUIP.DistributedBuffer<1x32x103x512x!qElemType, #NHWC, @CMX_NN

  // CHECK: [[ORIG_ELTWISE_1_INPUT_BUF_2:%.+]] = VPURT.AllocDistributed
  // CHECK: [[ORIG_ELTWISE_1_INPUT_2:%.+]] = VPUIP.Copy
  // CHECK-SAME: inputs([[DDR_BUF_1]] : memref<1x32x103x512x!qElemType, #NHWC>)
  // CHECK-SAME: outputs([[ORIG_ELTWISE_1_INPUT_BUF_2]] : !VPUIP.DistributedBuffer<1x32x103x512x!qElemType, #NHWC, @CMX_NN

  // CHECK: [[SPILL_DDR_BUF_1:%.+]] = memref.alloc
  // CHECK: [[SPILL_TO_DDR:%.+]] = VPUIP.Copy
  // CHECK-SAME: inputs([[ORIG_ELTWISE_1_INPUT_2]] : !VPUIP.DistributedBuffer<1x32x103x512x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
  // CHECK-SAME: outputs([[SPILL_DDR_BUF_1]] : memref<1x32x103x512x!qElemType, #NHWC, @DDR>)

  // CHECK: [[ELTWISE_1_INPUT_BUF_2:%.+]] = VPURT.AllocDistributed
  // CHECK: [[ELTWISE_1_INPUT_2:%.+]] = VPUIP.Copy
  // CHECK-SAME: inputs([[SPILL_TO_DDR]] : memref<1x32x103x512x!qElemType, #NHWC, @DDR>)
  // CHECK-SAME: outputs([[ELTWISE_1_INPUT_BUF_2]] : !VPUIP.DistributedBuffer<1x32x103x512x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

  // CHECK: [[ELTWISE:%.+]] = VPUIP.NCEClusterTask
  // CHECK-SAME:      is_inplace = true
  // CHECK-SAME:  input([[ELTWISE_1_INPUT_1]] : !VPUIP.DistributedBuffer<1x32x103x512x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
  // CHECK-SAME:         weights([[ELTWISE_1_INPUT_2]] : !VPUIP.DistributedBuffer<1x32x103x512x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
  // CHECK-SAME:  outputs([[ELTWISE_1_INPUT_BUF_2]] : !VPUIP.DistributedBuffer<1x32x103x512x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0047491008160161037:146>

!qTypeCMX = memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>
!qTypeDDR = memref<1x32x103x512x!qElemType, #NHWC>
!qTypeEltCMX = memref<1x32x52x512x!qElemType, #NHWC, @CMX_NN>
!qTypeEltDDR = memref<1x32x52x512x!qElemType, #NHWC>

!DistributedType1 = !VPUIP.DistributedBuffer<
    1x32x103x512x!qElemType,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 1, 2],
        num_clusters = 2 : i64
}>

!DistributedType2 = !VPUIP.DistributedBuffer<
    1x32x52x512x!qElemType,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 1, 2],
        num_clusters = 2 : i64
}>

!DistributedTypeSubview = !VPUIP.DistributedBuffer<
    1x32x52x512x!qElemType,
    {order = #NHWC, strides = [1687552, 1, 16384, 32]}, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 1, 2],
        num_clusters = 2 : i64
}>

// CHECK-LABEL: InsertSpillingCopiesOnFirstInputWithViewParent
func.func @InsertSpillingCopiesOnFirstInputWithViewParent(
    %weights0 : memref<32x32x1x1xf16, #NHWC, @CMX_NN>)
      -> (!DistributedTypeSubview, !DistributedType1) {
    %in1 = memref.alloc() : !qTypeDDR
    %in2 = memref.alloc() : !qTypeEltDDR
    %eltwiseIn1CMXBuff = VPURT.AllocDistributed -> !DistributedType1
    %eltwiseIn1CMX = VPUIP.Copy inputs(%in1 : !qTypeDDR) outputs(%eltwiseIn1CMXBuff : !DistributedType1) -> !DistributedType1

    %eltwiseIn2CMXBuff = VPURT.AllocDistributed -> !DistributedType2
    %eltwiseIn2CMX = VPUIP.Copy inputs(%in2 : !qTypeEltDDR) outputs(%eltwiseIn2CMXBuff : !DistributedType2) -> !DistributedType2

    %eltwiseInput1Subview = VPUIP.SubView %eltwiseIn1CMX [0, 0, 0, 0] [1, 32, 52, 512] : !DistributedType1 to !DistributedTypeSubview

    %eltwise = VPUIP.NCEClusterTask {
      is_inplace = true,
      minimumHardwareExecutionCost = 21125 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%eltwiseInput1Subview : !DistributedTypeSubview)
        weights(%eltwiseIn2CMX : !DistributedType2)
        parent_input(%eltwiseInput1Subview : !DistributedTypeSubview)
        parent_output(%eltwiseInput1Subview : !DistributedTypeSubview)
        outputs(%eltwiseInput1Subview : !DistributedTypeSubview) -> !DistributedTypeSubview
    variants : {
      DPUTask {
        cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
        outEnd = [255, 51, 32], outStart = [0, 0, 0],
        pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
      DPUTask {
        cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
        outEnd = [255, 51, 32], outStart = [0, 0, 0],
        pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
    } PPE : {
      PPETask {
        ppe = #VPU.PPEStub<>
        }
    }

    %convOutBuff0 = VPURT.AllocDistributed -> !DistributedType1
    %conv0 = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
    }
    input(%eltwiseIn1CMX : !DistributedType1)
    weights(%weights0 : memref<32x32x1x1xf16, #NHWC, @CMX_NN>)
    parent_input(%eltwiseIn1CMX : !DistributedType1)
    parent_output(%convOutBuff0 : !DistributedType1)
    outputs(%convOutBuff0 : !DistributedType1)
        -> !DistributedType1 variants : {
      DPUTask {
        cluster_id = 0 : i64,
        inEnd = [255, 102, 31], inStart = [0, 0, 0],
        mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
        outEnd = [255, 102, 31], outStart = [0, 0, 0],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
      }
      DPUTask {
        cluster_id = 1 : i64,
        inEnd = [255, 102, 31], inStart = [0, 0, 0],
        mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
        outEnd = [255, 102, 31], outStart = [0, 0, 0],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
      }
    } PPE : {
      PPETask {
        ppe = #VPU.PPEStub<>
      }
    }

    return %eltwise, %conv0 : !DistributedTypeSubview, !DistributedType1

  // CHECK: [[DDR_BUF_0:%.+]] = memref.alloc
  // CHECK: [[DDR_BUF_1:%.+]] = memref.alloc

  // CHECK: [[ELTWISE_1_INPUT_FULL_BUF_1:%.+]] = VPURT.AllocDistributed
  // CHECK: [[ELTWISE_1_INPUT_FULL_1:%.+]] = VPUIP.Copy
  // CHECK-SAME: inputs([[DDR_BUF_0]] : memref<1x32x103x512x!qElemType, #NHWC>)
  // CHECK-SAME: outputs([[ELTWISE_1_INPUT_FULL_BUF_1]] : !VPUIP.DistributedBuffer<1x32x103x512x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>)

  // CHECK: [[ELTWISE_1_INPUT_BUF_2:%.+]] = VPURT.AllocDistributed
  // CHECK: [[ELTWISE_1_INPUT_2:%.+]] = VPUIP.Copy
  // CHECK-SAME: inputs([[DDR_BUF_1]] : memref<1x32x52x512x!qElemType, #NHWC>)
  // CHECK-SAME: outputs([[ELTWISE_1_INPUT_BUF_2]] : !VPUIP.DistributedBuffer<1x32x52x512x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>)

  // CHECK: [[SUBVIEW_BUFF:%.+]] = VPUIP.SubView [[ELTWISE_1_INPUT_FULL_1]] [0, 0, 0, 0] [1, 32, 52, 512]
  // CHECK-SAME:        to !VPUIP.DistributedBuffer<1x32x52x512x!qElemType, {order = #NHWC, strides = [1687552, 1, 16384, 32]}, @CMX_NN

  // CHECK: [[SPILL_DDR_BUF_1:%.+]] = memref.alloc
  // CHECK: [[SPILL_TO_DDR:%.+]] = VPUIP.Copy
  // CHECK-SAME: inputs([[SUBVIEW_BUFF]] : !VPUIP.DistributedBuffer<1x32x52x512x!qElemType, {order = #NHWC, strides = [1687552, 1, 16384, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>)
  // CHECK-SAME: outputs([[SPILL_DDR_BUF_1]] : memref<1x32x52x512x!qElemType, #NHWC, @DDR>)

  // CHECK: [[ELTWISE_1_INPUT_BUF_2:%.+]] = VPURT.AllocDistributed
  // CHECK: [[ELTWISE_1_INPUT_1:%.+]] = VPUIP.Copy
  // CHECK-SAME: inputs([[SPILL_TO_DDR]] : memref<1x32x52x512x!qElemType, #NHWC, @DDR>)
  // CHECK-SAME: outputs([[ELTWISE_1_INPUT_BUF_2]] : !VPUIP.DistributedBuffer<1x32x52x512x!qElemType, {order = #NHWC, strides = [1687552, 1, 16384, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>)


  // CHECK: [[ELTWISE:%.+]] = VPUIP.NCEClusterTask
  // CHECK-SAME:      is_inplace = true
  // CHECK-SAME:  input([[ELTWISE_1_INPUT_BUF_2]] : !VPUIP.DistributedBuffer<1x32x52x512x!qElemType, {order = #NHWC, strides = [1687552, 1, 16384, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>)
  // CHECK-SAME:         weights([[ELTWISE_1_INPUT_2]] : !VPUIP.DistributedBuffer<1x32x52x512x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>
  // CHECK-SAME:  outputs([[ELTWISE_1_INPUT_BUF_2]] : !VPUIP.DistributedBuffer<1x32x52x512x!qElemType, {order = #NHWC, strides = [1687552, 1, 16384, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 1, 2], num_clusters = 2 : i64}>)

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0047491008160161037:146>

!qTypeCMX = memref<1x32x52x256x!qElemType, #NHWC, @CMX_NN>
!qTypeDDR = memref<1x32x52x256x!qElemType, #NHWC>
!qTypeEltCMX = memref<1x32x26x256x!qElemType, {order = #NHWC, strides = [425984, 1, 8192, 32]}, @CMX_NN>
!qTypeEltDDR = memref<1x32x26x256x!qElemType, #NHWC>

// CHECK-LABEL: InsertSpillingCopiesOnFirstInputWithViewParentSingleCluster
func.func @InsertSpillingCopiesOnFirstInputWithViewParentSingleCluster(
    %weights0 : memref<32x32x1x1xf16, #NHWC, @CMX_NN>)
      -> (!qTypeEltCMX, !qTypeCMX) {
    %in1 = memref.alloc() : !qTypeDDR
    %in2 = memref.alloc() : !qTypeEltDDR
    %eltwiseIn1CMXBuff = memref.alloc() : !qTypeCMX
    %eltwiseIn1CMX = VPUIP.Copy inputs(%in1 : !qTypeDDR) outputs(%eltwiseIn1CMXBuff : !qTypeCMX) -> !qTypeCMX

    %eltwiseIn2CMXBuff = memref.alloc() : !qTypeEltCMX
    %eltwiseIn2CMX = VPUIP.Copy inputs(%in2 : !qTypeEltDDR) outputs(%eltwiseIn2CMXBuff : !qTypeEltCMX) -> !qTypeEltCMX

    %eltwiseInput1Subview = VPUIP.SubView %eltwiseIn1CMX [0, 0, 0, 0] [1, 32, 26, 256] : !qTypeCMX to !qTypeEltCMX

    %eltwise = VPUIP.NCEClusterTask {
        is_inplace = true,
        minimumHardwareExecutionCost = 21125 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%eltwiseInput1Subview : !qTypeEltCMX)
        weights(%eltwiseIn2CMX : !qTypeEltCMX)
        parent_input(%eltwiseInput1Subview : !qTypeEltCMX)
        parent_output(%eltwiseInput1Subview : !qTypeEltCMX)
        outputs(%eltwiseInput1Subview : !qTypeEltCMX) -> !qTypeEltCMX
    variants : {
        DPUTask {
        cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
        outEnd = [255, 25, 16], outStart = [0, 0, 0],
        pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
        DPUTask {
        cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
        outEnd = [255, 25, 16], outStart = [0, 0, 0],
        pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
    } PPE : {
        PPETask {
        ppe = #VPU.PPEStub<>
        }
    }

    %convOutBuff0 = memref.alloc() : !qTypeCMX
    %conv0 = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
      }
      input(%eltwiseIn1CMX : !qTypeCMX)
      weights(%weights0 : memref<32x32x1x1xf16, #NHWC, @CMX_NN>)
      parent_input(%eltwiseIn1CMX : !qTypeCMX)
      parent_output(%convOutBuff0 : !qTypeCMX)
      outputs(%convOutBuff0 : !qTypeCMX)
          -> !qTypeCMX variants : {
        DPUTask {
          cluster_id = 0 : i64,
          inEnd = [255, 51, 15], inStart = [0, 0, 0],
          mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
          outEnd = [255, 51, 15], outStart = [0, 0, 0],
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
        DPUTask {
          cluster_id = 1 : i64,
          inEnd = [255, 51, 15], inStart = [0, 0, 0],
          mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
          outEnd = [255, 51, 15], outStart = [0, 0, 0],
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
      } PPE : {
        PPETask {
          ppe = #VPU.PPEStub<>
        }
    }
    return %eltwise, %conv0 : !qTypeEltCMX, !qTypeCMX
    // CHECK: [[DDR_BUF_0:%.+]] = memref.alloc
    // CHECK: [[DDR_BUF_1:%.+]] = memref.alloc

    // CHECK: [[ELTWISE_1_INPUT_FULL_BUF_1:%.+]] = memref.alloc
    // CHECK: [[ELTWISE_1_INPUT_FULL_1:%.+]] = VPUIP.Copy
    // CHECK-SAME: inputs([[DDR_BUF_0]] : memref<1x32x52x256x!qElemType, #NHWC>)
    // CHECK-SAME: outputs([[ELTWISE_1_INPUT_FULL_BUF_1]] : memref<1x32x52x256x!qElemType, #NHWC, @CMX_NN>)

    // CHECK: [[ELTWISE_1_INPUT_BUF_2:%.+]] = memref.alloc
    // CHECK: [[ELTWISE_1_INPUT_2:%.+]] = VPUIP.Copy
    // CHECK-SAME: inputs([[DDR_BUF_1]] : memref<1x32x26x256x!qElemType, #NHWC>)
    // CHECK-SAME: outputs([[ELTWISE_1_INPUT_BUF_2]] : memref<1x32x26x256x!qElemType, {order = #NHWC, strides = [425984, 1, 8192, 32]}, @CMX_NN>)

    // CHECK: [[SUBVIEW_BUFF:%.+]] = VPUIP.SubView [[ELTWISE_1_INPUT_FULL_1]] [0, 0, 0, 0] [1, 32, 26, 256]
    // CHECK-SAME:        to memref<1x32x26x256x!qElemType, {order = #NHWC, strides = [425984, 1, 8192, 32]}, @CMX_NN>

    // CHECK: [[SPILL_DDR_BUF_1:%.+]] = memref.alloc
    // CHECK: [[SPILL_TO_DDR:%.+]] = VPUIP.Copy
    // CHECK-SAME: inputs([[SUBVIEW_BUFF]] : memref<1x32x26x256x!qElemType, {order = #NHWC, strides = [425984, 1, 8192, 32]}, @CMX_NN>)
    // CHECK-SAME: outputs([[SPILL_DDR_BUF_1]] : memref<1x32x26x256x!qElemType, #NHWC, @DDR>)

    // CHECK: [[ELTWISE_1_INPUT_BUF_2:%.+]] = memref.alloc
    // CHECK: [[ELTWISE_1_INPUT_1:%.+]] = VPUIP.Copy
    // CHECK-SAME: inputs([[SPILL_TO_DDR]] : memref<1x32x26x256x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME: outputs([[ELTWISE_1_INPUT_BUF_2]] : memref<1x32x26x256x!qElemType, {order = #NHWC, strides = [425984, 1, 8192, 32]}, @CMX_NN>)


    // CHECK: [[ELTWISE:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      is_inplace = true
    // CHECK-SAME:  input([[ELTWISE_1_INPUT_BUF_2]] : memref<1x32x26x256x!qElemType, {order = #NHWC, strides = [425984, 1, 8192, 32]}, @CMX_NN>)
    // CHECK-SAME:  weights([[ELTWISE_1_INPUT_2]] : memref<1x32x26x256x!qElemType, {order = #NHWC, strides = [425984, 1, 8192, 32]}, @CMX_NN>)
    // CHECK-SAME:  outputs([[ELTWISE_1_INPUT_BUF_2]] : memref<1x32x26x256x!qElemType, {order = #NHWC, strides = [425984, 1, 8192, 32]}, @CMX_NN>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0047491008160161037:146>

!qTypeCMX = memref<1x32x103x512x!qElemType, #NHWC, @CMX_NN>
!qTypeDDR = memref<1x32x103x512x!qElemType, {order = #NHWC, strides = [3375104, 1, 16384, 32]}>

!DistributedType1 = !VPUIP.DistributedBuffer<
    1x32x103x512x!qElemType,
    #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

// CHECK-LABEL: DontInsertSpill
func.func @DontInsertSpill(
    %weights0 : memref<32x32x1x1xf16, #NHWC, @CMX_NN>)
      -> (!DistributedType1, !DistributedType1) {
    %in = memref.alloc() : memref<1x32x206x512x!qElemType, #NHWC>
    %subview = VPUIP.SubView %in [0, 0, 0, 0] [1, 32, 103, 512] : memref<1x32x206x512x!qElemType, #NHWC> to !qTypeDDR

    %eltwiseIn1CMXBuff = VPURT.AllocDistributed -> !DistributedType1
    %eltwiseIn1CMX = VPUIP.Copy inputs(%subview : !qTypeDDR) outputs(%eltwiseIn1CMXBuff : !DistributedType1) -> !DistributedType1

    %eltwiseIn2CMXBuff = VPURT.AllocDistributed -> !DistributedType1
    %eltwiseIn2CMX = VPUIP.Copy inputs(%subview : !qTypeDDR) outputs(%eltwiseIn2CMXBuff : !DistributedType1) -> !DistributedType1

    %eltwise = VPUIP.NCEClusterTask {
      is_inplace = true,
      minimumHardwareExecutionCost = 21125 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%eltwiseIn1CMX : !DistributedType1)
        weights(%eltwiseIn2CMX : !DistributedType1)
        parent_input(%eltwiseIn1CMX : !DistributedType1)
        parent_output(%eltwiseIn1CMXBuff : !DistributedType1)
        outputs(%eltwiseIn1CMXBuff : !DistributedType1) -> !DistributedType1
    variants : {
      DPUTask {
        cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
        outEnd = [511, 51, 31], outStart = [0, 0, 0],
        pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
      DPUTask {
        cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
        outEnd = [511, 50, 31], outStart = [0, 0, 0],
        pad = #VPU.Padding<bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64>}
    } PPE : {
      PPETask {
        ppe = #VPU.PPEStub<>
        }
    }

    %convOutBuff0 = VPURT.AllocDistributed -> !DistributedType1
    %conv0 = VPUIP.NCEClusterTask {
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
    }
    input(%eltwiseIn2CMX : !DistributedType1)
    weights(%weights0 : memref<32x32x1x1xf16, #NHWC, @CMX_NN>)
    parent_input(%eltwiseIn2CMX : !DistributedType1)
    parent_output(%convOutBuff0 : !DistributedType1)
    outputs(%convOutBuff0 : !DistributedType1)
        -> !DistributedType1 variants : {
      DPUTask {
        cluster_id = 0 : i64,
        inEnd = [511, 51, 31], inStart = [0, 0, 0],
        mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
        outEnd = [511, 51, 31], outStart = [0, 0, 0],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
      }
      DPUTask {
        cluster_id = 1 : i64,
        inEnd = [511, 50, 31], inStart = [0, 0, 0],
        mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
        outEnd = [511, 50, 31], outStart = [0, 0, 0],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
      }
    } PPE : {
      PPETask {
        ppe = #VPU.PPEStub<>
      }
    }

    return %eltwise, %conv0 : !DistributedType1, !DistributedType1

  // CHECK: [[DDR_BUF_0:%.+]] = memref.alloc
  // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[DDR_BUF_0]] [0, 0, 0, 0] [1, 32, 103, 512]

  // CHECK: [[ELTWISE_1_INPUT_BUF_1:%.+]] = VPURT.AllocDistributed
  // CHECK: [[ELTWISE_1_INPUT_1:%.+]] = VPUIP.Copy
  // CHECK-SAME: inputs([[SUBVIEW]] : memref<1x32x103x512x!qElemType, {order = #NHWC, strides = [3375104, 1, 16384, 32]}>)
  // CHECK-SAME: outputs([[ELTWISE_1_INPUT_BUF_1]] : !VPUIP.DistributedBuffer<1x32x103x512x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

  // CHECK: [[ELTWISE_1_INPUT_BUF_2:%.+]] = VPURT.AllocDistributed
  // CHECK: [[ELTWISE_1_INPUT_2:%.+]] = VPUIP.Copy
  // CHECK-SAME: inputs([[SUBVIEW]] : memref<1x32x103x512x!qElemType, {order = #NHWC, strides = [3375104, 1, 16384, 32]}>)
  // CHECK-SAME: outputs([[ELTWISE_1_INPUT_BUF_2]] : !VPUIP.DistributedBuffer<1x32x103x512x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

  // CHECK: [[ELTWISE:%.+]] = VPUIP.NCEClusterTask
  // CHECK-SAME:      is_inplace = true
  // CHECK-SAME:  input([[ELTWISE_1_INPUT_1]] : !VPUIP.DistributedBuffer<1x32x103x512x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
  // CHECK-SAME:         weights([[ELTWISE_1_INPUT_2]] : !VPUIP.DistributedBuffer<1x32x103x512x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
  // CHECK-SAME:  outputs([[ELTWISE_1_INPUT_BUF_1]] : !VPUIP.DistributedBuffer<1x32x103x512x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.18452063167796415:128>
!qElemType1 = !quant.uniform<u8:f16, 0.14060529821059284:128>
!qElemType2 = !quant.uniform<u8:f16, 0.14128440408145679:128>

// Note: the order of element type changed
// CHECK: !qElemType = !quant.uniform<u8:f16, 0.14060529821059284:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.18452063167796415:128>
// CHECK: !qElemType2 = !quant.uniform<u8:f16, 0.14128440408145679:128>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedBuffer = !VPUIP.DistributedBuffer<1x160x65x65x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 10, 65]], compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]], memory_shapes = [[1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 10, 65]], memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]]}>
!DistributedBuffer1 = !VPUIP.DistributedBuffer<1x160x65x65x!qElemType1, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 10, 65]], compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]], memory_shapes = [[1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 10, 65]], memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]]}>
!DistributedBuffer2 = !VPUIP.DistributedBuffer<1x160x65x65x!qElemType2, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 10, 65]], compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]], memory_shapes = [[1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 11, 65], [1, 160, 10, 65]], memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]]}>
!DistributedBuffer3 = !VPUIP.DistributedBuffer<1x80x65x65x!qElemType1, {order = #NHWC, strides = [676000, 1, 10400, 160]}, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 80, 11, 65], [1, 80, 11, 65], [1, 80, 11, 65], [1, 80, 11, 65], [1, 80, 11, 65], [1, 80, 10, 65]], compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]], memory_shapes = [[1, 80, 11, 65], [1, 80, 11, 65], [1, 80, 11, 65], [1, 80, 11, 65], [1, 80, 11, 65], [1, 80, 10, 65]], memory_offsets = [[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 22, 0], [0, 0, 33, 0], [0, 0, 44, 0], [0, 0, 55, 0]]}>

//                            CONV0     CONV1
//                              |       |
//        Eltwise Input0     ConcatView as Input1      Eltwise Input0
//                     |      |                 |       |
//                  Eltwise ADD0               Eltwise ADD1
//                            |                 |
//                            Same output buffer
//
//
//                              ||
//                              \/
//
//                            CONV0     CONV1
//                              |       |
//        Eltwise Input0     ConcatView as Input1     Eltwise Input0
//                     |      |                 |     |
//                     |    Copy0             Copy1   |
//                     |      |                 |     |
//                    Eltwise ADD0            Eltwise ADD1
//                            |                 |
//                output(buffer of Copy0)   output(buffer of Copy1)
//

// CHECK-LABEL: OutputTypeMismatch
func.func @OutputTypeMismatch(%conv0: !DistributedBuffer3, %conv1: !DistributedBuffer3) -> !DistributedBuffer {
    // Eltwise Input0
    %0 = VPURT.AllocDistributed -> !DistributedBuffer2
    // Output buffer
    %1 = VPURT.AllocDistributed -> !DistributedBuffer1

    %2 = VPUIP.ViewOp %1 : !DistributedBuffer1 to !DistributedBuffer
    %3 = VPUIP.ViewOp %1 : !DistributedBuffer1 to !DistributedBuffer
    // ConcatView as Input1
    %4 = VPUIP.ConcatView inputs(%conv0, %conv1 : !DistributedBuffer3, !DistributedBuffer3) outputs(%1 : !DistributedBuffer1) -> !DistributedBuffer1
    // Eltwise ADD0
    %5 = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, minimumHardwareExecutionCost = 2909 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%0 : !DistributedBuffer2)
        weights(%4 : !DistributedBuffer1)
        parent_input(%0 : !DistributedBuffer2)
        parent_output(%3 : !DistributedBuffer)
        outputs(%3 : !DistributedBuffer )
          -> !DistributedBuffer  variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [64, 10, 159], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [64, 10, 159], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [64, 10, 159], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [64, 10, 159], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [64, 10, 159], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [64, 10, 159], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [64, 10, 159], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [64, 10, 159], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 4 : i64, inEnd = [64, 10, 159], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [64, 10, 159], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 5 : i64, inEnd = [64, 9, 159], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [64, 9, 159], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [22198], quant_shift = [29], quant_post_shift = 0 : i64, in1_quant_mult = [18518], in2_quant_mult = [18429], fp_prelu_alpha = 1.000000e+00 : f64>}
    }

    // Eltwise ADD1
    %6 = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, minimumHardwareExecutionCost = 2909 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%0 : !DistributedBuffer2)
        weights(%4 : !DistributedBuffer1)
        parent_input(%0 : !DistributedBuffer2)
        parent_output(%2 : !DistributedBuffer)
        outputs(%2 : !DistributedBuffer)
          -> !DistributedBuffer  variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [64, 10, 159], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [64, 10, 159], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [64, 10, 159], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [64, 10, 159], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [64, 10, 159], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [64, 10, 159], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [64, 10, 159], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [64, 10, 159], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 4 : i64, inEnd = [64, 10, 159], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [64, 10, 159], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 5 : i64, inEnd = [64, 9, 159], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [64, 9, 159], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [22198], quant_shift = [29], quant_post_shift = 0 : i64, in1_quant_mult = [18518], in2_quant_mult = [18429], fp_prelu_alpha = 1.000000e+00 : f64>}
    }
    return %6 : !DistributedBuffer

    // CHECK: [[ELTWISE_INPUT0_BUFFER:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x160x65x65x!qElemType2
    // CHECK: [[OUTPUT_BUFFER:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x160x65x65x!qElemType,

    // After inserting copies for eltwise inputs, viewOp was not being used anymore.

    // CHECK: [[DISABLE_VIEWOP0:%.*]] = VPUIP.ViewOp [[OUTPUT_BUFFER]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType,
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x160x65x65x!qElemType1
    // CHECK: [[DISABLE_VIEWOP1:%.*]] = VPUIP.ViewOp [[OUTPUT_BUFFER]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType,
    // CHECK-SAME: to !VPUIP.DistributedBuffer<1x160x65x65x!qElemType1

    // ConcatView as Input1

    // CHECK: [[OUTPUT_BUFFER_CONCATVIEW:%.*]] = VPUIP.ConcatView inputs([[CONV0:%.*]], [[CONV1:%.*]]

    // Copy0

    // CHECK: [[OUTPUT_COPY0_BUFFER:%.*]] = memref.alloc() : memref<1x160x65x65x!qElemType, #NHWC, @DDR>
    // CHECK: [[OUTPUT_BUFFER_COPY0:%.*]] = VPUIP.Copy
    // CHECK:   inputs([[OUTPUT_BUFFER_CONCATVIEW]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType,
    // CHECK:   outputs([[OUTPUT_COPY0_BUFFER]] : memref<1x160x65x65x!qElemType, #NHWC, @DDR>)
    // CHECK: [[ELTWISE_OUTPUT_BUFFER0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x160x65x65x!qElemType,
    // CHECK: [[ELTWISE_INPUT1_BUFFER_COPY0:%.*]] = VPUIP.Copy
    // CHECK:    inputs([[OUTPUT_BUFFER_COPY0]]
    // CHECK:    outputs([[ELTWISE_OUTPUT_BUFFER0]]

    // QuantizeCast

    // CHECK: [[ELTWISE_OUTPUT_BUFFER0_CAST:%.*]] = VPUIP.QuantizeCast
    // CHECK:   inputs([[ELTWISE_OUTPUT_BUFFER0]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType,
    // CHECK:   -> !VPUIP.DistributedBuffer<1x160x65x65x!qElemType1

    // Eltwise Add0

    // CHECK: [[ELTWISE_ADD0:%.*]] = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, minimumHardwareExecutionCost = 2909 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK:    input([[ELTWISE_INPUT0_BUFFER]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType2
    // CHECK:    weights([[ELTWISE_INPUT1_BUFFER_COPY0]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType,
    // CHECK:    parent_input([[ELTWISE_INPUT0_BUFFER]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType2
    // CHECK:    parent_output([[ELTWISE_OUTPUT_BUFFER0_CAST]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType1
    // CHECK:    outputs([[ELTWISE_OUTPUT_BUFFER0_CAST]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType1
    // CHECK:    -> !VPUIP.DistributedBuffer<1x160x65x65x!qElemType1
    // CHECK: }

    // Copy1

    // CHECK: [[OUTPUT_COPY1_BUFFER:%.*]] = memref.alloc() : memref<1x160x65x65x!qElemType, #NHWC, @DDR>
    // CHECK: [[OUTPUT_BUFFER_COPY1:%.*]] = VPUIP.Copy
    // CHECK:    inputs([[OUTPUT_BUFFER_CONCATVIEW]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType,
    // CHECK:    outputs([[OUTPUT_COPY1_BUFFER]] : memref<1x160x65x65x!qElemType, #NHWC, @DDR>)
    // CHECK: [[ELTWISE_OUTPUT_BUFFER1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x160x65x65x!qElemType,
    // CHECK: [[ELTWISE_INPUT1_BUFFER_COPY1:%.*]] = VPUIP.Copy
    // CHECK:    inputs([[OUTPUT_BUFFER_COPY1]]
    // CHECK:    outputs([[ELTWISE_OUTPUT_BUFFER1]]

    // QuantizeCast

    // CHECK: [[ELTWISE_OUTPUT_BUFFER1_CAST:%.*]] = VPUIP.QuantizeCast
    // CHECK:   inputs([[ELTWISE_OUTPUT_BUFFER1]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType,
    // CHECK:   -> !VPUIP.DistributedBuffer<1x160x65x65x!qElemType1

    // Eltwise Add1

    // CHECK: [[ELTWISE_ADD1:%.*]] = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, minimumHardwareExecutionCost = 2909 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK:    input([[ELTWISE_INPUT0_BUFFER]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType2
    // CHECK:    weights([[ELTWISE_INPUT1_BUFFER_COPY1]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType
    // CHECK:    parent_input([[ELTWISE_INPUT0_BUFFER]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType2
    // CHECK:    parent_output([[ELTWISE_OUTPUT_BUFFER1_CAST]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType1
    // CHECK:    outputs([[ELTWISE_OUTPUT_BUFFER1_CAST]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType1
    // CHECK:    -> !VPUIP.DistributedBuffer<1x160x65x65x!qElemType1
    // CHECK: }

    // CHECK: return [[ELTWISE_ADD1]] : !VPUIP.DistributedBuffer<1x160x65x65x!qElemType1
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: NotInsertCopyDueToSharedBufferIsOutput
// CHECK-SAME: [[INPUT1:%.+]]: memref<1x1x512x512xf16>,
// CHECK-SAME: [[INPUT2:%.+]]: memref<1x1x512x512xf16>
func.func @NotInsertCopyDueToSharedBufferIsOutput(
    %arg0 :  memref<1x1x512x512xf16>, %arg1 :  memref<1x1x512x512xf16>)
      -> memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]> {

    %cst = const.Declare memref<1x16x128x128xf16, #NHWC> = dense<1.0> : tensor<1x16x128x128xf16>, [#const.Reorder<#NHWC>]
    %in1_CMX = memref.alloc() : memref<1x1x512x512xf16, [@CMX_NN, 0]>
    %copy_in1CMX = VPUIP.Copy inputs(%arg0 : memref<1x1x512x512xf16>) outputs(%in1_CMX : memref<1x1x512x512xf16, [@CMX_NN, 0]>) -> memref<1x1x512x512xf16, [@CMX_NN, 0]>

    %in2_CMX = memref.alloc() : memref<1x1x512x512xf16, [@CMX_NN, 0]>
    %copy_in2CMX = VPUIP.Copy inputs(%arg1 : memref<1x1x512x512xf16>) outputs(%in2_CMX : memref<1x1x512x512xf16, [@CMX_NN, 0]>) -> memref<1x1x512x512xf16, [@CMX_NN, 0]>

    %in1_permute = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%copy_in1CMX : memref<1x1x512x512xf16, [@CMX_NN, 0]>) -> memref<1x1x512x512xf16, #NHWC, [@CMX_NN, 0]>
    %in2_permute = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%copy_in2CMX : memref<1x1x512x512xf16, [@CMX_NN, 0]>) -> memref<1x1x512x512xf16, #NHWC, [@CMX_NN, 0]>
    %in1_shapecast = VPUIP.ShapeCast {shape = [1, 16, 128, 128]} inputs(%in1_permute : memref<1x1x512x512xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>
    %in2_shapecast = VPUIP.ShapeCast {shape = [1, 16, 128, 128]} inputs(%in2_permute : memref<1x1x512x512xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>

    %nce_1 = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, minimumHardwareExecutionCost = 20422 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%in1_shapecast : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>) weights(%in2_shapecast : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%in1_shapecast : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%in1_shapecast : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%in1_shapecast : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [127, 127, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 127, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }

    %buffer_cst = memref.alloc() : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>
    %copy_cst = VPUIP.Copy inputs(%cst : memref<1x16x128x128xf16, #NHWC>) outputs(%buffer_cst : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>

    %nce_2 = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<ADD>, is_inplace = true, minimumHardwareExecutionCost = 20422 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%nce_1 : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>) weights(%copy_cst : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>) parent_input(%nce_1 : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%in1_shapecast : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>) outputs(%in1_shapecast : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [127, 127, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [127, 127, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }

    return %nce_2 :  memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK-DAG: [[CST_WEIGHTS:%.+]] = const.Declare memref<1x16x128x128xf16, #NHWC>
    // CHECK: [[IN1_CMX:%.+]] = memref.alloc() : memref<1x1x512x512xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY_IN1:%.+]] = VPUIP.Copy inputs([[INPUT1]] : memref<1x1x512x512xf16>) outputs([[IN1_CMX]] : memref<1x1x512x512xf16, [@CMX_NN, 0]>)
    // CHECK: [[IN2_CMX:%.+]] = memref.alloc() : memref<1x1x512x512xf16, [@CMX_NN, 0]>
    // CHECK: [[COPY_IN2:%.+]] = VPUIP.Copy inputs([[INPUT2]] : memref<1x1x512x512xf16>) outputs([[IN2_CMX]] : memref<1x1x512x512xf16, [@CMX_NN, 0]>)

    // CHECK: [[IN1_PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[COPY_IN1]] : memref<1x1x512x512xf16, [@CMX_NN, 0]>)
    // CHECK: [[IN2_PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[COPY_IN2]] : memref<1x1x512x512xf16, [@CMX_NN, 0]>)

    // CHECK: [[IN1_SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 128, 128]} inputs([[IN1_PERMUTE]] : memref<1x1x512x512xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK: [[IN2_SHAPECAST:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 128, 128]} inputs([[IN2_PERMUTE]] : memref<1x1x512x512xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK: [[NCE_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK:   input([[IN1_SHAPECAST]] : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:   weights([[IN2_SHAPECAST]] : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:   parent_input([[IN1_SHAPECAST]] : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:   parent_output([[IN1_SHAPECAST]] : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:   outputs([[IN1_SHAPECAST]] : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK: [[BUFFER_CST:%.+]] = memref.alloc() : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[COPY_CST:%.+]] = VPUIP.Copy inputs([[CST_WEIGHTS]] : memref<1x16x128x128xf16, #NHWC>) outputs([[BUFFER_CST]] : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK: [[NCE_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK:   input([[NCE_1]] : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:   weights([[COPY_CST]] : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:   parent_input([[NCE_1]] : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:   parent_output([[IN1_SHAPECAST]] : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:   outputs([[IN1_SHAPECAST]] : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>)

    // return [[NCE_2]] : memref<1x16x128x128xf16, #NHWC, [@CMX_NN, 0]>
}
