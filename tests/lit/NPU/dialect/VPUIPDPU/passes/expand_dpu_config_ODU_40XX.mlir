//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --expand-dpu-config %s | FileCheck %s
// REQUIRES: platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @ODU_TEST_1 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @ODU_TEST_1() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [2, 2],
        kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.ODUCfg {
    // CHECK-NEXT:    VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16){{$}}
    // CHECK-NEXT:    ODUDataReuse activation_reuse(NTHW_4){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUOutActivations out_activations({{%[^:]+}}: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>){{$}}
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) <{end = [7, 7, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63){{$}}
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @ODU_TEST_2 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @ODU_TEST_2() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <50192> : memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(1)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOutSparsityMap !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) output_sparsity_map(@buffer.CMX_NN.0::@DeclareBuffer_ActOutSparsityMap)
        waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.ODUCfg {
    // CHECK-NEXT:    VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16){{$}}
    // CHECK-NEXT:    ODUDataReuse activation_reuse(NTHW_8){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUSparsity {{%[^:]+}}: memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUOutActivations out_activations({{%[^:]+}}: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>){{$}}
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) <{end = [7, 7, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 7 : i64, yStart = 0 : i64, xEnd = 7 : i64, yEnd = 45 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -224 : i64, sparsityOffset = -14 : i64, targetWidth = 10 : i64, targetClusters = [1, 2, 4]>]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:         VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUHaloCfg {
    // CHECK-NEXT:    VPUIPDPU.ODUHaloRegion begin_coord_x(7) begin_coord_y(0) end_coord_x(7) end_coord_y(45) activations_offset(-224) sparsity_offset(-14) target_width(10) cast_to_tile(DPU_TILE_1|DPU_TILE_2|DPU_TILE_4){{$}}
    // CHECK-NEXT:    }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01>
module {
  net.NetworkInfo entryPoint : @ODU_TEST_3 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xui8>
  }

  func.func @ODU_TEST_3() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <50192> : memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <40976> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NCWH}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOutSparsityMap !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn)
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) output_sparsity_map(@buffer.CMX_NN.0::@DeclareBuffer_ActOutSparsityMap)
         waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [2, 2], kernel_strides = [2, 2], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.ODUCfg {
    // CHECK-NEXT:    VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16){{$}}
    // CHECK-NEXT:    ODUDataReuse activation_reuse(NTHW_16){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUPermuteData permute_mode(PERMUTE_YXZ){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUSparsity {{%[^:]+}}: memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]>{{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUOutActivations out_activations({{%[^:]+}}: memref<1x16x64x64x!qElemType, {order = #NCWH}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT){{$}}
    // CHECK-NEXT:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0)
        invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0 weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights)
         weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) <{end = [7, 7, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 7 : i64, yStart = 0 : i64, xEnd = 7 : i64, yEnd = 45 : i64, zStart = 0 : i64, zEnd = 15 : i64,
          targetOffset = -224 : i64, sparsityOffset = -14 : i64, targetWidth = 10 : i64, targetClusters = [1, 2, 4]>, #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, yStart = 7 : i64, xEnd = 0 : i64, yEnd = 14 : i64, zStart = 0 : i64, zEnd = 15 : i64,
          targetOffset = -224 : i64, sparsityOffset = -14 : i64, targetWidth = 10 : i64, targetClusters = [3]>
        ]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:         VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUHaloCfg {
    // CHECK-NEXT:    VPUIPDPU.ODUHaloRegion begin_coord_x(7) begin_coord_y(0) end_coord_x(7) end_coord_y(45) activations_offset(-224) sparsity_offset(-14) target_width(10) cast_to_tile(DPU_TILE_1|DPU_TILE_2|DPU_TILE_4){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(7) end_coord_x(0) end_coord_y(14) activations_offset(-224) sparsity_offset(-14) target_width(10) cast_to_tile(DPU_TILE_3){{$}}
    // CHECK-NEXT:    }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f32, 1.600000e+01:10>
module {
  net.NetworkInfo entryPoint : @ODU_TEST_4 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xui8>
  }

  func.func @ODU_TEST_4() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <50192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(1)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOutSparsityMap !VPUASM.Buffer< "CMX_NN"[0] <42000> : memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) output_sparsity_map(@buffer.CMX_NN.0::@DeclareBuffer_ActOutSparsityMap) waits([0 : ui16]) updates([1 : ui16]) <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 0 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64, variant_count = 1 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>}
        }
      }

    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.ODUCfg {
    // CHECK-NEXT:    VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16){{$}}
    // CHECK-NEXT:    ODUDataReuse activation_reuse(NTHW_8){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUSparsity {{%[^:]+}}: memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]> sparse_value(10){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUSwizzleData swizzle_key(SWIZZLE_KEY_1){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUOutActivations out_activations({{%[^:]+}}: memref<1x16x64x64x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_8BIT){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUMemoryMode mem_mode(MODE_SUPERDENSE){{$}}

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0) invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0
        weights(@buffer.CMX_NN.0::@DeclareBuffer_Weights) weight_table(@buffer.CMX_NN.0::@DeclareBuffer_WeightsTable) <{end = [7, 7, 63], inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0], haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 7 : i64, yStart = 0 : i64, xEnd = 7 : i64, yEnd = 45 : i64, zStart = 0 : i64, zEnd = 15 : i64,
          targetOffset = -224 : i64, sparsityOffset = -14 : i64, targetWidth = 10 : i64, targetClusters = [1, 2, 3]>, #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, yStart = 7 : i64, xEnd = 0 : i64, yEnd = 14 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 0 : i64,
          sparsityOffset = -14 : i64, targetWidth = 10 : i64, targetClusters = [2, 3, 4]>, #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, yStart = 7 : i64, xEnd = 0 : i64, yEnd = 14 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 224 : i64, sparsityOffset = -14 : i64,
          targetWidth = 10 : i64, targetClusters = [1, 2, 3, 4, 5]>, #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, yStart = 7 : i64, xEnd = 0 : i64, yEnd = 14 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -100 : i64, sparsityOffset = -14 : i64,
          targetWidth = 10 : i64, targetClusters = [1]>, #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, yStart = 7 : i64, xEnd = 0 : i64, yEnd = 14 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 100 : i64, sparsityOffset = -14 : i64, targetWidth = 10 : i64,
          targetClusters = [2]>]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:         VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUHaloCfg {
    // CHECK-NEXT:    VPUIPDPU.ODUHaloRegion begin_coord_x(7) begin_coord_y(0) end_coord_x(7) end_coord_y(45) activations_offset(-224) sparsity_offset(-14) target_width(10) cast_to_tile(DPU_TILE_1|DPU_TILE_2|DPU_TILE_3){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(7) end_coord_x(0) end_coord_y(14) activations_offset(0) sparsity_offset(-14) target_width(10) cast_to_tile(DPU_TILE_2|DPU_TILE_3|DPU_TILE_4){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(7) end_coord_x(0) end_coord_y(14) activations_offset(224) sparsity_offset(-14) target_width(10) cast_to_tile(DPU_TILE_1|DPU_TILE_2|DPU_TILE_3|DPU_TILE_4|DPU_TILE_5){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(7) end_coord_x(0) end_coord_y(14) activations_offset(-100) sparsity_offset(-14) target_width(10) cast_to_tile(DPU_TILE_1){{$}}
    // CHECK-NEXT:    VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(7) end_coord_x(0) end_coord_y(14) activations_offset(100) sparsity_offset(-14) target_width(10) cast_to_tile(DPU_TILE_2){{$}}
    // CHECK-NEXT:    }
    }
    return
  }
}
