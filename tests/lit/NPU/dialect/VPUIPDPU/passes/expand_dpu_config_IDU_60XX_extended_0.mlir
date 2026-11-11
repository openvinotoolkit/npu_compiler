//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --expand-dpu-config %s | FileCheck %s
// REQUIRES: platform-NPU6010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module {
  net.NetworkInfo entryPoint : @IDU_MAXPOOL_HALO_WEIGHT_TABLE_SCALE_BIAS inputsInfo : {
    DataInfo "input_0" : tensor<1x1280x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x1280x16x16xf16>
  }

  func.func @IDU_MAXPOOL_HALO_WEIGHT_TABLE_SCALE_BIAS() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_1 idx(!VPURegMapped.Index<0:0:1>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x1280x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WtScale !VPUASM.Buffer< "CMX_NN"[0] <655360> : memref<1280x1x1x1xf32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WtBias !VPUASM.Buffer< "CMX_NN"[0] <660480> : memref<1280x1x1x1xf32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <665600> : memref<1x1280x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUInvariant @DPUInvariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
        input(@buffer.CMX_NN.0::@DeclareBuffer_ActIn) weight_table_scale(@buffer.CMX_NN.0::@DeclareBuffer_WtScale)
        weight_table_bias(@buffer.CMX_NN.0::@DeclareBuffer_WtBias) output(@buffer.CMX_NN.0::@DeclareBuffer_ActOut) waits([0 : ui16]) updates([1 : ui16])
        <{clean_after = 1 : ui64, cm_sp_pattern = 32 : i64, first_variant_index = 0 : ui32, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1], kernel_strides = [1, 1], last_variant_index = 1 : ui32, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>,
        start_after = 0 : ui64, variant_count = 2 : ui64}> PPE : {
          VPUASM.PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.402823e+38 : f64, clamp_high = 3.402823e+38 : f64, prelu_alpha = [1.000000e+00], scale = 1.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
      }

    // CHECK-LABEL: func.func @IDU_MAXPOOL_HALO_WEIGHT_TABLE_SCALE_BIAS
    // CHECK:       VPUIPDPU.DPUInvariant
    // CHECK:       VPUIPDPU.IDUCfg {
    // CHECK-NEXT:    VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x1280x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-NEXT:    VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT) <{wt_sparse}>
    // CHECK-NEXT:    VPUIPDPU.IDUInputLayerCfg sparsity_pattern(32)
    // CHECK-NEXT:    VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUStride stride_x(1) stride_y(1)
    // CHECK-NEXT:    VPUIPDPU.IDUWorkloadCfg workload_type(MAXPOOL)
    // CHECK-NEXT:    VPUIPDPU.IDUDepthWiseCfg dw_3x3s1_opt_dis(true)
    // CHECK:  }

      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
        VPUASM.DPUVariant @DPUVariant_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0)
        invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0
        weight_table_scale(@buffer.CMX_NN.0::@DeclareBuffer_WtScale) weight_table_bias(@buffer.CMX_NN.0::@DeclareBuffer_WtBias)
        <{end = [15, 15, 639], inEnd = [15, 15, 639], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0],
        haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 15 : i64, yEnd = 15 : i64, zStart = 0 : i64, zEnd = 639 : i64,
        targetOffset = -40960 : i64, targetClusters = [1], targetWidth = 16 : i64>]}>

        VPUASM.DPUVariant @DPUVariant_0_1 idx(!VPURegMapped.Index<0:0:1>) taskLocation(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_1)
        invariant @task.dpu.invariant.0.0::@DPUInvariant_0_0 calls @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0
        weight_table_scale(@buffer.CMX_NN.0::@DeclareBuffer_WtScale) weight_table_bias(@buffer.CMX_NN.0::@DeclareBuffer_WtBias)
        <{end = [15, 15, 1279], inEnd = [15, 15, 1279], inStart = [0, 0, 640], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 640],
        haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 0 : i64, yEnd = 0 : i64, zStart = 640 : i64, zEnd = 1279 : i64,
        targetOffset = 40960 : i64, targetClusters = [0], targetWidth = 16 : i64>]}>
      }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(640){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(640) weight_size(640){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense{{$}}
    // CHECK:  VPUIPDPU.IDUWeightTableOffsets {bias_offset = 0 : i64, scale_offset = 0 : i64}
    // CHECK:  VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(15) end_coord_y(15) end_coord_z(639){{$}}
    // CHECK-NEXT:  VPUIPDPU.ODUHaloCfg {
    // CHECK-NEXT:    VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(15) end_coord_x(15) end_coord_y(15) activations_offset(-40960) target_width(16) cast_to_tile(DPU_TILE_1){{$}}
    // CHECK-NEXT:  }

    // CHECK:       VPUIPDPU.DPUVariant
    // CHECK-SAME:    invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0)
    // CHECK:       VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(640) size_x(16) size_y(16) size_z(640){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUWeightSet weight_start(0) weight_num(640) weight_size(640){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4){{$}}
    // CHECK-NEXT:  VPUIPDPU.IDUSEDense{{$}}
    // CHECK:  VPUIPDPU.IDUWeightTableOffsets {bias_offset = 80 : i64, scale_offset = 80 : i64}
    // CHECK-NEXT:  VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(640) end_coord_x(15) end_coord_y(15) end_coord_z(1279){{$}}
    // CHECK-NEXT:  VPUIPDPU.ODUHaloCfg {
    // CHECK-NEXT:    VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(15) end_coord_y(0) activations_offset(40960) target_width(16) cast_to_tile(DPU_TILE_0){{$}}
    // CHECK-NEXT:  }
    }
    return
  }
}
