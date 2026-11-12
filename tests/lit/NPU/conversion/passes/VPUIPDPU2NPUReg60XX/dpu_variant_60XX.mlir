//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file  --init-compiler="platform=%platform%" --convert-VPUIPDPU-to-NPUReg60XX %s | FileCheck %s
// REQUIRES: dev-build && platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input_0" : tensor<1x16x16x16xf16>
        DataInfo "input_1" : tensor<16x1x1x1xi64>
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x16x64x64xf16>
    }
    func.func @main() {
        ELF.Main {
            ELF.CreateLogicalSection @builtin.data.nncmx0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
                VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <196736> : memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
                VPUASM.DeclareBuffer @Weights !VPUASM.Buffer< "CMX_NN"[0] <213632> : memref<64x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
                VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <128> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            }
            ELF.CreateLogicalSection @builtin.tasks.DPUVariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
                VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
            }
            ELF.CreateLogicalSection @builtin.tasks.DPUInvariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
                VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            }
            ELF.CreateSection @text.invariants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
                VPUIPDPU.DPUInvariant @DPUInvariant_0 {mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64} <{task_index = !VPURegMapped.Index<0:0:0>, task_location = @builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0,
                input = @builtin.data.nncmx0::@DeclareBuffer_ActIn, output = @builtin.data.nncmx0::@DeclareBuffer_ActOut,
                nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}>
                    DPUCfg : {
                    ^bb0(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                        %weight_table: memref<16x1x1x1xi64, {order = #NHWC}, [@CMX_NN, 0]>,
                        %act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>,
                        %sparse_out: memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]>):
                    VPUIPDPU.IDUCfg {
                        VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x16x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                    }
                    VPUIPDPU.PPECfg {
                        VPUIPDPU.PPEFpSprLUTMode sprlut_mode(ON)
                    }
                    VPUIPDPU.ODUCfg {
                        VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
                        VPUIPDPU.ODUSparsity %sparse_out: memref<1x16x64x64xi1, {order = #NHWC}, [@CMX_NN, 0]>
                        VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
                    }
                }
            }
            ELF.CreateSection @text.variants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
                VPUIPDPU.DPUVariant @DPUVariant11 invariant(@builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @builtin.tasks.DPUVariant0::@DeclareTaskBuffer_DPUVariant_0, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
                DPUCfg: {
                // CHECK-NOT:   VPUIPDPU.DPUVariant
                // CHECK:       NPUReg60XX.DPUVariant

                    VPUIPDPU.IDUPadding pad_count(<left = 4, right = 6, top = 3, bottom = 5>)
                    VPUIPDPU.IDUWorkloadSet start_x(3) start_y(4) start_z(5) size_x(32) size_y(33) size_z(34)
                    VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_8_8)
                    VPUIPDPU.IDUBinaryConfig
                    VPUIPDPU.IDUConvContinue
                    VPUIPDPU.IDUSEDense
                    VPUIPDPU.IDUSEOnly
                    VPUIPDPU.IDUActSwizzle swizzle_key(SWIZZLE_KEY_1)
                    VPUIPDPU.IDUWeightSwizzle wt_swizzle_key(SWIZZLE_KEY_2)
                    VPUIPDPU.IDUPerOutputChannelScaling {tensor2_act_dense}

                    // CHECK:  UINT workload_size_x = 0x20
                    // CHECK:  UINT workload_size_y = 0x21
                    // CHECK:  UINT pad_count_up = 3
                    // CHECK:  UINT pad_count_left = 4
                    // CHECK:  UINT pad_count_down = 5
                    // CHECK:  UINT pad_count_right = 6
                    // CHECK:  UINT workload_start_x = 3
                    // CHECK:  UINT workload_start_y = 4
                    // CHECK:  UINT nthw_ntk = 0
                    // CHECK:  UINT bin_cfg = 1
                    // CHECK:  UINT conv_cond = 1
                    // CHECK:  UINT dense_se = 1
                    // CHECK:  UINT swizzle_key_offset = 1
                    // CHECK:  UINT noc_clk_en = 1
                    // CHECK:  UINT se_only_en = 1
                    // CHECK:  UINT sb_read_en = 1
                    // CHECK:  UINT tensor2_act_dense = 1
                    // CHECK:  UINT wt_swizzle_key = 2
                    // CHECK:  UINT wt_swizzle_sel = 1

                    VPUIPDPU.IDUpalletLUTRead
                    // CHECK:  UINT invar_plt_rd_en = 1

                    VPUIPDPU.PPEsprLUTRead
                    // CHECK: UINT invar_lut_rd_en = 1

                    VPUIPDPU.BarrierCfg waits([1 : ui16, 2 : ui16]) updates([3 : ui16, 4 : ui16, 5 : ui16]) start_after(0) clean_after(0)
                    VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(4) <{isFirstVariant}>
                    // CHECK:  cbarrier_lo = UINT 6
                    // CHECK:  pbarrier_lo = UINT 0


                    VPUIPDPU.ODUOutSubtensor begin_coord_x(1) begin_coord_y(32) begin_coord_z(64) end_coord_x(63) end_coord_y(63) end_coord_z(15)
                    // CHECK:  UINT te_beg_y = 0x20
                    // CHECK:  UINT te_beg_z = 0x40
                    // CHECK:  UINT te_beg_x = 1
                    // CHECK:  UINT te_end_y = 0x3F
                    // CHECK:  UINT te_end_z = 0xF
                    // CHECK:  UINT te_end_x = 0x3F

                    VPUIPDPU.ODUHaloCfg {
                        VPUIPDPU.ODUHaloRegion begin_coord_x(0) begin_coord_y(0) end_coord_x(15) end_coord_y(15) activations_offset(10) sparsity_offset(10) target_width(32) cast_to_tile("DPU_TILE_1|DPU_TILE_2")
                        // CHECK:  halo_region0A
                        // CHECK:  SINT sp_adr_offset = 0xA
                        // CHECK:  UINT tile_select = 6
                        // CHECK:  UINT enable = 1
                        // CHECK:  halo_region0B
                        // CHECK:  SINT ac_adr_offset = 0xA
                        // CHECK:  UINT target_width_lsb = 0x20
                        // CHECK:  halo_region0C
                        // CHECK:  UINT begin_x = 0
                        // CHECK:  UINT begin_y = 0
                        // CHECK:  UINT target_width_msb = 0
                        // CHECK:  halo_region0D
                        // CHECK:  UINT end_x = 0xF
                        // CHECK:  UINT end_y = 0xF

                        VPUIPDPU.ODUHaloRegion begin_coord_x(4) begin_coord_y(5) end_coord_x(15) end_coord_y(16) activations_offset(9) target_width(64) cast_to_tile("DPU_TILE_1|DPU_TILE_3")
                        // CHECK:  halo_region1A
                        // CHECK:  SINT sp_adr_offset = 0
                        // CHECK:  UINT tile_select = 0xA
                        // CHECK:  UINT enable = 1
                        // CHECK:  halo_region1B
                        // CHECK:  SINT ac_adr_offset = 9
                        // CHECK:  UINT target_width_lsb = 0x40
                        // CHECK:  halo_region1C
                        // CHECK:  UINT begin_x = 4
                        // CHECK:  UINT begin_y = 5
                        // CHECK:  UINT target_width_msb = 0
                        // CHECK:  halo_region1D
                        // CHECK:  UINT end_x = 0xF
                        // CHECK:  UINT end_y = 0x10

                        VPUIPDPU.ODUHaloRegion begin_coord_x(9) begin_coord_y(0) end_coord_x(63) end_coord_y(63) activations_offset(400) target_width(15424) cast_to_tile("DPU_TILE_0|DPU_TILE_1|DPU_TILE_2|DPU_TILE_3|DPU_TILE_4|DPU_TILE_5")
                        // CHECK:  halo_region2A
                        // CHECK:  SINT sp_adr_offset = 0
                        // CHECK:  UINT tile_select = 0x3F
                        // CHECK:  UINT enable = 1
                        // CHECK:  halo_region2B
                        // CHECK:  SINT ac_adr_offset = 0x190
                        // CHECK:  UINT target_width_lsb = 0x40
                        // CHECK:  halo_region2C
                        // CHECK:  UINT begin_x = 9
                        // CHECK:  UINT begin_y = 0
                        // CHECK:  UINT target_width_msb = 0xF
                        // CHECK:  halo_region2D
                        // CHECK:  UINT end_x = 0x3F
                        // CHECK:  UINT end_y = 0x3F

                        VPUIPDPU.ODUHaloRegion begin_coord_x(9) begin_coord_y(0) end_coord_x(63) end_coord_y(63) activations_offset(9) target_width(15488) cast_to_tile("DPU_TILE_0|DPU_TILE_5")
                        // CHECK:  halo_region3A
                        // CHECK:  SINT sp_adr_offset = 0
                        // CHECK:  UINT tile_select = 0x21
                        // CHECK:  UINT enable = 1
                        // CHECK:  halo_region3B
                        // CHECK:  SINT ac_adr_offset = 9
                        // CHECK:  UINT target_width_lsb = 0x80
                        // CHECK:  halo_region3C
                        // CHECK:  UINT begin_x = 9
                        // CHECK:  UINT begin_y = 0
                        // CHECK:  UINT target_width_msb = 0xF
                        // CHECK:  halo_region3D
                        // CHECK:  UINT end_x = 0x3F
                        // CHECK:  UINT end_y = 0x3F

                        VPUIPDPU.ODUHaloRegion begin_coord_x(9) begin_coord_y(0) end_coord_x(63) end_coord_y(63) activations_offset(9) target_width(3328) cast_to_tile("DPU_TILE_2")
                        // CHECK:  halo_region4A
                        // CHECK:  SINT sp_adr_offset = 0
                        // CHECK:  UINT tile_select = 4
                        // CHECK:  UINT enable = 1
                        // CHECK:  halo_region4B
                        // CHECK:  SINT ac_adr_offset = 9
                        // CHECK:  UINT target_width_lsb = 0x100
                        // CHECK:  halo_region4C
                        // CHECK:  UINT begin_x = 9
                        // CHECK:  UINT begin_y = 0
                        // CHECK:  UINT target_width_msb = 3
                        // CHECK:  halo_region4D
                        // CHECK:  UINT end_x = 0x3F
                        // CHECK:  UINT end_y = 0x3F
                    }
                }

                VPUIPDPU.DPUVariant @DPUVariant22 invariant(@builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @builtin.tasks.DPUVariant0::@DeclareTaskBuffer_DPUVariant_0, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
                DPUCfg: {
                // CHECK-NOT:   VPUIPDPU.DPUVariant
                // CHECK:       NPUReg60XX.DPUVariant

                    VPUIPDPU.IDUPadding pad_count(<left = 4, right = 6, top = 3, bottom = 5>)
                    VPUIPDPU.IDUWorkloadSet start_x(3) start_y(4) start_z(5) size_x(32) size_y(33) size_z(34)
                    VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_4_16)
                    VPUIPDPU.IDUActSwizzle swizzle_key(SWIZZLE_KEY_3)

                    // CHECK:  UINT workload_size_x = 0x20
                    // CHECK:  UINT workload_size_y = 0x21
                    // CHECK:  UINT pad_count_up = 3
                    // CHECK:  UINT pad_count_left = 4
                    // CHECK:  UINT pad_count_down = 5
                    // CHECK:  UINT pad_count_right = 6
                    // CHECK:  UINT workload_start_x = 3
                    // CHECK:  UINT workload_start_y = 4
                    // CHECK:  UINT nthw_ntk = 1
                    // CHECK:  UINT bin_cfg = 0
                    // CHECK:  UINT conv_cond = 0
                    // CHECK:  UINT dense_se = 0
                    // CHECK:  UINT swizzle_key_offset = 3
                    // CHECK:  UINT noc_clk_en = 1
                    // CHECK:  UINT wt_swizzle_key = 0
                    // CHECK:  UINT wt_swizzle_sel = 1

                    // CHECK: UINT invar_lut_rd_en = 0

                    VPUIPDPU.BarrierCfg waits([1 : ui16, 2 : ui16]) updates([3 : ui16, 4 : ui16, 5 : ui16]) start_after(0) clean_after(0)
                    VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(4)
                    // CHECK:  cbarrier_lo = UINT 0
                    // CHECK:  pbarrier_lo = UINT 0

                    VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
                    // CHECK:  UINT te_beg_y = 0
                    // CHECK:  UINT te_beg_z = 0
                    // CHECK:  UINT te_beg_x = 0
                    // CHECK:  UINT te_end_y = 0x3F
                    // CHECK:  UINT te_end_z = 0xF
                    // CHECK:  UINT te_end_x = 0x3F
                }

                VPUIPDPU.DPUVariant @DPUVariant33 invariant(@builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @builtin.tasks.DPUVariant0::@DeclareTaskBuffer_DPUVariant_0,
                nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
                DPUCfg: {
                // CHECK-NOT:   VPUIPDPU.DPUVariant
                // CHECK:       NPUReg60XX.DPUVariant
                    ^bb0(%weights_tensor: memref<32x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>):

                    VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4)
                    VPUIPDPU.IDUActSwizzle swizzle_key(SWIZZLE_KEY_4)
                    // CHECK:  UINT nthw_ntk = 3
                    // CHECK:  UINT swizzle_key_offset = 4
                    // CHECK:  UINT noc_clk_en = 1

                    // CHECK: UINT invar_lut_rd_en = 0

                    VPUIPDPU.BarrierCfg waits([1 : ui16, 2 : ui16]) updates([3 : ui16, 4 : ui16, 5 : ui16]) start_after(0) clean_after(0)
                    VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(4)
                    // CHECK:  cbarrier_lo = UINT 0
                    // CHECK:  pbarrier_lo = UINT 0

                    VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(5) begin_coord_z(4) end_coord_x(63) end_coord_y(63) end_coord_z(15)
                    // CHECK:  UINT te_beg_y = 5
                    // CHECK:  UINT te_beg_z = 4
                    // CHECK:  UINT te_beg_x = 0
                    // CHECK:  UINT te_end_y = 0x3F
                    // CHECK:  UINT te_end_z = 0xF
                    // CHECK:  UINT te_end_x = 0x3F

                    VPUIPDPU.IDUWeightSet weight_start(0) weight_num(32) weight_size(144)
                    // CHECK:  UINT weight_size = 0x90
                    // CHECK:  UINT weight_num = 0x20
                    // CHECK:  weight_start = UINT 0
                }

                // Use default initialized values
                VPUIPDPU.DPUVariant @DPUVariant44 invariant(@builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @builtin.tasks.DPUVariant0::@DeclareTaskBuffer_DPUVariant_0, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>}
                DPUCfg: {
                // CHECK-NOT:   VPUIPDPU.DPUVariant
                // CHECK:       NPUReg60XX.DPUVariant
                    ^bb0(%weights_tensor: memref<32x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>):
                    // CHECK:  UINT noc_clk_en = 1
                    // CHECK:  UINT shave_l2_cache_en = 0

                    // CHECK: UINT invar_lut_rd_en = 0

                    VPUIPDPU.BarrierCfg waits([1 : ui16, 2 : ui16]) updates([3 : ui16, 4 : ui16, 5 : ui16]) start_after(0) clean_after(0)
                    VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(4) <{isLastVariant}>
                    // CHECK:  cbarrier_lo = UINT 0
                    // CHECK:  pbarrier_lo = UINT 0x38

                    // CHECK:  UINT workload_prm_sel = 0
                    VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(63) end_coord_y(63) end_coord_z(15)
                    // CHECK:  UINT te_beg_y = 0
                    // CHECK:  UINT te_beg_z = 0
                    // CHECK:  UINT te_beg_x = 0
                    // CHECK:  UINT te_end_y = 0x3F
                    // CHECK:  UINT te_end_z = 0xF
                    // CHECK:  UINT te_end_x = 0x3F
                }
            }
            ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
                ELF.Symbol @elfsym.builtin.data.nncmx0 of(@builtin.data.nncmx0) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.builtin.tasks.DPUVariant0 of(@builtin.tasks.DPUVariant0) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.builtin.tasks.DPUInvariant0 of(@builtin.tasks.DPUInvariant0) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.text.invariants of(@text.invariants) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.text.variants of(@text.variants) type(<STT_SECTION>) size(0) value(0)
            }
        }

        return
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
    net.NetworkInfo entryPoint : @dynamic_shape_var_primitives_0 inputsInfo : {
        DataInfo "input_0" : tensor<1x64x16x16xf16>
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x64x8x8xf16>
    }
    func.func @dynamic_shape_var_primitives_0() {
        ELF.Main {
            ELF.CreateLogicalSection @builtin.data.nncmx0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
                VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
                VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            }
            ELF.CreateLogicalSection @builtin.tasks.DPUVariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
                VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
            }
            ELF.CreateLogicalSection @builtin.tasks.DPUInvariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
                VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            }
            ELF.CreateSection @text.variants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
                VPUIPDPU.DPUVariant @DPUVariant0 invariant(@builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @builtin.tasks.DPUVariant0::@DeclareTaskBuffer_DPUVariant_0, nce_task_type = #VPUIP.nce_task_type<CONV>}
                DPUCfg: {
                // CHECK-NOT:   VPUIPDPU.DPUVariant
                // CHECK:       NPUReg60XX.DPUVariant

                        // CHECK:  UINT noc_clk_en = 1

                        VPUIPDPU.DynPrimitiveIdVar dyn_primitive_id_var(4) dyn_primitive_id2_var(6)
                        // CHECK:  UINT dyn_primitive_id_var = 4
                        // CHECK:  UINT dyn_primitive_id_var_en = 1
                        // CHECK:  UINT dyn_primitive_id2_var = 6
                        VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
                        VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(64)
                        VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
                        VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4)
                        VPUIPDPU.IDUSEDense
                        VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
                        VPUIPDPU.BarrierCfg waits([2 : ui16]) updates([3 : ui16]) start_after(0) clean_after(0)
                        VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1) <{isFirstVariant}>
                  }
            }
            ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
                ELF.Symbol @elfsym.builtin.data.nncmx0 of(@builtin.data.nncmx0) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.builtin.tasks.DPUVariant0 of(@builtin.tasks.DPUVariant0) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.builtin.tasks.DPUInvariant0 of(@builtin.tasks.DPUInvariant0) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.text.invariants of(@text.invariants) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.text.variants of(@text.variants) type(<STT_SECTION>) size(0) value(0)
            }
        }

        return
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @Test {
    net.NetworkInfo entryPoint : @dynamic_shape_var_primitives_1 inputsInfo : {
        DataInfo "input_0" : tensor<1x64x16x16xf16>
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x64x8x8xf16>
    }
    func.func @dynamic_shape_var_primitives_1() {
        ELF.Main {
            ELF.CreateLogicalSection @builtin.data.nncmx0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
                VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <8192> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
                VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <40960> : memref<64x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x64x8x8xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
            }
            ELF.CreateLogicalSection @builtin.tasks.DPUVariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
                VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUVariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
            }
            ELF.CreateLogicalSection @builtin.tasks.DPUInvariant0 aligned(64) secType(SHT_NOBITS) secFlags(SHF_ALLOC) secLocation(<CMX_NN>) {
                VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
            }
            ELF.CreateSection @text.variants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
                VPUIPDPU.DPUVariant @DPUVariant0 invariant(@builtin.tasks.DPUInvariant0::@DeclareTaskBuffer_DPUInvariant_0) {task_index = !VPURegMapped.Index<0:0:0>, taskLocation = @builtin.tasks.DPUVariant0::@DeclareTaskBuffer_DPUVariant_0, nce_task_type = #VPUIP.nce_task_type<CONV>}
                DPUCfg: {
                // CHECK-NOT:   VPUIPDPU.DPUVariant
                // CHECK:       NPUReg60XX.DPUVariant

                        // CHECK:  UINT noc_clk_en = 1
                        VPUIPDPU.DynPrimitiveIdVar dyn_primitive_id_var(4)
                        // CHECK:  UINT dyn_primitive_id_var = 4
                        // CHECK:  UINT dyn_primitive_id_var_en = 1
                        // CHECK:  UINT dyn_primitive_id2_var = 0
                        VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
                        VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(64)
                        VPUIPDPU.IDUPadding pad_count(<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>)
                        VPUIPDPU.IDUNthwNtk nthw_ntk(NTHW_NTK_16_4)
                        VPUIPDPU.IDUSEDense
                        VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
                        VPUIPDPU.BarrierCfg waits([2 : ui16]) updates([3 : ui16]) start_after(0) clean_after(0)
                        VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1) <{isFirstVariant}>
                  }
            }
            ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
                ELF.Symbol @elfsym.builtin.data.nncmx0 of(@builtin.data.nncmx0) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.builtin.tasks.DPUVariant0 of(@builtin.tasks.DPUVariant0) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.builtin.tasks.DPUInvariant0 of(@builtin.tasks.DPUInvariant0) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.text.invariants of(@text.invariants) type(<STT_SECTION>) size(0) value(0)
                ELF.Symbol @elfsym.text.variants of(@text.variants) type(<STT_SECTION>) size(0) value(0)
            }
        }

        return
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @IDU_no_partial_IC_CONV_input_f16_weights_fp16 {
  net.NetworkInfo entryPoint : @IDU_no_partial_IC_CONV_input_f16_weights_fp16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xui16>
  }

  func.func @IDU_no_partial_IC_CONV_input_f16_weights_fp16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <163840> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut,
        task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0,
        weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}>
        DPUCfg : {
        ^bb0(%arg0: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg2: memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg3: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
          VPUIPDPU.IDUCfg {
            VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
            VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
            VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
          }
          VPUIPDPU.PPECfg {
            VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
          }
          VPUIPDPU.ODUCfg {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%arg3 : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
          }
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
        }
      }
      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUVariant @DPUVariant_0_0 invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) {nce_task_type = #VPUIP.nce_task_type<CONV>, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights} DPUCfg : {
          VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
          VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256)
          VPUIPDPU.IDUSEDense
          VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1) <{isFirstVariant}>
          //CHECK:  UINT workload_size_z = 0x40,
          //CHECK:  UINT workload_start_z = 0,
          //CHECK:  UINT noc_clk_en = 1,
          //CHECK:  UINT grp_conv_en = 0,
          //CHECK:  UINT weight_size = 0x100,
          //CHECK:  UINT weight_num = 0x40,
        }
      }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @IDU_partial_32_IC_CONV_first_half_input_f16_weights_fp16 {
  net.NetworkInfo entryPoint : @IDU_partial_32_IC_CONV_first_half_input_f16_weights_fp16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xui16>
  }

  func.func @IDU_partial_32_IC_CONV_first_half_input_f16_weights_fp16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <163840> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut,
        task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}>
        DPUCfg : {
        ^bb0(%arg0: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg2: memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg3: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
          VPUIPDPU.IDUCfg {
            VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
            VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
            VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
          }
          VPUIPDPU.PPECfg {
            VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
          }
          VPUIPDPU.ODUCfg {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%arg3 : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
          }
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
        }
      }
      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUVariant @DPUVariant_0_0 invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) {nce_task_type = #VPUIP.nce_task_type<CONV>, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights} DPUCfg : {
          VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(32)
          VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(128)
          VPUIPDPU.IDUSEDense
          VPUIPDPU.IDUGroupConvOp
          VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1) <{isFirstVariant}>
          //CHECK:  UINT workload_size_z = 0x20,
          //CHECK:  UINT workload_start_z = 0,
          //CHECK:  UINT noc_clk_en = 1,
          //CHECK:  UINT grp_conv_en = 1,
          //CHECK:  UINT weight_size = 0x80,
          //CHECK:  UINT weight_num = 0x40,
        }
      }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @IDU_partial_16_IC_CONV_offset_group_input_f16_weights_fp16 {
  net.NetworkInfo entryPoint : @IDU_partial_16_IC_CONV_offset_group_input_f16_weights_fp16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xui16>
  }

  func.func @IDU_partial_16_IC_CONV_offset_group_input_f16_weights_fp16() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <163840> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut,
        task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable,
        weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}> DPUCfg : {
        ^bb0(%arg0: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg2: memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg3: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
          VPUIPDPU.IDUCfg {
            VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
            VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
            VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
          }
          VPUIPDPU.PPECfg {
            VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
          }
          VPUIPDPU.ODUCfg {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%arg3 : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
          }
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
        }
      }
      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUVariant @DPUVariant_0_0 invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) {nce_task_type = #VPUIP.nce_task_type<CONV>, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights} DPUCfg : {
          VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(16) size_x(16) size_y(16) size_z(16)
          VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(64)
          VPUIPDPU.IDUSEDense
          VPUIPDPU.IDUGroupConvOp
          VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1) <{isFirstVariant}>
          //CHECK:  UINT workload_size_z = 0x10,
          //CHECK:  UINT workload_start_z = 0x10,
          //CHECK:  UINT noc_clk_en = 1,
          //CHECK:  UINT grp_conv_en = 1,
          //CHECK:  UINT weight_size = 0x40,
          //CHECK:  UINT weight_num = 0x40,
        }
      }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @IDU_CONV_wt_offset_dataptr_zp {
  net.NetworkInfo entryPoint : @IDU_CONV_wt_offset_dataptr_zp inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xui16>
  }

  func.func @IDU_CONV_wt_offset_dataptr_zp() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <163840> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}> DPUCfg : {
        ^bb0(%arg0: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg2: memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg3: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
          VPUIPDPU.IDUCfg {
            VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
            VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
            VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
          }
          VPUIPDPU.PPECfg {
            VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
          }
          VPUIPDPU.ODUCfg {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%arg3 : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
          }
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
        }
      }
      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUVariant @DPUVariant_0_0 invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) {nce_task_type = #VPUIP.nce_task_type<CONV>, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights} DPUCfg : {
          VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
          VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256)
          VPUIPDPU.IDUSEDense
          VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.IDUWeightTableOffsets {data_ptr_zp_offset = 8 : i64}
          //CHECK:  UINT weight_d_ptr_start = 8,
          //CHECK:  UINT weight_sp_ptr_start = 0,
          //CHECK:  weight_start = UINT 0,
        }
      }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @IDU_CONV_wt_offset_dataptr_zp_spptr {
  net.NetworkInfo entryPoint : @IDU_CONV_wt_offset_dataptr_zp_spptr inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xui16>
  }

  func.func @IDU_CONV_wt_offset_dataptr_zp_spptr() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <163840> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}> DPUCfg : {
        ^bb0(%arg0: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg2: memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg3: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
          VPUIPDPU.IDUCfg {
            VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
            VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
            VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
          }
          VPUIPDPU.PPECfg {
            VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
          }
          VPUIPDPU.ODUCfg {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%arg3 : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
          }
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
        }
      }
      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUVariant @DPUVariant_0_0 invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) {nce_task_type = #VPUIP.nce_task_type<CONV>, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights} DPUCfg : {
          VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
          VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256)
          VPUIPDPU.IDUSEDense
          VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.IDUWeightTableOffsets {data_ptr_zp_offset = 8 : i64, sp_ptr_offset = 8 : i64}
          //CHECK:  UINT weight_d_ptr_start = 8,
          //CHECK:  UINT weight_sp_ptr_start = 8,
          //CHECK:  weight_start = UINT 0,
        }
      }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @IDU_CONV_wt_offset_scale {
  net.NetworkInfo entryPoint : @IDU_CONV_wt_offset_scale inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xui16>
  }

  func.func @IDU_CONV_wt_offset_scale() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <163840> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}> DPUCfg : {
        ^bb0(%arg0: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg2: memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg3: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
          VPUIPDPU.IDUCfg {
            VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
            VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
            VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
          }
          VPUIPDPU.PPECfg {
            VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
          }
          VPUIPDPU.ODUCfg {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%arg3 : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
          }
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
        }
      }
      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUVariant @DPUVariant_0_0 invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) {nce_task_type = #VPUIP.nce_task_type<CONV>, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights} DPUCfg : {
          VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
          VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256)
          VPUIPDPU.IDUSEDense
          VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.IDUWeightTableOffsets {scale_offset = 2 : i64}
          //CHECK:  UINT weight_d_ptr_start = 0,
          //CHECK:  UINT weight_sp_ptr_start = 0,
          //CHECK:  weight_start = UINT 0x20000,
        }
      }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @IDU_CONV_wt_offset_bias {
  net.NetworkInfo entryPoint : @IDU_CONV_wt_offset_bias inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xui16>
  }

  func.func @IDU_CONV_wt_offset_bias() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <163840> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}> DPUCfg : {
        ^bb0(%arg0: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg2: memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg3: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
          VPUIPDPU.IDUCfg {
            VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
            VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
            VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
          }
          VPUIPDPU.PPECfg {
            VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
          }
          VPUIPDPU.ODUCfg {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%arg3 : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
          }
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
        }
      }
      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUVariant @DPUVariant_0_0 invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) {nce_task_type = #VPUIP.nce_task_type<CONV>, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights} DPUCfg : {
          VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
          VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256)
          VPUIPDPU.IDUSEDense
          VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.IDUWeightTableOffsets {bias_offset = 2 : i64}
          //CHECK:  UINT weight_d_ptr_start = 0,
          //CHECK:  UINT weight_sp_ptr_start = 0,
          //CHECK:  weight_start = UINT 2,
        }
      }
    }
    return
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @IDU_CONV_wt_offset_scale_bias {
  net.NetworkInfo entryPoint : @IDU_CONV_wt_offset_scale_bias inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xui16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xui16>
  }

  func.func @IDU_CONV_wt_offset_scale_bias() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(32) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUvariant_0_0_0 idx(!VPURegMapped.Index<0:0:0>) <DPUVariant>
      }
      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(1) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer_ActIn !VPUASM.Buffer< "CMX_NN"[0] <131072> : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_Weights !VPUASM.Buffer< "CMX_NN"[0] <164864> : memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_WeightsTable !VPUASM.Buffer< "CMX_NN"[0] <163840> : memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer_ActOut !VPUASM.Buffer< "CMX_NN"[0] <0> : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }
      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUInvariant @DPUInvariant_0_0 <{input = @buffer.CMX_NN.0::@DeclareBuffer_ActIn, nce_task_type = #VPUIP.nce_task_type<CONV>, output = @buffer.CMX_NN.0::@DeclareBuffer_ActOut, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights}> DPUCfg : {
        ^bb0(%arg0: memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<64x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>, %arg2: memref<64x64x2x2xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg3: memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>):
          VPUIPDPU.IDUCfg {
            VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
            VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
            VPUIPDPU.IDUKernel kernel_x(2) kernel_y(2)
            VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
          }
          VPUIPDPU.PPECfg {
            VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
          }
          VPUIPDPU.ODUCfg {
            VPUIPDPU.ODUOutTensorSize dim_x(64) dim_y(64) dim_z(16)
            VPUIPDPU.ODUOutActivations out_activations(%arg3 : memref<1x16x64x64xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)
          }
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:0>) variantCount(1)
        }
      }
      ELF.CreateSection @task.dpu.variant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUVariant @DPUVariant_0_0 invariant(@program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_0) {nce_task_type = #VPUIP.nce_task_type<CONV>, task_index = !VPURegMapped.Index<0:0:0>, task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_0, weight_table = @buffer.CMX_NN.0::@DeclareBuffer_WeightsTable, weights = @buffer.CMX_NN.0::@DeclareBuffer_Weights} DPUCfg : {
          VPUIPDPU.IDUWorkloadSet start_x(0) start_y(0) start_z(0) size_x(16) size_y(16) size_z(64)
          VPUIPDPU.IDUWeightSet weight_start(0) weight_num(64) weight_size(256)
          VPUIPDPU.IDUSEDense
          VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(7) end_coord_y(7) end_coord_z(63)
          VPUIPDPU.BarrierCfg waits([0 : ui16]) updates([1 : ui16]) start_after(0) clean_after(1)
          VPUIPDPU.IDUWeightTableOffsets {bias_offset = 2 : i64, scale_offset = 2 : i64}
          //CHECK:  UINT weight_d_ptr_start = 0,
          //CHECK:  UINT weight_sp_ptr_start = 0,
          //CHECK:  weight_start = UINT 0x20002,
        }
      }
    }
    return
  }
}
