//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --convert-VPUASM-to-NPUReg40XX --convert-VPUIPDPU-to-NPUReg40XX --set-elf-op-offsets %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @mainModule attributes {config.arch = #config.arch_kind<NPU40XX>} {
  config.ExecutorResource 1 of @DMA_NN
  config.Resources 1 of @NCE at 6.000000e+02 MHz
  net.NetworkInfo entryPoint : @tests inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @tests() {
    ELF.Main @elfMain {
      VPUASM.DeclareBuffer @stub !VPUASM.Buffer< "NetworkInput"[0] <0> : memref<1x64x16x16xf16, @DDR> :  swizzling(0)>
    ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<DDR>) {
        VPUASM.DeclareTaskBuffer @stub idx(!VPURegMapped.Index<0:0:0>) <DPUInvariant>
    }
      ELF.CreateSection @text.Barriers aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.ConfigureBarrier @ConfigureBarrier0 idx(!VPURegMapped.Index<0:0:0>) (0) => (-1) counts(4 : 2) {elfMemOffsetAttrKey = 0 : ui64}
        VPUASM.ConfigureBarrier @ConfigureBarrier1 idx(!VPURegMapped.Index<0:0:1>) (1) => (-1) counts(2 : 2) {elfMemOffsetAttrKey = 8 : ui64}
        VPUASM.ConfigureBarrier @ConfigureBarrier2 idx(!VPURegMapped.Index<0:0:1>) (1) => (-1) counts(2 : 2) {elfMemOffsetAttrKey = 8 : ui64}
      }
      ELF.CreateSection @text.nndma0 aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUASM.NNDMA @NNDMA_0_0_0 idx(!VPURegMapped.Index<0:0:0>) taskLocation(@stub) links(@text.nndma0::@NNDMA_0_0_1) input(@stub) outputs([@stub]) waits([]) updates([0 : ui8]) start_after(0) clean_after(0) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>) {elfMemOffsetAttrKey = 0 : ui64}
        VPUASM.NNDMA @NNDMA_0_0_1 idx(!VPURegMapped.Index<0:0:1>) taskLocation(@stub) links(@text.nndma0::@NNDMA_0_0_2) input(@stub) outputs([@stub]) waits([]) updates([0 : ui8]) start_after(0) clean_after(0) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>) {elfMemOffsetAttrKey = 224 : ui64}
        VPUASM.NNDMA @NNDMA_0_0_2 idx(!VPURegMapped.Index<0:0:2>) taskLocation(@stub) links(@text.nndma0::@NNDMA_0_0_3) input(@stub) outputs([@stub]) waits([]) updates([0 : ui8]) start_after(0) clean_after(0) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>) {elfMemOffsetAttrKey = 448 : ui64}
        VPUASM.NNDMA @NNDMA_0_0_3 idx(!VPURegMapped.Index<0:0:3>) taskLocation(@stub) links(@text.nndma0::@NNDMA_0_0_4) input(@stub) outputs([@stub]) waits([]) updates([0 : ui8]) start_after(0) clean_after(0) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>) {elfMemOffsetAttrKey = 672 : ui64}
        VPUASM.NNDMA @NNDMA_0_0_4 idx(!VPURegMapped.Index<0:0:4>) taskLocation(@stub) links(@text.nndma0::@NNDMA_0_0_5) input(@stub) outputs([@stub]) waits([1 : ui8]) updates([]) start_after(0) clean_after(0) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>) {elfMemOffsetAttrKey = 896 : ui64}
        VPUASM.NNDMA @NNDMA_0_0_5 idx(!VPURegMapped.Index<0:0:5>) taskLocation(@stub) input(@stub) outputs([@stub]) waits([1 : ui8]) updates([]) start_after(0) clean_after(0) dma_descriptor(#VPUIP.DMADescriptorAttr<numPlanes = 0 : i4, len = 0 : i4, srcWidth = 0 : i4, srcStride = 0 : i4, srcPlaneStride = 0 : i4, dstWidth = 0 : i4, dstStride = 0 : i4, dstPlaneStride = 0 : i4>) acceleration_mode(<DISABLE>) {elfMemOffsetAttrKey = 1120 : ui64}
      }
      ELF.CreateSection @text.invariants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUInvariant @DPUInvariant0 {task_index = !VPURegMapped.Index<0:0:0>, task_location = @stub, input = @stub, weight_table = @stub, output = @stub, nce_task_type = #VPUIP.nce_task_type<CONV>, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
            DPUCfg : {
              ^bb0(%act_in: memref<1x64x16x16xf16, @DDR>,
                  %act_out: memref<1x64x16x16xf16, @DDR>):
              VPUIPDPU.IDUCfg {
                  VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x64x16x16xf16, @DDR>)
              }
              VPUIPDPU.PPECfg {
                  VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
              }
              VPUIPDPU.ODUCfg {
                  VPUIPDPU.ODUOutTensorSize dim_x(1) dim_y(1) dim_z(1)
                  VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x64x16x16xf16, @DDR>)
              }
          }
        VPUIPDPU.DPUInvariant @DPUInvariant1 {task_index = !VPURegMapped.Index<0:0:1>, task_location = @stub, input = @stub, weight_table = @stub, output = @stub, nce_task_type = #VPUIP.nce_task_type<CONV>, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, start_after = 0 : ui64, clean_after = 0 : ui64}
            DPUCfg : {
              ^bb0(%act_in: memref<1x64x16x16xf16, @DDR>,
                  %act_out: memref<1x64x16x16xf16, @DDR>):
              VPUIPDPU.IDUCfg {
                  VPUIPDPU.IDUInActivations in_activations(%act_in: memref<1x64x16x16xf16, @DDR>)
              }
              VPUIPDPU.PPECfg {
                  VPUIPDPU.PPEFpAddMultBypass bypass_mode(ON)
              }
              VPUIPDPU.ODUCfg {
                  VPUIPDPU.ODUOutTensorSize dim_x(1) dim_y(1) dim_z(1)
                  VPUIPDPU.ODUOutActivations out_activations(%act_out: memref<1x64x16x16xf16, @DDR>)
              }
        }
      }
      ELF.CreateSection @text.variants aligned(64) secType(SHT_PROGBITS) secFlags(SHF_ALLOC) secLocation(<DDR>) {
        VPUIPDPU.DPUVariant @DPUVariant0 invariant(@stub) {task_index = !VPURegMapped.Index<0:0:0>, task_location = @stub, weights = @stub, weight_table = @stub, nce_task_type = #VPUIP.nce_task_type<CONV>, elfMemOffsetAttrKey = 0 : ui64}
        DPUCfg: {
          VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(0) end_coord_y(0) end_coord_z(0)
        }
        VPUIPDPU.DPUVariant @DPUVariant1 invariant(@stub) {task_index = !VPURegMapped.Index<0:0:1>, task_location = @stub, weights = @stub, weight_table = @stub, nce_task_type = #VPUIP.nce_task_type<CONV>, elfMemOffsetAttrKey = 224 : ui64}
        DPUCfg: {
          VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(0) end_coord_y(0) end_coord_z(0)
        }
        ELF.Pad size(224)
        VPUIPDPU.DPUVariant @DPUVariant2 invariant(@stub) {task_index = !VPURegMapped.Index<0:0:2>, task_location = @stub, weights = @stub, weight_table = @stub, nce_task_type = #VPUIP.nce_task_type<CONV>, elfMemOffsetAttrKey = 224 : ui64}
        DPUCfg: {
          VPUIPDPU.ODUOutSubtensor begin_coord_x(0) begin_coord_y(0) begin_coord_z(0) end_coord_x(0) end_coord_y(0) end_coord_z(0)
        }
      }
    }
    return
  }
}

//CHECK: ELF.Main @elfMain

//CHECK: ELF.CreateSection @text.Barriers
//CHECK: NPUReg40XX.ConfigureBarrier descriptor = <
//CHECK:  > {elfMemOffsetAttrKey = 0 : ui64, sym_name = "ConfigureBarrier0"}
//CHECK: NPUReg40XX.ConfigureBarrier descriptor = <
//CHECK:  > {elfMemOffsetAttrKey = 12 : ui64, sym_name = "ConfigureBarrier1"}
//CHECK: NPUReg40XX.ConfigureBarrier descriptor = <
//CHECK:  > {elfMemOffsetAttrKey = 24 : ui64, sym_name = "ConfigureBarrier2"}

//CHECK: ELF.CreateSection @text.nndma0
//CHECK: NPUReg40XX.NNDMA descriptor = <
//CHECK:  > {elfMemOffsetAttrKey = 0 : ui64
//CHECK:   sym_name = "NNDMA_0_0_0"}

//CHECK: NPUReg40XX.NNDMA descriptor = <
//CHECK:  > {elfMemOffsetAttrKey = 224 : ui64
//CHECK:   sym_name = "NNDMA_0_0_1"}

//CHECK: NPUReg40XX.NNDMA descriptor = <
//CHECK:  > {elfMemOffsetAttrKey = 448 : ui64
//CHECK:   sym_name = "NNDMA_0_0_2"}

//CHECK: NPUReg40XX.NNDMA descriptor = <
//CHECK:  > {elfMemOffsetAttrKey = 672 : ui64
//CHECK:   sym_name = "NNDMA_0_0_3"}

//CHECK: NPUReg40XX.NNDMA descriptor = <
//CHECK:  > {elfMemOffsetAttrKey = 896 : ui64
//CHECK:   sym_name = "NNDMA_0_0_4"}

//CHECK: NPUReg40XX.NNDMA descriptor = <
//CHECK:  > {elfMemOffsetAttrKey = 1120 : ui64
//CHECK:   sym_name = "NNDMA_0_0_5"}

//CHECK: ELF.CreateSection @text.invariants
//CHECK: NPUReg40XX.DPUInvariant descriptor = <
//CHECK:   > {elfMemOffsetAttrKey = 0 : ui64
//CHECK:   sym_name = "DPUInvariant0"
//CHECK: NPUReg40XX.DPUInvariant descriptor = <
//CHECK:   > {elfMemOffsetAttrKey = 352 : ui64
//CHECK:   sym_name = "DPUInvariant1"

//CHECK: ELF.CreateSection @text.variants
//CHECK: NPUReg40XX.DPUVariant descriptor = <
//CHECK    > {elfMemOffsetAttrKey = 0 : ui64
//CHECK:   sym_name = "DPUVariant0"
//CHECK:   task_index = !VPURegMapped.Index<0:0:0>
//CHECK: NPUReg40XX.DPUVariant descriptor = <
//CHECK    > {elfMemOffsetAttrKey = 224 : ui64
//CHECK:   sym_name = "DPUVariant1"
//CHECK:   task_index = !VPURegMapped.Index<0:0:1>
//CHECK: NPUReg40XX.DPUVariant descriptor = <
//CHECK    > {elfMemOffsetAttrKey = 672 : ui64
//CHECK:   sym_name = "DPUVariant2"
//CHECK:   task_index = !VPURegMapped.Index<0:0:2>
