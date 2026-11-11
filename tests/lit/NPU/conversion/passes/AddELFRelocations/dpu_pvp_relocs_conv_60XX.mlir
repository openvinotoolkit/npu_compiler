//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform%" --convert-VPUIPDPU-to-NPUReg60XX --convert-VPUASM-to-NPUReg60XX --create-elf-relocations %s | FileCheck %s
// REQUIRES: dev-build && platform-NPU60XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DPUPVPRelocConvTest
module @DPUPVPRelocConvTest {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x16x16x16xf16>
    DataInfo "input_1" : tensor<16x1x1x1xi64>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x16x64x64xf16>
  }

  func.func @main() {
    ELF.Main {
      ELF.CreateLogicalSection @program.metadata.cmx aligned(64) secType(VPU_SHT_CMX_METADATA) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareTaskBuffer @DeclareTaskBuffer_DPUInvariant_0_0_22 idx(!VPURegMapped.Index<0:0:22>) <DPUInvariant>
      }

      ELF.CreateLogicalSection @buffer.CMX_NN.0 aligned(64) secType(VPU_SHT_CMX_WORKSPACE) secFlags("SHF_NONE") secLocation(<CMX_NN>) {
        VPUASM.DeclareBuffer @DeclareBuffer115 !VPUASM.Buffer< "CMX_NN"[0] <3072> : memref<160x1280x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer123 !VPUASM.Buffer< "CMX_NN"[0] <1659392> : memref<160x1x1x4xsi32, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer397 !VPUASM.Buffer< "CMX_NN"[0] <1679040> : memref<1x1280x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
        VPUASM.DeclareBuffer @DeclareBuffer445 !VPUASM.Buffer< "CMX_NN"[0] <1668800> : memref<1x640x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]> :  swizzling(0)>
      }

      ELF.CreateSection @task.dpu.invariant.0.0 aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|VPU_SHF_PROC_DMA") secLocation(<DDR>) {
        VPUIPDPU.DPUInvariant @DPUInvariant_0_2
            <{input = @buffer.CMX_NN.0::@DeclareBuffer397, is_zero_offset_weights_table, nce_task_type = #VPUIP.nce_task_type<CONV>,
            output = @buffer.CMX_NN.0::@DeclareBuffer445, task_index = !VPURegMapped.Index<0:0:2>,
            task_location = @program.metadata.cmx::@DeclareTaskBuffer_DPUInvariant_0_0_22,
            weight_table = @buffer.CMX_NN.0::@DeclareBuffer123, weights = @buffer.CMX_NN.0::@DeclareBuffer115}>
            DPUCfg : {
            ^bb0(%arg0: memref<1x1280x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>,
            %arg1: memref<160x1x1x4xsi32, [@CMX_NN, 0]>,
            %arg2: memref<160x1280x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>,
            %arg3: memref<1x640x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>):
            VPUIPDPU.IDUCfg {
                VPUIPDPU.IDUInActivations in_activations(%arg0 : memref<1x1280x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
                VPUIPDPU.IDUWeights wmode(f16) wt_plt_cfg(NO_PLT)
                VPUIPDPU.IDUKernel kernel_x(1) kernel_y(1)
                VPUIPDPU.IDUStride stride_x(1) stride_y(1)
                VPUIPDPU.IDUWorkloadCfg workload_type(CONV)
            }
            VPUIPDPU.MPECfg {
                VPUIPDPU.MPEEngineMode mpe_engine_mode(<SCL>)
            }
            VPUIPDPU.PPECfg {
                VPUIPDPU.PPEFpBiasAdd %arg1 : memref<160x1x1x4xsi32, [@CMX_NN, 0]>
                VPUIPDPU.PPEFpScaleMult %arg1 : memref<160x1x1x4xsi32, [@CMX_NN, 0]>
                VPUIPDPU.PPEFpSprLUTMode sprlut_mode(OFF)
                VPUIPDPU.PPEFpPreluMult prelu_alpha(1.000000e+00)
                VPUIPDPU.PPEFpClamp clamp_low(-3.40282347E+38) clamp_high(3.40282347E+38)
                VPUIPDPU.PPEFpConvert convert_mode(FP16) clamp_mode(ON)
                VPUIPDPU.PPEIntZeroPointOffset zero_point_static(0)
            }
            VPUIPDPU.ODUCfg {
                VPUIPDPU.ODUOutTensorSize dim_x(1) dim_y(4) dim_z(640)
                VPUIPDPU.ODUDataReuse activation_reuse(NTHW_4)
                VPUIPDPU.ODUOutActivations out_activations(%arg3 : memref<1x640x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) data_width(ODU_DTYPE_16BIT)

            }
            VPUIPDPU.BarrierCfg waits([15 : ui8]) updates([16 : ui8]) start_after(0) clean_after(0)
            VPUIPDPU.DPUGroup invariantIdx(!VPURegMapped.Index<0:0:2>) variantCount(1)
        }
      }

      ELF.CreateSection @note.MappedInferenceVersion aligned(4) secType(SHT_NOTE) secFlags("SHF_NONE") secLocation(<DDR>) {
        VPUASM.MappedInferenceVersion @MappedInferenceVersion_0_0(11 _ 4 _ 10)
      }

      ELF.CreateSection @program.nnrt_config aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") secLocation(<DDR>) {
        VPUASM.nnrtConfig @MappedInference_nnrtConfigManaged :
      }

      ELF.CreateSection @program.mapped_inference aligned(64) secType(SHT_PROGBITS) secFlags("SHF_ALLOC") secLocation(<DDR>) {
        VPUASM.ManagedMappedInference @MappedInference_managed : dmas ([]) workItems() barrierTasks() bootstrapBarriers() nnrtConfig(@program.nnrt_config::@MappedInference_nnrtConfigManaged) mappedInferenceVersion(@note.MappedInferenceVersion::@MappedInferenceVersion_0_0) {actshv_used = 0 : ui8, barrierConfigurationStride = 0 : i64, barrierConfigurationTasksCount = 0 : i64, barrierCount = 0 : i64, barriersReprogrammingCount = 0 : i64, bootstrapWorkItemsCount = 0 : i64, bootstrapBarriersCount = 0 : i64, disableDmaSwFifo = true, dmaCount = [[0, 0], [0, 0]], dma_from_cmx_used = 0 : ui8, dma_from_ddr_used = 0 : ui8, dpu_used = 1 : ui8, final_barrier_id = 0 : i64, workItemsCount = 0 : i64, workloadManagementBarrierProgrammingMode = #VPURegMapped.workload_management_barrier_programming_mode<ALL_BARRIER_DMAS_SCHEDULED>}
      }

      ELF.CreateSymbolTableSection @symtab secFlags("SHF_NONE") {
        ELF.Symbol @elfsym.program.metadata.cmx of(@program.metadata.cmx) type(<STT_SECTION>) size(82944) value(1075854336)
        ELF.Symbol @elfsym.buffer.CMX_NN.0 of(@buffer.CMX_NN.0) type(<STT_SECTION>) size(1474560) value(1075937280)
        ELF.Symbol @elfsym.task.dpu.invariant.0.0 of(@task.dpu.invariant.0.0) type(<STT_SECTION>)
        ELF.Symbol @elfsym.program.nnrt_config of(@program.nnrt_config) type(<STT_SECTION>)
        ELF.Symbol @elfsym.program.mapped_inference of(@program.mapped_inference) type(<STT_SECTION>)
      }
    }
    return
  }
}

// CHECK:      ELF.CreateRelocationSection @rela.program.mapped_inference.symtab
// DPU PVP relocation points to DpuPVP section
// CHECK:      ELF.Reloc offset({{[0-9]+}}) sourceSym(@symtab::@elfsym.dpu.pvp)
// CHECK-SAME:   relocType(<R_VPU_64>)
// CHECK-SAME:   addend({{[0-9]+}})
