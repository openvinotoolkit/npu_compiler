//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg40XX/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/lower_to_registers.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace vpux::VPURegMapped;
using namespace npu40xx;
using namespace NPUReg40XX;

namespace vpux {
namespace vpuipdpu2npureg40xx {

DPUInvariantRewriter::DPUInvariantRewriter(mlir::MLIRContext* ctx, Logger log, VPU::DPUDryRunMode dryRunMode)
        : mlir::OpRewritePattern<VPUIPDPU::DPUInvariantOp>(ctx), _log(log), _dryRunMode(dryRunMode) {
    setDebugName("DPUInvariant_VPUASM2NPUReg40XXRewriter");
}

mlir::LogicalResult DPUInvariantRewriter::matchAndRewrite(VPUIPDPU::DPUInvariantOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    NPUReg40XX::Descriptors::DpuInvariantRegister descriptor;
    // fill default configuration
    descriptor.write<Fields::cmx_slice0_low_addr>(0x4000000);
    descriptor.write<Fields::cmx_slice1_low_addr>(0x4000000);
    descriptor.write<Fields::cmx_slice2_low_addr>(0x4000000);
    descriptor.write<Fields::cmx_slice3_low_addr>(0x4000000);
    descriptor.write<Fields::cmx_slice_size>(0x00018000);
    descriptor.write<Fields::ppe_scale_round>(3);
    descriptor.write<Fields::ppe_prelu_mult>(0x1);
    descriptor.write<Fields::ppe_scale_hclamp>(0x7FFFFFFF);
    descriptor.write<Fields::ppe_scale_lclamp>(int64_t(-2147483648));  // 0x80000000

    if (_dryRunMode == VPU::DPUDryRunMode::STUB) {
        _log.trace("DPU dry run mode = 'stub', updating invariant descriptor");
        fillStubCfg(descriptor);
    } else {
        fillIDUCfg(origOp.getRegion(), descriptor);
        fillMPECfg(origOp.getRegion(), descriptor);
        fillPPECfg(origOp.getRegion(), descriptor);
        fillODUCfg(origOp.getRegion(), descriptor);
    }
    fillBarrierCfg(origOp, descriptor);
    fillProfilingCfg(origOp, descriptor);

    auto taskListCfgOp = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::DPUGroupOp>());
    if (taskListCfgOp.size() == 1) {
        descriptor.write<Fields::variant_count_>(taskListCfgOp[0].getVariantCount());
    }

    descriptor.write<Fields::nvar_tag>(origOp.getIndex() + 1);

    auto regDPUInvariantAttr = DpuInvariantRegisterAttr::get(rewriter.getContext(), std::move(descriptor));

    rewriter.create<NPUReg40XX::DPUInvariantOp>(
            origOp->getLoc(), origOp.getSymNameAttr(), origOp.getTaskIndexAttr(), regDPUInvariantAttr,
            origOp.getTaskLocationAttr(), origOp.getInputAttr(), origOp.getInputSparsityMapAttr(),
            origOp.getInputStorageElementTableAttr(), origOp.getWeightsAttr(), origOp.getWeightsSparsityMapAttr(),
            origOp.getWeightTableAttr(), origOp.getSprLookupTableAttr(), origOp.getOutputAttr(),
            origOp.getOutputSparsityMapAttr(), origOp.getProfilingDataAttr(), origOp.getIsZeroOffsetWeightsTableAttr(),
            origOp.getNceTaskTypeAttr(), origOp.getIsContinuedAttr());

    rewriter.eraseOp(origOp);

    return mlir::success();
}

using DpuInvariantDescriptorType = NPUReg40XX::Descriptors::DpuInvariantRegister;

void DPUInvariantRewriter::fillIDUCfg(mlir::Region& DPURegion, DpuInvariantDescriptorType& descriptor) const {
    auto IDUCfgOps = DPURegion.getOps<VPUIPDPU::IDUCfgOp>();
    if (!IDUCfgOps.empty()) {
        auto IDUCfgOp = *IDUCfgOps.begin();

        for (auto& IDUOp : IDUCfgOp.getRegion().getOps()) {
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUStorageElementOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUStorageElementOp<VPUIPDPU::arch40xx::FieldsIDUStorageElementOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUKernelOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUKernelOp<VPUIPDPU::arch40xx::FieldsIDUKernelOp>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUStrideOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUStrideOp<VPUIPDPU::arch40xx::FieldsIDUStrideOp>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUInActivationsOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUInActivationsOp<VPUIPDPU::arch40xx::FieldsIDUInActivationsOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUInputLayerCfgOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUInputLayerCfgOp<VPUIPDPU::arch40xx::FieldsIDUInputLayerCfgOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWeightsOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUWeightsOp<VPUIPDPU::arch40xx::FieldsIDUWeightsOp>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWorkloadCfgOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUWorkloadCfgOp<VPUIPDPU::arch40xx::FieldsIDUWorkloadCfgOp>(op,
                                                                                                           descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUDepthWiseCfgOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUDepthWiseCfgOp<VPUIPDPU::arch40xx::FieldsIDUDepthWiseCfgOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUEltWiseCfgOp>(&IDUOp)) {
                VPUIPDPU::arch40xx::lowerToRegIDUEltWiseCfgOp<VPUIPDPU::arch40xx::FieldsIDUEltWiseCfgOp>(op,
                                                                                                         descriptor);
            } else {
                VPUX_THROW("Unknown IDU operation: {0}", IDUOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillMPECfg(mlir::Region& DPURegion, DpuInvariantDescriptorType& descriptor) const {
    auto MPECfgOps = DPURegion.getOps<VPUIPDPU::MPECfgOp>();
    if (!MPECfgOps.empty()) {
        auto MPECfgOp = *MPECfgOps.begin();

        for (auto& MPEOp : MPECfgOp.getRegion().getOps()) {
            if (mlir::dyn_cast_or_null<VPUIPDPU::MPEDenormalOperandsFTZOp>(&MPEOp)) {
                VPUIPDPU::lowerToRegMPEDenormalOperandsFTZOp<Fields::mpe_daz>(descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::MPEActivationBiasOp>(&MPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegMPEActivationBiasOp<Fields::mpe_actbias>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::MPEWeightsBiasOp>(&MPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegMPEWeightsBiasOp<Fields::mpe_wtbias>(op, descriptor);
            } else {
                VPUX_THROW("Unknown MPE operation: {0}", MPEOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillPPECfg(mlir::Region& DPURegion, DpuInvariantDescriptorType& descriptor) const {
    auto PPECgfOps = DPURegion.getOps<VPUIPDPU::PPECfgOp>();
    if (!PPECgfOps.empty()) {
        auto PPECgfOp = *PPECgfOps.begin();

        for (auto& PPEOp : PPECgfOp.getRegion().getOps()) {
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpBiasAddOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEFpBiasAddOp<VPUIPDPU::arch40xx::FieldsPPEFpBiasAddOp>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpScalePreluMultOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEFpScalePreluMultOp<VPUIPDPU::arch40xx::FieldsPPEFpScalePreluMultOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpAddMultBypassOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEFpAddMultBypassOp<Fields::ppe_fp_bypass>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpConvertOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEFpConvertOp<VPUIPDPU::arch40xx::FieldsPPEFpConvertOp>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntBiasAddOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntBiasAddOp<VPUIPDPU::arch40xx::FieldsPPEIntBiasAddOp>(op,
                                                                                                         descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntScaleMultOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntScaleMultOp<VPUIPDPU::arch40xx::FieldsPPEIntScaleMultOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntPreluMultOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntPreluMultOp<Fields::ppe_prelu_mult>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntScaleShiftOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntScaleShiftOp<VPUIPDPU::arch40xx::FieldsPPEIntScaleShiftOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntPreluShiftOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntPreluShiftOp<Fields::ppe_prelu_shift>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntRoundOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntRoundOp<Fields::ppe_scale_round>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntZeroPointOffsetOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntZeroPointOffsetOp<Fields::ppe_g8_bias_c>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntClampOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntClampOp(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntConvertOp>(&PPEOp)) {
                VPUIPDPU::arch40xx::lowerToRegPPEIntConvertOp<Fields::ppe_i32_convert>(op, descriptor);
            } else {
                VPUX_THROW("Unknown PPE operation {0}", PPEOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillODUCfg(mlir::Region& DPURegion, DpuInvariantDescriptorType& descriptor) const {
    auto ODUCfgOps = DPURegion.getOps<VPUIPDPU::ODUCfgOp>();
    if (!ODUCfgOps.empty()) {
        auto ODUCfgOp = *ODUCfgOps.begin();

        // TODO: E#80766 select optimal write combine mode and serialize based on VPUIPDU instruction
        descriptor.write<Fields::wcb_bypass>(0);

        // Statically set bits that should not be part of functional defaults

        // Not used by HW. Setting to 1 to be coeherent with GFile.
        descriptor.write<Fields::addr_format_sel>(1);

        // TODO: E#81883 need to figure this out why it's always set to 1?
        descriptor.write<Fields::rst_ctxt>(1);

        // TODO: E#82814 should it be a  defailt value? this is hardcoded and directly copied from POC runtime...
        descriptor.write<Fields::base_offset_a>(0x200);

        for (auto& ODUOp : ODUCfgOp.getRegion().getOps()) {
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUOutTensorSizeOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUOutTensorSizeOp<VPUIPDPU::arch40xx::FieldsODUOutTensorSizeOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUDataReuseOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUDataReuseOp<Fields::nthw>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUPermuteDataOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUPermuteDataOp<Fields::permutation>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUSparsityOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUSparsityOp<VPUIPDPU::arch40xx::FieldsODUSparsityOp>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUSwizzleDataOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUSwizzleDataOp<Fields::swizzle_key>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUOutActivationsOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUOutActivationsOp<VPUIPDPU::arch40xx::FieldsODUOutActivationsOp>(
                        op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUMemoryModeOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUMemoryModeOp<Fields::mode>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUCmxPortsOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUCmxPortsOp<Fields::cmx_port_muxing_disable>(op, descriptor);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUWriteCombineBufferOp>(&ODUOp)) {
                VPUIPDPU::arch40xx::lowerToRegODUWriteCombineBufferOp<
                        VPUIPDPU::arch40xx::FieldsODUWriteCombineBufferOp>(op, descriptor);
            } else {
                VPUX_THROW("Unknown ODU operation {0}", ODUOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillBarrierCfg(VPUIPDPU::DPUInvariantOp origOp,
                                          DpuInvariantDescriptorType& descriptor) const {
    VPUIPDPU::arch40xx::lowerToRegBarrierCfgOpWithDPUInvariantParent<VPUIPDPU::arch40xx::RegistersInvariantBarrierCfg,
                                                                     VPUIPDPU::arch40xx::FieldsInvariantBarrierCfg>(
            origOp, descriptor);
}

void DPUInvariantRewriter::fillProfilingCfg(VPUIPDPU::DPUInvariantOp origOp,
                                            DpuInvariantDescriptorType& descriptor) const {
    if (!origOp.getProfilingData().has_value()) {
        return;
    }
    descriptor.write<Fields::hwp_en>(1);
    descriptor.write<Fields::hwp_stat_mode>(3);
}

void DPUInvariantRewriter::fillStubCfg(DpuInvariantDescriptorType& descriptor) const {
    descriptor.write<Fields::tensor_size_x>(0x1);
    descriptor.write<Fields::tensor_size_y>(0x1);
    descriptor.write<Fields::tensor_size_z>(0x10);
    descriptor.write<Fields::workload_operation>(0x0);
    descriptor.write<Fields::zm_input>(0x1);
    descriptor.write<Fields::kernel_y>(0x1);
    descriptor.write<Fields::kernel_x>(0x1);
    descriptor.write<Fields::elop_wload>(0x1);
    descriptor.write<Fields::elop_wload_type>(0x1);
    descriptor.write<Fields::te_dim_y>(0x0);
    descriptor.write<Fields::te_dim_z>(0xF);
    descriptor.write<Fields::te_dim_x>(0x0);
    descriptor.write<Fields::nthw>(0x1);
}

}  // namespace vpuipdpu2npureg40xx
}  // namespace vpux
