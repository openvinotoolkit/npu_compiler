//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace VPU {

mlir::LogicalResult verifyOpLayout(mlir::Operation* op);

//
// SameInOutDefaultDimsOrder
//
mlir::LogicalResult verifySameInOutDimsOrder(mlir::Operation* op);

mlir::LogicalResult verifySameInOutDefaultDimsOrder(mlir::Operation* op);
void inferLayoutInfoSameInOutDefaultDimsOrder(IE::LayerLayoutInfo& info);

mlir::LogicalResult verifyDefaultDimsOrder(mlir::Operation* op);
void inferLayoutInfoDefaultDimsOrder(IE::LayerLayoutInfo& info);

mlir::LogicalResult verifySameAnyDimsOrder(mlir::Operation* op);
void inferLayoutInfoSameAnyDimsOrder(IE::LayerLayoutInfo& info);

mlir::LogicalResult verifySameInOutSpecificDimsOrder(mlir::Operation* op, ArrayRef<DimsOrder> supportedLayouts);
void inferLayoutInfoSameInOutSpecificDimsOrder(IE::LayerLayoutInfo& info, ArrayRef<DimsOrder> supportedLayouts);

mlir::LogicalResult verifySameMultipleInOutSpecificDimsOrder(mlir::Operation* op, ArrayRef<DimsOrder> supportedLayouts);
void inferSameMultipleInOutSpecificDimsOrder(IE::LayerLayoutInfo& info, ArrayRef<DimsOrder> supportedLayouts);

mlir::LogicalResult verifyReduceLayoutInfo(mlir::Operation* op);
void inferReduceLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info);

mlir::LogicalResult verifyAffineReshapeLayoutInfo(mlir::Operation* op);
void inferAffineReshapeLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info);

mlir::LogicalResult verifyRegionYoloLayoutInfo(mlir::Operation* op);
void inferRegionYoloLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info);

void inferLSTMSequenceLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info);
mlir::LogicalResult verifyLSTMSequenceLayoutInfo(mlir::Operation* op);

mlir::LogicalResult verifyInterpolateLayoutInfo(mlir::Operation* op);
void inferInterpolateLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info);

mlir::LogicalResult verifyQuantizeLayoutInfo(mlir::Operation* op);
void inferQuantizeLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info);
mlir::LogicalResult verifyDequantizeLayoutInfo(mlir::Operation* op);
void inferDequantizeLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info);

DimsOrder inferSqueezeOutputLayout(const DimArr& inPerm, const SmallVector<int64_t>& axesVec,
                                   ArrayRef<int64_t> inShape);
DimsOrder inferUnsqueezeOutputLayout(const DimArr& inPerm, const SmallVector<int64_t>& axesVec,
                                     ArrayRef<int64_t> inShape);
void inferSqueezeUnsqueezeLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info);

mlir::LogicalResult verifyNCEConvolutionLayoutInfo(mlir::Operation* op);
mlir::LogicalResult verifyTopKLayoutInfo(mlir::Operation* op);
mlir::LogicalResult verifyScatterNDUpdateLayoutInfo(mlir::Operation* op);
mlir::LogicalResult verifyNCEPermuteLayoutInfo(mlir::Operation* op);
mlir::LogicalResult verifySWGroupConvolutionLayoutInfo(mlir::Operation* op);

mlir::LogicalResult verifyRollLayoutInfo(mlir::Operation* op);
void inferRollLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info);

template <class OrigOpType, class FallbackSWImplOpType, class FallbackHWImplOpType>
class LayoutInfoOpModelForHW final :
        public IE::LayoutInfoOpInterface::ExternalModel<
                LayoutInfoOpModelForHW<OrigOpType, FallbackSWImplOpType, FallbackHWImplOpType>, OrigOpType> {
public:
    void inferLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info, const bool seOpsEnabled,
                         const bool seExperimentalOpsEnabled) const {
        if (!canBeExecutedOnNCE(origOp, seOpsEnabled, seExperimentalOpsEnabled)) {
            FallbackSWImplOpType::inferLayoutInfo(origOp, info, seOpsEnabled, seExperimentalOpsEnabled);
            return;
        }

        FallbackHWImplOpType::inferLayoutInfo(origOp, info, seOpsEnabled, seExperimentalOpsEnabled);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return IE::verifyLayout(origOp);
    }

private:
    static bool canBeExecutedOnNCE(mlir::Operation* op, const bool seOpsEnabled, const bool seExperimentalOpsEnabled) {
        if (VPU::getCompilationMode(op) == VPU::CompilationMode::ReferenceSW) {
            // We are in reference SW compilation mode
            return false;
        }

        if (!seExperimentalOpsEnabled && mlir::isa<IE::PadOp, IE::RollOp>(op)) {
            return false;
        }

        if (!seOpsEnabled && mlir::isa<IE::SEOpInterface>(op) && !mlir::isa<IE::PadOp, IE::RollOp>(op)) {
            return false;
        }

        if (VPU::NCEInvariant::isSupported(op).failed()) {
            // Basic NCE invariants check failed, the operation will fallback to SW mode
            return false;
        }

        return true;
    }
};

//
// AnyDimsOrderOpModelForSW
//

class AnyDimsOrderOpModelForSW final : public IE::LayoutInfoOpInterface::FallbackModel<AnyDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& /*info*/, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return IE::verifyLayout(origOp);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// SameAnyDimsOrderOpModelForSW
//

class SameAnyDimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<SameAnyDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferLayoutInfoSameAnyDimsOrder(info);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifySameAnyDimsOrder(origOp);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// SameInOutAnyDimsOrderOpModelForSW
//

class SameInOutAnyDimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<SameInOutAnyDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        const auto inOrder = info.getInput(0);
        info.setOutput(0, inOrder);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifySameInOutDimsOrder(origOp);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// SameInOutDefaultDimsOrderOpModelForSW
//

class SameInOutDefaultDimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<SameInOutDefaultDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferLayoutInfoSameInOutDefaultDimsOrder(info);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifySameInOutDefaultDimsOrder(origOp);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// DefaultDimsOrderOpModelForSW
//

class DefaultDimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<DefaultDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferLayoutInfoDefaultDimsOrder(info);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifyDefaultDimsOrder(origOp);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC
//

class SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC final :
        public IE::LayoutInfoOpInterface::FallbackModel<SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(
                info, {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC});
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifySameInOutSpecificDimsOrder(
                origOp, {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC});
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

class SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC_NCDHW_NDHWC final :
        public IE::LayoutInfoOpInterface::FallbackModel<SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC_NCDHW_NDHWC> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW,
                                                              DimsOrder::NHWC, DimsOrder::NCDHW, DimsOrder::NDHWC});
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifySameInOutSpecificDimsOrder(origOp, {DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW,
                                                              DimsOrder::NHWC, DimsOrder::NCDHW, DimsOrder::NDHWC});
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// SameInOutDimsOrderOpModelForSW_NC_CHW_HWC_NCHW_NHWC
//

class SameInOutDimsOrderOpModelForSW_NC_CHW_HWC_NCHW_NHWC final :
        public IE::LayoutInfoOpInterface::FallbackModel<SameInOutDimsOrderOpModelForSW_NC_CHW_HWC_NCHW_NHWC> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(
                info, {DimsOrder::NC, DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC});
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifySameInOutSpecificDimsOrder(
                origOp, {DimsOrder::NC, DimsOrder::CHW, DimsOrder::HWC, DimsOrder::NCHW, DimsOrder::NHWC});
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// SameInOutDimsOrderOpModelForSW_NCHW_NHWC
//

class SameInOutDimsOrderOpModelForSW_NCHW_NHWC final :
        public IE::LayoutInfoOpInterface::FallbackModel<SameInOutDimsOrderOpModelForSW_NCHW_NHWC> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::NCHW, DimsOrder::NHWC});
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifySameInOutSpecificDimsOrder(origOp, {DimsOrder::NCHW, DimsOrder::NHWC});
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// SameInOutDimsOrderOpModelForSW_NCHW
//

class SameInOutDimsOrderOpModelForSW_NCHW final :
        public IE::LayoutInfoOpInterface::FallbackModel<SameInOutDimsOrderOpModelForSW_NCHW> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(info, {DimsOrder::NCHW});
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifySameInOutSpecificDimsOrder(origOp, {DimsOrder::NCHW});
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// SameMultipleInOutDimsOrderOpModelForSW_NCHW_NCWH_NHWC_NWHC
//

class SameMultipleInOutDimsOrderOpModelForSW_NCHW_NCWH_NHWC_NWHC final :
        public IE::LayoutInfoOpInterface::FallbackModel<SameMultipleInOutDimsOrderOpModelForSW_NCHW_NCWH_NHWC_NWHC> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferSameMultipleInOutSpecificDimsOrder(
                info, {DimsOrder::NCHW, DimsOrder::NCWH, DimsOrder::NHWC, DimsOrder::NWHC});
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifySameMultipleInOutSpecificDimsOrder(
                origOp, {DimsOrder::NCHW, DimsOrder::NCWH, DimsOrder::NHWC, DimsOrder::NWHC});
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// ReduceDimsOrderOpModelForSW
//

class ReduceDimsOrderOpModelForSW final : public IE::LayoutInfoOpInterface::FallbackModel<ReduceDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferReduceLayoutInfo(op, info);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifyReduceLayoutInfo(origOp);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// AffineReshapeDimsOrderOpModelForSW
//

class AffineReshapeDimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<AffineReshapeDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferAffineReshapeLayoutInfo(op, info);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifyAffineReshapeLayoutInfo(origOp);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// QuantizeDimsOrderOpModelForSW
//

class QuantizeDimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<QuantizeDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferQuantizeLayoutInfo(op, info);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        return VPU::verifyQuantizeLayoutInfo(op);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// DequantizeDimsOrderOpModelForSW
//

class DequantizeDimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<DequantizeDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferDequantizeLayoutInfo(op, info);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        return VPU::verifyDequantizeLayoutInfo(op);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// SqueezeUnsqueezeDimsOrderOpModelForSW
//

class SqueezeUnsqueezeDimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<SqueezeUnsqueezeDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferSqueezeUnsqueezeLayoutInfo(op, info);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation*) const {
        return mlir::success();
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// RegionYoloDimsOrderOpModelForSW
//

class RegionYoloDimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<RegionYoloDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferRegionYoloLayoutInfo(op, info);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        return VPU::verifyRegionYoloLayoutInfo(op);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// InterpolateDimsOrderOpModelForSW
//

class InterpolateDimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<InterpolateDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferInterpolateLayoutInfo(op, info);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        return VPU::verifyInterpolateLayoutInfo(op);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// NCEConvolutionDimsOrderOpModelForHW
//

class NCEConvolutionDimsOrderOpModelForHW final :
        public IE::LayoutInfoOpInterface::FallbackModel<NCEConvolutionDimsOrderOpModelForHW> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        info.setInput(0, DimsOrder::NHWC);
        info.setInput(1, DimsOrder::OYXI);
        info.setOutput(0, DimsOrder::NHWC);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        return VPU::verifyNCEConvolutionLayoutInfo(op);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// PermuteQuantizeDimsOrderOpModelForSW
//

class PermuteQuantizeDimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<PermuteQuantizeDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        info.setInput(0, DimsOrder::NHWC);
        info.setOutput(0, DimsOrder::NWCH);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation*) const {
        // Tracking number [E#86928]
        return mlir::success();
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// SameInOutDimsOrderOpModel_NCHW_NHWC
//

class SameInOutDimsOrderOpModel_NCHW_NHWC final :
        public IE::LayoutInfoOpInterface::FallbackModel<SameInOutDimsOrderOpModel_NCHW_NHWC> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        const auto inOrder = info.getInput(0);
        if (inOrder != DimsOrder::NHWC && inOrder != DimsOrder::NCHW) {
            info.setInput(0, DimsOrder::NCHW);
            info.setOutput(0, DimsOrder::NCHW);
        } else {
            VPU::inferLayoutInfoSameAnyDimsOrder(info);
        }
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifySameAnyDimsOrder(origOp);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// SameInOutDimsOrderOpModel_NHWC
//

class SameInOutDimsOrderOpModel_NHWC final :
        public IE::LayoutInfoOpInterface::FallbackModel<SameInOutDimsOrderOpModel_NHWC> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        info.fill(DimsOrder::NHWC);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifySameInOutSpecificDimsOrder(origOp, {DimsOrder::NHWC});
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// TopKSameInOutDimsOrderOpModelForSW
//

class TopKSameInOutDimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<TopKSameInOutDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        const auto inOrder = info.getInput(0);

        info.setInput(0, inOrder);
        info.setOutput(0, inOrder);
        info.setOutput(1, inOrder);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        return VPU::verifyTopKLayoutInfo(op);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

template <class FallbackSWImplOpType, class FallbackHWImplOpType>
class GroupConvolutionLayoutInfoOpModel final :
        public IE::LayoutInfoOpInterface::ExternalModel<
                GroupConvolutionLayoutInfoOpModel<FallbackSWImplOpType, FallbackHWImplOpType>,
                VPU::GroupConvolutionOp> {
public:
    void inferLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info, const bool seOpsEnabled,
                         const bool seExperimentalOpsEnabled) const {
        auto groupConvOp = mlir::dyn_cast<VPU::GroupConvolutionOp>(origOp);
        VPUX_THROW_WHEN(groupConvOp == nullptr,
                        "GroupConvolutionLayoutInfoOpModel is only supported for VPU_GroupConvolutionOp");
        if (!canBeExecutedOnNCE(origOp, seExperimentalOpsEnabled)) {
            FallbackSWImplOpType::inferLayoutInfo(origOp, info, seOpsEnabled, seExperimentalOpsEnabled);
            return;
        }

        FallbackHWImplOpType::inferLayoutInfo(origOp, info, seOpsEnabled, seExperimentalOpsEnabled);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        auto groupConvOp = mlir::dyn_cast<VPU::GroupConvolutionOp>(origOp);
        VPUX_THROW_WHEN(groupConvOp == nullptr,
                        "GroupConvolutionLayoutInfoOpModel is only supported for VPU_GroupConvolutionOp");
        return vpux::VPU::isDilatedGroupConv(groupConvOp) ? IE::verifyLayout(origOp)
                                                          : VPU::verifySWGroupConvolutionLayoutInfo(origOp);
    }

private:
    static bool canBeExecutedOnNCE(mlir::Operation* op, const bool seExperimentalOpsEnabled) {
        if (VPU::getCompilationMode(op) == VPU::CompilationMode::ReferenceSW) {
            // We are in reference SW compilation mode
            return false;
        }

        if (!seExperimentalOpsEnabled) {
            return false;
        }
        auto log = Logger::global().nest("can-be-executed-on-nce-group-conv", 0);
        const auto logCb = [&](const formatv_object_base& msg) {
            log.trace("{0}", msg.str());
        };
        auto groupConvOp = mlir::cast<VPU::GroupConvolutionOp>(op);
        VPUX_THROW_WHEN(groupConvOp == nullptr,
                        "GroupConvolutionLayoutInfoOpModel is only supported for VPU_GroupConvolutionOp");
        return vpux::VPU::isSupportedSEPDilatedConv(groupConvOp, logCb, /*checkLayout=*/false,
                                                    /*checkChannelAlignment=*/true);
    }
};

//
// NCEPermuteDimsOrderOpModelForHW
//

class NCEPermuteDimsOrderOpModelForHW final :
        public IE::LayoutInfoOpInterface::FallbackModel<NCEPermuteDimsOrderOpModelForHW> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        info.setInput(0, DimsOrder::NCHW);
        info.setOutput(0, DimsOrder::NHWC);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        return VPU::verifyNCEPermuteLayoutInfo(op);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// RollDimsOrderOpModelForHW
//

class RollDimsOrderOpModelForHW final : public IE::LayoutInfoOpInterface::FallbackModel<RollDimsOrderOpModelForHW> {
public:
    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seTransposedConvEnabled*/) {
        info.setInput(0, DimsOrder::NHWC);
        info.setInput(1, DimsOrder::C);
        info.setInput(2, DimsOrder::C);

        info.setOutput(0, DimsOrder::NHWC);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
        if (layer == nullptr) {
            return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
        }
        const auto data = layer.getInputs()[0];
        const auto shift = layer.getInputs()[1];
        const auto axes = layer.getInputs()[2];
        const auto output = layer.getOutputs()[0];

        const auto dataOrder = DimsOrder::fromValue(data);
        const auto shiftOrder = DimsOrder::fromValue(shift);
        const auto axesOrder = DimsOrder::fromValue(axes);
        const auto outOrder = DimsOrder::fromValue(output);

        if (dataOrder != DimsOrder::NHWC) {
            return errorAt(op->getLoc(), "Operation input order is not as expected. inL={0}, expectedInL=NHWC",
                           dataOrder);
        }
        if (outOrder.numDims() != dataOrder.numDims()) {
            return errorAt(op->getLoc(),
                           "Operation output order is not as expected. outL={0}, it is expected to have same number of "
                           "dims as input",
                           outOrder);
        }
        if (shiftOrder != DimsOrder::C || axesOrder != DimsOrder::C) {
            return errorAt(op->getLoc(), "Operation shift/axes order is not as expected");
        }
        return mlir::success();
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// RollDimsOrderOpModelForSW
//

class RollDimsOrderOpModelForSW final : public IE::LayoutInfoOpInterface::FallbackModel<RollDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seTransposedConvEnabled*/) {
        VPU::inferRollLayoutInfo(op, info);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        return VPU::verifyRollLayoutInfo(op);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// LSTMSequenceDimsOrderOpModel
//

class LSTMSequenceDimsOrderOpModel final :
        public IE::LayoutInfoOpInterface::FallbackModel<LSTMSequenceDimsOrderOpModel> {
public:
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        inferLSTMSequenceLayoutInfo(op, info);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        return verifyLSTMSequenceLayoutInfo(op);
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

}  // namespace VPU
}  // namespace vpux
