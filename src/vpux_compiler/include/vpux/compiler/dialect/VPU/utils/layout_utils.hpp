//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
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
    void inferLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info) const {
        if (!canBeExecutedOnNCE(origOp)) {
            FallbackSWImplOpType::inferLayoutInfo(origOp, info);
            return;
        }

        FallbackHWImplOpType::inferLayoutInfo(origOp, info);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return IE::verifyLayout(origOp);
    }

private:
    static bool canBeExecutedOnNCE(mlir::Operation* op) {
        if (config::getCompilationMode(op) == config::CompilationMode::ReferenceSW) {
            // We are in reference SW compilation mode
            return false;
        }

        const auto moduleOp = getModuleOp(op);
        if (!config::hasEnableSEPtrsOperations(moduleOp) && mlir::isa<IE::SEOpInterface>(op) &&
            !mlir::isa<IE::RollOp, IE::GroupConvolutionOp, IE::GroupTransposedConvolutionOp>(op)) {
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
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& /*info*/) {
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
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
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
// SameMultipleInOutDimsOrderOpModelForSW_NCHW_CHW_NCWH_CWH_NHWC_HWC_NWHC_WHC
//

class SameMultipleInOutDimsOrderOpModelForSW_NCHW_CHW_NCWH_CWH_NHWC_HWC_NWHC_WHC final :
        public IE::LayoutInfoOpInterface::FallbackModel<
                SameMultipleInOutDimsOrderOpModelForSW_NCHW_CHW_NCWH_CWH_NHWC_HWC_NWHC_WHC> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
        VPU::inferSameMultipleInOutSpecificDimsOrder(
                info, {DimsOrder::NCHW, DimsOrder::CHW, DimsOrder::NCWH, DimsOrder::CWH, DimsOrder::NHWC,
                       DimsOrder::HWC, DimsOrder::NWHC, DimsOrder::WHC});
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifySameMultipleInOutSpecificDimsOrder(
                origOp, {DimsOrder::NCHW, DimsOrder::CHW, DimsOrder::NCWH, DimsOrder::CWH, DimsOrder::NHWC,
                         DimsOrder::HWC, DimsOrder::NWHC, DimsOrder::WHC});
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
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
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
// ReduceL2DimsOrderOpModelForSW
//

class ReduceL2DimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<ReduceL2DimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
        const auto axesValue = mlir::dyn_cast_or_null<mlir::ArrayAttr>(op->getAttr("axes_value"));
        if (axesValue != nullptr) {
            const auto axes = parseIntArrayAttr<int64_t>(axesValue);
            const auto inputOrder = DimsOrder::fromValue(op->getOperand(0));

            if (axes.size() == 1) {
                const auto axis = axes.front();
                // Preserve innermost axis to achieve best performance for the current kernel implementation, follow
                // for more details E-231049
                if (inputOrder.dimPos(Dim(axis)) == inputOrder.numDims() - 1) {
                    info.setInput(0, inputOrder);
                }
            }
        }

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
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
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
// SqueezeUnsqueezeDimsOrderOpModelForSW
//

class SqueezeUnsqueezeDimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<SqueezeUnsqueezeDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
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
// FlashSDPADimsOrderOpModel
//

class FlashSDPADimsOrderOpModel final : public IE::LayoutInfoOpInterface::FallbackModel<FlashSDPADimsOrderOpModel> {
public:
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
        auto valueTensorIdx = 2u;
        for (auto i : irange(op->getNumOperands())) {
            if (i == valueTensorIdx) {
                info.setInput(i, DimsOrder::NCWH);
            } else {
                info.setInput(i, DimsOrder::NCHW);
            }
        }

        for (auto i : irange(op->getNumResults())) {
            info.setOutput(i, DimsOrder::NCHW);
        }
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
        if (layer == nullptr) {
            return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
        }

        const auto valueIdx = 2;
        for (auto [idx, opOperand] : enumerate(op->getOpOperands())) {
            auto expectedOrder = DimsOrder::NCHW;
            if (opOperand.getOperandNumber() == valueIdx) {
                expectedOrder = DimsOrder::NCWH;
            }

            auto inOrder = DimsOrder::fromValue(opOperand.get());
            if (inOrder != expectedOrder) {
                return errorAt(op->getLoc(),
                               "Operation '{0}' has unsupported input layout '{1}' for operand '{2}'. Expected '{3}'.",
                               op->getName(), inOrder, idx, expectedOrder);
            }
        }

        for (auto [idx, result] : enumerate(op->getResults())) {
            auto expectedOrder = DimsOrder::NCHW;
            auto outOrder = DimsOrder::fromValue(result);

            if (outOrder != expectedOrder) {
                return errorAt(op->getLoc(),
                               "Operation '{0}' has unsupported output layout '{1}' for result '{2}'. Expected '{3}'.",
                               op->getName(), outOrder, idx, expectedOrder);
            }
        }

        return mlir::success();
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* op) const {
        return IE::getLayoutInfo(op);
    }
};

//
// NCEConvolutionDimsOrderOpModelForHW
//

class NCEConvolutionDimsOrderOpModelForHW final :
        public IE::LayoutInfoOpInterface::FallbackModel<NCEConvolutionDimsOrderOpModelForHW> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
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
    void inferLayoutInfo(mlir::Operation* origOp, IE::LayerLayoutInfo& info) const {
        auto groupConvOp = mlir::dyn_cast<VPU::GroupConvolutionOp>(origOp);
        VPUX_THROW_WHEN(groupConvOp == nullptr,
                        "GroupConvolutionLayoutInfoOpModel is only supported for VPU_GroupConvolutionOp");

        if (!canBeExecutedOnNCE(origOp)) {
            FallbackSWImplOpType::inferLayoutInfo(origOp, info);
            return;
        }

        FallbackHWImplOpType::inferLayoutInfo(origOp, info);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        auto groupConvOp = mlir::dyn_cast<VPU::GroupConvolutionOp>(origOp);
        VPUX_THROW_WHEN(groupConvOp == nullptr,
                        "GroupConvolutionLayoutInfoOpModel is only supported for VPU_GroupConvolutionOp");
        return vpux::VPU::isDilatedGroupConv(groupConvOp) ? IE::verifyLayout(origOp)
                                                          : VPU::verifySWGroupConvolutionLayoutInfo(origOp);
    }

private:
    static bool canBeExecutedOnNCE(mlir::Operation* op) {
        if (config::getCompilationMode(op) == config::CompilationMode::ReferenceSW) {
            // We are in reference SW compilation mode
            return false;
        }

        const auto moduleOp = getModuleOp(op);
        if (!config::hasEnableExperimentalSEPtrsOperations(moduleOp)) {
            return false;
        }
        auto log = Logger::global().nest("can-be-executed-on-nce-group-conv", 0);
        const auto logCb = [&](const formatv_object_base& msg) {
            log.trace("{0}", msg.str());
        };
        auto groupConvOp = mlir::cast<VPU::GroupConvolutionOp>(op);
        VPUX_THROW_WHEN(groupConvOp == nullptr,
                        "GroupConvolutionLayoutInfoOpModel is only supported for VPU_GroupConvolutionOp");
        auto seOp = mlir::dyn_cast<IE::SEOpInterface>(groupConvOp.getOperation());
        return seOp && seOp.isSupported(logCb, /*checkLayout=*/false, /*checkChannelAlignment=*/true);
    }
};

//
// NCEPermuteDimsOrderOpModelForHW
//

class NCEPermuteDimsOrderOpModelForHW final :
        public IE::LayoutInfoOpInterface::FallbackModel<NCEPermuteDimsOrderOpModelForHW> {
public:
    static void inferLayoutInfo(mlir::Operation* /*op*/, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
        info.setInput(0, DimsOrder::NHWC);

        const auto shiftRank = static_cast<int64_t>(getShape(op->getOperand(1)).size());
        if (shiftRank == 4) {
            info.setInput(1, DimsOrder::NCHW);
            info.setInput(2, DimsOrder::NCHW);
        } else if (shiftRank == 1) {
            info.setInput(1, DimsOrder::C);
            info.setInput(2, DimsOrder::C);
        }

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
        const auto shiftRank = static_cast<int64_t>(getShape(shift).size());
        if (shiftRank == 4 && (shiftOrder != DimsOrder::NCHW || axesOrder != DimsOrder::NCHW)) {
            return errorAt(op->getLoc(), "Operation shift order is not as expected. inL={0}, expectedInL=NCHW",
                           shiftOrder);
        }
        if (shiftRank == 1 && (shiftOrder != DimsOrder::C || axesOrder != DimsOrder::C)) {
            return errorAt(op->getLoc(), "Operation shift order is not as expected. inL={0}, expectedInL=C",
                           shiftOrder);
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
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
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
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info) {
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
