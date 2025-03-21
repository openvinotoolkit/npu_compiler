//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/shave_kernel_info.hpp"
#include "vpux/compiler/dialect/VPU/utils/layout_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

namespace {

//
// ConvolutionDimsOrderOpModelForSW
//

class ConvolutionDimsOrderOpModelForSW final :
        public IE::LayoutInfoOpInterface::FallbackModel<ConvolutionDimsOrderOpModelForSW> {
public:
    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(
                info, {DimsOrder::NCHW, DimsOrder::NHWC, DimsOrder::CHW, DimsOrder::HWC});
        info.setInput(1, DimsOrder::OIYX);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
        if (layer == nullptr) {
            return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
        }

        const auto filter = layer.getInputs()[1];

        const auto filterOrder = DimsOrder::fromValue(filter);

        if (filterOrder != DimsOrder::OIYX) {
            return errorAt(op->getLoc(), "Operation filter order is not as expected. filterL={0}, expectedFilterL=OIYX",
                           filterOrder);
        }

        return VPU::verifySameInOutSpecificDimsOrder(
                op, {DimsOrder::NCHW, DimsOrder::NHWC, DimsOrder::CHW, DimsOrder::HWC});
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
    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        info.setInput(0, DimsOrder::NHWC);
        info.setInput(1, DimsOrder::OYXI);

        // FIXME [E#-87197]: NPU37XX ODU supports reordering of the output tensor, so we could use any layout here.
        // But right now current behavior of AdjustLayouts and OptimizeReorder passes might introduce extra Reorders in
        // that case. We need to update the passes to properly handle various Reorder propagation and fusing cases prior
        // enabling ODU permutation feature in NPU37XX.
        info.setOutput(0, DimsOrder::NHWC);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
        if (layer == nullptr) {
            return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
        }
        const auto input = layer.getInputs()[0];
        const auto filter = layer.getInputs()[1];
        const auto output = layer.getOutputs()[0];

        const auto inOrder = DimsOrder::fromValue(input);
        const auto filterOrder = DimsOrder::fromValue(filter);
        const auto outOrder = DimsOrder::fromValue(output);

        if (inOrder != DimsOrder::NHWC) {
            return errorAt(op->getLoc(), "Operation input order is not as expected. inL={0}, expectedInL=NHWC",
                           inOrder);
        }
        if (outOrder.numDims() != inOrder.numDims()) {
            return errorAt(op->getLoc(),
                           "Operation output order is not as expected. outL={0}, it is expected to have same number of "
                           "dims as input",
                           outOrder);
        }
        if (filterOrder != DimsOrder::OYXI) {
            return errorAt(op->getLoc(), "Operation filter order is not as expected. filterL={0}, expectedFilterL=OYXI",
                           filterOrder);
        }
        return mlir::success();
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }
};

//
// NCEDimsOrderOpModelForHW
//

class NCEDimsOrderOpModelForHW final : public IE::LayoutInfoOpInterface::FallbackModel<NCEDimsOrderOpModelForHW> {
public:
    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        info.fill(DimsOrder::NHWC);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
        if (layer == nullptr) {
            return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
        }
        const auto input = layer.getInputs()[0];
        const auto output = layer.getOutputs()[0];

        const auto inOrder = DimsOrder::fromValue(input);
        const auto outOrder = DimsOrder::fromValue(output);

        if (inOrder != DimsOrder::NHWC) {
            return errorAt(op->getLoc(), "Operation input order is not as expected. inL={0}, expectedInL=NHWC",
                           inOrder);
        }
        if (outOrder.numDims() != inOrder.numDims()) {
            return errorAt(op->getLoc(),
                           "Operation output order is not as expected. outL={0}, it is expected to have same number of "
                           "dims as input",
                           outOrder);
        }
        return mlir::success();
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* op) const {
        return IE::getLayoutInfo(op);
    }
};

//
// NCEEltwiseDimsOrderOpModelForHW
//

class NCEEltwiseDimsOrderOpModelForHW final :
        public IE::LayoutInfoOpInterface::FallbackModel<NCEEltwiseDimsOrderOpModelForHW> {
public:
    static void inferLayoutInfo(mlir::Operation*, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        info.fill(DimsOrder::NHWC);
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* op) const {
        auto layer = mlir::dyn_cast<VPU::LayerOpInterface>(op);
        if (layer == nullptr) {
            return errorAt(op, "Operation '{0}' doesn't implement Layer interface", op->getName());
        }
        const auto inputFirst = layer.getInputs()[0];
        const auto inputSecond = layer.getInputs()[1];
        const auto output = layer.getOutputs()[0];

        const auto inOrderFirst = DimsOrder::fromValue(inputFirst);
        const auto inOrderSecond = DimsOrder::fromValue(inputSecond);
        const auto outOrder = DimsOrder::fromValue(output);

        if (inOrderFirst != DimsOrder::NHWC) {
            return errorAt(op->getLoc(), "Operation input order is not as expected. inL={0}, expectedInL=NHWC",
                           inOrderFirst);
        }
        if (inOrderSecond != DimsOrder::NHWC) {
            return errorAt(op->getLoc(), "Operation input order is not as expected. inL={0}, expectedInL=NHWC",
                           inOrderSecond);
        }
        if (outOrder.numDims() != inOrderFirst.numDims()) {
            return errorAt(op->getLoc(),
                           "Operation output order is not as expected. outL={0}, it is expected to have same number of "
                           "dims as input",
                           outOrder);
        }
        return mlir::success();
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* op) const {
        return IE::getLayoutInfo(op);
    }
};

//
// MVNLayoutInfoOpModelForSW
//

class MVNLayoutInfoOpModelForSW final : public IE::LayoutInfoOpInterface::FallbackModel<MVNLayoutInfoOpModelForSW> {
private:
    // Helps setting #NHWC layout for fuse-reshape-mvn pass
    static bool isReshapeFusePattern(mlir::Operation* mvnOp) {
        if (!mlir::isa_and_nonnull<IE::MVNOp>(mvnOp)) {
            return false;
        }
        auto mvnShape = mvnOp->getOperand(0).getType().cast<NDTypeInterface>().getShape();
        auto W = mvnShape[Dims4D::Act::W];
        auto H = mvnShape[Dims4D::Act::H];
        auto C = mvnShape[Dims4D::Act::C];
        // Experimental value, refer to E#148629 for more information
        auto minSize = 32 * 1024;
        if ((W * H < minSize) || (C % 32)) {
            // running small instances through MVN-decomposition degrades perf
            return false;
        }

        auto reshapeBefore = mvnOp->getOperand(0).getDefiningOp<IE::ReshapeOp>();
        if (!reshapeBefore) {
            return false;
        }
        mlir::Operation* reshapeAfter = *mvnOp->getResult(0).getUsers().begin();
        if (!mlir::isa_and_nonnull<IE::ReshapeOp, IE::AffineReshapeOp>(reshapeAfter)) {
            return false;
        }

        // To be removed after E#123528 gets implemented
        if (mlir::isa_and_nonnull<IE::AffineReshapeOp>(reshapeAfter)) {
            mlir::Operation* groupConv = *reshapeAfter->getResult(0).getUsers().begin();
            if (!mlir::isa_and_nonnull<IE::GroupConvolutionOp>(groupConv)) {
                return false;
            }
            mlir::Operation* finReshape = *groupConv->getResult(0).getUsers().begin();
            if (!mlir::isa_and_nonnull<IE::ReshapeOp>(finReshape)) {
                return false;
            }
            reshapeAfter = finReshape;
        }

        // Check Reshapes are symmetrical
        const auto inType = reshapeBefore->getOperand(0).getType().cast<NDTypeInterface>();
        const auto outType = reshapeAfter->getResult(0).getType().cast<NDTypeInterface>();
        const auto inC = inType.getShape()[Dims4D::Act::C];
        const auto outC = outType.getShape()[Dims4D::Act::C];
        if (inC != outC) {
            return false;
        }

        // Check channel constraints
        if ((inC <= C) || (inC % C)) {
            return false;
        }

        return true;
    }

public:
    static void inferLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info, const bool /*seOpsEnabled*/,
                                const bool /*seExperimentalOpsEnabled*/) {
        VPU::inferLayoutInfoSameInOutSpecificDimsOrder(
                info, {DimsOrder::NCHW, DimsOrder::NCWH, DimsOrder::NHWC, DimsOrder::NWHC});

        auto shaveInfo = VPU::getShaveKernelInfo(op);
        const auto shaveVecSize = shaveInfo->getShaveVectorSize();
        if (shaveVecSize.count() == 0) {
            return;
        }

        const auto inputType = op->getOperand(0).getType().cast<NDTypeInterface>();
        const auto inputShape = inputType.getShape();

        if (isReshapeFusePattern(op)) {
            info.setInput(0, DimsOrder::NHWC);
            info.setOutput(0, DimsOrder::NHWC);
            return;
        }

        if (isAcrossChannels(op)) {
            return;
        }

        const auto origOrder = info.getInput(0);
        // only adjusting NHWC MVN to NCHW
        if (origOrder != DimsOrder::NHWC) {
            return;
        }
        // to ensure that we can use odu permute on parent layer
        auto maybeInputReorder = op->getOperand(0).getDefiningOp<IE::ReorderOp>();
        auto parentLayerWithPermute = maybeInputReorder == nullptr
                                              ? getFusableLayerWithPermuteInterface(op)
                                              : getFusableLayerWithPermuteInterface(maybeInputReorder.getOperation());
        if ((maybeInputReorder != nullptr && !maybeInputReorder->hasOneUse()) || parentLayerWithPermute == nullptr) {
            return;
        }
        // If W is not aligned, the output reorder can not be fused into ODU
        auto alignIface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(parentLayerWithPermute.getOperation());
        if (inputShape[Dims4D::Act::W] % alignIface.getOutputChannelAlignment() != 0) {
            return;
        }
        const auto shaveVecElemSize = shaveVecSize.count() / inputType.getElemTypeSize().count();
        const auto isMVNEfficientWithNCHWLayout = [&](int64_t channels) {
            const auto trailingChannels = channels % shaveVecElemSize;
            const auto vectorChannels = channels - trailingChannels;
            // NHWC can be faster when "[(C/VEC_SIZE)*VEC_SIZE]*W > W*H" and "C%VEC_SIZE" is "small" relative to C.
            // But it's hard to decide how "small" the "C%VEC_SIZE" should be, so only convert NHWC back to NCHW
            // when "[(C/VEC_SIZE)*VEC_SIZE]*W <= W*H". We can make this pattern more precise when cost model ready.
            // for details, see: E#105260
            return vectorChannels <= inputShape[Dims4D::Act::H];
        };

        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
        auto tileExec = IE::getTileExecutor(moduleOp);
        auto shaveActExec = tileExec.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT);
        const auto numSplits = tileExec.getCount() * shaveActExec.getCount();

        const auto minChannelsPerSplit = inputShape[Dims4D::Act::C] / numSplits;
        const auto maxChannelsPerSplit = (inputShape[Dims4D::Act::C] + numSplits - 1) / numSplits;
        if (isMVNEfficientWithNCHWLayout(minChannelsPerSplit) || isMVNEfficientWithNCHWLayout(maxChannelsPerSplit)) {
            info.setInput(0, DimsOrder::NCHW);
            info.setOutput(0, DimsOrder::NCHW);
        }
    }

    mlir::LogicalResult verifyLayout(mlir::Operation* origOp) const {
        return VPU::verifySameInOutSpecificDimsOrder(
                origOp, {DimsOrder::NCHW, DimsOrder::NCWH, DimsOrder::NHWC, DimsOrder::NWHC});
    }

    IE::LayerLayoutInfo getLayoutInfo(mlir::Operation* origOp) const {
        return IE::getLayoutInfo(origOp);
    }

    static bool isAcrossChannels(mlir::Operation* op) {
        if (auto mvnOp = mlir::dyn_cast<IE::MVNOp>(op)) {
            return mvnOp.getAcrossChannels();
        } else if (auto mvnOp = mlir::dyn_cast<VPU::MVNOp>(op)) {
            return mvnOp.getAcrossChannels();
        }
        VPUX_THROW("Operation is neither IE.MVNOp nor VPU.MVNOp");
    }
};

//
// redirectLayoutOpInterfacesForVPU
//

void redirectLayoutOpInterfacesForVPU(mlir::DialectRegistry& registry) {
    // Tracking number [E#84955]
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::ConvertOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::SoftMaxOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::SpaceToDepthOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::SwishOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::PReluOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::TileOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::AddOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::SigmoidOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::SignOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::TanOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::SinOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::CosOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::SeluOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::LeakyReluOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::HardSigmoidOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::SoftPlusOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::NegativeOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::ClampOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::ReLUOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::EluOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::HSwishOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::ErfOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::MishOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::FloorOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::RoundOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::TanhOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::SqrtOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::SinhOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::CoshOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::AsinhOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::AcoshOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::AbsOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::AtanOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::AsinOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::AcosOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::AtanhOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::CeilingOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::ExpOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::LogOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::HSigmoidOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::LogSoftmaxOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::PadOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::NegativeOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::CumSumOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::MaximumOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::SubtractOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::DivideOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::InverseOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::DynamicDequantizeOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::PowerOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::ModOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::MinimumOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::LogicalOrOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::LogicalXorOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::BitwiseAndOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::BitwiseOrOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::BitwiseXorOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::LessOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::LessEqualOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::NotEqualOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::GreaterOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::GreaterEqualOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::EqualOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        VPU::FloorModOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);

        VPU::StridedSliceOp::attachInterface<vpux::VPU::SameInOutAnyDimsOrderOpModelForSW>(*ctx);
        VPU::LRNOp::attachInterface<vpux::VPU::SameInOutAnyDimsOrderOpModelForSW>(*ctx);

        VPU::AdaptiveAvgPoolOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::AdaptiveMaxPoolOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        // TODO: E#115645 support HW/SW option for SEP VPU ops
        VPU::RollOp::attachInterface<vpux::VPU::RollDimsOrderOpModelForSW>(*ctx);
        VPU::SelectOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::GridSampleOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::BucketizeOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::PerAxisTileOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::LSTMCellOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::LSTMSequenceOp::attachInterface<vpux::VPU::LSTMSequenceDimsOrderOpModel>(*ctx);
        VPU::NormalizeL2Op::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::NormalizeIEOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::DynamicQuantizeOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::SquaredDifferenceOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::BitwiseNotOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::LogicalNotOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::ScaleShiftOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::GatherTreeOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::FullyConnectedOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::MatMulOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::AvgPoolOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC_NCDHW_NDHWC>(*ctx);
        VPU::ScatterUpdateOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);

        VPU::OneHotOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::DFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::RDFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::IDFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::IRDFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::RDFTUncutOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::IRDFTLastAxisOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::CTCGreedyDecoderOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::NonMaxSuppressionOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::RandomUniformOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::EyeOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::CTCGreedyDecoderSeqLenOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::ExtractImagePatchesOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::ReverseSequenceOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::ScatterNDUpdateOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::GatherOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::GatherNDOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::SpaceToBatch::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::BatchToSpace::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::ReorgYoloOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::RangeOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::NonZeroOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        VPU::DynamicReshapeOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);

        VPU::LRN_IEOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::UpsamplingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::ROIPoolingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::PSROIPoolingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        VPU::DeformablePSROIPoolingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(
                *ctx);
        VPU::MaxPoolOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC_NCDHW_NDHWC>(*ctx);
        VPU::MaxPool8Op::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC_NCDHW_NDHWC>(*ctx);

        VPU::DetectionOutputOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::EmbeddingBagOffsetsSumOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::EmbeddingSegmentsSumOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::EmbeddingBagPackedSumOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::BroadcastOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::ProposalOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::YuvToRgbOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::GRUSequenceOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::GRUSequenceLastPartOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::MemPermuteOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::ScatterElementsUpdateOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        VPU::GatherElementsOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);

        VPU::FakeQuantizeOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW_NHWC>(*ctx);
        VPU::GRNOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW_NHWC>(*ctx);

        VPU::DepthToSpaceOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModel_NHWC>(*ctx);

        VPU::ExtractImagePatchesOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW>(*ctx);

        VPU::MVNOp::attachInterface<MVNLayoutInfoOpModelForSW>(*ctx);
        VPU::MVN6Op::attachInterface<vpux::VPU::SameMultipleInOutDimsOrderOpModelForSW_NCHW_NCWH_NHWC_NWHC>(*ctx);

        VPU::NCEMaxPoolOp::attachInterface<NCEDimsOrderOpModelForHW>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<NCEDimsOrderOpModelForHW>(*ctx);
        VPU::NCEInterpolateOp::attachInterface<NCEDimsOrderOpModelForHW>(*ctx);

        VPU::NCEEltwiseOp::attachInterface<NCEEltwiseDimsOrderOpModelForHW>(*ctx);
        VPU::NCEReduceOp::attachInterface<NCEDimsOrderOpModelForHW>(*ctx);

        VPU::ReduceMaxOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceMeanOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceLogicalOrOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceLogicalAndOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceProdOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceSumOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceMinOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceL1Op::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        VPU::ReduceL2Op::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);

        VPU::AffineReshapeOp::attachInterface<vpux::VPU::AffineReshapeDimsOrderOpModelForSW>(*ctx);

        VPU::TopKOp::attachInterface<vpux::VPU::TopKSameInOutDimsOrderOpModelForSW>(*ctx);

        VPU::QuantizeOp::attachInterface<vpux::VPU::QuantizeDimsOrderOpModelForSW>(*ctx);
        VPU::DequantizeOp::attachInterface<vpux::VPU::DequantizeDimsOrderOpModelForSW>(*ctx);

        VPU::SqueezeOp::attachInterface<vpux::VPU::SqueezeUnsqueezeDimsOrderOpModelForSW>(*ctx);
        VPU::UnsqueezeOp::attachInterface<vpux::VPU::SqueezeUnsqueezeDimsOrderOpModelForSW>(*ctx);

        VPU::RegionYoloOp::attachInterface<vpux::VPU::RegionYoloDimsOrderOpModelForSW>(*ctx);

        VPU::ConvolutionOp::attachInterface<ConvolutionDimsOrderOpModelForSW>(*ctx);
        VPU::GroupConvolutionOp::attachInterface<vpux::VPU::GroupConvolutionLayoutInfoOpModel<
                /*FallbackSWImplOpType=*/ConvolutionDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/NCEConvolutionDimsOrderOpModelForHW>>(*ctx);

        VPU::NCEConvolutionOp::attachInterface<NCEConvolutionDimsOrderOpModelForHW>(*ctx);
        VPU::NCEDepthConvolutionOp::attachInterface<NCEConvolutionDimsOrderOpModelForHW>(*ctx);

        VPU::InterpolateOp::attachInterface<vpux::VPU::InterpolateDimsOrderOpModelForSW>(*ctx);
        VPU::NCEPermuteOp::attachInterface<vpux::VPU::NCEPermuteDimsOrderOpModelForHW>(*ctx);

        VPU::GeluOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModel_NCHW_NHWC>(*ctx);
        VPU::RMSOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        VPU::RoPEOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
    });
}

//
// redirectLayoutOpInterfacesForIE
//

void redirectLayoutOpInterfacesForIE(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::ConvertOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::SoftMaxOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::SpaceToDepthOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::SwishOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::PReluOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::TileOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::SigmoidOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::SignOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::ClampOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::ReLUOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::EluOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::HSwishOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::ErfOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::MishOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::FloorOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::RoundOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::TanhOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::SqrtOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::SinhOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::CoshOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::AsinhOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::AcoshOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::AbsOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::AtanOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::AsinOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::AcosOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::AtanhOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::TanOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::SinOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::CosOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::SeluOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::LeakyReluOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::HardSigmoidOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::NegativeOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::SoftPlusOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::ExpOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::LogOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::CeilingOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::HSigmoidOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::LogSoftmaxOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::NegativeOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::CumSumOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::DivideOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::InverseOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::PowerOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::FloorModOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::ModOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::MinimumOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::MaximumOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::LogicalOrOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::LogicalXorOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::BitwiseAndOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::BitwiseOrOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::BitwiseXorOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::LessOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::LessEqualOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::NotEqualOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::GreaterOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::GreaterEqualOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);
        IE::EqualOp::attachInterface<vpux::VPU::SameAnyDimsOrderOpModelForSW>(*ctx);

        IE::StridedSliceOp::attachInterface<vpux::VPU::SameInOutAnyDimsOrderOpModelForSW>(*ctx);
        IE::LRNOp::attachInterface<vpux::VPU::SameInOutAnyDimsOrderOpModelForSW>(*ctx);

        IE::AdaptiveAvgPoolOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::AdaptiveMaxPoolOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::SelectOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::GridSampleOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::BucketizeOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::PerAxisTileOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::LSTMCellOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::LSTMSequenceOp::attachInterface<vpux::VPU::LSTMSequenceDimsOrderOpModel>(*ctx);
        IE::NormalizeL2Op::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::NormalizeIEOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::DynamicQuantizeOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::SquaredDifferenceOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::BitwiseNotOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::LogicalNotOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::GatherTreeOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::FullyConnectedOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::MatMulOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::AndOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::ScatterUpdateOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);

        IE::ScaleShiftOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);

        IE::OneHotOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::DFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::RDFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::IDFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::IRDFTOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::CTCGreedyDecoderOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::NonMaxSuppressionOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::RandomUniformOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::EyeOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::CTCGreedyDecoderSeqLenOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::ExtractImagePatchesOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::ReverseSequenceOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::ScatterNDUpdateOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::GatherOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::GatherNDOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::SpaceToBatch::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::ReorgYoloOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::RangeOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::NonZeroOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::DynamicReshapeOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);
        IE::DynamicTileOp::attachInterface<vpux::VPU::DefaultDimsOrderOpModelForSW>(*ctx);

        IE::LRN_IEOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::UpsamplingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::ROIPoolingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::PSROIPoolingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(*ctx);
        IE::DeformablePSROIPoolingOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC>(
                *ctx);

        IE::DetectionOutputOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::EmbeddingBagOffsetsSumOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::EmbeddingSegmentsSumOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::EmbeddingBagPackedSumOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::BroadcastOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::ProposalOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::YuvToRgbOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::GRUSequenceOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::GRUSequenceLastPartOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::ReorderOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::MemPermuteOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::ScatterElementsUpdateOp::attachInterface<vpux::VPU::AnyDimsOrderOpModelForSW>(*ctx);
        IE::GatherElementsOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);

        IE::FakeQuantizeOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW_NHWC>(*ctx);
        IE::GRNOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW_NHWC>(*ctx);

        IE::DepthToSpaceOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModel_NHWC>(*ctx);

        IE::ExtractImagePatchesOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModelForSW_NCHW>(*ctx);

        IE::MVNOp::attachInterface<MVNLayoutInfoOpModelForSW>(*ctx);
        IE::MVN6Op::attachInterface<vpux::VPU::SameMultipleInOutDimsOrderOpModelForSW_NCHW_NCWH_NHWC_NWHC>(*ctx);

        IE::ReduceMaxOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        IE::ReduceLogicalOrOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        IE::ReduceLogicalAndOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        IE::ReduceProdOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        IE::ReduceMinOp::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        IE::ReduceL1Op::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);
        IE::ReduceL2Op::attachInterface<vpux::VPU::ReduceDimsOrderOpModelForSW>(*ctx);

        IE::AffineReshapeOp::attachInterface<vpux::VPU::AffineReshapeDimsOrderOpModelForSW>(*ctx);

        IE::TopKOp::attachInterface<vpux::VPU::TopKSameInOutDimsOrderOpModelForSW>(*ctx);

        IE::QuantizeOp::attachInterface<vpux::VPU::QuantizeDimsOrderOpModelForSW>(*ctx);
        IE::DequantizeOp::attachInterface<vpux::VPU::DequantizeDimsOrderOpModelForSW>(*ctx);

        IE::SqueezeOp::attachInterface<vpux::VPU::SqueezeUnsqueezeDimsOrderOpModelForSW>(*ctx);
        IE::UnsqueezeOp::attachInterface<vpux::VPU::SqueezeUnsqueezeDimsOrderOpModelForSW>(*ctx);

        IE::RegionYoloOp::attachInterface<vpux::VPU::RegionYoloDimsOrderOpModelForSW>(*ctx);
        IE::AccumulateOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModel_NHWC>(*ctx);
        IE::RMSOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);
        IE::RoPEOp::attachInterface<vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW>(*ctx);

        IE::GeluOp::attachInterface<vpux::VPU::SameInOutDimsOrderOpModel_NCHW_NHWC>(*ctx);

        // clang-format off
        IE::ConvolutionOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::ConvolutionOp,
                /*FallbackSWImplOpType=*/ConvolutionDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/NCEConvolutionDimsOrderOpModelForHW>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::GroupConvolutionOp,
                /*FallbackSWImplOpType=*/ConvolutionDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/NCEConvolutionDimsOrderOpModelForHW>>(*ctx);
        IE::MaxPoolOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::MaxPoolOp,
                /*FallbackSWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC_NCDHW_NDHWC,
                /*FallbackHWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModel_NHWC>>(*ctx);
        IE::MaxPool8Op::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::MaxPool8Op,
                /*FallbackSWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC_NCDHW_NDHWC,
                /*FallbackHWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModel_NHWC>>(*ctx);
        IE::AvgPoolOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::AvgPoolOp,
                /*FallbackSWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModelForSW_CHW_HWC_NCHW_NHWC_NCDHW_NDHWC,
                /*FallbackHWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModel_NHWC>>(*ctx);
        IE::AddOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::AddOp,
                /*FallbackSWImplOpType=*/vpux::VPU::SameAnyDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModel_NHWC>>(*ctx);
        IE::MultiplyOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::MultiplyOp,
                /*FallbackSWImplOpType=*/vpux::VPU::SameAnyDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModel_NHWC>>(*ctx);
        IE::SubtractOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::SubtractOp,
                /*FallbackSWImplOpType=*/vpux::VPU::SameAnyDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModel_NHWC>>(*ctx);
        IE::ReduceMeanOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::ReduceMeanOp,
                /*FallbackSWImplOpType=*/vpux::VPU::ReduceDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModel_NHWC>>(*ctx);
        IE::ReduceSumOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::ReduceSumOp,
                /*FallbackSWImplOpType=*/vpux::VPU::ReduceDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModel_NHWC>>(*ctx);
        IE::InterpolateOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::InterpolateOp,
                /*FallbackSWImplOpType=*/vpux::VPU::InterpolateDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModel_NHWC>>(*ctx);
        IE::TransposedConvolutionOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::TransposedConvolutionOp,
                /*FallbackSWImplOpType=*/ConvolutionDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/NCEConvolutionDimsOrderOpModelForHW>>(*ctx);
        IE::PadOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::PadOp,
                /*FallbackSWImplOpType=*/vpux::VPU::SameAnyDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::SameInOutDimsOrderOpModel_NHWC>>(*ctx);
        IE::RollOp::attachInterface<
                vpux::VPU::LayoutInfoOpModelForHW<IE::RollOp,
                /*FallbackSWImplOpType=*/vpux::VPU::SameInOutDefaultDimsOrderOpModelForSW,
                /*FallbackHWImplOpType=*/vpux::VPU::RollDimsOrderOpModelForHW>>(*ctx);
        // clang-format on
    });
}

}  // namespace

//
// registerLayoutInfoOpInterfaces
//

void vpux::VPU::arch37xx::registerLayoutInfoOpInterfaces(mlir::DialectRegistry& registry) {
    redirectLayoutOpInterfacesForVPU(registry);
    redirectLayoutOpInterfacesForIE(registry);
}
