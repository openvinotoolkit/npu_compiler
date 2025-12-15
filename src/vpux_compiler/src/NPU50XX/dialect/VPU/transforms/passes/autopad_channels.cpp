//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/attr_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>
#include <memory>

namespace vpux::VPU::arch50xx {
#define GEN_PASS_DECL_AUTOPADCHANNELS
#define GEN_PASS_DEF_AUTOPADCHANNELS
#include "vpux/compiler/NPU50XX/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU::arch50xx

using namespace vpux;

namespace {

class AutopadChannelsPass final : public VPU::arch50xx::impl::AutopadChannelsBase<AutopadChannelsPass> {
public:
    explicit AutopadChannelsPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final {
        auto funcOp = getOperation();
        auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
        const auto autoPaddingIDUEnabled = config::hasAutoPaddingIDU(moduleOp);
        const auto autoPaddingODUEnabled = config::hasAutoPaddingODU(moduleOp);
        if (!autoPaddingIDUEnabled && !autoPaddingODUEnabled) {
            _log.trace("Autopad is disabled");
            return;
        }

        funcOp.walk([&](VPU::NCEOpInterface nceOp) {
            _log.trace("Got {0} at {1}", nceOp->getName(), nceOp->getLoc());

            const auto hasInputPaddingAttr = nceOp->hasAttr(VPU::INPUT_PADDING_ATTR_NAME);
            const auto hasOutputPaddingAttr = nceOp->hasAttr(VPU::OUTPUT_PADDING_ATTR_NAME);
            if (!hasInputPaddingAttr && !hasOutputPaddingAttr) {
                _log.nest().trace("Operation has no padding attribute");
                return;
            }

            // NCECompressConvolution is a special variant of NCEConvolution which can consume IC<=4. This operation is
            // not meant to be used when autopad is enabled, as autopad allows both IC and OC to be lower than 16
            // for NCE operations
            if (mlir::isa<VPU::NCECompressConvolutionOp>(nceOp)) {
                _log.nest().trace("NCECompressConvolutionOp does not support autopad");
                return;
            }

            if (autoPaddingIDUEnabled) {
                tryEnablingIDUAutopad(nceOp);
            }
            if (autoPaddingODUEnabled) {
                tryEnablingODUAutopad(nceOp);
            }
        });
    }

    // Check whether the operation is compatible with IDU autopad and whether the parent operation is a compatible
    // Expand operation. If the constraints are satisfied, the parent Expand operation is discarded and the weights /
    // weight table constants are adjusted for the new input channel size
    void tryEnablingIDUAutopad(VPU::NCEOpInterface nceOp) {
        _log.nest().trace("Trying to autopad IDU");

        const auto nceConvOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(nceOp.getOperation());
        if (nceConvOp == nullptr) {
            _log.nest().trace("Only NCE convolutions support IDU autopad");
            return;
        }

        const auto logCb = [&](const formatv_object_base& msg) {
            _log.nest().trace("{0}", msg.str());
        };
        if (!VPU::canConsumeIDUAutopad(nceConvOp, logCb)) {
            _log.nest().trace("Operation cannot consume IDU autopad");
            return;
        }

        const auto inputType = mlir::cast<NDTypeInterface>(nceOp->getOperand(0).getType());
        const auto inputChannels = inputType.getShape()[Dims4D::Act::C];
        const auto inputPadding =
                parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(nceOp->getAttr(VPU::INPUT_PADDING_ATTR_NAME)));
        auto unpaddedInputChannels = inputChannels - inputPadding[Dims4D::Act::C.ind()];

        Const::DeclareOp weightsTableOp = nullptr;
        if (auto weightsTable = nceOp.getWeightsTableOperand()) {
            const auto constOp = weightsTable.getDefiningOp<Const::DeclareOp>();
            if (constOp == nullptr) {
                _log.nest().trace("Cannot find the weights table");
                return;
            }
            weightsTableOp = constOp;
        }

        auto parentExpandOp = mlir::dyn_cast_if_present<VPU::ExpandOp>(nceOp->getOperand(0).getDefiningOp());
        if (parentExpandOp == nullptr) {
            _log.nest().trace("Parent operation is not Expand");
            return;
        }

        const auto padsBegin = parseIntArrayAttr<int64_t>(parentExpandOp.getPadsBeginAttr());
        const auto padsEnd = parseIntArrayAttr<int64_t>(parentExpandOp.getPadsEndAttr());
        if (padsBegin.size() != padsEnd.size() || static_cast<int64_t>(padsBegin.size()) != inputType.getRank()) {
            _log.nest().trace("Unsupported Expansion pads: {0}, {1}", padsBegin, padsEnd);
            return;
        }

        if (padsBegin.front() != 0 || !llvm::all_equal(padsBegin)) {
            _log.nest().trace("Expected all beginning pads to be zero, but got: {0}", padsBegin);
            return;
        }
        for (int i = 0; i < static_cast<int>(padsEnd.size()); ++i) {
            if (i == Dims4D::Act::C.ind()) {
                if (padsEnd[i] > inputPadding[Dims4D::Act::C.ind()]) {
                    _log.nest().trace("Expected end padding at index {0} to be <={1}, but got: {2}", i,
                                      inputPadding[Dims4D::Act::C.ind()], padsEnd);
                    return;
                } else if (padsEnd[i] < inputPadding[Dims4D::Act::C.ind()]) {
                    unpaddedInputChannels = inputChannels - padsEnd[i];
                }
                continue;
            }
            if (padsEnd[i] != 0) {
                _log.nest().trace("Expected end padding at index {0} to be zero, but got: {1}", i, padsEnd);
                return;
            }
        }

        // Experiments have shown that it is faster for ODU to write data and for IDU to read data when the channels are
        // aligned to four, compared to when they are unpadded and smaller than four. A common pattern seen in networks
        // is NCEPermute -> NCEConvolution, where the NCEPermute produces three channels; in this case, the NCEPermute
        // could write four channels (where one is padded), to gain a bit of performance
        // Note: this is only done if the data is not float, as float data may require explicit padding for the input of
        // NCEPermute, in order to avoid NaNs / INFs (see AddExplicitPaddingBeforeNCEPermutePass)
        auto parentNCEPermuteOp = parentExpandOp.getInput().getDefiningOp<VPU::NCEPermuteOp>();
        auto parentNCEPermuteElementType =
                mlir::cast<vpux::NDTypeInterface>(parentExpandOp.getInput().getType()).getElementType();
        const auto padExtraChannels = unpaddedInputChannels < 4 && parentNCEPermuteOp != nullptr &&
                                      !parentNCEPermuteElementType.isF16() && parentExpandOp.getInput().hasOneUse() &&
                                      parentExpandOp.getOutput().hasOneUse();
        if (padExtraChannels) {
            unpaddedInputChannels = 4;
        }

        // The value of `unpaddedInputChannels` could have been changed while verifying the padding introduced by Expand
        // (e.g. if the Expand only partly pads the data, as it receives pre-padded data itself). In this case, it is
        // necessary to make sure that the NCE operation can consume this partly-padded data
        if (!VPU::areChannelsCompatibleWithIDUAutoPad(unpaddedInputChannels, inputType.getElemTypeSize().count())) {
            logCb(formatv("Unpadded input channels {0} are not supported", unpaddedInputChannels));
            return;
        };

        autopadInput(nceOp, unpaddedInputChannels, weightsTableOp);

        if (padExtraChannels) {
            auto permuteType = mlir::cast<NDTypeInterface>(parentNCEPermuteOp.getOutput().getType());
            auto permuteShape = Shape(permuteType.getShape());
            permuteShape[Dims4D::Act::C] = unpaddedInputChannels;
            parentNCEPermuteOp.getOutput().setType(permuteType.changeShape(permuteShape));
            parentNCEPermuteOp.setExpandedChannels(unpaddedInputChannels);
        }

        nceOp->setOperand(0, parentExpandOp.getInput());
        if (parentExpandOp->use_empty()) {
            parentExpandOp->erase();
        }
    }

    void tryEnablingODUAutopad(VPU::NCEOpInterface nceOp) {
        _log.nest().trace("Trying to autopad ODU");

        if (!nceOp->hasAttr(VPU::OUTPUT_PADDING_ATTR_NAME)) {
            _log.nest().trace("Missing output_padding attribute");
            return;
        }
        const auto outputPaddingAttr = nceOp->getAttr(VPU::OUTPUT_PADDING_ATTR_NAME);
        if (outputPaddingAttr == nullptr) {
            _log.nest().trace("output_padding attribute is null");
            return;
        }

        const auto outputType = mlir::cast<NDTypeInterface>(nceOp->getResult(0).getType());
        if (outputType.getRank() != 4) {
            _log.nest().trace("Only 4D results are currently supported for autopad");
            return;
        }

        const auto outputChannels = outputType.getShape()[Dims4D::Act::C];
        const auto outputPadding = parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(outputPaddingAttr));
        const auto unpaddedOutputChannels = outputChannels - outputPadding[Dims4D::Act::C.ind()];
        if (unpaddedOutputChannels < 0) {
            _log.nest().trace("Invalid number of unpadded output channels: {0}", unpaddedOutputChannels);
            return;
        }

        const auto elemTypeBitWidth = outputType.getElemTypeSize().count();
        const auto canUseAutopad = outputChannels == VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT &&
                                   VPU::areChannelsCompatibleWithODUAutoPad(unpaddedOutputChannels, elemTypeBitWidth);
        if (!canUseAutopad) {
            _log.nest().trace("Cannot autopad output");
            return;
        }

        if (nceOp.getWeightTableDataPtrOperand() != nullptr || nceOp.getWeightTableSpPtrOperand() != nullptr ||
            nceOp.getWeightZeroPointsOperand() != nullptr) {
            _log.nest().trace("The split weight / sparsity pointer / zero-points tables are not currently "
                              "supported with ODU autopad");
            return;
        }

        // Find and verify the weights constant in case the weights are sparse, as it may be necessary to remove the
        // presence of sparsity in case the weights become dense when ODU autopad is enabled
        if (auto weights = nceOp.getWeightsOperand()) {
            if (mlir::isa<VPU::SparseTensorType>(weights.getType())) {
                // Skip intermediate operations that could have been introduced if IDU autopad was enabled
                auto parentOp = weights.getDefiningOp();
                while (isSupportedIntermediateOpForSparseWeights(parentOp)) {
                    parentOp = parentOp->getOperand(0).getDefiningOp();
                }
                // Skip intermediate grouping operation for sparsity
                if (auto groupOp = mlir::dyn_cast_if_present<VPU::GroupSparseTensorOp>(parentOp)) {
                    parentOp = groupOp.getData().getDefiningOp();
                }
                // Find the actual weights constant
                const auto constOp = mlir::dyn_cast_if_present<Const::DeclareOp>(parentOp);
                if (constOp == nullptr) {
                    _log.nest().trace("Cannot find the weights constant");
                    return;
                }
                auto contentAttr = constOp.getContentAttr();
                const auto transformations = contentAttr.getTransformations();
                if (transformations.empty() || !mlir::isa<Const::SparsifyAttr>(transformations.back())) {
                    _log.nest().trace("Weights constant does not have a Sparsify transformation at the end, despite "
                                      "the weights type being sparse");
                    return;
                }
            }
        }

        Const::DeclareOp weightsTableOp;
        if (auto weightsTable = nceOp.getWeightsTableOperand()) {
            const auto constOp = weightsTable.getDefiningOp<Const::DeclareOp>();
            if (constOp == nullptr) {
                _log.nest().trace("Cannot find the weights table");
                return;
            }
            weightsTableOp = constOp;
        }

        // Note: This could be extended to cases where any Shave operation is an user. These operations would need
        // an extra Expand operation to be added after them, in order for their output to remain compatible with the
        // rest of the IR.
        // This sort of cases should be enabled if identified in some specific models, as the introduction of the
        // Expand operation could affect the performance in some situations
        _log.nest().trace("Checking if all users are compatible");
        if (!hasCompatibleUsers(nceOp, unpaddedOutputChannels)) {
            _log.nest(2).trace("Not all users are compatible");
            return;
        }
        _log.nest(2).trace("All users are compatible");

        autopadOutput(nceOp, unpaddedOutputChannels, weightsTableOp);
    }

    void autopadInput(VPU::NCEOpInterface nceOp, int64_t unpaddedInputChannels, Const::DeclareOp weightsTableOp) {
        _log.nest().debug("Autopadding input to {0} channels for {1} at {2}", unpaddedInputChannels, nceOp->getName(),
                          nceOp.getLoc());

        // Slice weights as well, if they exist:
        // - slice the input channels from the weights; e.g. from 16x16x1x1xf16 to 16x3x1x1xf16
        // - flatten the weights shape; e.g. from 16x3x1x1xf16 to 16x1x1x3xf16
        // - align the weight set size to a multiple of 16 bytes; e.g. from 16x1x1x3xf16 to 16x1x1x8xf16
        // The flattening and alignment is done to satisfy the hardware requirement where each weight pointer is aligned
        // to 16 byte. The layout of the weights is set as OYXI, to maintain compatibility with the operation
        if (nceOp.getWeightsOperand() != nullptr) {
            mlir::OpBuilder builder(nceOp);
            const auto weightsType = mlir::cast<NDTypeInterface>(nceOp.getWeightsOperand().getType());
            const auto origWeightsShape = weightsType.getShape();
            SmallVector<int64_t> offsets(weightsType.getRank(), 0);
            SmallVector<int64_t> shape(origWeightsShape.raw());
            shape[Dims4D::Filter::IC.ind()] = unpaddedInputChannels;
            auto sliceOp = builder.create<VPU::SliceOp>(takeOpLoc(nceOp, "weights_slice"), nceOp.getWeightsOperand(),
                                                        offsets, shape);

            const auto weightSetSize =
                    shape[Dims4D::Filter::IC.ind()] * shape[Dims4D::Filter::KY.ind()] * shape[Dims4D::Filter::KX.ind()];
            SmallVector<int64_t> newShape(shape.size());
            newShape[Dims4D::Filter::OC.ind()] = shape[Dims4D::Filter::OC.ind()];
            newShape[Dims4D::Filter::IC.ind()] = 1;
            newShape[Dims4D::Filter::KY.ind()] = 1;
            newShape[Dims4D::Filter::KX.ind()] = weightSetSize;
            auto reshapeOp =
                    builder.create<VPU::ReshapeOp>(takeOpLoc(nceOp, "weights_flatten"), sliceOp.getResult(),
                                                   /*shape=*/nullptr,
                                                   /*special_zero=*/false, getIntArrayAttr(&getContext(), newShape));

            auto layoutCastOp =
                    builder.create<VPU::LayoutCastOp>(takeOpLoc(nceOp, "weights_layout_cast"), reshapeOp.getOutput(),
                                                      DimsOrder::OYXI.toAffineMap(&getContext()));

            // When the weight pointers are computed by the DPU during the inference, the weight sets offsets are
            // computed based on the `weight_size` register. This value of this register must have the input channels
            // aligned to 16, so it is therefore necessary to pad the weights constant as if it had the input channels
            // aligned to 16 as well
            const auto weightSetsNeedPaddedIC =
                    nceOp.getWeightsTableOperand() == nullptr && nceOp.getWeightTableDataPtrOperand() == nullptr;
            const auto paddedWeightSetSize = VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT *
                                             origWeightsShape[Dims4D::Filter::KY] *
                                             origWeightsShape[Dims4D::Filter::KX];

            const auto alignment = weightSetsNeedPaddedIC
                                           ? paddedWeightSetSize
                                           : VPU::NCEInvariant::getAlignment(weightsType.getElementType());
            const auto remainder = weightSetSize % alignment;
            auto alignedWeights = layoutCastOp.getOutput();
            if (remainder != 0) {
                const auto padsBegin = std::move(offsets);
                auto padsEnd = padsBegin;
                padsEnd[Dims4D::Filter::KX.ind()] += alignment - remainder;
                auto expandOp = builder.create<VPU::ExpandOp>(takeOpLoc(nceOp, "weights_align"), alignedWeights,
                                                              getIntArrayAttr(&getContext(), padsBegin),
                                                              getIntArrayAttr(&getContext(), padsEnd));
                alignedWeights = expandOp->getResult(0);
            }

            nceOp.getWeightsOperand().replaceUsesWithIf(alignedWeights, [&](mlir::OpOperand& operand) -> bool {
                return operand.getOwner() == nceOp.getOperation();
            });
        }

        // Update the raw filter shape attribute
        if (auto op = mlir::dyn_cast<VPU::NCEConvolutionOp>(nceOp.getOperation())) {
            auto filterShape = parseIntArrayAttr<int64_t>(op.getRawFilterShape());
            filterShape[Dims4D::Filter::IC.ind()] = unpaddedInputChannels;
            op.setRawFilterShapeAttr(getIntArrayAttr(nceOp.getContext(), filterShape));
        } else if (auto op = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(nceOp.getOperation())) {
            auto filterShape = parseIntArrayAttr<int64_t>(op.getRawFilterShape());
            filterShape[Dims4D::Filter::IC.ind()] = unpaddedInputChannels;
            op.setRawFilterShapeAttr(getIntArrayAttr(nceOp.getContext(), filterShape));
        }

        // The weights table must still have its weights pointers updated
        if (weightsTableOp != nullptr) {
            const auto weightsTableContent = weightsTableOp.getContent();
            auto weightsTableValues = weightsTableContent.vec<int32_t>();

            auto weightsType = mlir::cast<NDTypeInterface>(nceOp.getWeightsOperand().getType());
            auto weightsShape = weightsType.getShape();
            const auto weightPtrStep = weightsShape.back() * weightsType.getElemTypeSize().to<Byte>().count();
            auto sparsityPtrStep = static_cast<int64_t>(0);
            if (auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(weightsType)) {
                if (sparseType.getSparsityMap() != nullptr) {
                    // The sparsity map type is expected to be in the format OCx1x1xWEIGHT_SET_SIZExi1, where
                    // WEIGHT_SET_SIZE is the flattened weight set aligned to 128-bits
                    auto sparsityMapShape = mlir::cast<NDTypeInterface>(sparseType.getSparsityMap()).getShape();
                    sparsityPtrStep = sparsityMapShape.back() / CHAR_BIT;
                }
            }
            const auto outputChannels = weightsShape[Dims4D::Filter::OC];
            constexpr auto numElemPerOC = VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC;
            for (int64_t oc = 0; oc < outputChannels; ++oc) {
                weightsTableValues[oc * numElemPerOC + 0] = checked_cast<int32_t>(oc * weightPtrStep);
                weightsTableValues[oc * numElemPerOC + 1] = checked_cast<int32_t>(oc * sparsityPtrStep);
            }

            mlir::OpBuilder builder(weightsTableOp);
            const auto dataType = mlir::cast<mlir::RankedTensorType>(weightsTableOp.getContentAttr().getType());
            const auto newWeightsTable =
                    Const::createConst(builder, weightsTableOp->getLoc(), dataType, ArrayRef(weightsTableValues));
            weightsTableOp->getResult(0).replaceUsesWithIf(newWeightsTable, [&](mlir::OpOperand& operand) -> bool {
                return operand.getOwner() == nceOp.getOperation();
            });
        }

        nceOp->setAttr(VPU::INPUT_PADDING_ATTR_NAME, nullptr);
    }

    void autopadOutput(VPU::NCEOpInterface nceOp, int64_t unpaddedOutputChannels, Const::DeclareOp weightsTableOp) {
        _log.nest().debug("Autopadding output to {0} channels for {1} at {2}", unpaddedOutputChannels, nceOp->getName(),
                          nceOp.getLoc());

        const auto unpadChannels = [&](NDTypeInterface type) -> NDTypeInterface {
            auto tileShape = Shape(type.getShape());
            tileShape[Dims4D::Act::C] = unpaddedOutputChannels;
            const auto tileOffsets = Shape(tileShape.size(), 0);
            return type.extractDenseTile(tileOffsets, tileShape);
        };

        const auto outputType = mlir::cast<NDTypeInterface>(nceOp->getResult(0).getType());
        const auto origOutputChannels = outputType.getShape()[Dims4D::Act::C];

        const auto newType = unpadChannels(outputType);
        nceOp->getResult(0).setType(newType);

        // Update the users to receive the new output directly
        for (auto userOp : nceOp->getUsers()) {
            auto actualUserOp = userOp;
            while (isCompatibleViewLikeOp(actualUserOp)) {
                auto outputType = mlir::cast<NDTypeInterface>(actualUserOp->getResult(0).getType());
                const auto newType = unpadChannels(outputType);
                actualUserOp->getResult(0).setType(newType);
                actualUserOp = *actualUserOp->getResult(0).getUsers().begin();
            }

            if (auto sliceOp = mlir::dyn_cast_if_present<VPU::SliceOp>(actualUserOp)) {
                actualUserOp = *sliceOp->getResult(0).getUsers().begin();
                sliceOp->getResult(0).replaceAllUsesWith(sliceOp->getOperand(0));
            } else if (auto nceUserOp = mlir::dyn_cast_if_present<VPU::NCEOpInterface>(actualUserOp)) {
                Const::DeclareOp userWeightsTableOp = nullptr;
                if (auto weightsTable = nceUserOp.getWeightsTableOperand()) {
                    userWeightsTableOp = weightsTable.getDefiningOp<Const::DeclareOp>();
                }
                autopadInput(nceUserOp, unpaddedOutputChannels, userWeightsTableOp);
            }
        }

        // Slice weights as well, if they exist
        bool droppedWeightsSparsity = false;
        if (nceOp.getWeightsOperand() != nullptr) {
            mlir::OpBuilder builder(nceOp);
            const auto weightsType = mlir::cast<NDTypeInterface>(nceOp.getWeightsOperand().getType());
            SmallVector<int64_t> offsets(weightsType.getRank(), 0);
            SmallVector<int64_t> shape(weightsType.getShape().raw());
            shape[Dims4D::Filter::OC.ind()] = unpaddedOutputChannels;
            auto sliceOp = builder.create<VPU::SliceOp>(takeOpLoc(nceOp, "weights_slice"), nceOp.getWeightsOperand(),
                                                        offsets, shape);
            nceOp.getWeightsOperand().replaceUsesWithIf(sliceOp->getResult(0), [&](mlir::OpOperand& operand) -> bool {
                return operand.getOwner() == nceOp.getOperation();
            });

            // If the weights are of SparseTensorType and now they are dense, drop weights sparsity
            if (auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(nceOp.getWeightsOperand().getType())) {
                if (sparseType.getSparsityCompression() != nullptr) {
                    const auto compressedSize =
                            sparseType.getSparsityCompression().getAllocSize(weightsType.getElementType()).count();
                    const auto sparsityRatio =
                            VPU::NCESparsity::getSparsityRatio(mlir::cast<NDTypeInterface>(sparseType), compressedSize);
                    _log.nest(2).trace("Sparsity ratio: {0}", sparsityRatio);
                    if (sparsityRatio == 0) {
                        _log.nest(3).trace("Dropping weights sparsity");
                        droppedWeightsSparsity = true;
                        dropWeightsSparsity(nceOp);
                    }
                }
            }
        }

        // Update the raw filter shape attribute
        if (auto op = mlir::dyn_cast<VPU::NCEConvolutionOp>(nceOp.getOperation())) {
            auto filterShape = parseIntArrayAttr<int64_t>(op.getRawFilterShape());
            filterShape[Dims4D::Filter::OC.ind()] = unpaddedOutputChannels;
            op.setRawFilterShapeAttr(getIntArrayAttr(nceOp.getContext(), filterShape));
        } else if (auto op = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(nceOp.getOperation())) {
            auto filterShape = parseIntArrayAttr<int64_t>(op.getRawFilterShape());
            filterShape[Dims4D::Filter::OC.ind()] = unpaddedOutputChannels;
            op.setRawFilterShapeAttr(getIntArrayAttr(nceOp.getContext(), filterShape));
        }

        // The weights table must still remain aligned to 16 channels.
        // Ensure the extra pointers from the weights table still point to data in CMX, even if the weights /
        // weights sparsity map have been sliced
        if (weightsTableOp != nullptr) {
            const auto weightsTableContent = weightsTableOp.getContent();
            auto weightsTableValues = weightsTableContent.vec<int32_t>();
            const auto lastChannelIdx = unpaddedOutputChannels - 1;
            constexpr auto numElemsPerOC = VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC;
            for (int64_t oc = unpaddedOutputChannels; oc < origOutputChannels; ++oc) {
                weightsTableValues[oc * numElemsPerOC] = weightsTableValues[lastChannelIdx * numElemsPerOC];
                weightsTableValues[oc * numElemsPerOC + 1] = weightsTableValues[lastChannelIdx * numElemsPerOC + 1];
            }
            if (droppedWeightsSparsity) {
                for (int64_t oc = 0; oc < origOutputChannels; ++oc) {
                    weightsTableValues[oc * numElemsPerOC + 1] = VPU::NCESparsity::SPARSITY_PTR_WHEN_NO_SPARSITY;
                }
            }

            mlir::OpBuilder builder(weightsTableOp);
            const auto dataType = mlir::cast<mlir::RankedTensorType>(weightsTableOp.getContentAttr().getType());
            const auto newWeightsTable =
                    Const::createConst(builder, weightsTableOp->getLoc(), dataType, ArrayRef(weightsTableValues));
            weightsTableOp->getResult(0).replaceUsesWithIf(newWeightsTable, [&](mlir::OpOperand& operand) -> bool {
                return operand.getOwner() == nceOp.getOperation();
            });
        }

        nceOp->setAttr(VPU::OUTPUT_PADDING_ATTR_NAME, nullptr);
    }

    void dropWeightsSparsity(VPU::NCEOpInterface nceOp) const {
        SmallVector<mlir::Operation*> parentOpChain;
        auto operand = nceOp.getWeightsOperand();
        while (auto parentOp = operand.getDefiningOp()) {
            assert((isSupportedIntermediateOpForSparseWeights(parentOp) ||
                    mlir::isa<Const::DeclareOp, VPU::GroupSparseTensorOp>(parentOp)) &&
                   "Unexpected operation in weights parent chain");
            parentOpChain.push_back(parentOp);
            if (parentOp->getNumOperands() == 0) {
                break;
            }
            operand = parentOp->getOperand(0);
        }

        // Create a new chain of parent operations, where the sparsity is dropped
        mlir::OpBuilder builder(nceOp);
        mlir::Operation* newOp = nullptr;
        for (auto op : parentOpChain | reversed) {
            if (auto cstOp = mlir::dyn_cast<Const::DeclareOp>(op)) {
                // Create a new weights constant which does not contain the Sparsify transform
                auto contentAttr = cstOp.getContentAttr();
                auto transformations = contentAttr.getTransformations();
                assert(!transformations.empty() && mlir::isa<Const::SparsifyAttr>(transformations.back()) &&
                       "Expected last transformation to be Sparsify");
                auto newContentAttr =
                        Const::ContentAttr::get(contentAttr.getBaseContent(), transformations.drop_back());
                newOp = builder.create<Const::DeclareOp>(cstOp.getLoc(), cstOp.getType(), newContentAttr);
                continue;
            } else if (mlir::isa<VPU::GroupSparseTensorOp>(op)) {
                // Skip the sparsity grouping operation, as the new types are no longer sparse
                continue;
            }

            mlir::IRMapping mapping;
            if (newOp != nullptr) {
                // All the expected intermediate operations should have a single operand
                mapping.map(op->getOperand(0), newOp->getResult(0));
            }
            newOp = builder.clone(*op, mapping);
            // For all the expected operations, the operation attributes are expected to be compatible with the dense
            // types as well, so it should be sufficient to simply override the result type with the data part of the
            // sparse type
            if (auto sparseType = mlir::dyn_cast<VPU::SparseTensorType>(newOp->getResult(0).getType())) {
                newOp->getResult(0).setType(sparseType.getData());
            }
        }

        nceOp.getWeightsOperand().replaceUsesWithIf(newOp->getResult(0), [&](mlir::OpOperand& use) -> bool {
            return use.getOwner() == nceOp.getOperation();
        });
    }

    // Only view-like operations that are likely to not modify the shape are supported for now
    // More operations could be added if necessary, based on subgraphs identified in models
    bool isCompatibleViewLikeOp(mlir::Operation* op) const {
        return mlir::isa_and_nonnull<VPU::QuantizeCastOp, VPU::LayoutCastOp, VPU::PermuteCastOp>(op);
    }

    // Only some operations are expected to be present in the chain of operations that represent the sparse weights:
    // operations that are introduced when IDU or ODU autopad is enabled for the NCE operation
    bool isSupportedIntermediateOpForSparseWeights(mlir::Operation* op) const {
        return mlir::isa_and_nonnull<VPU::SliceOp, VPU::ExpandOp, VPU::ReshapeOp, VPU::LayoutCastOp>(op);
    }

    bool hasCompatibleUsers(mlir::Operation* op, int64_t unpaddedOutputChannels) const {
        const auto isTheShapeChanged = [](mlir::Operation* op) -> bool {
            // Note: Operations that could modify the shape (e.g. PermuteCast) must currently preserve the shape, so
            // that when the channels are unpadded, we can be sure that the channel dimension will be the one unpadded.
            // This could be optimized to cover the case where the channel dimension is permuted, in case the new
            // position of the channel dimension can be inferred
            const auto inputShape = mlir::cast<NDTypeInterface>(op->getOperand(0).getType()).getShape();
            const auto outputShape = mlir::cast<NDTypeInterface>(op->getResult(0).getType()).getShape();
            return inputShape != outputShape;
        };

        const auto getNonViewOp = [&](mlir::Operation* op) -> mlir::Operation* {
            while (isCompatibleViewLikeOp(op) && !isTheShapeChanged(op)) {
                const auto userOps = op->getResult(0).getUsers();
                const auto numUsers = std::distance(userOps.begin(), userOps.end());
                if (numUsers > 1) {
                    _log.nest(3).trace("User {0} at {1} has {2} users. Stopping search", op->getName(), op->getLoc(),
                                       numUsers);
                    break;
                }
                _log.nest(3).trace("Skipping view-like user {0} at {1}", op->getName(), op->getLoc());
                op = *userOps.begin();
            }
            return op;
        };

        const auto isCompatibleSliceUser = [&](VPU::SliceOp sliceOp) -> bool {
            // The Slice should only extract the first `unpaddedOutputChannels` channels from the data
            const auto offsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
            for (int64_t i = 0; i < static_cast<int64_t>(offsets.size()); ++i) {
                if (offsets[i] != 0) {
                    _log.nest(3).trace("User {0} at {1} has an unsupported slice (non-zero offset for dimension {2}). "
                                       "Stopping search",
                                       op->getName(), op->getLoc(), i);
                    return false;
                }
            }
            auto sliceInputShape = mlir::cast<NDTypeInterface>(sliceOp->getOperand(0).getType()).getShape().raw();
            auto sliceOutputShape = mlir::cast<NDTypeInterface>(sliceOp->getResult(0).getType()).getShape().raw();
            if (sliceInputShape.size() != sliceOutputShape.size()) {
                _log.nest(3).trace("User {0} at {1} has different ranks for input and output shapes ({2} vs {3}). "
                                   "Stopping search",
                                   op->getName(), op->getLoc(), sliceInputShape.size(), sliceOutputShape.size());
                return false;
            }
            for (int64_t i = 0; i < static_cast<int64_t>(sliceInputShape.size()); ++i) {
                if (i == Dims4D::Act::C.ind()) {
                    const auto slicedOnlyUnpaddedChannels = sliceOutputShape[i] == unpaddedOutputChannels;
                    if (!slicedOnlyUnpaddedChannels) {
                        _log.nest(3).trace("User {0} at {1} has an unsupported slice (sliced {2} channels, when the "
                                           "unpadded number of channels is {3}). Stopping search",
                                           op->getName(), op->getLoc(), sliceOutputShape[i], unpaddedOutputChannels);
                        return false;
                    }
                    continue;
                }
                const auto sliceOverNonChannelDim = sliceInputShape[i] != sliceOutputShape[i];
                if (sliceOverNonChannelDim) {
                    _log.nest(3).trace("User {0} at {1} has an unsupported slice (over dimension {2}). Stopping search",
                                       op->getName(), op->getLoc(), i);
                    return false;
                }
            }
            return true;
        };

        const auto logCb = [&](const formatv_object_base& msg) {
            _log.nest().trace("{0}", msg.str());
        };

        // For each user op, allow the following pattern, where operations in square brackets are optional:
        // [view-like op] -> slice
        //                \> NCEConv
        for (auto userOp : op->getUsers()) {
            _log.nest(2).trace("Found user {0} at {1}", userOp->getName(), userOp->getLoc());

            auto actualUserOp = getNonViewOp(userOp);

            _log.nest(3).trace("Checking compatibility with user operation {0} at {1}", actualUserOp->getName(),
                               actualUserOp->getLoc());
            if (auto sliceOp = mlir::dyn_cast_if_present<VPU::SliceOp>(actualUserOp)) {
                // Using ODU autopad here would allow the NCE->Interpolate operations to be potentially vertically
                // fused. The cost model does not currently have support for Interpolate, which means that the strategy
                // manager is not aware of the actual cost of this decision and it would result in using vertical
                // fusion, despite the fact that this decision would lead to a significantly worse performance. This has
                // been observed on networks such as isv_wondershare_EffectSeg_onnx_dense. To prevent this from
                // happening, skip such cases of ODU autopad, until the cost model has support for such SHAVE operators.
                // TODO: remove this workaround E#179769
                auto hasInterpolateUser =
                        llvm::any_of(sliceOp.getOutput().getUsers(), [](mlir::Operation* sliceUserOp) {
                            return mlir::isa_and_present<VPU::InterpolateOp>(sliceUserOp);
                        });
                if (hasInterpolateUser) {
                    return false;
                }
                if (!isCompatibleSliceUser(sliceOp)) {
                    return false;
                }
            } else if (auto nceConvOp = mlir::dyn_cast_if_present<VPU::NCEConvolutionOp>(actualUserOp)) {
                if (!VPU::canConsumeIDUAutopad(nceConvOp, logCb)) {
                    return false;
                }
            } else {
                _log.nest(3).trace("User {0} at {1} is incompatible. Stopping search", actualUserOp->getName(),
                                   actualUserOp->getLoc());
                return false;
            }
        }

        return true;
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::arch50xx::createAutopadChannelsPass(const Logger& log) {
    return std::make_unique<AutopadChannelsPass>(log);
}
