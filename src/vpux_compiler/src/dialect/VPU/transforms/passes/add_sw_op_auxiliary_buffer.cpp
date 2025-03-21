//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <openvino/op/op.hpp>
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_ADDSWOPAUXILIARYBUFFER
#define GEN_PASS_DEF_ADDSWOPAUXILIARYBUFFER
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// AddSwOpBufferPass
//

class AddSwOpAuxiliaryBufferPass final : public VPU::impl::AddSwOpAuxiliaryBufferBase<AddSwOpAuxiliaryBufferPass> {
public:
    explicit AddSwOpAuxiliaryBufferPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// AddProposalBuffer
//

void insertProposalBuffer(Logger& log, VPU::ProposalOp origOp) {
    log.trace("Found Proposal Operation '{0}'", origOp->getLoc());

    constexpr int32_t proposalBoxSize = 10;         // see: sw_runtime_kernels/kernels/src/proposal.cpp (proposalBox)
    constexpr int32_t anchorsBuffElementSize = 16;  // see: sw_runtime_kernels / kernels / src / proposal.cpp (anchors)
    const auto inType = origOp.getClassProbs().getType().cast<vpux::NDTypeInterface>();

    const auto inShape = inType.getShape().raw();
    // [ num_batches, 2 * K, H, W ]
    auto rank = inShape.size();

    VPUX_THROW_UNLESS(rank == 4, "Unsupported rank {0}", rank);
    const auto k = inShape[rank - 3] / 2;
    const auto h = inShape[rank - 2];
    const auto w = inShape[rank - 1];
    const auto numProposals = k * h * w;
    const auto auxiliaryBuffSize = alignValUp(numProposals * proposalBoxSize, static_cast<int64_t>(7)) +
                                   alignValUp(k * anchorsBuffElementSize, static_cast<int64_t>(7));
    std::vector<uint8_t> vals(auxiliaryBuffSize, 0.0f);

    mlir::OpBuilder builder(origOp);
    const SmallVector<int64_t> shape({auxiliaryBuffSize});
    const auto auxiliaryType = mlir::RankedTensorType::get(shape, getUInt8Type(origOp.getContext()));
    auto auxBuff =
            Const::createConst(builder, mlir::UnknownLoc::get(origOp.getContext()), auxiliaryType, ArrayRef(vals));

    auto newProposalOp =
            builder.create<VPU::ProposalOp>(origOp->getLoc(), origOp.getClassProbs(), origOp.getBboxDeltas(),
                                            origOp.getImageShape(), auxBuff, origOp.getProposalAttrsAttr());

    origOp.replaceAllUsesWith(newProposalOp.getOperation());
    origOp->erase();
}

//
// AddTopKBuffer
//

void insertTopKBuffer(Logger& log, VPU::TopKOp origOp) {
    log.trace("Found TopK Operation '{0}'", origOp->getLoc());

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto inputShape = inputType.getShape().raw();
    auto axis = origOp.getAxis();
    constexpr int64_t int32Size = sizeof(int32_t);

    int64_t bufferSizePerShave =
            inputShape[axis] * (2 * std::max(int32Size, Byte(vpux::getElemTypeSize(inputType)).count()));

    const SmallVector<int64_t> shape({1, 1, 1, 2 * bufferSizePerShave});
    const auto topKBufferType = mlir::RankedTensorType::get(shape, getUInt8Type(origOp.getContext()));
    std::vector<uint8_t> values(2 * bufferSizePerShave, 0.0f);

    mlir::OpBuilder builder(origOp);

    auto topKBuffer =
            Const::createConst(builder, mlir::UnknownLoc::get(origOp.getContext()), topKBufferType, ArrayRef(values));

    auto newTopKOp = builder.create<VPU::TopKOp>(
            origOp->getLoc(), origOp.getInput(), origOp.getK(), topKBuffer, origOp.getKValueAttr(), origOp.getAxis(),
            origOp.getMode(), origOp.getSort(), origOp.getElementType(), /*multiClusterStrategy=*/nullptr);

    origOp.replaceAllUsesWith(newTopKOp.getOperation());
    origOp->erase();
}

void insertExperimentalDetectronROIFeatureExtractorBuffer(Logger& log,
                                                          VPU::ExperimentalDetectronROIFeatureExtractorOp origOp) {
    log.trace("Found ExperimentalDetectronROIFeatureExtractor Operation '{0}'", origOp->getLoc());

    mlir::OpBuilder builder(origOp);

    const auto shapeROI = getShape(origOp.getInputs()[0]);
    const auto shapeFeature = getShape(origOp.getInputs()[1]);
    const auto outputSize = origOp.getAttr().getOutputSize().getInt();

    // store reordered rois coords
    const int32_t reorderedRoisBuffSize = 4 * shapeROI[Dim(0)];

    std::vector<float> vals0(reorderedRoisBuffSize, 0);

    const auto reorderedRoisType =
            mlir::RankedTensorType::get(reorderedRoisBuffSize, mlir::Float32Type::get(origOp.getContext()));
    auto reorderedRoisBuff =
            Const::createConst(builder, mlir::UnknownLoc::get(origOp.getContext()), reorderedRoisType, ArrayRef(vals0));

    // store original roi mapping
    const int32_t originalRoiMapBuffSize = shapeROI[Dim(0)];

    std::vector<uint32_t> vals1(originalRoiMapBuffSize, 0);

    const auto originalRoiMapType =
            mlir::RankedTensorType::get(originalRoiMapBuffSize, getUInt32Type(origOp.getContext()));
    auto originalRoiMapBuff = Const::createConst(builder, mlir::UnknownLoc::get(origOp.getContext()),
                                                 originalRoiMapType, ArrayRef(vals1));

    // store output rois features, which will be reordered back to the initial
    const int32_t outputRoisFeaturesTempBuffSize = shapeFeature[Dim(1)] * outputSize * outputSize * shapeROI[Dim(0)];

    std::vector<float> vals2(outputRoisFeaturesTempBuffSize, 0);

    const auto outputRoisFeaturesTempType =
            mlir::RankedTensorType::get(outputRoisFeaturesTempBuffSize, mlir::Float32Type::get(origOp.getContext()));
    auto outputRoisFeaturesTempBuff = Const::createConst(builder, mlir::UnknownLoc::get(origOp.getContext()),
                                                         outputRoisFeaturesTempType, ArrayRef(vals2));

    // store level indices
    const int32_t levelsBuffSize = shapeROI[Dim(0)];

    std::vector<uint32_t> vals3(levelsBuffSize, 0);

    const auto levelsType = mlir::RankedTensorType::get(levelsBuffSize, getUInt32Type(origOp.getContext()));
    auto levelsBuff =
            Const::createConst(builder, mlir::UnknownLoc::get(origOp.getContext()), levelsType, ArrayRef(vals3));

    auto newExperimentalDetectronROIFeatureExtractorOp =
            builder.create<VPU::ExperimentalDetectronROIFeatureExtractorOp>(
                    origOp->getLoc(), origOp.getInputs(), reorderedRoisBuff, originalRoiMapBuff,
                    outputRoisFeaturesTempBuff, levelsBuff, origOp.getAttrAttr());

    origOp.replaceAllUsesWith(newExperimentalDetectronROIFeatureExtractorOp.getOperation());
    origOp->erase();
}

//
// safeRunOnFunc
//

void AddSwOpAuxiliaryBufferPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPU::ProposalOp origOp) {
        if (origOp.getAuxiliary() == nullptr) {
            insertProposalBuffer(_log, origOp);
        }
    });

    func.walk([&](VPU::TopKOp origOp) {
        if (origOp.getLineBuffer() == nullptr) {
            insertTopKBuffer(_log, origOp);
        }
    });

    func.walk([&](VPU::ExperimentalDetectronROIFeatureExtractorOp origOp) {
        if (origOp.getReorderedRois() == nullptr && origOp.getOriginalRoiMap() == nullptr &&
            origOp.getOutputRoisFeaturesTemp() == nullptr && origOp.getLevels() == nullptr) {
            insertExperimentalDetectronROIFeatureExtractorBuffer(_log, origOp);
        }
    });
}

}  // namespace

//
// createAddSwOpAuxiliaryBufferPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createAddSwOpAuxiliaryBufferPass(Logger log) {
    return std::make_unique<AddSwOpAuxiliaryBufferPass>(log);
}
