//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVWEIGHTSCOMPRESSION
#define GEN_PASS_DEF_CONVWEIGHTSCOMPRESSION
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

mlir::Value reduceWeightsConstant(VPUIP::NCEClusterTaskOp nceOp, VPUIP::CopyOp weightsCopyOp,
                                  NDTypeInterface weightsCopyOutputType, ShapeRef origShape,
                                  const int64_t origChannelVal, mlir::OpBuilder& builder) {
    const auto currentOffset = SmallVector<int64_t>{0, 0, 0, 0};
    mlir::Value weightsCopyInput = weightsCopyOp.getInput();
    auto weightsConstOp = weightsCopyInput.getDefiningOp<vpux::Const::DeclareOp>();
    mlir::OpBuilder constBuilder(weightsConstOp);

    auto newContentAttr = weightsConstOp.transformContentAttr().subview(Shape(currentOffset), origShape).get();

    auto newConstType = weightsConstOp.getType().cast<NDTypeInterface>().changeShape(origShape);
    auto newWeightsConstOp = constBuilder.create<Const::DeclareOp>(
            weightsConstOp.getLoc(), newConstType.cast<mlir::MemRefType>(), std::move(newContentAttr));
    weightsConstOp.replaceAllUsesWith(newWeightsConstOp.getOperation());

    constexpr int64_t requiredAlignment = 16;
    const auto elemSize = getElemTypeSize(weightsCopyOutputType).count();

    // Check if weights set is aligned to 16B
    // If not, stride output of VPUIP.Copy accordingly
    if ((weightsCopyOutputType.getStrides()[Dims4D::Filter::OC].count() / elemSize) % requiredAlignment != 0) {
        auto paddedStrideValue = alignValUp(
                (origChannelVal * origShape[Dims4D::Filter::KY] * origShape[Dims4D::Filter::KX]), requiredAlignment);
        const SmallVector<Bit> newStrides({Bit(paddedStrideValue * elemSize), Bit(1 * elemSize),
                                           Bit(origChannelVal * origShape[Dims4D::Filter::KX] * elemSize),
                                           Bit(origChannelVal * elemSize)});
        weightsCopyOutputType = weightsCopyOutputType.changeStrides(vpux::StridesRef(newStrides));
    }
    bool isTiled = vpux::VPUIP::hasDistributedOperand(nceOp);

    if (isTiled) {
        auto oldDistrType = weightsCopyOp.getResult().getType().dyn_cast<VPUIP::DistributedBufferType>();
        const auto orderAttr =
                mlir::AffineMapAttr::get(oldDistrType.getDimsOrder().toAffineMap(oldDistrType.getContext()));
        const auto elemStrides =
                to_small_vector(weightsCopyOutputType.getStrides() | transformed([&](Bit stride) {
                                    return stride.count() / weightsCopyOutputType.getElemTypeSize().count();
                                }));
        const auto stridesAttr = getIntArrayAttr(oldDistrType.getContext(), elemStrides);
        const auto layout = vpux::MemRefAttr::get(orderAttr, stridesAttr,
                                                  /*allocSize=*/nullptr, {oldDistrType.getSparsityCompression()},
                                                  oldDistrType.getContext());

        auto distributedAttr = oldDistrType.getDistribution();
        if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributedAttr)) {
            // Small optimization:
            // convert to native type so when getPerCluster... to avoid
            // multiple parsing for same values
            auto distribution = vpux::VPU::DistributionInfo::getClassFromAttr(distributedAttr);

            auto perClusterMemoryShapes =
                    VPU::getPerClusterMemoryShapes(weightsCopyOutputType.getShape(), distribution).value();
            auto perClusterMemoryOffsets =
                    VPU::getPerClusterMemoryShapeOffsets(weightsCopyOutputType.getShape(), distribution);
            auto perClusterComputeShapes =
                    VPU::getPerClusterComputeShapes(weightsCopyOutputType.getShape(), distribution);
            auto perClusterComputeOffsets =
                    VPU::getPerClusterComputeShapeOffsets(weightsCopyOutputType.getShape(), distribution);

            distribution.setMemoryShapes(VPU::arrayOfArrayFromShape(perClusterMemoryShapes));
            distribution.setMemoryOffsets(VPU::arrayOfArrayFromShape(perClusterMemoryOffsets));
            distribution.setComputeShapes(VPU::arrayOfArrayFromShape(perClusterComputeShapes));
            distribution.setComputeOffsets(VPU::arrayOfArrayFromShape(perClusterComputeOffsets));

            distributedAttr = vpux::VPU::DistributionInfo::getAttrFromClass(oldDistrType.getContext(), distribution);
        }

        auto distrib = VPUIP::DistributedBufferType::get(
                oldDistrType.getContext(), weightsCopyOutputType.getShape().raw(), oldDistrType.getElementType(),
                layout, oldDistrType.getMemSpace(), distributedAttr);

        auto copyAllocOp = weightsCopyOp.getOutputBuff().getDefiningOp<VPURT::AllocDistributed>();
        mlir::IntegerAttr alignment = nullptr;
        if (copyAllocOp.getAlignmentAttr() != nullptr) {
            alignment = copyAllocOp.getAlignmentAttr();
        }
        mlir::Location loc = nceOp->getLoc();
        auto allocOp = builder.create<VPURT::AllocDistributed>(loc, distrib, alignment, nullptr);

        return builder.create<VPUIP::CopyOp>(loc, newWeightsConstOp.getOutput(), allocOp).getResult();
    }

    auto allocOp =
            builder.create<mlir::memref::AllocOp>(nceOp->getLoc(), weightsCopyOutputType.cast<mlir::MemRefType>());

    auto copyOp = builder.create<VPUIP::CopyOp>(nceOp->getLoc(), newWeightsConstOp.getOutput(), allocOp);

    return copyOp.getOutput();
}

mlir::Value getAdjustWeightsTable(Const::DeclareOp weightsTableConstOp, const int64_t weightsOC,
                                  const int64_t weightsOCstride) {
    auto weightsTableContent = weightsTableConstOp.getContent();
    auto weightsTableVals = to_std_vector(weightsTableContent.getValues<std::int32_t>());

    int64_t weightPtrOffset = 0;
    for (auto oc : irange(checked_cast<size_t>(weightsOC))) {
        const auto wtInd = oc * static_cast<size_t>(VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC);

        weightsTableVals[wtInd] = checked_cast<int32_t>(weightPtrOffset);

        weightPtrOffset += weightsOCstride / CHAR_BIT;
    }

    mlir::OpBuilder constBuilder(weightsTableConstOp);
    const auto elemType = getSInt32Type(constBuilder.getContext());
    const auto weightTableShape = weightsTableContent.getType().getShape();
    const auto dataStorageType = mlir::RankedTensorType::get(weightTableShape.raw(), elemType);
    const auto dataAttr = Const::createConstContent(dataStorageType, ArrayRef(weightsTableVals));

    auto dataConstOp = constBuilder.create<Const::DeclareOp>(
            weightsTableConstOp.getLoc(), weightsTableConstOp.getType(), Const::ContentAttr::get(dataAttr));

    return dataConstOp;
}

//
// CompressConvWeights
//

/*
    For weights with original IC < VPU_CHANNEL_ALIGNMENT, weights compression optimization can be applied.
    When we only have padding on weights ICs, we do the following:
        * strip weights of extra padding
        * set cm_sp_pattern for the original input channels
        * add ShapeCast to weights input of NCEClusterTask op
    Otherwise, we do:
        * set cm_sp_pattern for channels padded to VPU_COMPRESSED_INPUT_CHANNEL_NUM
        * add ShapeCast to weights input of NCEClusterTask op

    One case were we might have weights padded over another dimension is when weights IC <=
    VPU_COMPRESSED_INPUT_CHANNEL_NUM and we also have input activations compression. Then, the weights are padded up to
    VPU_COMPRESSED_INPUT_CHANNEL_NUM, not VPU_CHANNEL_ALIGNMENT. But that might lead to weights sets not being aligned
    to 16B. In that case, weights sets are flattened on W dimension and padded up to 16B. Therefore, the resulting shape
    will be: {OC x 1 x 1 x (IC padded up to VPU_COMPRESSED_INPUT_CHANNEL_NUM) * KY * KX + padding}

*/
void compressConvWeights(Logger& log, VPUIP::NCEClusterTaskOp origOp) {
    if (!VPUIP::canWeightsBeCompressed(origOp)) {
        return;
    }
    auto weights = origOp.getWeights().getDefiningOp<VPUIP::CopyOp>();
    auto weightsInput = weights.getInput().getDefiningOp<Const::DeclareOp>();

    if (!weightsInput.getResult().hasOneUse()) {
        // TODO: support case with multi users
        return;
    }
    log.trace("Compressing weights for operation '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    mlir::OpBuilder builder(origOp);
    const auto& weightsContentAttr = weightsInput.getContentAttr();

    const auto origChannelVal =
            weightsContentAttr.getBaseContent().getType().cast<NDTypeInterface>().getShape()[Dims4D::Filter::IC];
    const auto kernelSz = parseIntArrayAttr<int64_t>(origOp.getKernelSizeAttr());
    const auto outputChannels = origOp.getOutput().getType().cast<NDTypeInterface>().getShape()[Dims4D::Act::C];
    const auto origShape = Shape(
            {outputChannels, origChannelVal, kernelSz[Dims4D::Kernel::Y.ind()], kernelSz[Dims4D::Kernel::X.ind()]});

    auto weightsCopyOp = weights.getOutput();
    auto weightsTableCopyOp = origOp.getWeightTable().getDefiningOp<VPUIP::CopyOp>();
    auto weightsTableConstOp = weightsTableCopyOp.getInput().getDefiningOp<Const::DeclareOp>();
    vpux::NDTypeInterface weightsType;

    const bool paddingDoneOnlyOnC = VPUIP::isOnlyPadOverIC(weightsContentAttr);
    if (paddingDoneOnlyOnC) {
        auto distributedBufferType = weights.getType().dyn_cast<VPUIP::DistributedBufferType>();
        auto distribution = distributedBufferType ? distributedBufferType.getDistribution() : nullptr;
        if (distribution && VPU::isDistributedAttrWithExplicitShapesAndOffsets(distribution)) {
            // we don't support OVERLAPPED weights
            auto mode = distribution ? distribution.getMode() : nullptr;
            VPUX_THROW_WHEN(mode == nullptr || mode.getValue() == VPU::DistributionMode::OVERLAPPED,
                            "mode cannot be nullptr or OVERLAPPED");

            // can't reshape DistributedBuffers with explicit per cluster shapes/offsets so we need to clone it with
            // proper shape & DistributedAttr
            auto newDistribution = VPU::getNonOverlappedDistributedAttr(
                    origShape, mode, distribution.getNumTiles(), distribution.getNumClusters(),
                    distribution.getAlignment(), distribution.getUniformDistributedSegments(), origOp.getContext());
            weightsType = VPUIP::DistributedBufferType::get(
                    distributedBufferType.getContext(), origShape.raw(), distributedBufferType.getElementType(),
                    distributedBufferType.getLayout(), distributedBufferType.getMemSpace(), newDistribution);
        } else {
            weightsType = weights.getType().cast<vpux::NDTypeInterface>();
            weightsType = weightsType.changeShape(origShape);
        }

        weightsCopyOp = reduceWeightsConstant(origOp, weights, weightsType, origShape, origChannelVal, builder);

        // Removing the input channel padding for the weights const will lead to differences in
        // weight sets offsets. This will make the necessary adjustments.
        const auto weightsCopyOpDstStrides = weightsCopyOp.getType().cast<NDTypeInterface>().getStrides();
        auto newWeightsTableConst = getAdjustWeightsTable(weightsTableConstOp, origShape[Dims4D::Filter::OC],
                                                          weightsCopyOpDstStrides[Dims4D::Filter::OC].count());

        if (weightsTableConstOp.getOutput().hasOneUse()) {
            weightsTableConstOp.replaceAllUsesWith(newWeightsTableConst);
            weightsTableConstOp.erase();
        } else {
            // In this case, the weights table is shared between multiple ops,
            // compressing current op should not affect the others
            auto newWeightsTableCopy = builder.create<VPUIP::CopyOp>(weightsTableCopyOp->getLoc(), newWeightsTableConst,
                                                                     weightsTableCopyOp.getOutputBuff());
            weightsTableCopyOp.replaceAllUsesWith(newWeightsTableCopy.getOperation());

            weightsTableCopyOp.erase();
        }
    }

    const auto channelAlignValue = VPU::NCEInvariant::getAlignment(weightsType.getElementType());
    const auto finalShape = SmallVector<int64_t>({origShape[Dims4D::Filter::OC], channelAlignValue,
                                                  origShape[Dims4D::Filter::KY], origShape[Dims4D::Filter::KX]});
    auto shapeCastOp = builder.create<VPUIP::ShapeCastOp>(origOp.getLoc(), weightsCopyOp,
                                                          getIntArrayAttr(origOp.getContext(), finalShape));

    const int64_t cmSpPattern =
            paddingDoneOnlyOnC ? (static_cast<int64_t>(1) << origChannelVal) - 1
                               : (static_cast<int64_t>(1) << VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM) - 1;
    auto cmSpPatternAttr = getIntAttr(origOp->getContext(), cmSpPattern);

    mlir::IRMapping mapper;
    mapper.map(origOp.getWeights(), shapeCastOp.getResult());
    auto newOp = mlir::cast<VPUIP::NCEClusterTaskOp>(builder.clone(*origOp.getOperation(), mapper));
    newOp.setCmSpPatternAttr(cmSpPatternAttr);
    origOp.replaceAllUsesWith(newOp);
    origOp->erase();

    if (paddingDoneOnlyOnC && weights.getResult().use_empty()) {
        weights->erase();
        weightsInput->erase();
    }
}

//
// ConvWeightsCompression
//

class ConvWeightsCompression final : public VPUIP::impl::ConvWeightsCompressionBase<ConvWeightsCompression> {
public:
    explicit ConvWeightsCompression(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvWeightsCompression::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPUIP::NCEClusterTaskOp origOp) {
        compressConvWeights(_log, origOp);
    });
}

}  // namespace

//
// createConvWeightsCompressionPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvWeightsCompressionPass(Logger log) {
    return std::make_unique<ConvWeightsCompression>(log);
}
