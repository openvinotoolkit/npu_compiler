//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/dialect/VPU/transforms/factories/shave_controls_dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::SDPAExtendedOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SDPAExtendedOpAdaptor sdpaExtended(operands, attrs, prop);
    if (mlir::failed(sdpaExtended.verify(loc))) {
        return mlir::failure();
    }

    const auto inQType = mlir::cast<vpux::NDTypeInterface>(sdpaExtended.getInputQ().getType());
    const auto inQShape = inQType.getShape().raw();
    const auto rank = inQType.getShape().size();

    const auto inKType = mlir::cast<vpux::NDTypeInterface>(sdpaExtended.getInputK().getType());
    const auto inKShape = inKType.getShape().raw();

    const auto inVType = mlir::cast<vpux::NDTypeInterface>(sdpaExtended.getInputV().getType());
    const auto inVShape = inVType.getShape().raw();

    const auto isTransposedV = inKShape[rank - 2] != inVShape[rank - 2];
    const auto Ev = isTransposedV ? inVShape[rank - 2] : inVShape[rank - 1];
    SmallVector<int64_t> outShape(inQShape.begin(), inQShape.end());
    outShape[rank - 1] = Ev;
    auto outputType =
            mlir::RankedTensorType::get(outShape, inQType.getElementType(), createTensorAttrFromType(inQType));
    inferredReturnTypes.push_back(outputType);
    return mlir::success();
}

//
// ClusteredOpInterface
//

bool vpux::VPU::SDPAExtendedOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t numTiles) {
    const auto outputShape = getShape(getOutput());

    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return outputShape[Dims4D::Act::C] >= checked_cast<int64_t>(numTiles);
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return outputShape[Dims4D::Act::H] >= checked_cast<int64_t>(numTiles);
    }

    return false;
}

static mlir::ModuleOp getModule(::mlir::OpBuilder& odsBuilder) {
    auto block = odsBuilder.getInsertionBlock();
    auto parentOp = block->getParentOp();
    while (parentOp && !llvm::isa<mlir::ModuleOp>(parentOp)) {
        parentOp = parentOp->getParentOp();
    }
    return llvm::cast<mlir::ModuleOp>(parentOp);
}

mlir::Value insertSDPAExtendedBuffer(Logger& log, mlir::OpBuilder& rewriter, ::mlir::Value inputQ, ::mlir::Value inputV,
                                     ::mlir::IntegerAttr padSizeAttr) {
    log.trace("Found SDPAExtended Operation '{0}'", mlir::UnknownLoc::get(rewriter.getContext()));
    const auto module = getModule(rewriter);

    const auto inputVType = mlir::cast<vpux::NDTypeInterface>(inputV.getType());
    const auto inputVShape = inputVType.getShape();
    const auto inputQType = mlir::cast<vpux::NDTypeInterface>(inputQ.getType());
    const auto inputQShape = inputQType.getShape();

    const auto dimS = inputVShape[Dim(3)];
    const auto dimEv = inputVShape[Dim(2)];
    const auto dimE = inputQShape[Dim(3)];
    const auto dimL = inputQShape[Dim(2)];

    auto tileOp = config::getTileExecutor(module);
    const auto numShavesPerTile = tileOp.getSubExecutor(VPU::ExecutorKind::SHAVE_ACT).getCount();

    // dpu storage buffer
    int64_t size1DpuDescriptor = VPU::getDpuDebugDataSize(config::getArch(module)) +
                                 VPU::getDPUVariantDataSize(config::getArch(module)) +
                                 VPU::getDPUInvariantDataSize(config::getArch(module));

    const auto dpuBufferDescriptorSize = (size1DpuDescriptor * 2 * numShavesPerTile) / sizeof(int32_t);
    std::vector<int32_t> dpuDescriptorVals(dpuBufferDescriptorSize, 0);

    //  Create weight table
    auto dimSReal = dimS;

    if (padSizeAttr) {
        dimSReal = dimS - padSizeAttr.getValue().getSExtValue();
    }
    log.trace("insertSDPAExtendedBuffer, create weight with params: dimL '{0}', dimEv '{1}', dimS '{2}', dimSReal "
              "'{3}', dimE '{2}'",
              dimL, dimEv, dimS, dimSReal, dimE);

    const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(config::getArch(module));
    const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(config::getArch(module));
    const auto elemType = inputQType.getElementType();

    // MatMull1
    auto inputShMM1 = Shape{dimL, 1, 1, dimE};
    auto outputShMM1 = Shape{dimL, 1, 1, dimSReal};
    const auto inputMM1 = mlir::RankedTensorType::get(inputShMM1.raw(), elemType);
    const auto outputMM1 = mlir::RankedTensorType::get(outputShMM1.raw(), elemType);
    std::vector<int32_t> matMull1WtTblDataValuesVec = VPU::NCESparsity::getWeightsTable(
            inputMM1, outputMM1, /*weightsPtrs*/ std::nullopt, static_cast<int32_t>(dimE * 2),
            /*sparsityPtr*/ std::nullopt, static_cast<int32_t>(0), ppeConverter, biasConverter, dimSReal);
    for (auto oc = dimSReal; oc < dimS; oc++) {
        matMull1WtTblDataValuesVec.push_back(matMull1WtTblDataValuesVec[0]);
        matMull1WtTblDataValuesVec.push_back(matMull1WtTblDataValuesVec[1]);
        matMull1WtTblDataValuesVec.push_back(matMull1WtTblDataValuesVec[2]);
        matMull1WtTblDataValuesVec.push_back(matMull1WtTblDataValuesVec[3]);
    }
    log.trace("matMull1WtTblDataValuesVec '{0}'", matMull1WtTblDataValuesVec);

    auto inputShMM2 = Shape{dimL, 1, 1, dimS};
    auto outputShMM2 = Shape{dimL, 1, 1, dimEv};
    const auto inputMM2 = mlir::RankedTensorType::get(inputShMM2.raw(), elemType);
    const auto outputMM2 = mlir::RankedTensorType::get(outputShMM2.raw(), elemType);
    std::vector<int32_t> matMull2WtTblDataValuesVec = VPU::NCESparsity::getWeightsTable(
            inputMM2, outputMM2, /*weightsPtrs*/ std::nullopt, static_cast<int32_t>(dimS * 2),
            /*sparsityPtr*/ std::nullopt, static_cast<int32_t>(0), ppeConverter, biasConverter, dimEv);
    log.trace("matMull2WtTblDataValuesVec '{0}'", matMull2WtTblDataValuesVec);

    std::vector<int32_t> resultDpuBufData;
    resultDpuBufData.reserve(dpuDescriptorVals.size() + matMull1WtTblDataValuesVec.size() +
                             matMull2WtTblDataValuesVec.size());
    resultDpuBufData.insert(resultDpuBufData.end(), std::make_move_iterator(dpuDescriptorVals.begin()),
                            std::make_move_iterator(dpuDescriptorVals.end()));
    resultDpuBufData.insert(resultDpuBufData.end(), std::make_move_iterator(matMull1WtTblDataValuesVec.begin()),
                            std::make_move_iterator(matMull1WtTblDataValuesVec.end()));
    resultDpuBufData.insert(resultDpuBufData.end(), std::make_move_iterator(matMull2WtTblDataValuesVec.begin()),
                            std::make_move_iterator(matMull2WtTblDataValuesVec.end()));
    const SmallVector<int64_t> shape({1, 1, 1, static_cast<int64_t>(resultDpuBufData.size())});
    const auto dpuBufferType = mlir::RankedTensorType::get(shape, getSInt32Type(rewriter.getContext()));
    return Const::createConst(rewriter,
                              appendLoc(mlir::UnknownLoc::get(rewriter.getContext()), "SDPA_Extended_dpuBuffer"),
                              dpuBufferType, ArrayRef(resultDpuBufData));
}

void vpux::VPU::SDPAExtendedOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                      ::mlir::Value inputQ, ::mlir::Value inputK, ::mlir::Value inputV,
                                      ::mlir::Value inputMask, ::mlir::Value inputScale, ::mlir::Value dataStorage,
                                      ::mlir::Value dpuStorage, ::mlir::IntegerAttr padSizeAttr) {
    build(odsBuilder, odsState, inputQ, inputK, inputV, inputMask, inputScale, dataStorage, dpuStorage, padSizeAttr,
          {});
}

void vpux::VPU::SDPAExtendedOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                      ::mlir::Value inputQ, ::mlir::Value inputK, ::mlir::Value inputV,
                                      ::mlir::Value inputMask, ::mlir::Value inputScale, ::mlir::Value dataStorage,
                                      ::mlir::IntegerAttr padSizeAttr) {
    auto log = vpux::Logger::global();
    auto dpuStorage = insertSDPAExtendedBuffer(log, odsBuilder, inputQ, inputV, padSizeAttr);
    build(odsBuilder, odsState, inputQ, inputK, inputV, inputMask, inputScale, dataStorage, dpuStorage, padSizeAttr,
          {});
}

vpux::VPU::DistributionInfo vpux::VPU::SDPAExtendedOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::SDPAExtendedOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() >= 5 && buffers.size() <= 9,
                      "SDPAOp requires 4-9 inputs and 1 output, but the number of buffers is {0}", buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::SDPAExtendedOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::SDPAExtendedOp::supportCycleCostCalculation() {
    return false;
}

//
// TilingBuilderOpInterface
//

InputTiling vpux::VPU::SDPAExtendedOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    TileInfo inQTile(getShape(getInputQ()));
    TileInfo inKTile(getShape(getInputK()));
    TileInfo inVTile(getShape(getInputV()));
    TileInfo dataStorageTile(getShape(getDataStorage()));

    transferTilingInfo(inQTile, outputTile, {Dim(Dims4D::Act::H), Dim(Dims4D::Act::C), Dim(Dims4D::Act::N)});
    transferTilingInfo(inVTile, outputTile, {Dim(Dims4D::Act::C), Dim(Dims4D::Act::N)});
    transferTilingInfo(inKTile, outputTile, {Dim(Dims4D::Act::C), Dim(Dims4D::Act::N)});
    transferTilingInfo(dataStorageTile, outputTile, {Dim(Dims4D::Act::H)});

    // InputQ, inputK and InputV are mandatory
    InputTiling inTiles = TilingInfo{{std::move(inQTile), std::move(inKTile), std::move(inVTile)}};

    // Mask is optional, but if it is present, it should be tiled if possible
    if (getInputMask() != nullptr) {
        TileInfo inMaskTile(getShape(getInputMask()));
        if (inMaskTile.shape[Dims4D::Act::H] != 1) {
            transferTilingInfo(inMaskTile, outputTile, {Dim(Dims4D::Act::H)});
        }
        if (inMaskTile.shape[Dims4D::Act::C] != 1) {
            transferTilingInfo(inMaskTile, outputTile, {Dim(Dims4D::Act::C)});
        }
        inTiles.tiles.push_back(inMaskTile);
    }

    // ScaleTensor is optional and can't be tiled
    if (getInputScale() != nullptr) {
        TileInfo inScaleTile(getShape(getInputScale()));
        inTiles.tiles.push_back(inScaleTile);
    }

    inTiles.tiles.push_back(std::move(dataStorageTile));

    // dpuStorageTensor is optional and can't be tiled
    if (getDpuStorage() != nullptr) {
        TileInfo dpuStorageTile(getShape(getDpuStorage()));
        inTiles.tiles.push_back(dpuStorageTile);
    }

    return inTiles;
}

void vpux::VPU::SDPAExtendedOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::SDPAExtendedOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

SmallVector<mlir::Value> VPU::SDPAExtendedOp::getAuxiliaryBuffers() {
    return {getDataStorage()};
}

mlir::LogicalResult VPU::SDPAExtendedOp::setAuxiliaryBuffers(ArrayRef<mlir::Value> buffers) {
    if (buffers.size() != 1 || buffers.front() == nullptr) {
        return mlir::failure();
    }
    getDataStorageMutable().assign(buffers.front());
    return mlir::success();
}

SmallVector<mlir::Type> VPU::SDPAExtendedOp::getBufferTypes() {
    const auto inputVType = mlir::cast<vpux::NDTypeInterface>(getInputV().getType());
    const auto inputQType = mlir::cast<vpux::NDTypeInterface>(getInputQ().getType());
    const auto inputQShape = inputQType.getShape();
    const auto inputVShape = inputVType.getShape();
    const auto dimS = inputVShape[Dim(3)];
    const auto dimL = inputQShape[Dim(2)];
    const int64_t noBuffersForSoftmax = 2;
    SmallVector<int64_t> bufferShape({1, 1, dimL, noBuffersForSoftmax * dimS * static_cast<int64_t>(sizeof(uint16_t))});
    const auto auxBuffType = mlir::RankedTensorType::get(bufferShape, getUInt8Type(getContext()));
    return {auxBuffType};
}
