//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"

#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/compiler/dialect/VPU/transforms/factories/shave_controls_dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"

using namespace vpux;

int64_t getAuxDataBufferLineWidthFromCSize(mlir::ModuleOp moduleOp, int64_t shapeC, int64_t dimS, int64_t dimEv,
                                           /*optional*/ vpux::VPU::MultiClusterStrategyAttr multiClusterStrategy,
                                           bool hasSink, bool hasMask) {
    bool noPingPongBuffer = (shapeC == 1);
    if (multiClusterStrategy) {
        vpux::VPU::MultiClusterStrategy msVal = multiClusterStrategy.getValue();
        auto tileOp = config::getTileExecutor(moduleOp);
        int64_t numClusters = tileOp.getCount();
        if (msVal == VPU::MultiClusterStrategy::SplitOverKernel) {
            if (numClusters >= shapeC) {
                noPingPongBuffer = true;
            }
        }
    }
    int64_t noBuffersForSoftmax = 2;
    if (noPingPongBuffer) {
        noBuffersForSoftmax = 1;  // no ping-pong buffers needed for batch size 1
    }
    auto auxLineSize = dimS;
    const bool useSmPostNorm = (dimEv + 256) < dimS;
    // Extra space for max/sumexp buffers when post-norm is required.
    // hasMask allocates post-norm space unconditionally: mask-aware kernel always uses post-norm,
    // and the minor over-allocation when the feature is disabled is harmless.
    if (useSmPostNorm || hasSink || hasMask) {
        // extra space for softmax post normalization: targetSequenceL * sizeof(float).
        // As targetSequenceL is tilled, I added for above buffer 32 bytes for every float entry on every line, to keep
        // alignment requirements of DPU accesses buffers.
        constexpr auto softmaxNormBuf = 32;
        auxLineSize = auxLineSize + softmaxNormBuf;
    }
    auto auxBufDataSize = noBuffersForSoftmax * auxLineSize * static_cast<int64_t>(sizeof(uint16_t));
    return auxBufDataSize;
}

int64_t getAuxDataBufferLineWidthFromInputs(mlir::ModuleOp moduleOp, mlir::Value inputQ, mlir::Value inputK,
                                            mlir::Value inputV,
                                            /*optional*/ vpux::VPU::MultiClusterStrategyAttr multiClusterStrategy,
                                            bool hasSink, bool hasMask) {
    const auto inputQType = mlir::cast<vpux::NDTypeInterface>(inputQ.getType());
    const auto inputKType = mlir::cast<vpux::NDTypeInterface>(inputK.getType());
    const auto inputVType = mlir::cast<vpux::NDTypeInterface>(inputV.getType());
    // extract possible broadcasted C size from inputs
    auto maxC = std::max({inputQType.getShape()[Dims4D::Act::C], inputKType.getShape()[Dims4D::Act::C],
                          inputVType.getShape()[Dims4D::Act::C]});

    const auto dimS = inputVType.getShape()[Dims4D::Act::W];
    const auto dimEv = inputVType.getShape()[Dims4D::Act::H];

    return getAuxDataBufferLineWidthFromCSize(moduleOp, maxC, dimS, dimEv, multiClusterStrategy, hasSink, hasMask);
}

mlir::Type calculateDpuStorageType(mlir::ModuleOp moduleOp, mlir::Value inputQ, int64_t dimS, int64_t dimEv,
                                   int64_t dimE, int64_t dimL, mlir::IntegerAttr padSizeAttr,
                                   std::vector<int32_t>& resultDpuStorageData) {
    const auto inputQType = mlir::cast<vpux::NDTypeInterface>(inputQ.getType());

    auto tileOp = config::getTileExecutor(moduleOp);
    const auto numShavesPerTile = tileOp.getSubExecutor(config::ExecutorKind::SHAVE_ACT).getCount();

    // dpu storage buffer
    int64_t size1DpuDescriptor = VPU::getDpuDebugDataSize(config::getArch(moduleOp)) +
                                 VPU::getDPUVariantDataSize(config::getArch(moduleOp)) +
                                 VPU::getDPUInvariantDataSize(config::getArch(moduleOp));

    const auto dpuBufferDescriptorSize = (size1DpuDescriptor * 2 * numShavesPerTile) / sizeof(int32_t);
    std::vector<int32_t> dpuDescriptorVals(dpuBufferDescriptorSize, 0);

    //  Create weight table
    auto dimSReal = dimS;

    if (padSizeAttr) {
        dimSReal = dimS - padSizeAttr.getValue().getSExtValue();
    }

    std::vector<int32_t> matMul2WtTblDataValuesVec = {};
    std::vector<int32_t> matMul1WtTblDataValuesVec = {};
    auto needWeightTable = VPU::getShaveDpuNeedWeightTable(config::getArch(moduleOp));
    if (needWeightTable) {
        const auto ppeConverter = VPU::NCESparsity::getPPEConverterCb(config::getArch(moduleOp));
        const auto biasConverter = VPU::NCESparsity::getBiasConverterCb(config::getArch(moduleOp));
        const auto elemType = inputQType.getElementType();

        // MatMul1
        auto inputShMM1 = Shape{dimL, 1, 1, dimE};
        auto outputShMM1 = Shape{dimL, 1, 1, dimSReal};
        const auto inputMM1 = mlir::RankedTensorType::get(inputShMM1.raw(), elemType);
        const auto outputMM1 = mlir::RankedTensorType::get(outputShMM1.raw(), elemType);
        matMul1WtTblDataValuesVec = VPU::NCESparsity::getWeightsTable(
                inputMM1, outputMM1, /*weightsPtrs*/ std::nullopt, static_cast<int32_t>(dimE * 2),
                /*sparsityPtr*/ std::nullopt, static_cast<int32_t>(0), ppeConverter, biasConverter, dimSReal);
        // Need padded outputs width values to use weight from input buffer. Will not be used.
        for (auto oc = dimSReal; oc < dimS; oc++) {
            matMul1WtTblDataValuesVec.push_back(matMul1WtTblDataValuesVec[0]);
            matMul1WtTblDataValuesVec.push_back(matMul1WtTblDataValuesVec[1]);
            matMul1WtTblDataValuesVec.push_back(matMul1WtTblDataValuesVec[2]);
            matMul1WtTblDataValuesVec.push_back(matMul1WtTblDataValuesVec[3]);
        }

        auto inputShMM2 = Shape{dimL, 1, 1, dimS};
        auto outputShMM2 = Shape{dimL, 1, 1, dimEv};
        const auto inputMM2 = mlir::RankedTensorType::get(inputShMM2.raw(), elemType);
        const auto outputMM2 = mlir::RankedTensorType::get(outputShMM2.raw(), elemType);
        matMul2WtTblDataValuesVec = VPU::NCESparsity::getWeightsTable(
                inputMM2, outputMM2, /*weightsPtrs*/ std::nullopt, static_cast<int32_t>(dimS * 2),
                /*sparsityPtr*/ std::nullopt, static_cast<int32_t>(0), ppeConverter, biasConverter, dimEv);
    }
    resultDpuStorageData.clear();
    resultDpuStorageData.reserve(dpuDescriptorVals.size() + matMul1WtTblDataValuesVec.size() +
                                 matMul2WtTblDataValuesVec.size());
    resultDpuStorageData.insert(resultDpuStorageData.end(), std::make_move_iterator(dpuDescriptorVals.begin()),
                                std::make_move_iterator(dpuDescriptorVals.end()));
    resultDpuStorageData.insert(resultDpuStorageData.end(), std::make_move_iterator(matMul1WtTblDataValuesVec.begin()),
                                std::make_move_iterator(matMul1WtTblDataValuesVec.end()));
    resultDpuStorageData.insert(resultDpuStorageData.end(), std::make_move_iterator(matMul2WtTblDataValuesVec.begin()),
                                std::make_move_iterator(matMul2WtTblDataValuesVec.end()));
    const SmallVector<int64_t> shape({1, 1, 1, static_cast<int64_t>(resultDpuStorageData.size())});
    const auto dpuStorageType = mlir::RankedTensorType::get(shape, getSInt32Type(inputQ.getContext()));
    return dpuStorageType;
}

SmallVector<mlir::Type> getAuxiliaryBufferTypes(mlir::ModuleOp moduleOp, mlir::Value inputQ, mlir::Value inputK,
                                                mlir::Value inputV, mlir::IntegerAttr padSizeAttr,
                                                std::vector<int32_t>& resultDpuStorageData,
                                                vpux::VPU::MultiClusterStrategyAttr multiClusterStrategy, bool hasSink,
                                                bool hasMask) {
    const auto inputVType = mlir::cast<vpux::NDTypeInterface>(inputV.getType());
    const auto inputVShape = inputVType.getShape();
    const auto inputQType = mlir::cast<vpux::NDTypeInterface>(inputQ.getType());
    const auto inputQShape = inputQType.getShape();

    const auto dimS = inputVShape[Dim(3)];
    const auto dimEv = inputVShape[Dim(2)];
    const auto dimE = inputQShape[Dim(3)];
    const auto dimL = inputQShape[Dim(2)];

    const auto dataStorageType = [&]() -> mlir::Type {
        const auto lineWidth = getAuxDataBufferLineWidthFromInputs(moduleOp, inputQ, inputK, inputV,
                                                                   multiClusterStrategy, hasSink, hasMask);
        SmallVector<int64_t> bufferShape({1, 1, dimL, lineWidth});
        return mlir::RankedTensorType::get(bufferShape, getUInt8Type(inputQ.getContext()));
    }();

    const auto dpuStorageType =
            calculateDpuStorageType(moduleOp, inputQ, dimS, dimEv, dimE, dimL, padSizeAttr, resultDpuStorageData);

    return {dataStorageType, dpuStorageType};
}

mlir::Value createDpuStorageConstant(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type dpuStorageType,
                                     ArrayRef<int32_t> dpuStorageData) {
    return Const::createConst(builder, appendLoc(loc, "Attention_dpuBuffer"),
                              mlir::cast<mlir::RankedTensorType>(dpuStorageType), dpuStorageData);
}

void vpux::VPU::AttentionOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value inputQ,
                                   mlir::Value inputK, mlir::Value inputV, mlir::Value inputMask,
                                   mlir::Value inputScale, ::mlir::Value inputSink, ::mlir::Value inputBias,
                                   mlir::IntegerAttr padSizeAttr) {
    auto block = odsBuilder.getInsertionBlock();
    const auto moduleOp = getModuleOp(block->getParentOp());

    std::vector<int32_t> dpuStorageData;
    auto auxBufferTypes = getAuxiliaryBufferTypes(moduleOp, inputQ, inputK, inputV, padSizeAttr, dpuStorageData,
                                                  nullptr, inputSink != nullptr, inputMask != nullptr);
    VPUX_THROW_WHEN(auxBufferTypes.size() != 2, "Expected 2 auxiliary buffer types, got {0}", auxBufferTypes.size());
    auto dataStorage = VPU::createEmptyAuxiliaryBuffer(odsBuilder, odsState.location, auxBufferTypes[0]);
    auto dpuStorage = createDpuStorageConstant(odsBuilder, odsState.location, auxBufferTypes[1], dpuStorageData);

    build(odsBuilder, odsState, inputQ, inputK, inputV, inputMask, inputScale, inputSink, inputBias, dataStorage,
          dpuStorage, padSizeAttr,
          /*multiClusterStrategy=*/nullptr);
}

mlir::LogicalResult vpux::VPU::AttentionOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::AttentionOpAdaptor attention(operands, attrs, prop);
    if (mlir::failed(attention.verify(loc))) {
        return mlir::failure();
    }

    const auto inQType = mlir::cast<vpux::NDTypeInterface>(attention.getInputQ().getType());
    const auto inQShape = inQType.getShape().raw();
    const auto rank = inQType.getShape().size();

    const auto inKType = mlir::cast<vpux::NDTypeInterface>(attention.getInputK().getType());
    const auto inKShape = inKType.getShape().raw();

    const auto inVType = mlir::cast<vpux::NDTypeInterface>(attention.getInputV().getType());
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

llvm::LogicalResult VPU::AttentionOp::verify() {
    const auto moduleOp = getModuleOp(getOperation()->getParentOp());
    std::vector<int32_t> dpuStorageData;
    auto expectedAuxBuffTypes =
            getAuxiliaryBufferTypes(moduleOp, getInputQ(), getInputK(), getInputV(), getPadSizeSAttr(), dpuStorageData,
                                    MultiClusterStrategyAttr(), getInputSink() != nullptr, getInputMask() != nullptr);
    if (expectedAuxBuffTypes.size() != 2) {
        return errorAt(getOperation(), "Expected two reference auxiliary buffer types, but got {0}",
                       expectedAuxBuffTypes.size());
    }
    auto loc = getOperation()->getLoc();
    if (mlir::failed(VPU::compareTypes(loc, getDpuStorage().getType(), expectedAuxBuffTypes[1]))) {
        return errorAt(getOperation(), "Invalid DPU storage auxiliary buffer");
    }
    return mlir::success();
}
mlir::FailureOr<OutputTiling> vpux::VPU::AttentionOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(getOperation(), tilingMode, std::move(log));
}

//
// ClusteredOpInterface
//

bool vpux::VPU::AttentionOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t numTiles) {
    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }

    const auto queryShape = getShape(getInputQ());
    const auto numHeads = checked_cast<size_t>(queryShape[Dims4D::Act::C]);
    const auto targetSeqLen = checked_cast<size_t>(queryShape[Dims4D::Act::H]);

    if (strategy == VPU::MultiClusterStrategy::SplitOverBatch) {
        return queryShape[Dims4D::Act::N] >= checked_cast<int64_t>(numTiles);
    }

    if (targetSeqLen >= numTiles) {
        return strategy == VPU::MultiClusterStrategy::SplitOverHeight;
    } else if (numHeads >= numTiles) {
        return strategy == VPU::MultiClusterStrategy::SplitOverKernel;
    }

    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::AttentionOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams,
        const std::optional<ArrayRef<int64_t>> /* memoryNumTiles */) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

///
/// isSupported
///
/// This function verifies if the minimum memory footprint of the operation fits within
/// the available CMX memory. The minimum size is calculated by setting all tilable
/// dimensions (batch N, query heads qH, key-value heads kvH, and target sequence length tSL)
/// to 1, while keeping non-tilable dimensions at their original values aligned to 32.
///
/// The function calculates memory requirements for:
/// - Input tensors: Q [1,1,1,e], K [1,1,sSL,e], V [1,1,eV,sSL]
/// - Optional inputs: mask [1,1,1,sSL], scale [1,1,1,1], bias [1,1,1,sSL]
/// - Auxiliary data buffer
/// - DPU storage buffer
/// - Output tensor [1,1,1,eV]
///
bool vpux::VPU::AttentionOp::isSupported(IE::AttentionOp origOp) {
    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    if (!module) {
        return false;
    }

    const auto qType = mlir::cast<NDTypeInterface>(origOp.getInputQ().getType());
    const auto kType = mlir::cast<NDTypeInterface>(origOp.getInputK().getType());
    const auto vType = mlir::cast<NDTypeInterface>(origOp.getInputV().getType());

    const auto qShape = qType.getShape().raw();
    const auto kShape = kType.getShape().raw();
    const auto vShape = vType.getShape().raw();
    const auto rank = qType.getRank();

    // Align fixed dimensions to 32
    const auto e = alignValUp((int64_t)qShape[rank - 1], (int64_t)32);
    const auto sSL = alignValUp((int64_t)kShape[rank - 2], (int64_t)32);
    const auto eV = alignValUp((int64_t)vShape[rank - 2], (int64_t)32);
    constexpr int64_t fp16Size = 2;

    SmallVector<Byte> buffersSize = {
            Byte(e * fp16Size),         // Q: [1,1,1,e]
            Byte(sSL * e * fp16Size),   // K: [1,1,sSL,e]
            Byte(eV * sSL * fp16Size),  // V: [1,1,eV,sSL]
            Byte(eV * fp16Size)         // Output: [1,1,1,eV]
    };

    if (origOp.getInputMask()) {
        buffersSize.push_back(Byte(sSL * fp16Size));
    }
    if (origOp.getInputScale()) {
        buffersSize.push_back(Byte(fp16Size));
    }
    if (origOp.getInputSink()) {
        buffersSize.push_back(Byte(1 * fp16Size));
    }
    if (origOp.getInputBias()) {
        buffersSize.push_back(Byte(sSL * fp16Size));
    }

    buffersSize.push_back(Byte(getAuxDataBufferLineWidthFromCSize(
            module, 1, sSL, eV, nullptr, origOp.getInputSink() != nullptr, origOp.getInputMask() != nullptr)));

    std::vector<int32_t> dpuStorageData;
    auto dpuStorageType = calculateDpuStorageType(module, origOp.getInputQ(), sSL, eV, e, qShape[rank - 2],
                                                  origOp.getPadSizeSAttr(), dpuStorageData);
    if (auto dpuStorageNDType = mlir::dyn_cast<NDTypeInterface>(dpuStorageType)) {
        buffersSize.push_back(dpuStorageNDType.getTotalAllocSize());
    }

    const auto totalSize = vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(module), buffersSize);
    return totalSize <= getTotalCMXSize(module);
}

//
// SWOpInterface
//

bool vpux::VPU::AttentionOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() >= 5 && buffers.size() <= 9,
                      "SDPAOp requires  4-10 inputs and 1 output, but the number of buffers is {0}", buffers.size());

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

bool vpux::VPU::AttentionOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::AttentionOp::supportCycleCostCalculation() {
    return false;
}

//
// TilingBuilderOpInterface
//

void transferTilingInfoBroadcastAware(vpux::TileInfo& dst, const vpux::TileInfo& src) {
    if (dst.shape[Dims4D::Act::H] != 1) {
        transferTilingInfo(dst, src, {Dim(Dims4D::Act::H)});
    }
    if (dst.shape[Dims4D::Act::C] != 1) {
        transferTilingInfo(dst, src, {Dim(Dims4D::Act::C)});
    }
    if (dst.shape[Dims4D::Act::N] != 1) {
        transferTilingInfo(dst, src, {Dim(Dims4D::Act::N)});
    }
}

InputTiling vpux::VPU::AttentionOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger log) {
    log.trace("AttentionOp - backInferTileInfo outputTile: {}", outputTile);
    TileInfo inQTile(getShape(getInputQ()));
    TileInfo inKTile(getShape(getInputK()));
    TileInfo inVTile(getShape(getInputV()));
    TileInfo dataStorageTile(getShape(getDataStorage()));
    bool isMQA = inQTile.shape[Dims4D::Act::C] > inKTile.shape[Dims4D::Act::C] && inKTile.shape[Dims4D::Act::C] == 1;

    transferTilingInfo(inQTile, outputTile, {Dim(Dims4D::Act::H), Dim(Dims4D::Act::C), Dim(Dims4D::Act::N)});
    transferTilingInfo(inVTile, outputTile, {Dim(Dims4D::Act::N)});
    transferTilingInfo(inKTile, outputTile, {Dim(Dims4D::Act::N)});
    if (isMQA == false) {
        transferTilingInfo(inKTile, outputTile, {Dim(Dims4D::Act::C)});
        transferTilingInfo(inVTile, outputTile, {Dim(Dims4D::Act::C)});
    }
    transferTilingInfo(dataStorageTile, outputTile, {Dim(Dims4D::Act::H)});
    // if output tile head become 1, ping-pong buffers are not needed anymore
    dataStorageTile.shape[Dims4D::Act::W] = getAuxDataBufferLineWidthFromCSize(
            getOperation()->getParentOfType<mlir::ModuleOp>(), outputTile.shape[Dims4D::Act::C],
            inVTile.shape[Dims4D::Act::W], outputTile.shape[Dims4D::Act::W], getMultiClusterStrategyAttr(),
            getInputSink() != nullptr, getInputMask() != nullptr);

    // InputQ, inputK and InputV are mandatory
    InputTiling inTiles = TilingInfo{{std::move(inQTile), std::move(inKTile), std::move(inVTile)}};

    // Mask, Scale and Bias are optional, but if they are present, they should be tiled if possible
    if (getInputMask() != nullptr) {
        TileInfo inMaskTile(getShape(getInputMask()));
        transferTilingInfoBroadcastAware(inMaskTile, outputTile);
        inTiles.tiles.push_back(inMaskTile);
    }
    if (getInputScale() != nullptr) {
        TileInfo inScaleTile(getShape(getInputScale()));
        transferTilingInfoBroadcastAware(inScaleTile, outputTile);
        inTiles.tiles.push_back(inScaleTile);
    }
    if (getInputSink() != nullptr) {
        TileInfo inSinkTile(getShape(getInputSink()));
        transferTilingInfoBroadcastAware(inSinkTile, outputTile);
        inTiles.tiles.push_back(inSinkTile);
    }
    if (getInputBias() != nullptr) {
        TileInfo inBiasTile(getShape(getInputBias()));
        transferTilingInfoBroadcastAware(inBiasTile, outputTile);
        inTiles.tiles.push_back(inBiasTile);
    }

    inTiles.tiles.push_back(std::move(dataStorageTile));

    TileInfo dpuStorageTile(getShape(getDpuStorage()));
    inTiles.tiles.push_back(dpuStorageTile);

    return inTiles;
}

void vpux::VPU::AttentionOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

SmallVector<mlir::OpOperand*> VPU::AttentionOp::getAuxiliaryBuffers() {
    return {&getDataStorageMutable(), &getDpuStorageMutable()};
}
