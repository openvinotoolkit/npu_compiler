//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/dma_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/core/IR/strided_dmas_utils.hpp"
#include "vpux/compiler/utils/dma_transaction_utils.hpp"

#include <algorithm>

namespace vpux {
namespace vpumi40xx2vpuasm {

VPUIP::DMADescriptorAttr NNDMARewriter::getDmaDescriptorAttr(VPUMI40XX::NNDMAOp op, mlir::MLIRContext* ctx) const {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto inputTotalSizeBits =
            alignMemSize(inputType.getNumElements() * vpux::getElemTypeSize(inputType), Byte(1));
    const auto inputTotalLength = vpux::Byte(inputTotalSizeBits).count();

    auto [inputMemShape, inputMemStrides, inputElemSize] = getTypeInfo(inputType);
    auto reducedDimsInput =
            vpux::reduceDimsForDma(std::move(inputMemShape), std::move(inputMemStrides), inputElemSize, false);

    vpux::patchDimsForNPU37XX(reducedDimsInput);

    VPUX_THROW_WHEN(reducedDimsInput.dims.size() != reducedDimsInput.strides.size(),
                    "Non matching rank between dims {0} and strides {1} for input", reducedDimsInput.dims.size(),
                    reducedDimsInput.strides.size());

    auto inputTransferRank = reducedDimsInput.dims.size();
    size_t outputTransferRank = 0;

    const auto inputInnerMostDim = inputTransferRank - 1;
    size_t outputInnerMostDim = 0;

    auto srcWidth = reducedDimsInput.dims[inputInnerMostDim];
    auto srcStride = reducedDimsInput.strides[inputInnerMostDim];
    size_t dstWidth = 0;
    size_t dstStride = 0;

    uint32_t srcPlaneStride = 0;
    uint32_t dstPlaneStride = 0;
    uint32_t planeLen = inputTotalLength;
    uint32_t numPlanes = 0;

    const auto outputBuffers = op.getOutputBuffs();
    if (outputBuffers.empty()) {
        if (inputTransferRank == 2) {
            srcPlaneStride = reducedDimsInput.strides[0];
            numPlanes = inputTotalLength / reducedDimsInput.dims[0];
            planeLen = reducedDimsInput.dims[0];
        }
    } else {
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(outputBuffers[0].getType());

        auto [outputMemShape, outputMemStrides, outputElemSize] = getTypeInfo(outputType);
        auto reducedDimsOutput =
                vpux::reduceDimsForDma(std::move(outputMemShape), std::move(outputMemStrides), outputElemSize, false);

        vpux::patchDimsForNPU37XX(reducedDimsOutput);
        VPUX_THROW_WHEN(reducedDimsOutput.dims.size() != reducedDimsOutput.strides.size(),
                        "Non matching rank between dims {0} and strides {1} for output", reducedDimsOutput.dims.size(),
                        reducedDimsOutput.strides.size());
        outputTransferRank = reducedDimsOutput.dims.size();

        if ((inputTransferRank > 2 || outputTransferRank > 2)) {
            _log.warning("cannot reduce dims to 2 for DMA; Reduced InSize: {0}, OutSize: {1}", inputTransferRank,
                         outputTransferRank);
            return nullptr;
        }

        outputInnerMostDim = outputTransferRank - 1;
        dstWidth = reducedDimsOutput.dims[outputInnerMostDim];
        dstStride = reducedDimsOutput.strides[outputInnerMostDim];

        if ((inputTransferRank == 2 && outputTransferRank == 2)) {
            // 3D to 3D transaction
            if (reducedDimsInput.dims[0] != reducedDimsOutput.dims[0]) {
                _log.error("DMA's don't have equal plane size {0} != {1}", reducedDimsInput.dims[0],
                           reducedDimsOutput.dims[0]);
                return nullptr;
            }
            srcPlaneStride = reducedDimsInput.strides[0];
            dstPlaneStride = reducedDimsOutput.strides[0];
            numPlanes = inputTotalLength / reducedDimsInput.dims[0];
            planeLen = reducedDimsInput.dims[0];
        } else if (inputTransferRank == 2) {
            const auto outputTotalSizeBits =
                    alignMemSize(outputType.getNumElements() * vpux::getElemTypeSize(outputType), Byte(1));
            const auto outputTotalLength = vpux::Byte(outputTotalSizeBits).count();

            // 3D to 2D transaction
            srcPlaneStride = reducedDimsInput.strides[0];
            numPlanes = inputTotalLength / reducedDimsInput.dims[0];
            planeLen = reducedDimsInput.dims[0];

            const uint32_t outputPlaneLen = outputTotalLength / numPlanes;
            if (outputTotalLength == static_cast<int64_t>(dstWidth)) {
                dstWidth = outputPlaneLen;
                dstStride = outputPlaneLen;
                dstPlaneStride = outputPlaneLen;
            } else {
                dstPlaneStride = (outputPlaneLen * dstStride) / dstWidth;
                dstWidth = std::min(static_cast<uint32_t>(dstWidth), outputPlaneLen);
                dstStride = std::min(static_cast<uint32_t>(dstStride), outputPlaneLen);
            }

        } else if (outputTransferRank == 2) {
            // 2D to 3D transaction
            auto inputElemTypeSize = vpux::Byte(vpux::getElemTypeSize(inputType)).count();
            auto outputElemTypeSize = vpux::Byte(vpux::getElemTypeSize(outputType)).count();

            dstPlaneStride = reducedDimsOutput.strides[0];

            if (inputElemTypeSize != outputElemTypeSize && inputElemTypeSize > outputElemTypeSize) {
                numPlanes = inputTotalLength / (reducedDimsOutput.dims[0] * (inputElemTypeSize / outputElemTypeSize));
            } else {
                numPlanes = inputTotalLength / reducedDimsOutput.dims[0];
            }

            planeLen = inputTotalLength / numPlanes;
            if (inputTotalLength == static_cast<int64_t>(srcWidth)) {
                srcWidth = planeLen;
                srcStride = planeLen;
                srcPlaneStride = planeLen;
            } else {
                srcPlaneStride = (planeLen * srcStride) / srcWidth;
                srcWidth = std::min(static_cast<uint32_t>(srcWidth), planeLen);
                srcStride = std::min(static_cast<uint32_t>(srcStride), planeLen);
            }
        }
    }

    VPUX_THROW_WHEN((numPlanes > 0) && ((inputTotalLength % numPlanes) != 0),
                    "Number of planes is not a divisor of total transaction length");
    VPUX_THROW_WHEN((numPlanes > 0) && ((planeLen % srcWidth) != 0),
                    "Source width is not a divisor of transaction plane length");

    auto attr = [&ctx](uint64_t val) -> mlir::IntegerAttr {
        auto i32Type = mlir::IntegerType::get(ctx, sizeof(uint32_t) * CHAR_BIT);
        return mlir::IntegerAttr::get(i32Type, static_cast<int64_t>(val));
    };

    auto transactionAttr =
            VPUIP::DMADescriptorAttr::get(ctx, attr(numPlanes), attr(planeLen), attr(srcWidth), attr(srcStride),
                                          attr(srcPlaneStride), attr(dstWidth), attr(dstStride), attr(dstPlaneStride));

    return transactionAttr;
}

mlir::FailureOr<SymbolizationResult> NNDMARewriter::symbolize(VPUMI40XX::NNDMAOp op, SymbolMapper& mapper,
                                                              mlir::ConversionPatternRewriter& rewriter) const {
    constexpr auto maxTilesPerDma = 6;

    mlir::MLIRContext* ctx = rewriter.getContext();
    auto result = op.getResult();

    auto symName = findSym(result).getRootReference();
    auto taskLocation = op.getTaskLocation() ? findSym(op.getTaskLocation()) : nullptr;
    auto input = findSym(op.getInput());

    // Checking for CMX broadcast conditions, so first buff should be the same with all other buffers in the list
    auto outputBuffers = op.getOutputBuffs();
    const auto numOutputs = outputBuffers.size();

    SmallVector<mlir::Attribute> outputSyms;
    outputSyms.reserve(numOutputs);
    SmallVector<int64_t, maxTilesPerDma> tileIdx;
    if (!outputBuffers.empty()) {
        tileIdx.reserve(numOutputs);
    }

    bool isCmxNN = false;
    if (!outputBuffers.empty()) {
        const auto firstOutputType = mlir::cast<vpux::NDTypeInterface>(outputBuffers.front().getType());
        isCmxNN = firstOutputType.getMemoryKind() == vpux::VPU::MemoryKind::CMX_NN;
    }

    for (auto outputBuff : outputBuffers) {
        auto outputIt = mapper.find(outputBuff);
        VPUX_THROW_WHEN(outputIt == mapper.end(), "Cannot find symbol name entry for {0}", op.getOperationName());

        outputSyms.push_back(outputIt->getSecond());
        if (isCmxNN) {
            tileIdx.push_back(mlir::cast<vpux::NDTypeInterface>(outputBuff.getType()).getMemSpace().getIndex().value());
        }
    }

    auto outputs = mlir::ArrayAttr::get(ctx, llvm::ArrayRef<mlir::Attribute>(outputSyms));
    auto cmxTiles = tileIdx.empty() ? nullptr : rewriter.getI64ArrayAttr(ArrayRef(tileIdx));

    auto nextDmaIt = std::find_if(result.user_begin(), result.user_end(), [](mlir::Operation* op) -> bool {
        return mlir::isa<VPUMI40XX::NNDMAOp>(op);
    });

    mlir::SymbolRefAttr nextLink = nullptr;
    if (nextDmaIt != result.user_end()) {
        auto nextDma = mlir::cast<VPUMI40XX::NNDMAOp>(*nextDmaIt);
        auto nextTaskLocation = nextDma.getTaskLocation();
        auto nextDmaTaskLink = nextDma.getTaskLink();
        if (nextTaskLocation || nextDmaTaskLink.has_value()) {
            assert(!nextDmaTaskLink.has_value() || nextDmaTaskLink.value() == op.getType());
            auto nextLinkIt = mapper.find(nextTaskLocation ? nextTaskLocation : nextDma.getResult());
            VPUX_THROW_WHEN(nextLinkIt == mapper.end(), "Cannot find symbol name entry for {0}",
                            nextDma.getOperationName());
            nextLink = nextLinkIt->getSecond();
        }
    }

    auto accelerationMode = VPUIP::DMAAccModeAttr::get(ctx, op.getAccelerationMode());
    auto startAfter = op.getStartAfterAttr();
    auto cleanAfter = op.getCleanAfterAttr();
    auto transaction = op.getDmaTransactionAttr();
    mlir::SymbolRefAttr actCompressionSizeEntryAttr =
            op.getActCompressionSizeEntry() ? findSym(op.getActCompressionSizeEntry()) : nullptr;

    auto indices = op.getIndices();
    mlir::SymbolRefAttr indicesAttr = indices ? findSym(indices) : nullptr;

    mlir::SymbolRefAttr sparsityMapAttr =
            op.getActCompressionSparsityMap() ? findSym(op.getActCompressionSparsityMap()) : nullptr;

    auto waitAttr = vectorizeBarriers(op.getWaitBarriers());
    auto updateAttr = vectorizeBarriers(op.getUpdateBarriers());

    auto taskIdx = mlir::TypeAttr::get(op.getType());

    auto dmaHwpIdAttr = op.getDmaHwpIdAttr();
    auto addressingModeAttr = op.getAddressingModeAttr();

    auto skipDmaAttr = op.getSkipDmaAttr();
    auto fetchDmaAttr = op.getFetchDmaAttr();

    auto taskDynIdAttr = op.getTaskDynIdAttr();
    mlir::SymbolRefAttr dynamicSequenceLenBuffAttr =
            op.getDynamicSequenceLengthBuff() ? findSym(op.getDynamicSequenceLengthBuff()) : nullptr;

    auto descriptor = op.getDmaDescriptorAttr();

    // Prioritize the newer DMATransactionAttr
    if (!transaction && !descriptor) {
        descriptor = getDmaDescriptorAttr(op, ctx);
        VPUX_THROW_WHEN(!descriptor, "Failed to lower DMA descriptor parameters");
    }

    auto newOp = rewriter.create<VPUASM::NNDMAOp>(
            op.getLoc(), symName, taskIdx, taskLocation, nextLink, input, outputs, waitAttr, updateAttr, startAfter,
            cleanAfter, accelerationMode, op.getDmaEncodingAlgoAttr(), op.getIsOutOfOrder(), op.getIsCritical(),
            op.getEnableMsc(), actCompressionSizeEntryAttr, sparsityMapAttr, transaction, descriptor, dmaHwpIdAttr,
            cmxTiles, indicesAttr, addressingModeAttr, skipDmaAttr, fetchDmaAttr, taskDynIdAttr,
            dynamicSequenceLenBuffAttr);

    if (auto strided = op->getAttr(vpux::stridedInputAttrName)) {
        newOp->setAttr(vpux::stridedInputAttrName, strided);
    }

    if (auto strided = op->getAttr(vpux::stridedOutputAttrName)) {
        newOp->setAttr(vpux::stridedOutputAttrName, strided);
    }

    mlir::SmallVector<mlir::StringAttr> refsToUpdate;
    if (nextLink && nextLink.getNestedReferences().empty()) {
        refsToUpdate.push_back(newOp.getNextLinkAttrName());
    }

    rewriter.eraseOp(op);

    return SymbolizationResult(newOp, std::move(refsToUpdate));
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
