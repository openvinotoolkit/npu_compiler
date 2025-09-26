//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/dma_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/dma_transaction_utils.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

llvm::SmallVector<mlir::FlatSymbolRefAttr> NNDMARewriter::getSymbolicNames(VPUMI40XX::NNDMAOp op, size_t) {
    return createSymbolicName(op);
}

VPUIP::DMADescriptorAttr NNDMARewriter::getDmaDescriptorAttr(VPUMI40XX::NNDMAOp op, mlir::MLIRContext* ctx) const {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto inputTotalSizeBits =
            alignMemSize(inputType.getNumElements() * vpux::getElemTypeSize(inputType), Byte(1));
    const auto inputTotalLength = vpux::Byte(inputTotalSizeBits).count();

    auto [inputMemShape, inputMemStrides, inputElemSize] = getTypeInfo(inputType);
    auto reducedDimsInput = vpux::reduceDimsForDma(std::move(inputMemShape), std::move(inputMemStrides), inputElemSize);

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
            planeLen = inputTotalLength / numPlanes;
        }
    } else {
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(outputBuffers[0].getType());

        auto [outputMemShape, outputMemStrides, outputElemSize] = getTypeInfo(outputType);
        auto reducedDimsOutput =
                vpux::reduceDimsForDma(std::move(outputMemShape), std::move(outputMemStrides), outputElemSize);

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
            planeLen = inputTotalLength / numPlanes;
        } else if (inputTransferRank == 2) {
            const auto outputTotalSizeBits =
                    alignMemSize(outputType.getNumElements() * vpux::getElemTypeSize(outputType), Byte(1));
            const auto outputTotalLength = vpux::Byte(outputTotalSizeBits).count();

            // 3D to 2D transaction
            srcPlaneStride = reducedDimsInput.strides[0];
            numPlanes = inputTotalLength / reducedDimsInput.dims[0];
            planeLen = inputTotalLength / numPlanes;

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
        return mlir::IntegerAttr::get(i32Type, val);
    };

    auto transactionAttr =
            VPUIP::DMADescriptorAttr::get(ctx, attr(numPlanes), attr(planeLen), attr(srcWidth), attr(srcStride),
                                          attr(srcPlaneStride), attr(dstWidth), attr(dstStride), attr(dstPlaneStride));

    return transactionAttr;
}

mlir::FailureOr<SymbolizationResult> NNDMARewriter::symbolize(VPUMI40XX::NNDMAOp op, SymbolMapper& mapper,
                                                              mlir::ConversionPatternRewriter& rewriter) const {
    mlir::MLIRContext* ctx = rewriter.getContext();
    auto result = op.getResult();

    auto symName = findSym(result).getRootReference();
    auto taskLocation = op.getTaskLocation() ? findSym(op.getTaskLocation()) : nullptr;
    auto input = findSym(op.getInput());

    // Checking for CMX broadcast conditions, so first buff should be the same with all other buffers in the list
    auto outputBuffers = op.getOutputBuffs();
    bool isCmxNN = false;
    if (!outputBuffers.empty()) {
        auto firstBuff = std::begin(op.getOutputBuffs());
        isCmxNN = mlir::cast<vpux::NDTypeInterface>(firstBuff.getBase()->get().getType()).getMemoryKind() ==
                  vpux::VPU::MemoryKind::CMX_NN;
    }
    llvm::SmallVector<mlir::Attribute> outputSyms(outputBuffers.size());
    llvm::SmallVector<int64_t, 6> tileIdx;
    for (auto output : llvm::enumerate(outputBuffers)) {
        auto outputIt = mapper.find(output.value());
        VPUX_THROW_WHEN(outputIt == mapper.end(), "Cannot find symbol name entry for {0}", op.getOperationName());

        outputSyms[output.index()] = outputIt->getSecond();
        if (isCmxNN) {
            tileIdx.push_back(
                    mlir::cast<vpux::NDTypeInterface>(output.value().getType()).getMemSpace().getIndex().value());
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
    auto isOutOfOrder = op.getIsOutOfOrderAttr();
    auto isCritical = op.getIsCriticalAttr();
    auto enableMSC = op.getEnableMscAttr();
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

    auto descriptor = op.getDmaDescriptor().has_value() ? op.getDmaDescriptorAttr() : getDmaDescriptorAttr(op, ctx);
    if (!descriptor) {
        _log.warning("Failed to lower DMA descriptor parameters");
    }
    auto newOp = rewriter.create<VPUASM::NNDMAOp>(
            op.getLoc(), symName, taskIdx, taskLocation, nextLink, input, outputs, waitAttr, updateAttr, startAfter,
            cleanAfter, accelerationMode, isOutOfOrder, isCritical, enableMSC, actCompressionSizeEntryAttr,
            sparsityMapAttr, transaction, descriptor, dmaHwpIdAttr, cmxTiles, indicesAttr, addressingModeAttr);

    mlir::SmallVector<mlir::StringAttr> refsToUpdate;
    if (nextLink && nextLink.getNestedReferences().empty()) {
        refsToUpdate.push_back(newOp.getNextLinkAttrName());
    }

    rewriter.eraseOp(op);

    return SymbolizationResult(newOp, std::move(refsToUpdate));
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
