//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/dma_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/dma_transaction_utils.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

llvm::SmallVector<mlir::FlatSymbolRefAttr> NNDMARewriter::getSymbolicNames(VPUMI40XX::NNDMAOp op, size_t) {
    auto fullName = VPUMI40XX::NNDMAOp::getOperationName();
    auto opName = fullName.drop_front(VPUMI40XX::VPUMI40XXDialect::getDialectNamespace().size() + 1);

    auto tileIdx = std::to_string(op.getType().getTileIdx());
    auto srcTypeIdx = std::to_string(op.getType().getListIdx());
    auto opIdx = std::to_string(op.getType().getValue());

    auto symName = mlir::StringAttr::get(op.getContext(), opName + "_" + tileIdx + "_" + srcTypeIdx + "_" + opIdx);

    return {mlir::FlatSymbolRefAttr::get(symName)};
}

VPUIP::DMADescriptorAttr NNDMARewriter::getDmaDescriptorAttr(VPUMI40XX::NNDMAOp op, mlir::MLIRContext* ctx) const {
    auto inputType = op.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outputType = op.getOutputBuffs()[0].getType().cast<vpux::NDTypeInterface>();
    const Bit elemSize = vpux::getElemTypeSize(inputType);
    auto totalSizeBits = alignMemSize(inputType.getNumElements() * elemSize, Byte(1));
    auto totalLength = vpux::Byte(totalSizeBits).count();

    auto reduced_dims_input = vpux::reduceDimsForDma(inputType);
    vpux::patchDimsForNPU37XX(reduced_dims_input);
    auto reduced_dims_output = vpux::reduceDimsForDma(outputType);
    vpux::patchDimsForNPU37XX(reduced_dims_output);

    VPUX_THROW_WHEN(reduced_dims_input.dims.size() != reduced_dims_input.strides.size(),
                    "Non matching rank between dims {0} and strides {1} for input", reduced_dims_input.dims.size(),
                    reduced_dims_input.strides.size());
    VPUX_THROW_WHEN(reduced_dims_output.dims.size() != reduced_dims_output.strides.size(),
                    "Non matching rank between dims {0} and strides {1} for output", reduced_dims_output.dims.size(),
                    reduced_dims_output.strides.size());

    auto inputTransferRank = reduced_dims_input.dims.size();
    auto outputTransferRank = reduced_dims_output.dims.size();

    if (inputTransferRank > 2 || outputTransferRank > 2) {
        _log.warning("cannot reduce dims to 2 for DMA; Reduced InSize: {0}, OutSize: {1}", inputTransferRank,
                     outputTransferRank);
        return nullptr;
    }

    const auto inputInnerMostDim = inputTransferRank - 1;
    const auto outputInnerMostDim = outputTransferRank - 1;

    auto src_width = reduced_dims_input.dims[inputInnerMostDim];
    auto dst_width = reduced_dims_output.dims[outputInnerMostDim];
    auto src_stride = reduced_dims_input.strides[inputInnerMostDim];
    auto dst_stride = reduced_dims_output.strides[outputInnerMostDim];

    uint32_t src_plane_stride = 0;
    uint32_t dst_plane_stride = 0;
    uint32_t plane_len = 0;
    uint32_t num_planes = 0;

    if (inputTransferRank == 2 && outputTransferRank == 2) {
        // 3D to 3D transaction
        if (reduced_dims_input.dims[0] != reduced_dims_output.dims[0]) {
            _log.error("DMA's don't have equal plane size {0} != {1}", reduced_dims_input.dims[0],
                       reduced_dims_output.dims[0]);
            return nullptr;
        }

        src_plane_stride = reduced_dims_input.strides[0];
        dst_plane_stride = reduced_dims_output.strides[0];
        num_planes = totalLength / reduced_dims_input.dims[0];
        plane_len = totalLength / num_planes;
    } else if (inputTransferRank == 2) {
        // 3D to 2D transaction
        src_plane_stride = reduced_dims_input.strides[0];
        num_planes = totalLength / reduced_dims_input.dims[0];

        plane_len = totalLength / num_planes;
        if (totalLength == static_cast<int64_t>(dst_width)) {
            dst_width = plane_len;
            dst_stride = plane_len;
            dst_plane_stride = plane_len;
        } else {
            dst_plane_stride = (plane_len * dst_stride) / dst_width;
            dst_width = std::min(static_cast<uint32_t>(dst_width), plane_len);
            dst_stride = std::min(static_cast<uint32_t>(dst_stride), plane_len);
        }
    } else if (outputTransferRank == 2) {
        // 2D to 3D transaction
        dst_plane_stride = reduced_dims_output.strides[0];
        num_planes = totalLength / reduced_dims_output.dims[0];

        plane_len = totalLength / num_planes;
        if (totalLength == static_cast<int64_t>(src_width)) {
            src_width = plane_len;
            src_stride = plane_len;
            src_plane_stride = plane_len;
        } else {
            src_plane_stride = (plane_len * src_stride) / src_width;
            src_width = std::min(static_cast<uint32_t>(src_width), plane_len);
            src_stride = std::min(static_cast<uint32_t>(src_stride), plane_len);
        }
    } else {
        src_plane_stride = 0;
        dst_plane_stride = 0;
        num_planes = 0;
        plane_len = totalLength;
    }

    VPUX_THROW_WHEN((num_planes > 0) && ((totalLength % num_planes) != 0),
                    "Number of planes is not a divisor of total transaction length");
    VPUX_THROW_WHEN((num_planes > 0) && ((plane_len % src_width) != 0),
                    "Source width is not a divisor of transaction plane length");
    VPUX_THROW_WHEN((num_planes > 0) && ((plane_len % dst_width) != 0),
                    "Destination width is not a divisor of transaction plane length");

    auto attr = [&ctx](uint64_t val) -> mlir::IntegerAttr {
        auto i32Type = mlir::IntegerType::get(ctx, sizeof(uint32_t) * CHAR_BIT);
        return mlir::IntegerAttr::get(i32Type, val);
    };

    auto transactionAttr = VPUIP::DMADescriptorAttr::get(ctx, attr(num_planes), attr(plane_len), attr(src_width),
                                                         attr(src_stride), attr(src_plane_stride), attr(dst_width),
                                                         attr(dst_stride), attr(dst_plane_stride));

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
    auto firstBuff = std::begin(op.getOutputBuffs());
    auto isCmxNN = firstBuff.getBase()->get().getType().cast<NDTypeInterface>().getMemoryKind() ==
                   vpux::VPU::MemoryKind::CMX_NN;

    llvm::SmallVector<mlir::Attribute> outputSyms(op.getOutputBuffs().size());
    llvm::SmallVector<int64_t, 6> tileIdx;
    for (auto output : llvm::enumerate(op.getOutputBuffs())) {
        auto outputIt = mapper.find(output.value());
        VPUX_THROW_WHEN(outputIt == mapper.end(), "Cannot find symbol name entry for {0}", op.getOperationName());

        outputSyms[output.index()] = outputIt->getSecond();
        if (isCmxNN) {
            tileIdx.push_back(output.value().getType().cast<NDTypeInterface>().getMemSpace().getIndex().value());
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

    auto descriptor = op.getDmaDescriptor().has_value() ? op.getDmaDescriptorAttr() : getDmaDescriptorAttr(op, ctx);
    if (!descriptor) {
        _log.warning("Failed to lower DMA descriptor parameters");
    }
    auto newOp = rewriter.create<VPUASM::NNDMAOp>(
            op.getLoc(), symName, taskIdx, taskLocation, nextLink, input, outputs, waitAttr, updateAttr, startAfter,
            cleanAfter, accelerationMode, isOutOfOrder, isCritical, enableMSC, actCompressionSizeEntryAttr,
            sparsityMapAttr, transaction, descriptor, dmaHwpIdAttr, cmxTiles, indicesAttr);

    mlir::SmallVector<mlir::StringAttr> refsToUpdate;
    if (nextLink && nextLink.getNestedReferences().empty()) {
        refsToUpdate.push_back(newOp.getNextLinkAttrName());
    }

    rewriter.eraseOp(op);

    return SymbolizationResult(newOp, std::move(refsToUpdate));
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
