//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/dma_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

llvm::SmallVector<mlir::FlatSymbolRefAttr> NNDMARewriter::getSymbolicNames(VPUMI37XX::NNDMAOp op, size_t) {
    return createSymbolicName(op, std::nullopt, std::nullopt);
}

llvm::SmallVector<std::pair<uint32_t, int32_t>> NNDMARewriter::reduce_dims_for_dma(mlir::Value val) {
    auto ndType = mlir::cast<vpux::NDTypeInterface>(val.getType());
    const auto memShape = ndType.getMemShape();
    const auto memStrides = ndType.getMemStrides();
    const Bit ndTypeElemSize = ndType.getElemTypeSize();

    auto inner_most_index = memShape.size() - 1;
    llvm::SmallVector<std::pair<uint32_t, int32_t>> finalDims;

    auto previous_size = checked_cast<uint32_t>(memShape[MemDim(inner_most_index)]);
    auto previous_stride_bits = checked_cast<uint32_t>(vpux::Bit(memStrides[MemDim(inner_most_index)]).count());

    if (previous_size * ndTypeElemSize.count() < previous_stride_bits) {
        int32_t final_stride = previous_stride_bits / CHAR_BIT;
        uint32_t final_size = alignMemSize(previous_size * ndTypeElemSize, Byte(1)).to<Byte>().count();

        finalDims.push_back({final_size, final_stride});
    }

    // TODO: Could there be some way to iterate over all MemDim's of a particular shape/order?
    //       Please refer to the ticket E#36225.
    for (auto dim = checked_cast<int64_t>(inner_most_index) - 1; dim >= 0; --dim) {
        auto memDim = MemDim(dim);

        auto current_size = checked_cast<uint32_t>(memShape[memDim]);
        auto current_stride_bits = checked_cast<uint32_t>(vpux::Bit(memStrides[memDim]).count());

        if (previous_size * previous_stride_bits < current_stride_bits) {
            int32_t final_stride = current_stride_bits / CHAR_BIT;
            uint32_t final_size = (previous_size * previous_stride_bits) / CHAR_BIT;

            finalDims.push_back({final_size, final_stride});
        }

        previous_size = current_size;
        previous_stride_bits = current_stride_bits;
    }

    if (finalDims.size() == 0) {
        uint32_t final_size = (previous_size * previous_stride_bits) / CHAR_BIT;
        int32_t final_stride = final_size;
        finalDims.push_back({final_size, final_stride});
    }

    return finalDims;
}

VPUIP::DMADescriptorAttr NNDMARewriter::getDmaTransactionTraits(VPUMI37XX::NNDMAOp op, mlir::MLIRContext* ctx) const {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const Bit elemSize = vpux::getElemTypeSize(inputType);
    auto totalSizeBits = alignMemSize(inputType.getNumElements() * elemSize, Byte(1));
    auto length = vpux::Byte(totalSizeBits).count();

    auto reduced_dims_input = reduce_dims_for_dma(op.getInput());
    auto reduced_dims_output = reduce_dims_for_dma(op.getOutputBuffs()[0]);

    if (reduced_dims_input.size() > 2 || reduced_dims_output.size() > 2) {
        _log.error("cannot reduce dims to 2 for DMA; Reduced InSize: {0}, OutSize: {1}", reduced_dims_input.size(),
                   reduced_dims_output.size());
        return nullptr;
    }

    auto src_width = reduced_dims_input[0].first;
    auto dst_width = reduced_dims_output[0].first;
    auto src_stride = reduced_dims_input[0].second;
    auto dst_stride = reduced_dims_output[0].second;

    uint32_t src_plane_stride;
    uint32_t dst_plane_stride;
    uint32_t num_planes;

    if (reduced_dims_input.size() == 2 && reduced_dims_output.size() == 2) {
        if (reduced_dims_input[1].first != reduced_dims_output[1].first) {
            _log.error("DMA's don't have equal plane size {0} != {1}", reduced_dims_input[1].first,
                       reduced_dims_output[1].first);
            return nullptr;
        }

        src_plane_stride = reduced_dims_input[1].second;
        dst_plane_stride = reduced_dims_output[1].second;
        num_planes = length / reduced_dims_input[1].first;
    } else if (reduced_dims_input.size() == 2) {
        src_plane_stride = reduced_dims_input[1].second;
        dst_plane_stride = dst_stride;
        num_planes = length / reduced_dims_input[1].first;
    } else if (reduced_dims_output.size() == 2) {
        src_plane_stride = src_stride;
        dst_plane_stride = reduced_dims_output[1].second;
        num_planes = length / reduced_dims_output[1].first;
    } else {
        src_plane_stride = 0;
        dst_plane_stride = 0;
        num_planes = 0;
    }

    auto attr = [&ctx](uint64_t val) -> mlir::IntegerAttr {
        auto i32Type = mlir::IntegerType::get(ctx, sizeof(uint32_t));
        return mlir::IntegerAttr::get(i32Type, val);
    };

    auto transactionAttr = VPUIP::DMADescriptorAttr::get(ctx, attr(num_planes), attr(length), attr(src_width),
                                                         attr(src_stride), attr(src_plane_stride), attr(dst_width),
                                                         attr(dst_stride), attr(dst_plane_stride));

    return transactionAttr;
}

mlir::FailureOr<SymbolizationResult> NNDMARewriter::symbolize(VPUMI37XX::NNDMAOp op, SymbolMapper& mapper,
                                                              mlir::ConversionPatternRewriter& rewriter) const {
    mlir::MLIRContext* ctx = rewriter.getContext();
    auto result = op.getResult();

    auto symName = findSym(result).getRootReference();
    auto taskLocation = findSym(op.getTaskLocation());
    auto input = findSym(op.getInput());

    llvm::SmallVector<mlir::Attribute> outputSyms(op.getOutputBuffs().size());
    for (auto output : llvm::enumerate(op.getOutputBuffs())) {
        auto outputIt = mapper.find(output.value());
        VPUX_THROW_WHEN(outputIt == mapper.end(), "Cannot find symbol name entry for {0}", op.getOperationName());

        outputSyms[output.index()] = outputIt->getSecond();
    }
    auto outputs = mlir::ArrayAttr::get(ctx, llvm::ArrayRef<mlir::Attribute>(outputSyms));

    mlir::SymbolRefAttr nextLink;
    if (auto nextDmaIdx = op.getNextDMAIdx()) {
        auto nextDma = mlir::cast<VPUMI37XX::NNDMAOp>(nextDmaIdx.getDefiningOp());
        auto nextLinkIt = mapper.find(nextDmaIdx);
        VPUX_THROW_WHEN(nextLinkIt == mapper.end(), "Cannot find symbol name entry for {0}",
                        nextDma.getOperationName());
        nextLink = nextLinkIt->getSecond();
    }

    auto descriptor = op.getDmaDescriptor().value_or(getDmaTransactionTraits(op, ctx));
    if (!descriptor) {
        _log.error("Failed to lower DMA descriptor parameters");
        return mlir::failure();
    }

    auto accelerationMode = VPUIP::DMAAccModeAttr::get(ctx, op.getAccelerationMode());
    auto startAfter = op.getStartAfterAttr();
    auto cleanAfter = op.getCleanAfterAttr();
    auto isOutOfOrder = op.getIsOutOfOrderAttr();
    auto isCritical = op.getIsCriticalAttr();
    auto waitAttr = vectorizeBarriers(op.getWaitBarriers());
    auto updateAttr = vectorizeBarriers(op.getUpdateBarriers());

    auto taskIdx = mlir::TypeAttr::get(op.getType());

    auto newOp = rewriter.create<VPUASM::NNDMAOp>(
            op.getLoc(), symName, taskIdx, taskLocation, nextLink, input, outputs, waitAttr, updateAttr, startAfter,
            cleanAfter, accelerationMode, isOutOfOrder, isCritical,
            /*enable_msc*/ nullptr,
            /*act_compression_size_entry*/ nullptr, /*act_compression_sparsity_map*/ nullptr,
            /*dma_transaction*/ nullptr, descriptor,
            /*dma_hwp_id*/ nullptr, /*tile_indexes*/ nullptr, /*indices*/ nullptr, /*addressing_mode*/ nullptr);

    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux
