//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/common_utils/layer_permute_ie.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

DimsOrder vpux::VPU::getTargetOrder(mlir::Operation* permuteOp) {
    if (auto maybeMemPermute = mlir::dyn_cast_or_null<IE::MemPermuteOp>(permuteOp)) {
        // IE.MemPermute must produce such target orders that they are compatible with ODU.
        const auto inOrder = DimsOrder::fromValue(maybeMemPermute.getInput());
        const auto memPerm = maybeMemPermute.getMemPerm();
        return vpux::applyPermutation(inOrder, DimsOrder::fromAffineMap(memPerm));
    }

    return DimsOrder::fromValue(permuteOp->getResult(0));
}

bool vpux::VPU::isSupportedPermutation(mlir::Operation* nceOp, mlir::Operation* permuteOp) {
    if (config::getCompilationMode(permuteOp) == config::CompilationMode::ReferenceSW) {
        return false;
    }

    // Check if the operation is a valid NCE Op
    if (VPU::NCEInvariant::isSupported(nceOp).failed()) {
        return false;
    }

    if (!mlir::isa<IE::ReorderOp, IE::MemPermuteOp>(permuteOp)) {
        return false;
    }

    // Check that reorder is not applied to sub-byte element types:
    const auto elemType = mlir::cast<vpux::NDTypeInterface>(permuteOp->getResult(0).getType());
    const Bit elemSize = vpux::getElemTypeSize(elemType);
    if (elemSize.count() < CHAR_BIT) {
        return false;
    }

    // Check that permutation is supported by ODU
    std::unordered_set<DimsOrder> supportedOrders = {
            DimsOrder::NCHW, DimsOrder::NCWH, DimsOrder::NHCW, DimsOrder::NHWC, DimsOrder::NWCH, DimsOrder::NWHC,
    };

    /* SEP dilated convolution must have contiguous channels in output
       since it is always followed by strided concat.
       Strided concat interleaves H and W. If one of them is inner dimesion,
       it will lead to extremely slow DMA concatenation. */
    auto moduleOp = getModuleOp(nceOp);
    auto seOpsEnabled = config::hasEnableSEPtrsOperations(moduleOp);
    auto seExperimentalOpsEnabled = config::hasEnableExperimentalSEPtrsOperations(moduleOp);

    if (seExperimentalOpsEnabled && seOpsEnabled) {
        const auto logCb = [&](const formatv_object_base& msg) {
            Logger::global().trace("{0}", msg.str());
        };
        // TODO: E153229
        if (auto groupConv = mlir::dyn_cast_or_null<IE::GroupConvolutionOp>(nceOp)) {
            const auto isSepDilatedConv =
                    VPU::isSupportedSEPDilatedConv(groupConv, logCb,
                                                   /*checkLayout=*/false, /*checkChannelAlignment=*/false);
            if (isSepDilatedConv) {
                supportedOrders = {DimsOrder::NHWC, DimsOrder::NWHC};
            }
        }
    }

    const auto targetOrder = getTargetOrder(permuteOp);

    auto adjustOrder = vpux::moveD0ToTheFront(targetOrder);
    if (adjustOrder != targetOrder) {
        const auto inShape = getShape(permuteOp->getOperand(0));
        const auto inMemShape = adjustOrder.toMemoryOrder(inShape);
        auto affineMap = getPermutationFromOrders(adjustOrder, targetOrder, permuteOp->getContext());
        if (!isTrivialPermute(inMemShape, affineMap)) {
            return false;
        }
    }
    if (supportedOrders.count(adjustOrder) != 1) {
        return false;
    }

    const auto outputShape = getShape(nceOp->getResult(0));
    const auto outputBatch = outputShape[Dims4D::Act::N];
    if (outputBatch != vpux::VPU::NCEInvariant::SUPPORTED_BATCH_SIZE) {
        return false;
    }

    return true;
}
