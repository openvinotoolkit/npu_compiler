//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sep_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/asm.hpp"

#include "vpux/compiler/utils/permute_utils.hpp"

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops.cpp.inc>

using namespace vpux;

namespace {

//
// LayerWithPermuteInterface
//

template <class MainOpType>
class LayerWithPermuteInterface final :
        public IE::LayerWithPermuteInterface::ExternalModel<LayerWithPermuteInterface<MainOpType>, MainOpType> {
public:
    bool isSupportedPermutation(mlir::Operation* nceOp, mlir::Operation* permuteOp) const {
        if (VPU::getCompilationMode(permuteOp) == VPU::CompilationMode::ReferenceSW) {
            return false;
        }

        // Check if the operation is a valid NCE Op
        if (VPU::NCEInvariant::isSupported(nceOp).failed()) {
            return false;
        }

        if (!isSupportedODUPermute(nceOp, permuteOp)) {
            return false;
        }

        const auto outputShape = getShape(nceOp->getResult(0));
        const auto outputBatch = outputShape[Dims4D::Act::N];
        if (outputBatch != vpux::VPU::NCEInvariant::SUPPORTED_BATCH_SIZE) {
            return false;
        }

        return VPU::NCEInvariant::verifyKernel(mlir::cast<MainOpType>(nceOp)).succeeded();
    }

private:
    bool isSupportedODUPermute(mlir::Operation* nceOp, mlir::Operation* permuteOp) const {
        if (!mlir::isa<IE::ReorderOp, IE::MemPermuteOp>(permuteOp)) {
            return false;
        }

        // Check that reorder is not applied to sub-byte element types:
        const auto elemType = permuteOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
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
        auto seOpsEnabled = VPU::hasEnableSEPtrsOperations(moduleOp);
        auto seExperimentalOpsEnabled = VPU::hasEnableExperimentalSEPtrsOperations(moduleOp);

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

        DimsOrder targetOrder;
        if (auto maybeMemPermute = mlir::dyn_cast_or_null<IE::MemPermuteOp>(permuteOp)) {
            // IE.MemPermute must produce such target orders that they are compatible with ODU.
            const auto inOrder = DimsOrder::fromValue(maybeMemPermute.getInput());
            const auto memPerm = maybeMemPermute.getMemPerm();
            targetOrder = vpux::applyPermutation(inOrder, DimsOrder::fromAffineMap(memPerm));
        } else {
            targetOrder = DimsOrder::fromValue(permuteOp->getResult(0));
        }

        auto adjustOrder = vpux::moveD0ToTheFront(targetOrder);
        if (adjustOrder != targetOrder) {
            const auto inShape = getShape(permuteOp->getOperand(0));
            const auto inMemShape = adjustOrder.toMemoryOrder(inShape);
            auto affineMap = getPermutationFromOrders(adjustOrder, targetOrder, permuteOp->getContext());
            if (!isTrivialPermute(inMemShape, affineMap)) {
                return false;
            }
        }
        return supportedOrders.count(adjustOrder) == 1;
    }
};

}  // namespace

//
// registerLayerWithPermuteInterfaceForIE
//

void vpux::VPU::arch37xx::registerLayerWithPermuteInterfaceForIE(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        IE::ConvolutionOp::attachInterface<LayerWithPermuteInterface<IE::ConvolutionOp>>(*ctx);
        IE::GroupConvolutionOp::attachInterface<LayerWithPermuteInterface<IE::GroupConvolutionOp>>(*ctx);
        IE::TransposedConvolutionOp::attachInterface<LayerWithPermuteInterface<IE::TransposedConvolutionOp>>(*ctx);
        IE::MaxPoolOp::attachInterface<LayerWithPermuteInterface<IE::MaxPoolOp>>(*ctx);
        IE::AvgPoolOp::attachInterface<LayerWithPermuteInterface<IE::AvgPoolOp>>(*ctx);
        IE::AddOp::attachInterface<LayerWithPermuteInterface<IE::AddOp>>(*ctx);
    });
}
