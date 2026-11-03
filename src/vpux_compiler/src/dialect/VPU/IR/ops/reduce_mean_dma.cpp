//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <limits>

using namespace vpux;

namespace {

// The reduce_mean_dma kernel takes the whole fragmentation-aware CMX as its workspace, as AttentionDMA does.
// It needs about 9 KB of it on NPU40XX/50XX: 7 KB of DMA driver chains for 2 SHAVEs and 2 DMA ports, 0.5 KB
// of SHAVE synchronization buffers, and per SHAVE one 192-byte DMA descriptor and a minimum 512-byte input
// chunk. It splits the rest into its input chunks, so a larger workspace means fewer, larger input DMAs.
mlir::Type getAuxiliaryBufferType(mlir::ModuleOp module) {
    const auto auxBytes = VPU::getTotalCMXFragmentationAwareSize(module).count();
    return mlir::RankedTensorType::get({1, 1, 1, auxBytes}, getUInt8Type(module.getContext()));
}

// Smallest workspace the verifier accepts, with room to spare over the kernel's fixed needs.
constexpr int64_t minWorkspaceBytes = 32 * 1024;

}  // namespace

mlir::LogicalResult VPU::ReduceMeanDMAOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange,
                                                           SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    VPU::ReduceMeanDMAOpAdaptor adaptor(operands, attrs, prop);
    if (mlir::failed(adaptor.verify(loc))) {
        return mlir::failure();
    }
    SmallVector<int64_t> axes{Dims4D::Act::H.ind(), Dims4D::Act::W.ind()};
    return VPU::inferReduceReturnTypes(loc, adaptor.getInput(), true, axes, inferredReturnTypes);
}

void VPU::ReduceMeanDMAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input) {
    SmallVector<mlir::Type> outputs;
    SmallVector<int64_t> axes{Dims4D::Act::H.ind(), Dims4D::Act::W.ind()};
    VPUX_THROW_UNLESS(mlir::succeeded(VPU::inferReduceReturnTypes(state.location, input, true, axes, outputs)),
                      "Cannot infer DMA Mean output");
    const auto auxiliaryType = getAuxiliaryBufferType(getModuleOp(builder));
    const auto auxiliary = VPU::createEmptyAuxiliaryBuffer(builder, state.location, auxiliaryType);
    build(builder, state, outputs.front(), input, auxiliary, nullptr);
}

bool VPU::ReduceMeanDMAOp::isSupportedInput(mlir::Type inputType, vpux::LogCb logCb) {
    const auto input = mlir::cast<NDTypeInterface>(inputType);
    const auto shape = input.getShape();
    if (shape.size() != Dims4D::Act::numDims || shape[Dims4D::Act::N] != 1 || input.getDimsOrder() != DimsOrder::NCHW) {
        logCb(llvm::formatv("DMA Mean requires NCHW [1,C,H,W] input, got shape '{0}' with order '{1}'", shape,
                            input.getDimsOrder()));
        return false;
    }
    if (mlir::ShapedType::isDynamic(shape[Dims4D::Act::C]) || shape[Dims4D::Act::C] <= 0 ||
        shape[Dims4D::Act::C] > std::numeric_limits<int32_t>::max()) {
        logCb(llvm::formatv("DMA Mean requires NCHW [1,C,H,W] with positive int32 static C, got '{0}'",
                            shape[Dims4D::Act::C]));
        return false;
    }
    const auto bounds = getBoundedShape(inputType);
    if (bounds[Dims4D::Act::H] <= 0 || bounds[Dims4D::Act::W] <= 0 ||
        bounds[Dims4D::Act::H] > std::numeric_limits<int32_t>::max() / bounds[Dims4D::Act::W]) {
        logCb(llvm::formatv("DMA Mean requires positive spatial bounds with an int32 reduction count, got H '{0}' "
                            "and W '{1}'",
                            bounds[Dims4D::Act::H], bounds[Dims4D::Act::W]));
        return false;
    }
    return true;
}

mlir::LogicalResult VPU::ReduceMeanDMAOp::verify() {
    const auto input = mlir::cast<NDTypeInterface>(getInput().getType());
    const auto shape = input.getShape();
    const auto reportInputError = [&](const llvm::formatv_object_base& reason) {
        std::ignore = errorAt(getLoc(), "{0}", reason.str());
    };
    if (!isSupportedInput(input, reportInputError)) {
        return mlir::failure();
    }
    const auto output = mlir::cast<NDTypeInterface>(getOutput().getType());
    const Shape expectedOutput{1, shape[Dims4D::Act::C], 1, 1};
    if (output.getShape() != expectedOutput || output.getDimsOrder() != DimsOrder::NCHW) {
        return errorAt(getLoc(), "DMA Mean requires static NCHW [1,C,1,1] output");
    }
    const auto auxiliary = mlir::cast<NDTypeInterface>(getAuxBuffer().getType());
    if (!auxiliary.getElementType().isUnsignedInteger(8) || auxiliary.getNumElements() < minWorkspaceBytes) {
        return errorAt(getLoc(), "DMA Mean requires an adequate unsigned byte workspace");
    }
    if (getMultiClusterStrategy().has_value()) {
        return errorAt(getLoc(), "DMA Mean runs on a single tile and takes no multi-cluster strategy");
    }
    return mlir::success();
}

mlir::LogicalResult VPU::ReduceMeanDMAOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                            mlir::ReifiedRankedShapedTypeDims& shapes) {
    shapes.emplace_back(reifyTrivialTensor(builder, getOutput(), getLoc()));
    return mlir::success();
}
