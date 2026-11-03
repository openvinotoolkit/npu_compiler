//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/utils/attention_dma_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

mlir::Type calculateDpuStorageType(mlir::ModuleOp moduleOp, mlir::Value inputQ, int64_t dimS, int64_t dimEv,
                                   int64_t dimE, int64_t dimL, mlir::IntegerAttr padSizeAttr,
                                   std::vector<int32_t>& resultDpuStorageData);

namespace {

mlir::Type getAuxiliaryBufferType(mlir::ModuleOp module) {
    const auto auxBytes = VPU::getTotalCMXFragmentationAwareSize(module).count();
    return mlir::RankedTensorType::get({1, 1, 1, auxBytes}, getUInt8Type(module.getContext()));
}

}  // namespace

mlir::LogicalResult vpux::VPU::AttentionDMAOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::AttentionDMAOpAdaptor attention(operands, attrs, prop);
    if (mlir::failed(attention.verify(loc))) {
        return mlir::failure();
    }

    SmallVector<int64_t> outShape;
    SmallVector<int64_t> outBounds;
    mlir::Type outElementType;
    inferAttentionOutputShapeAndBounds(attention.getInputQ(), attention.getInputK(), attention.getInputV(), outShape,
                                       outBounds, outElementType);

    const auto rankedType = mlir::RankedTensorType::get(outShape, outElementType);
    inferredReturnTypes.push_back(outBounds.empty() ? mlir::Type(rankedType)
                                                    : Core::BoundedTensorType::get(rankedType, BoundsRef(outBounds)));

    // presentK/presentV keep the exact type of pastK/pastV:
    // the kernel writes the new entries back into the past cache layout.
    if (attention.getPastK() != nullptr && attention.getPastV() != nullptr) {
        inferredReturnTypes.push_back(attention.getPastK().getType());
        inferredReturnTypes.push_back(attention.getPastV().getType());
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::AttentionDMAOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    if (mlir::failed(reifyAttentionResultShape(builder, getInputQ(), getInputK(), getInputV(), getLoc(),
                                               reifiedReturnShapes))) {
        return mlir::failure();
    }

    if (getPresentK() != nullptr && getPresentV() != nullptr) {
        reifiedReturnShapes.emplace_back(reifyTrivialTensor(builder, getPastK(), getLoc()));
        reifiedReturnShapes.emplace_back(reifyTrivialTensor(builder, getPastV(), getLoc()));
    }

    return mlir::success();
}

void vpux::VPU::AttentionDMAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                      ::mlir::Value inputQ, ::mlir::Value inputK, ::mlir::Value inputV,
                                      ::mlir::Value pastK, ::mlir::Value pastV, ::mlir::Value inputMask,
                                      ::mlir::Value inputScale, ::mlir::Value inputSink, ::mlir::Value inputBias,
                                      ::mlir::Value seqLenK, ::mlir::Value kScale, ::mlir::Value vScale,
                                      ::mlir::IntegerAttr padSizeS, ::mlir::IntegerAttr localWindowSize,
                                      ::mlir::UnitAttr smoothSoftmax, ::mlir::UnitAttr slidingWindowCache,
                                      ::mlir::UnitAttr vTransposed) {
    auto loc = odsState.location;
    auto module = getModuleOp(odsBuilder);
    auto auxBufferType = getAuxiliaryBufferType(module);
    auto auxBuffer = VPU::createEmptyAuxiliaryBuffer(odsBuilder, loc, auxBufferType);

    build(odsBuilder, odsState, inputQ, inputK, inputV, pastK, pastV, inputMask, inputScale, inputSink, inputBias,
          auxBuffer, seqLenK, kScale, vScale, padSizeS, localWindowSize, smoothSoftmax, slidingWindowCache, vTransposed,
          nullptr);
}

//
// SWOpInterface
//

///
/// fitsAndLegalForNonFlashKernel
///
/// The non-flash attention_dma kernel does not support a KV cache, quantized (si8) K/V, or a sink
/// (only attention_dma_flash does); those are rejected outright. It also requires {eDim, eVDim, sSL}
/// to be 16-aligned. The remaining checks calculate memory requirements for:
/// - Input tensors: Q [1,1,1,e], K [1,1,sSL,e], V [1,1,eV,sSL]
/// - Optional inputs: mask [1,1,1,sSL], scale [1,1,1,1], bias [1,1,1,sSL], seqLenK [1,1,1,1]
/// - DPU storage buffer * numShaves
/// - DMA aux buffer
/// - Output tensor [1,1,1,eV]
///
/// "Legal" additionally requires the op's raw {qHeadSize, tSL, sSL, eDim, eVDim} to match one of
/// the static configs in isLegalNonFlashAttentionDMAConfig -- the same configs IE::FuseAttentionPass
/// uses to decide AttentionOp -> AttentionDMAOp conversion without relying on a dynamic
/// seqLenK/KV-cache. An op that only became AttentionDMAOp via that dynamic path will not match and
/// must run the flash kernel instead, regardless of whether its buffers would fit CMX.
///
bool vpux::VPU::AttentionDMAOp::fitsAndLegalForNonFlashKernel(VPU::AttentionDMAOp origOp) {
    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    if (!module) {
        return false;
    }

    if (origOp.getPastK() != nullptr && origOp.getPastV() != nullptr) {
        return false;
    }
    if (origOp.getInputSink() != nullptr) {
        return false;
    }
    // With a KV cache the quantized tensors would be pastK/pastV, not inputK/inputV, but a cache is
    // already rejected above, so inputK/inputV are the only possible quantized tensors here.
    const auto kElemType = mlir::cast<NDTypeInterface>(origOp.getInputK().getType()).getElementType();
    const auto vElemType = mlir::cast<NDTypeInterface>(origOp.getInputV().getType()).getElementType();
    if (kElemType.isSignedInteger(8) || vElemType.isSignedInteger(8) || origOp.getKScale() != nullptr ||
        origOp.getVScale() != nullptr) {
        return false;
    }

    const auto qType = mlir::cast<NDTypeInterface>(origOp.getInputQ().getType());
    const auto kType = mlir::cast<NDTypeInterface>(origOp.getInputK().getType());
    const auto vType = mlir::cast<NDTypeInterface>(origOp.getInputV().getType());

    if (hasUnboundedDynamicShape(qType) || hasUnboundedDynamicShape(kType) || hasUnboundedDynamicShape(vType)) {
        return false;
    }

    // A (bounded) dynamic sequence length has no static {qHeadSize, tSL, sSL, eDim, eVDim} entry to
    // match in isLegalNonFlashAttentionDMAConfig below, and only attention_dma (not the flash
    // kernel) handles it correctly -- always route it to the non-flash kernel.
    const auto hasDynamicShape = [](NDTypeInterface type) {
        return llvm::is_contained(type.getShape().raw(), mlir::ShapedType::kDynamic);
    };
    if (hasDynamicShape(qType) || hasDynamicShape(kType) || hasDynamicShape(vType)) {
        return true;
    }

    const auto qShape = getBoundedShape(qType).raw();
    const auto kShape = getBoundedShape(kType).raw();
    const auto vShape = getBoundedShape(vType).raw();
    const auto rank = qType.getRank();

    // V may be pre-transposed to [.., eV, sSL] (the layout FuseAttention inserts for the legal
    // static shapes) or not; pick eVDim accordingly.
    const auto isTransposedV = kShape[rank - 2] != vShape[rank - 2];
    const auto qHeadSize = (rank == 3) ? qShape[0] : qShape[0] * qShape[1];
    const auto tSL = qShape[rank - 2];
    const auto sSLRaw = kShape[rank - 2];
    const auto eDimRaw = kShape[rank - 1];
    const auto eVDimRaw = isTransposedV ? vShape[rank - 2] : vShape[rank - 1];
    if (!isLegalNonFlashAttentionDMAConfig(qHeadSize, tSL, sSLRaw, eDimRaw, eVDimRaw)) {
        return false;
    }
    if (eDimRaw % 16 != 0 || eVDimRaw % 16 != 0 || sSLRaw % 16 != 0) {
        return false;
    }

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
    if (origOp.getInputBias()) {
        buffersSize.push_back(Byte(sSL * fp16Size));
    }
    if (origOp.getSeqLenK()) {
        buffersSize.push_back(Byte(1 * sizeof(int32_t)));
    }

    int64_t numShaves = (static_cast<VPU::SwIoDmaOpInterface>(origOp)).getDuplicatedNumShaves();
    std::vector<int32_t> dpuStorageData;
    auto dpuStorageType = calculateDpuStorageType(module, origOp.getInputQ(), sSL, eV, e, qShape[rank - 2],
                                                  origOp.getPadSizeSAttr(), dpuStorageData);
    if (auto dpuStorageNDType = mlir::dyn_cast<NDTypeInterface>(dpuStorageType)) {
        buffersSize.push_back(dpuStorageNDType.getTotalAllocSize() * numShaves);
    }

    // Minimum global allocation for DMA is 10KB, which is used for DMA transfer of auxiliary buffer.
    constexpr uint32_t kGlobalAllocSize = 10 * 1024;  // 10KB
    buffersSize.push_back(Byte(kGlobalAllocSize));

    const auto totalSize = vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(module), buffersSize);
    return totalSize <= getTotalCMXSize(module);
}
