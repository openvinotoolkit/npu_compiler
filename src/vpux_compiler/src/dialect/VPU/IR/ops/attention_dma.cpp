//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
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

    const auto inQType = mlir::cast<vpux::NDTypeInterface>(attention.getInputQ().getType());
    const auto inQShape = inQType.getShape().raw();
    const auto rank = inQType.getShape().size();

    const auto inVType = mlir::cast<vpux::NDTypeInterface>(attention.getInputV().getType());
    const auto inVShape = inVType.getShape().raw();

    // Detect V transposition by comparing K and V sequence-length dimensions.
    // For dynamic shapes (BoundedTensorType), use upper bounds for the comparison.
    const auto kBounds = getBoundedShape(attention.getInputK());
    const auto vBounds = getBoundedShape(attention.getInputV());
    const auto isTransposedV = kBounds[Dim(rank - 2)] != vBounds[Dim(rank - 2)];
    const auto Ev = isTransposedV ? inVShape[rank - 2] : inVShape[rank - 1];
    SmallVector<int64_t> outShape(inQShape.begin(), inQShape.end());
    outShape[rank - 1] = Ev;

    // Propagate bounds to the output when it has dynamic dimensions.
    const bool outIsDynamic = llvm::any_of(outShape, [](int64_t d) {
        return d == mlir::ShapedType::kDynamic;
    });
    mlir::Type outputType;
    if (outIsDynamic) {
        const auto qBounds = getBoundedShape(attention.getInputQ());
        SmallVector<int64_t> outBounds(qBounds.begin(), qBounds.end());
        outBounds[rank - 1] = isTransposedV ? vBounds[Dim(rank - 2)] : vBounds[Dim(rank - 1)];
        auto ranked = mlir::RankedTensorType::get(outShape, inQType.getElementType());
        outputType = Core::BoundedTensorType::get(ranked, BoundsRef(outBounds));
    } else {
        outputType = mlir::RankedTensorType::get(outShape, inQType.getElementType());
    }
    inferredReturnTypes.push_back(outputType);

    return mlir::success();
}

mlir::LogicalResult vpux::VPU::AttentionDMAOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    const auto loc = getLoc();
    const auto inputQ = getInputQ();
    const auto inputV = getInputV();
    const auto qType = mlir::cast<mlir::RankedTensorType>(inputQ.getType());
    const auto rank = qType.getRank();

    // Output shape = Q dims [0..rank-2] + V last dim
    SmallVector<mlir::OpFoldResult> dims;
    dims.reserve(rank);
    for (int64_t i = 0; i < rank - 1; ++i) {
        dims.push_back(reifyDim(builder, inputQ, qType, i, loc));
    }
    const auto vType = mlir::cast<mlir::RankedTensorType>(inputV.getType());
    dims.push_back(reifyDim(builder, inputV, vType, rank - 1, loc));

    reifiedReturnShapes.emplace_back(std::move(dims));
    return mlir::success();
}

void vpux::VPU::AttentionDMAOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                      ::mlir::Value inputQ, ::mlir::Value inputK, ::mlir::Value inputV,
                                      ::mlir::Value inputMask, ::mlir::Value inputScale, ::mlir::Value inputSink,
                                      ::mlir::Value inputBias, ::mlir::Value seqLenK, ::mlir::IntegerAttr padSizeS) {
    auto loc = odsState.location;
    auto module = getModuleOp(odsBuilder);
    auto auxBufferType = getAuxiliaryBufferType(module);
    auto auxBuffer = VPU::createEmptyAuxiliaryBuffer(odsBuilder, loc, auxBufferType);

    build(odsBuilder, odsState, inputQ, inputK, inputV, inputMask, inputScale, inputSink, inputBias, auxBuffer, seqLenK,
          padSizeS, nullptr);
}

//
// SWOpInterface
//

///
/// fitsForNonFlashKernel
///
/// The function calculates memory requirements for:
/// - Input tensors: Q [1,1,1,e], K [1,1,sSL,e], V [1,1,eV,sSL]
/// - Optional inputs: mask [1,1,1,sSL], scale [1,1,1,1], bias [1,1,1,sSL], seqLenK [1,1,1,1]
/// - DPU storage buffer * numShaves
/// - DMA aux buffer
/// - Output tensor [1,1,1,eV]
///
bool vpux::VPU::AttentionDMAOp::fitsForNonFlashKernel(VPU::AttentionDMAOp origOp) {
    auto module = origOp->getParentOfType<mlir::ModuleOp>();
    if (!module) {
        return false;
    }

    const auto qType = mlir::cast<NDTypeInterface>(origOp.getInputQ().getType());
    const auto kType = mlir::cast<NDTypeInterface>(origOp.getInputK().getType());
    const auto vType = mlir::cast<NDTypeInterface>(origOp.getInputV().getType());

    const auto qShape = getBoundedShape(qType).raw();
    const auto kShape = getBoundedShape(kType).raw();
    const auto vShape = getBoundedShape(vType).raw();
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
