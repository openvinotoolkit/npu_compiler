//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {
namespace IE {

int64_t getExpandedCMXUsagePerGroup(IE::MatMulOp matmulOp, ShapeRef input1Shape, ShapeRef input2Shape) {
    VPUX_THROW_UNLESS(input1Shape.size() == 3 && input2Shape.size() == 3,
                      "Matmul Dimensions for batched Matmul must be 3d");
    const auto transposeA = matmulOp.getTransposeA();
    const auto transposeB = matmulOp.getTransposeB();
    const auto dimOfIC = transposeA ? Dims3D::Act::H : Dims3D::Act::IC;
    const auto dimOfOC = transposeB ? Dims3D::Filter::IC : Dims3D::Filter::OC;
    const int64_t sizeofIC{input1Shape[dimOfIC]};
    const int64_t sizeofOC{input2Shape[dimOfOC]};
    // Following expansion is done at expandActivationChannelsPass, example assuming no transpose A/B
    // We calculate CMX size following expansion and assumed tiling
    // WE mean W Expanded
    // Input1 Input2 Output  -> After Expansion Input1  Input2   Output
    // BxHxIC  BxICxOC BxHxOC ------------------- BxHxICE  BxICExOCE BxHxOCE

    // Here we preemptively estimate CMX usage in VPU after tiling, if data does not fit
    // into CMX we dont run with batched matmul
    // OC dimension must be expanded to be multiple of 16

    const auto input1ElementType = mlir::cast<vpux::NDTypeInterface>(matmulOp.getInput1().getType()).getElementType();
    const auto outputElementType = mlir::cast<vpux::NDTypeInterface>(matmulOp.getOutput().getType()).getElementType();
    const int64_t inputChannelAlignment = VPU::NCEInvariant::getAlignment(input1ElementType);
    const int64_t outputChannelAlignment = VPU::NCEInvariant::getAlignment(outputElementType);
    constexpr auto float16Size = sizeof(type::float16);
    constexpr auto int32Size = sizeof(int32_t);
    const auto sizeOfICE = alignValUp(sizeofIC, inputChannelAlignment);
    const auto sizeOfOCE = alignValUp(sizeofOC, outputChannelAlignment);
    const auto input1Size = vpux::details::calcTotalShapeSize(input1Shape) * float16Size * sizeOfICE / sizeofIC /
                            input1Shape[Dims3D::Act::B];
    // To cover transpose case without conditional we multiply all then remove expanded dimension
    const auto input2Size = sizeOfICE * sizeOfOCE * float16Size;

    const auto sizeH = transposeA ? input1Shape[Dim(2)] : input1Shape[Dim(1)];
    const auto outputSize = sizeH * sizeOfOCE * float16Size;

    const auto weightTableSize = sizeOfOCE * VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * int32Size;

    // Calculation is not totally accurate, we are not considering if input2 is being duplicated instead of being tiled
    return input1Size + input2Size + outputSize + weightTableSize;
}

bool isGroupBiggerThanTileCount(IE::MatMulOp matmulOp, ShapeRef inputShape) {
    const auto module = getModuleOp(matmulOp);
    auto tileOp = config::getTileExecutor(module);
    const auto numOfTiles = tileOp.getCount();
    auto batchSize = inputShape.size() == 3 ? inputShape[Dims3D::Act::B]
                                            : inputShape[Dims4D::Act::C] * inputShape[Dims4D::Act::N];
    // E-153097:
    // If the batch size is equal to the number of tiles we can at least fit it into CMX
    return numOfTiles == 3 ? batchSize > numOfTiles : batchSize >= numOfTiles;
}

bool isMatmulSoftmaxMatmulAndNotAligned(IE::MatMulOp matmulOp) {
    // If Matmul->Softmax-Matmul pattern, if the first Matmul is beneficial to unroll, the second Matmul is not
    // beneficial and it's not aligned. It will introduce extra slice and expand for alignment. This may be inefficient,
    // to avoid this, such matmuls will be unrolled and wont use grouped matmul optimization.
    auto softmaxOp = matmulOp.getInput1().getDefiningOp<IE::SoftMaxOp>();
    if (softmaxOp == nullptr) {
        softmaxOp = matmulOp.getInput2().getDefiningOp<IE::SoftMaxOp>();
    }
    if (softmaxOp == nullptr) {
        return false;
    }

    auto producerOp = softmaxOp.getInput().getDefiningOp();
    IE::ConcatOp concatOp(nullptr);
    while (mlir::isa_and_nonnull<IE::ViewLikeOpInterface>(producerOp) && producerOp->getResult(0).hasOneUse()) {
        producerOp = producerOp->getOperand(0).getDefiningOp();

        if (mlir::isa_and_nonnull<IE::ConcatOp>(producerOp)) {
            concatOp = mlir::dyn_cast<IE::ConcatOp>(*producerOp);
            break;
        }
    }
    if (concatOp == nullptr) {
        return false;
    }
    auto allInputsAreMatMul = llvm::all_of(concatOp->getOperands(), [](mlir::Value operand) {
        return mlir::isa<IE::MatMulOp>(operand.getDefiningOp());
    });

    // check matmulOp channel alignment.
    const auto input2Type = mlir::cast<vpux::NDTypeInterface>(matmulOp.getInput2().getType());
    auto input2Dims = input2Type.getShape().toValues();
    auto input2Rank = input2Type.getRank();

    auto inputChannelDim = input2Dims[matmulOp.getTransposeB() ? Dim(input2Rank - 1) : Dim(input2Rank - 2)];
    auto outputChannelDim = input2Dims[matmulOp.getTransposeB() ? Dim(input2Rank - 2) : Dim(input2Rank - 1)];

    bool isAligned = (inputChannelDim % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT == 0) &&
                     (outputChannelDim % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT == 0);

    // Small batch size is more efficient to unroll.
    static constexpr int64_t BATCH_SIZE_TO_UNROLL = 10;

    if (!isAligned && input2Dims[Dim(input2Rank - 3)] < BATCH_SIZE_TO_UNROLL && allInputsAreMatMul) {
        return true;
    }

    return false;
}

// Does single group (2D MatMul) fit into CMX
bool isGroupedMatMulBeneficial(IE::MatMulOp matmulOp, ShapeRef input1Shape, ShapeRef input2Shape) {
    const auto availableCMXBytes = vpux::VPU::getTotalCMXSize(matmulOp);
    Shape input1Shape3d =
            input1Shape.size() > 3 ? Shape(input1Shape.begin() + 1, input1Shape.end()) : input1Shape.toValues();
    if (input1Shape.size() > 3) {
        input1Shape3d[Dims3D::Act::B] *= *input1Shape.begin();
    }
    auto input2Shape3d =
            input2Shape.size() > 3 ? Shape(input2Shape.begin() + 1, input2Shape.end()) : input2Shape.toValues();
    if (input2Shape.size() > 3) {
        input2Shape3d[Dims3D::Act::B] *= *input2Shape.begin();
    }

    // Get CMX usage for full operation
    auto expandedCMXUsagePerGroup = IE::getExpandedCMXUsagePerGroup(matmulOp, input1Shape3d, input2Shape3d);

    if (IE::isMatmulSoftmaxMatmulAndNotAligned(matmulOp)) {
        return false;
    }

    // Currently NCE.Matmul multicluster strategy is not compatible with other layers so spill will happen, we do not
    // prefer grouped matmul execution when spill is more expensive than grouped matmul execution benefits, expensive
    // spills are normally avoided via VF, so we avoid grouped Matmul when output tensors are large. With support
    // of SOG for more layers and VF support we can disable these limitations. (#E154850)
    const auto outputType = mlir::cast<NDTypeInterface>(matmulOp.getOutput().getType());
    constexpr int64_t float16Size = sizeof(type::float16);
    const auto outputSize = vpux::details::calcTotalShapeSize(outputType.getShape()) * float16Size;
    constexpr int64_t perGroupOutputSizeLimit = 200'000;
    const auto smallOutputSize = (outputSize / input1Shape3d[Dims3D::Act::B]) < perGroupOutputSizeLimit;

    const double safetyFactor = 0.9;
    const auto fitIntoCMX =
            input1Shape3d[Dims3D::Act::B] == 1 || expandedCMXUsagePerGroup < availableCMXBytes.count() * safetyFactor;

    const auto groupBiggerThanTiles = isGroupBiggerThanTileCount(matmulOp, input1Shape);

    return fitIntoCMX && smallOutputSize && groupBiggerThanTiles;
}

bool isMatmulWithRHSTransposition(IE::MatMulOp matmulOp) {
    return !matmulOp.getTransposeA() && matmulOp.getTransposeB();
}

}  // namespace IE
}  // namespace vpux
