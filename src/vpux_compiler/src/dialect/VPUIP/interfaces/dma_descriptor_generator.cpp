//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/interfaces/dma_descriptor_generator.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dma_transaction_utils.hpp"

using namespace vpux;

vpux::VPUIP::PermuteDmaDescriptorGenerator::PermuteDmaDescriptorGenerator(mlir::MLIRContext* ctx,
                                                                          mlir::AffineMap mergedMemPerm,
                                                                          vpux::Logger log)
        : _ctx(ctx), _mergedMemPerm(mergedMemPerm), _log(log) {
}

VPUIP::DMADescriptorAttr vpux::VPUIP::PermuteDmaDescriptorGenerator::generate(ShapeRef mergedInputShape,
                                                                              ShapeRef mergedOutputShape,
                                                                              Byte elemTypeSize) const {
    VPUX_THROW_UNLESS(_mergedMemPerm.getNumResults() == 2 || _mergedMemPerm.getNumResults() == 3,
                      "Invalid merged mem perm {0}", _mergedMemPerm);
    if (mergedInputShape.size() == 2) {
        return generateWithTwoAxis(mergedInputShape, mergedOutputShape, elemTypeSize);
    } else if (_mergedMemPerm == mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{1, 0, 2}, _ctx)) {
        return generateWithSwapFront(mergedInputShape, elemTypeSize);
    } else if (_mergedMemPerm == mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{0, 2, 1}, _ctx)) {
        return generateWithSwapBack(mergedInputShape, elemTypeSize);
    } else {
        VPUX_THROW("Unsupported merged mem perm {0} ", _mergedMemPerm);
    }
}

SmallVector<VPUIP::DMADescriptorAttr> vpux::VPUIP::PermuteDmaDescriptorGenerator::generate(
        ShapeRef mergedInputShape, ShapeRef mergedOutputShape, ArrayRef<Shape> mergedSubOutputShapes, Dim tileDim,
        Byte elemTypeSize) const {
    VPUX_THROW_UNLESS(_mergedMemPerm.getNumResults() == 2, "Invalid merged mem perm {0}", _mergedMemPerm);
    return generateWithTwoAxis(mergedInputShape, mergedOutputShape, mergedSubOutputShapes, tileDim, elemTypeSize);
}

VPUIP::DMADescriptorAttr vpux::VPUIP::PermuteDmaDescriptorGenerator::generateWithTwoAxis(ShapeRef mergedInputShape,
                                                                                         ShapeRef mergedOutputShape,
                                                                                         Byte elemSize) const {
    // Permute pattern [x y] #NC -> [y z] #NC, note that expand op may be fused into permute op, so x may not equal
    // to z in that case
    VPUX_THROW_UNLESS(mergedInputShape.size() == 2, "The size of merged input shape {0} is not equal to 2",
                      mergedInputShape.size());
    VPUX_THROW_UNLESS(mergedOutputShape.size() == 2, "The size of merged output shape {0} is not equal to 2",
                      mergedOutputShape.size());
    VPUX_THROW_UNLESS(elemSize.count() > 0, "Invalid element size {0}", elemSize);
    auto IN = mergedInputShape.front();
    auto IC = mergedInputShape.back();
    auto OC = mergedOutputShape.back();
    auto elemTypeSize = elemSize.count();

    auto numPlane = vpux::getIntAttr(_ctx, IN);
    auto len = vpux::getIntAttr(_ctx, IC * elemTypeSize);
    auto srcWidth = vpux::getIntAttr(_ctx, IC * elemTypeSize);
    auto srcStride = vpux::getIntAttr(_ctx, elemTypeSize);
    auto srcPlaneStride = vpux::getIntAttr(_ctx, IC * elemTypeSize);
    auto dstWidth = vpux::getIntAttr(_ctx, elemTypeSize);
    auto dstStride = vpux::getIntAttr(_ctx, OC * elemTypeSize);
    auto dstPlaneStride = vpux::getIntAttr(_ctx, elemTypeSize);
    return VPUIP::DMADescriptorAttr::get(_ctx, numPlane, len, srcWidth, srcStride, srcPlaneStride, dstWidth, dstStride,
                                         dstPlaneStride);
}

SmallVector<VPUIP::DMADescriptorAttr> vpux::VPUIP::PermuteDmaDescriptorGenerator::generateWithTwoAxis(
        ShapeRef mergedInputShape, ShapeRef mergedOutputShape, ArrayRef<Shape> mergedSubOutputShapes, Dim tileDim,
        Byte elemSize) const {
    VPUX_THROW_UNLESS(_mergedMemPerm.getNumResults() == 2, "Unexpected merged mem perm {0} ", _mergedMemPerm);
    VPUX_THROW_UNLESS(mergedInputShape.size() == 2, "The size of merged input shape {0} is not equal to 2",
                      mergedInputShape.size());
    VPUX_THROW_UNLESS(mergedOutputShape.size() == 2, "The size of merged output shape {0} is not equal to 2",
                      mergedOutputShape.size());
    VPUX_THROW_UNLESS(elemSize.count() > 0, "Invalid element size {0}", elemSize);

    // Permute pattern: #NC -> #CN
    auto elemTypeSize = elemSize.count();
    auto IN = mergedInputShape.front();
    auto IC = mergedInputShape.back();

    auto OC = mergedOutputShape.back();
    SmallVector<VPUIP::DMADescriptorAttr> dmaDescriptorAttrs;

    int64_t usedNumPlaneCount = 0;
    for (auto& mergedSubOutputShape : mergedSubOutputShapes) {
        VPUX_THROW_UNLESS(mergedSubOutputShape.size() == 2, "The size of merged sub output shape {0} is not equal to 2",
                          mergedSubOutputShape);
        auto subON = mergedSubOutputShape.front();
        auto subOC = mergedSubOutputShape.back();

        auto srcStride = vpux::getIntAttr(_ctx, elemTypeSize);
        auto srcPlaneStride = vpux::getIntAttr(_ctx, IC * elemTypeSize);
        auto dstWidth = vpux::getIntAttr(_ctx, elemTypeSize);
        auto dstPlaneStride = vpux::getIntAttr(_ctx, elemTypeSize);

        auto dimIdx = tileDim.ind();
        if (dimIdx == 0) {
            auto numPlane = vpux::getIntAttr(_ctx, IN);
            auto len = vpux::getIntAttr(_ctx, subON * elemTypeSize);
            auto srcWidth = vpux::getIntAttr(_ctx, subON * elemTypeSize);
            auto dstStride = vpux::getIntAttr(_ctx, OC * elemTypeSize);
            dmaDescriptorAttrs.push_back(VPUIP::DMADescriptorAttr::get(
                    _ctx, numPlane, len, srcWidth, srcStride, srcPlaneStride, dstWidth, dstStride, dstPlaneStride));
        } else {
            auto numPlane = vpux::getIntAttr(_ctx, std::min(IN - usedNumPlaneCount, subOC));
            auto len = vpux::getIntAttr(_ctx, IC * elemTypeSize);
            auto srcWidth = vpux::getIntAttr(_ctx, IC * elemTypeSize);
            auto dstStride = vpux::getIntAttr(_ctx, subOC * elemTypeSize);
            dmaDescriptorAttrs.push_back(VPUIP::DMADescriptorAttr::get(
                    _ctx, numPlane, len, srcWidth, srcStride, srcPlaneStride, dstWidth, dstStride, dstPlaneStride));
            usedNumPlaneCount += numPlane.getInt();
        }
    }
    return dmaDescriptorAttrs;
}

VPUIP::DMADescriptorAttr vpux::VPUIP::PermuteDmaDescriptorGenerator::generateWithSwapFront(ShapeRef mergedInputShape,
                                                                                           Byte elemSize) const {
    VPUX_THROW_UNLESS(mergedInputShape.size() == 3, "The size of merged input shape {0} is not equal to 3",
                      mergedInputShape.size());
    VPUX_THROW_UNLESS(elemSize.count() > 0, "Invalid element size {0}", elemSize);
    // Permute pattern #HWC ->  #WHC
    auto elemTypeSize = elemSize.count();
    auto H = mergedInputShape[Dim(0)];
    auto W = mergedInputShape[Dim(1)];
    auto C = mergedInputShape[Dim(2)];

    auto numPlane = vpux::getIntAttr(_ctx, H);
    auto len = vpux::getIntAttr(_ctx, W * C * elemTypeSize);
    auto srcWidth = vpux::getIntAttr(_ctx, W * C * elemTypeSize);
    auto srcStride = vpux::getIntAttr(_ctx, elemTypeSize);
    auto srcPlaneStride = vpux::getIntAttr(_ctx, W * C * elemTypeSize);
    auto dstWidth = vpux::getIntAttr(_ctx, C * elemTypeSize);
    auto dstStride = vpux::getIntAttr(_ctx, H * C * elemTypeSize);
    auto dstPlaneStride = vpux::getIntAttr(_ctx, C * elemTypeSize);
    return VPUIP::DMADescriptorAttr::get(_ctx, numPlane, len, srcWidth, srcStride, srcPlaneStride, dstWidth, dstStride,
                                         dstPlaneStride);
}

VPUIP::DMADescriptorAttr vpux::VPUIP::PermuteDmaDescriptorGenerator::generateWithSwapBack(ShapeRef mergedInputShape,
                                                                                          Byte elemSize) const {
    // Permute pattern #HWC ->  #HCW
    VPUX_THROW_UNLESS(mergedInputShape.size() == 3, "The size of merged input shape {0} is not equal to 2",
                      mergedInputShape.size());
    VPUX_THROW_UNLESS(elemSize.count() > 0, "Invalid element size {0}", elemSize);
    auto elemTypeSize = elemSize.count();
    auto H = mergedInputShape[Dim(0)];
    auto W = mergedInputShape[Dim(1)];
    auto C = mergedInputShape[Dim(2)];

    auto numPlane = vpux::getIntAttr(_ctx, W);
    auto len = vpux::getIntAttr(_ctx, H * C * elemTypeSize);
    auto srcWidth = vpux::getIntAttr(_ctx, C * elemTypeSize);
    auto srcStride = vpux::getIntAttr(_ctx, C * W * elemTypeSize);
    auto srcPlaneStride = vpux::getIntAttr(_ctx, C * elemTypeSize);
    auto dstWidth = vpux::getIntAttr(_ctx, elemTypeSize);
    auto dstStride = vpux::getIntAttr(_ctx, W * elemTypeSize);
    auto dstPlaneStride = vpux::getIntAttr(_ctx, elemTypeSize);
    return VPUIP::DMADescriptorAttr::get(_ctx, numPlane, len, srcWidth, srcStride, srcPlaneStride, dstWidth, dstStride,
                                         dstPlaneStride);
}

vpux::VPUIP::DepthToSpaceDmaDescriptorGenerator::DepthToSpaceDmaDescriptorGenerator(mlir::MLIRContext* ctx,
                                                                                    vpux::Logger log)
        : _ctx(ctx), _log(log) {
}

VPUIP::DMADescriptorAttr vpux::VPUIP::DepthToSpaceDmaDescriptorGenerator::generate(
        vpux::NDTypeInterface inType, vpux::NDTypeInterface outType, ShapeRef inShape, ShapeRef outShape,
        vpux::IE::DepthToSpaceMode mode, int64_t blockSize, mlir::IntegerAttr paddedIC,
        mlir::IntegerAttr paddedOC) const {
    const auto inOrder = inType.getDimsOrder();
    const auto outOrder = outType.getDimsOrder();
    auto isLegalType = (inOrder == DimsOrder::NHWC && outOrder == DimsOrder::NHWC);
    VPUX_THROW_UNLESS(isLegalType, "Unsupported layout: input={0} and output={1}.", inOrder, outOrder);

    const auto elemTypeByteSize = Byte(inType.getElemTypeSize()).count();

    const auto inputH = inShape[Dims4D::Act::H];
    const auto inputW = inShape[Dims4D::Act::W];
    const auto inputC = inShape[Dims4D::Act::C];
    const auto outputW = outShape[Dims4D::Act::W];
    const auto outputC = outShape[Dims4D::Act::C];

    mlir::IntegerAttr len(0), srcWidth(0), srcStride(0), srcPlaneStride(0);
    mlir::IntegerAttr dstWidth(0), dstStride(0), dstPlaneStride(0), numPlanes(0);

    if (inOrder == DimsOrder::NHWC && mode == IE::DepthToSpaceMode::BLOCKS_FIRST) {
        if (paddedIC != nullptr && paddedOC != nullptr) {
            // new descriptor for padded case D2S
            int64_t padOutChannels = paddedOC.getInt();

            numPlanes = vpux::getIntAttr(_ctx, inputH);
            len = vpux::getIntAttr(_ctx, inputW * blockSize * (outputC - padOutChannels) * elemTypeByteSize);
            srcWidth = vpux::getIntAttr(_ctx, (outputC - padOutChannels) * blockSize * elemTypeByteSize);
            srcStride = vpux::getIntAttr(_ctx, inputC * elemTypeByteSize);
            srcPlaneStride = vpux::getIntAttr(_ctx, inputW * inputC * elemTypeByteSize);
            dstWidth = vpux::getIntAttr(_ctx, (outputC - padOutChannels) * elemTypeByteSize);
            dstStride = vpux::getIntAttr(_ctx, outputC * elemTypeByteSize);
            dstPlaneStride = vpux::getIntAttr(_ctx, outputC * outputW * blockSize * elemTypeByteSize);
        } else {
            numPlanes = vpux::getIntAttr(_ctx, inputH);
            len = vpux::getIntAttr(_ctx, outputC * outputW * elemTypeByteSize);
            srcWidth = vpux::getIntAttr(_ctx, outputC * blockSize * elemTypeByteSize);
            srcStride = vpux::getIntAttr(_ctx, outputC * blockSize * blockSize * elemTypeByteSize);
            srcPlaneStride = vpux::getIntAttr(_ctx, outputC * outputW * blockSize * elemTypeByteSize);
            dstWidth = vpux::getIntAttr(_ctx, outputC * outputW * elemTypeByteSize);
            dstStride = vpux::getIntAttr(_ctx, elemTypeByteSize);
            dstPlaneStride = vpux::getIntAttr(_ctx, outputC * outputW * blockSize * elemTypeByteSize);
        }

    } else if (inOrder == DimsOrder::NHWC && mode == IE::DepthToSpaceMode::DEPTH_FIRST) {
        numPlanes = vpux::getIntAttr(_ctx, inputH);
        len = vpux::getIntAttr(_ctx, outputW * elemTypeByteSize);
        srcWidth = vpux::getIntAttr(_ctx, blockSize * elemTypeByteSize);
        srcStride = vpux::getIntAttr(_ctx, outputC * blockSize * blockSize * elemTypeByteSize);
        srcPlaneStride = vpux::getIntAttr(_ctx, outputC * outputW * blockSize * elemTypeByteSize);
        dstWidth = vpux::getIntAttr(_ctx, elemTypeByteSize);
        dstStride = vpux::getIntAttr(_ctx, outputC * elemTypeByteSize);
        dstPlaneStride = vpux::getIntAttr(_ctx, outputC * outputW * blockSize * elemTypeByteSize);
    } else {
        VPUX_THROW("Unsupported order {0} and mode {1} for DepthToSpaceDMA op", inOrder, mode);
    }

    return VPUIP::DMADescriptorAttr::get(_ctx, numPlanes, len, srcWidth, srcStride, srcPlaneStride, dstWidth, dstStride,
                                         dstPlaneStride);
}

vpux::VPUIP::SpaceToDepthDmaDescriptorGenerator::SpaceToDepthDmaDescriptorGenerator(mlir::MLIRContext* ctx,
                                                                                    vpux::Logger log)
        : _ctx(ctx), _log(log) {
}

VPUIP::DMADescriptorAttr vpux::VPUIP::SpaceToDepthDmaDescriptorGenerator::generate(vpux::NDTypeInterface inType,
                                                                                   vpux::NDTypeInterface outType,
                                                                                   vpux::IE::SpaceToDepthMode mode,
                                                                                   int64_t blockSize) const {
    const auto inOrder = inType.getDimsOrder();
    const auto outOrder = outType.getDimsOrder();
    auto isLegalType = (inOrder == DimsOrder::NHWC && outOrder == DimsOrder::NHWC) ||
                       (inOrder == DimsOrder::NCHW && outOrder == DimsOrder::NCHW) ||
                       (inOrder == DimsOrder::NCHW && outOrder == DimsOrder::NHWC);
    VPUX_THROW_UNLESS(isLegalType, "Unsupported layout: input={0} and output={1}.", inOrder, outOrder);

    const auto elemTypeSize = Byte(inType.getElemTypeSize()).count();
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();

    if (inOrder == DimsOrder::NCHW && outOrder == DimsOrder::NCHW && mode == IE::SpaceToDepthMode::BLOCKS_FIRST) {
        return generateBlocksFirstNCHW2NCHW(inShape, outShape, elemTypeSize, blockSize);
    }

    if (inOrder == DimsOrder::NCHW && outOrder == DimsOrder::NCHW && mode == IE::SpaceToDepthMode::DEPTH_FIRST) {
        return generateDepthFirstNCHW2NCHW(inShape, outShape, elemTypeSize, blockSize);
    }

    if (inOrder == DimsOrder::NHWC && outOrder == DimsOrder::NHWC && mode == IE::SpaceToDepthMode::BLOCKS_FIRST) {
        return generateBlocksFirstNHWC2NHWC(inShape, outShape, elemTypeSize, blockSize);
    }

    if (inOrder == DimsOrder::NHWC && outOrder == DimsOrder::NHWC && mode == IE::SpaceToDepthMode::DEPTH_FIRST) {
        return generateDepthFirstNHWC2NHWC(inShape, outShape, elemTypeSize, blockSize);
    }

    if (inOrder == DimsOrder::NCHW && outOrder == DimsOrder::NHWC && mode == IE::SpaceToDepthMode::BLOCKS_FIRST) {
        return generateBlocksFirstNCHW2NHWC(inShape, outShape, elemTypeSize, blockSize);
    }

    if (inOrder == DimsOrder::NCHW && outOrder == DimsOrder::NHWC && mode == IE::SpaceToDepthMode::DEPTH_FIRST) {
        return generateDepthFirstNCHW2NHWC(inShape, outShape, elemTypeSize, blockSize);
    }

    VPUX_THROW("SpaceToDepthDMA layout '{0}->{1}' mode {2} is not supported yet.", inOrder, outOrder, mode);

    return nullptr;
}

VPUIP::DMADescriptorAttr vpux::VPUIP::SpaceToDepthDmaDescriptorGenerator::generateBlocksFirstNCHW2NCHW(
        vpux::ShapeRef inShape, vpux::ShapeRef outShape, int64_t elemTypeSize, int64_t blockSize) const {
    const auto IW = inShape[Dims4D::Act::W];
    const auto IC = inShape[Dims4D::Act::C];
    const auto OH = outShape[Dims4D::Act::H];
    const auto OW = outShape[Dims4D::Act::W];
    const auto OC = outShape[Dims4D::Act::C];

    auto len = vpux::getIntAttr(_ctx, IW * elemTypeSize);
    auto srcWidth = vpux::getIntAttr(_ctx, elemTypeSize);
    auto srcStride = vpux::getIntAttr(_ctx, blockSize * elemTypeSize);
    auto srcPlaneStride = vpux::getIntAttr(_ctx, elemTypeSize);
    auto dstWidth = vpux::getIntAttr(_ctx, OW * elemTypeSize);
    auto dstStride = vpux::getIntAttr(_ctx, OH * OW * OC / blockSize * elemTypeSize);
    auto dstPlaneStride = vpux::getIntAttr(_ctx, OH * OW * IC * elemTypeSize);
    auto numPlanes = vpux::getIntAttr(_ctx, blockSize);

    return VPUIP::DMADescriptorAttr::get(_ctx, numPlanes, len, srcWidth, srcStride, srcPlaneStride, dstWidth, dstStride,
                                         dstPlaneStride);
}

VPUIP::DMADescriptorAttr vpux::VPUIP::SpaceToDepthDmaDescriptorGenerator::generateBlocksFirstNHWC2NHWC(
        vpux::ShapeRef inShape, vpux::ShapeRef outShape, int64_t elemTypeSize, int64_t blockSize) const {
    const auto IW = inShape[Dims4D::Act::W];
    const auto IC = inShape[Dims4D::Act::C];
    const auto OH = outShape[Dims4D::Act::H];
    const auto OC = outShape[Dims4D::Act::C];

    auto len = vpux::getIntAttr(_ctx, IC * IW * OH * elemTypeSize);
    auto srcWidth = vpux::getIntAttr(_ctx, IC * IW * elemTypeSize);
    auto srcStride = vpux::getIntAttr(_ctx, IC * IW * blockSize * elemTypeSize);
    auto srcPlaneStride = vpux::getIntAttr(_ctx, IC * IW * elemTypeSize);
    auto dstWidth = vpux::getIntAttr(_ctx, IC * blockSize * elemTypeSize);
    auto dstStride = vpux::getIntAttr(_ctx, OC * elemTypeSize);
    auto dstPlaneStride = vpux::getIntAttr(_ctx, IC * blockSize * elemTypeSize);
    auto numPlanes = vpux::getIntAttr(_ctx, blockSize);

    return VPUIP::DMADescriptorAttr::get(_ctx, numPlanes, len, srcWidth, srcStride, srcPlaneStride, dstWidth, dstStride,
                                         dstPlaneStride);
}

VPUIP::DMADescriptorAttr vpux::VPUIP::SpaceToDepthDmaDescriptorGenerator::generateBlocksFirstNCHW2NHWC(
        vpux::ShapeRef inShape, vpux::ShapeRef outShape, int64_t elemTypeSize, int64_t blockSize) const {
    const auto IW = inShape[Dims4D::Act::W];
    const auto IC = inShape[Dims4D::Act::C];
    const auto OW = outShape[Dims4D::Act::W];
    const auto OC = outShape[Dims4D::Act::C];

    auto len = vpux::getIntAttr(_ctx, blockSize * blockSize * elemTypeSize);
    auto srcWidth = vpux::getIntAttr(_ctx, blockSize * elemTypeSize);
    auto srcStride = vpux::getIntAttr(_ctx, IW * elemTypeSize);
    auto srcPlaneStride = vpux::getIntAttr(_ctx, blockSize * elemTypeSize);
    auto dstWidth = vpux::getIntAttr(_ctx, elemTypeSize);
    auto dstStride = vpux::getIntAttr(_ctx, IC * elemTypeSize);
    auto dstPlaneStride = vpux::getIntAttr(_ctx, OC * elemTypeSize);
    auto numPlanes = vpux::getIntAttr(_ctx, OW);

    return VPUIP::DMADescriptorAttr::get(_ctx, numPlanes, len, srcWidth, srcStride, srcPlaneStride, dstWidth, dstStride,
                                         dstPlaneStride);
}

VPUIP::DMADescriptorAttr vpux::VPUIP::SpaceToDepthDmaDescriptorGenerator::generateDepthFirstNCHW2NCHW(
        vpux::ShapeRef inShape, vpux::ShapeRef outShape, int64_t elemTypeSize, int64_t blockSize) const {
    const auto IW = inShape[Dims4D::Act::W];
    const auto OH = outShape[Dims4D::Act::H];
    const auto OW = outShape[Dims4D::Act::W];

    auto len = vpux::getIntAttr(_ctx, IW * elemTypeSize);
    auto srcWidth = vpux::getIntAttr(_ctx, elemTypeSize);
    auto srcStride = vpux::getIntAttr(_ctx, blockSize * elemTypeSize);
    auto srcPlaneStride = vpux::getIntAttr(_ctx, elemTypeSize);
    auto dstWidth = vpux::getIntAttr(_ctx, OW * elemTypeSize);
    auto dstStride = vpux::getIntAttr(_ctx, OH * OW * blockSize * elemTypeSize);
    auto dstPlaneStride = vpux::getIntAttr(_ctx, OH * OW * elemTypeSize);
    auto numPlanes = vpux::getIntAttr(_ctx, blockSize);

    return VPUIP::DMADescriptorAttr::get(_ctx, numPlanes, len, srcWidth, srcStride, srcPlaneStride, dstWidth, dstStride,
                                         dstPlaneStride);
}

VPUIP::DMADescriptorAttr vpux::VPUIP::SpaceToDepthDmaDescriptorGenerator::generateDepthFirstNHWC2NHWC(
        vpux::ShapeRef inShape, vpux::ShapeRef outShape, int64_t elemTypeSize, int64_t blockSize) const {
    const auto IW = inShape[Dims4D::Act::W];
    const auto IC = inShape[Dims4D::Act::C];
    const auto OC = outShape[Dims4D::Act::C];

    auto len = vpux::getIntAttr(_ctx, IW * elemTypeSize);
    auto srcWidth = vpux::getIntAttr(_ctx, elemTypeSize);
    auto srcStride = vpux::getIntAttr(_ctx, IC * elemTypeSize);
    auto srcPlaneStride = vpux::getIntAttr(_ctx, elemTypeSize);
    auto dstWidth = vpux::getIntAttr(_ctx, blockSize * elemTypeSize);
    auto dstStride = vpux::getIntAttr(_ctx, OC * elemTypeSize);
    auto dstPlaneStride = vpux::getIntAttr(_ctx, blockSize * blockSize * elemTypeSize);
    auto numPlanes = vpux::getIntAttr(_ctx, IC);

    return VPUIP::DMADescriptorAttr::get(_ctx, numPlanes, len, srcWidth, srcStride, srcPlaneStride, dstWidth, dstStride,
                                         dstPlaneStride);
}

VPUIP::DMADescriptorAttr vpux::VPUIP::SpaceToDepthDmaDescriptorGenerator::generateDepthFirstNCHW2NHWC(
        vpux::ShapeRef inShape, vpux::ShapeRef outShape, int64_t elemTypeSize, int64_t blockSize) const {
    const auto IH = inShape[Dims4D::Act::H];
    const auto IW = inShape[Dims4D::Act::W];
    const auto OH = outShape[Dims4D::Act::H];
    const auto OW = outShape[Dims4D::Act::W];
    const auto OC = outShape[Dims4D::Act::C];

    auto len = vpux::getIntAttr(_ctx, IW * OH * elemTypeSize);
    auto srcWidth = vpux::getIntAttr(_ctx, IW * elemTypeSize);
    auto srcStride = vpux::getIntAttr(_ctx, IW * blockSize * elemTypeSize);
    auto srcPlaneStride = vpux::getIntAttr(_ctx, IW * elemTypeSize);
    auto dstWidth = vpux::getIntAttr(_ctx, blockSize * elemTypeSize);
    auto dstStride = vpux::getIntAttr(_ctx, OC * elemTypeSize);
    auto dstPlaneStride = vpux::getIntAttr(_ctx, OW * IH * elemTypeSize);
    auto numPlanes = vpux::getIntAttr(_ctx, blockSize);

    return VPUIP::DMADescriptorAttr::get(_ctx, numPlanes, len, srcWidth, srcStride, srcPlaneStride, dstWidth, dstStride,
                                         dstPlaneStride);
}

vpux::VPUIP::PerAxisTileDmaDescriptorGenerator::PerAxisTileDmaDescriptorGenerator(mlir::MLIRContext* ctx,
                                                                                  vpux::Logger log)
        : _ctx(ctx), _log(log) {
}

VPUIP::DMADescriptorAttr vpux::VPUIP::PerAxisTileDmaDescriptorGenerator::generate(vpux::ShapeRef inShape,
                                                                                  vpux::ShapeRef outShape,
                                                                                  int64_t repeats,
                                                                                  int64_t elemTypeSize) const {
    VPUX_THROW_UNLESS(inShape.size() == 3 && inShape[Dim(1)] * repeats == outShape[Dim(1)],
                      "Unexpected inShape '{0}' and  outShape '{1}'", inShape, outShape);

    const auto L = inShape[Dim(0)];
    const auto M = inShape[Dim(1)];
    const auto R = inShape[Dim(2)];

    auto len = vpux::getIntAttr(_ctx, M * R * repeats * elemTypeSize);
    auto srcWidth = vpux::getIntAttr(_ctx, M * R * elemTypeSize);
    auto srcStride = vpux::getIntAttr(_ctx, 0 * elemTypeSize);
    auto srcPlaneStride = vpux::getIntAttr(_ctx, M * R * elemTypeSize);
    auto dstWidth = vpux::getIntAttr(_ctx, M * R * repeats * elemTypeSize);
    auto dstStride = vpux::getIntAttr(_ctx, M * R * repeats * elemTypeSize);
    auto dstPlaneStride = vpux::getIntAttr(_ctx, M * R * repeats * elemTypeSize);
    auto numPlanes = vpux::getIntAttr(_ctx, L);

    return VPUIP::DMADescriptorAttr::get(_ctx, numPlanes, len, srcWidth, srcStride, srcPlaneStride, dstWidth, dstStride,
                                         dstPlaneStride);
}

vpux::VPUIP::ExpandDmaDescriptorGenerator::ExpandDmaDescriptorGenerator(mlir::MLIRContext* ctx, vpux::Logger log)
        : _ctx(ctx), _log(log) {
}

VPUIP::DMADescriptorAttr vpux::VPUIP::ExpandDmaDescriptorGenerator::generate(vpux::NDTypeInterface inType,
                                                                             vpux::NDTypeInterface outType,
                                                                             mlir::ArrayAttr padsBegin,
                                                                             mlir::ArrayAttr padsEnd,
                                                                             int64_t elemTypeSize) const {
    // Only support ExpandDMA padding at end
    // TODO: support padding at begin E65670
    VPUX_THROW_WHEN(llvm::any_of(parseIntArrayAttr<int64_t>(padsBegin),
                                 [](auto padValue) {
                                     return padValue != 0;
                                 }),
                    "ExpandDMA don't support padding at begin!");

    auto padEnd = parseIntArrayAttr<int64_t>(padsEnd);
    const auto nonZeroAxisPredicate = [](const int64_t dim) -> bool {
        return dim > 0;
    };
    const auto padEndAxisIter = std::find_if(padEnd.begin(), padEnd.end(), nonZeroAxisPredicate);
    VPUX_THROW_WHEN(padEndAxisIter == padEnd.end(), "Can not find padding axis");
    const auto padEndAxis = std::distance(padEnd.begin(), padEndAxisIter);
    const auto inOrder = inType.getDimsOrder();
    const auto padEndAxisPos = inOrder.dimPos(Dim(padEndAxis));

    const auto inputPaddingDimSize = inType.getShape()[Dim(padEndAxis)];
    const auto outputPaddingDimSize = outType.getShape()[Dim(padEndAxis)];
    int64_t totalLowerDimSize = 1;
    for (auto idx : irange(inOrder.numDims())) {
        auto curPos = inOrder.dimPos(Dim(idx));
        if (curPos > padEndAxisPos) {
            totalLowerDimSize *= inType.getShape()[Dim(idx)];
        }
    }

    const bool isHighestDim = padEndAxisPos > 0;
    auto getStrideOnPaddingDim = [&](NDTypeInterface type, const DimsOrder& order,
                                     const int64_t paddingDimSize) -> int64_t {
        const auto higherDim = isHighestDim ? order.dimAt(padEndAxisPos - 1) : order.dimAt(padEndAxisPos);
        return isHighestDim ? type.getStrides()[Dim(higherDim)].to<Byte>().count()
                            : type.getStrides()[Dim(higherDim)].to<Byte>().count() * paddingDimSize;
    };

    const auto inPaddingStride = getStrideOnPaddingDim(inType, inOrder, inputPaddingDimSize);

    const auto lowerDimByteSize = inputPaddingDimSize * totalLowerDimSize * elemTypeSize;
    const auto totalTrasferSize = inType.getShape().totalSize() * elemTypeSize;

    // If source buffer is compact, source width and stride should be set as the size of the whole transfer,
    // otherwise there might performance penalties.
    const bool isInputStridedOnExpandDim = inPaddingStride > lowerDimByteSize;
    auto len = vpux::getIntAttr(_ctx, totalTrasferSize);
    auto srcWidth = vpux::getIntAttr(_ctx, isInputStridedOnExpandDim ? lowerDimByteSize : totalTrasferSize);
    auto srcStride = vpux::getIntAttr(_ctx, isInputStridedOnExpandDim ? inPaddingStride : totalTrasferSize);
    auto srcPlaneStride = vpux::getIntAttr(_ctx, 0 * elemTypeSize);

    const auto outOrder = outType.getDimsOrder();
    const auto outPaddingStride = getStrideOnPaddingDim(outType, outOrder, outputPaddingDimSize);

    auto dstWidth = vpux::getIntAttr(_ctx, lowerDimByteSize);
    auto dstStride = vpux::getIntAttr(_ctx, outPaddingStride);
    auto dstPlaneStride = vpux::getIntAttr(_ctx, 0 * elemTypeSize);
    auto numPlanes = vpux::getIntAttr(_ctx, 1);

    return VPUIP::DMADescriptorAttr::get(_ctx, numPlanes, len, srcWidth, srcStride, srcPlaneStride, dstWidth, dstStride,
                                         dstPlaneStride);
}
