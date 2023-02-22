//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/const/utils/transformations.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

#include "vpux/utils/IE/blob.hpp"
#include "vpux/utils/IE/loop.hpp"
#include "vpux/utils/core/hash.hpp"

#include <unordered_map>
#include <unordered_set>

using namespace vpux;

//
// memPermuteTransformation
//

Const::Content Const::details::memPermuteTransformation(vpux::Const::Content& input, vpux::NDTypeInterface outType,
                                                        mlir::AffineMap memPerm) {
    const auto inOrder = input.getType().getDimsOrder();
    const auto outOrder = outType.getDimsOrder();
    const auto permOrder = DimsOrder::fromAffineMap(memPerm);
    VPUX_THROW_UNLESS(inOrder.numDims() == outOrder.numDims(), "Can't reorder from '{0}' to '{1}'", inOrder, outOrder);
    VPUX_THROW_UNLESS(inOrder.numDims() == permOrder.numDims(), "Can't reorder from '{0}' to '{1}'", inOrder,
                      permOrder);

    if (input.isSplat() || memPerm.isIdentity()) {
        return Const::Content::moveBuffer(outType, std::move(input));
    } else {
        const Byte elemSize = getElemTypeSize(input.getStorageElemType());
        const auto inShape = input.getType().getShape();
        const auto inMemShape = inOrder.toMemoryOrder(inShape);

        static const std::unordered_set<DimsOrder> optimizedCases = {
                {DimsOrder::NHWC},
                {DimsOrder::NDHWC},
        };

        static const std::unordered_map<size_t, InferenceEngine::Precision> elemSizeToPrecision = {
                {sizeof(uint8_t), InferenceEngine::Precision::U8},
                {sizeof(uint16_t), InferenceEngine::Precision::U16},
                {sizeof(uint32_t), InferenceEngine::Precision::U32},
                // U64 is not supported by cvtBlobLayout
                // {sizeof(uint64_t), InferenceEngine::Precision::U64},
        };

        const auto precision = elemSizeToPrecision.find(checked_cast<size_t>(elemSize.count()));
        auto output = Const::Content::allocTempBuffer(outType, input.getStorageElemType(), input.isSplat());
        auto outBuf = output.getRawTempBuf();
        const auto inBuf = input.getRawStorageBuf();
        VPUX_THROW_UNLESS(outBuf.size() == inBuf.size(), "Storage buffer size mismatch in 'memPermuteTransformation'");

        if (optimizedCases.count(permOrder) != 0 && precision != elemSizeToPrecision.end()) {
            // Use optimized algorithm from IE core

            const InferenceEngine::SizeVector ieShape(inMemShape.begin(), inMemShape.end());

            const InferenceEngine::TensorDesc inDesc(precision->second, ieShape,
                                                     InferenceEngine::TensorDesc::getLayoutByDims(ieShape));
            const InferenceEngine::TensorDesc outDesc(precision->second, ieShape, permOrder.toIE());

            const auto inBlob = makeBlob(inDesc, nullptr, const_cast<char*>(inBuf.data()));
            const auto outBlob = makeBlob(outDesc, nullptr, outBuf.data());

            cvtBlobLayout(inBlob, outBlob);
        } else {
            // Use generic algorithm
            const auto outShape = outType.getShape();
            const auto outMemShape = outOrder.toMemoryOrder(outShape);

            loop_1d(LoopExecPolicy::Parallel, input.getType().getNumElements(), [&](int64_t inMemInd1D) {
                const auto inMemIndND = getMemIndexND(inMemInd1D, inMemShape);
                const auto outMemIndND = permOrder.toMemoryOrder(ShapeRef(inMemIndND.raw()));
                const auto outMemInd1D = getMemIndex1D(outMemIndND, outMemShape);

                const auto inMemRawInd = checked_cast<size_t>(inMemInd1D * elemSize.count());
                VPUX_THROW_UNLESS(inMemRawInd < inBuf.size(), "Out-of-bound access in 'memPermuteTransformation'");

                const auto outMemRawInd = checked_cast<size_t>(outMemInd1D * elemSize.count());
                VPUX_THROW_UNLESS(outMemRawInd < outBuf.size(), "Out-of-bound access in 'memPermuteTransformation'");

                std::copy_n(inBuf.data() + inMemRawInd, checked_cast<size_t>(elemSize.count()),
                            outBuf.data() + outMemRawInd);
            });
        }
        return output;
    }
}
