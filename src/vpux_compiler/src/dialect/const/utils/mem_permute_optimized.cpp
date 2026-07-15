//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/utils/mem_permute_optimized.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

namespace {

size_t getContentRank(vpux::Const::Content& content) {
    return content.getType().getRank();
}

struct OvLike4DStrides {
    OvLike4DStrides(vpux::NDTypeInterface type) {
        const auto order = type.getDimsOrder();
        const auto strides = type.getMemStrides();
        const auto refStrides = order.toLogicalOrder(strides);
        const auto elemSizeBits = getElemTypeSize(type).count();

        this->N = refStrides[Dim(0)].count() / elemSizeBits;
        this->C = refStrides[Dim(1)].count() / elemSizeBits;
        this->H = refStrides[Dim(2)].count() / elemSizeBits;
        this->W = refStrides[Dim(3)].count() / elemSizeBits;
    }

    size_t N;
    size_t C;
    size_t H;
    size_t W;
};

struct OvLike5DStrides {
    OvLike5DStrides(vpux::NDTypeInterface type) {
        const auto order = type.getDimsOrder();
        const auto strides = type.getMemStrides();
        const auto refStrides = order.toLogicalOrder(strides);
        const auto elemSizeBits = getElemTypeSize(type).count();

        this->N = refStrides[Dim(0)].count() / elemSizeBits;
        this->C = refStrides[Dim(1)].count() / elemSizeBits;
        this->D = refStrides[Dim(2)].count() / elemSizeBits;
        this->H = refStrides[Dim(3)].count() / elemSizeBits;
        this->W = refStrides[Dim(4)].count() / elemSizeBits;
    }

    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
};

template <typename StorageType>
void blob_copy_4d_t(vpux::Const::Content& input, vpux::Const::Content& output, mlir::MLIRContext* ctx) {
    const StorageType* srcPtr = reinterpret_cast<const StorageType*>(input.getRawStorageBuf().data());
    StorageType* dstPtr = reinterpret_cast<StorageType*>(output.getRawTempBuf().data());

    auto shapeRef = input.getType().getShape();
    SmallVector<int64_t> shape(shapeRef.begin(), shapeRef.end());

    const size_t N = shape[0];
    const size_t C = shape[1];
    const size_t H = shape[2];
    const size_t W = shape[3];

    const auto inputStrides = OvLike4DStrides(input.getType());
    const auto outputStrides = OvLike4DStrides(output.getType());
    const auto loopPolicy = LoopExecPolicy::Parallel;
    if (input.getType().getDimsOrder() == DimsOrder::NHWC) {
        loop_2d(loopPolicy, ctx, N, C, [&](size_t n, size_t c) {
            StorageType* dst_ptr_l = dstPtr + n * outputStrides.N + c * outputStrides.C;
            const StorageType* src_ptr_l = srcPtr + n * inputStrides.N + c * inputStrides.C;
            for (size_t h = 0; h < H; h++) {
                const StorageType* src_ptr_l_l = src_ptr_l + h * inputStrides.H;
                for (size_t w = 0; w < W; w++) {
                    *dst_ptr_l = *src_ptr_l_l;
                    src_ptr_l_l += inputStrides.W;
                    dst_ptr_l++;
                }
            }
        });
    } else {
        loop_2d(loopPolicy, ctx, N, C, [&](size_t n, size_t c) {
            const StorageType* src_ptr_l = srcPtr + n * inputStrides.N + c * inputStrides.C;
            StorageType* dst_ptr_l = dstPtr + n * outputStrides.N + c;
            for (size_t h = 0; h < H; h++) {
                const StorageType* src_ptr_l_l = src_ptr_l + h * inputStrides.H;
                for (size_t w = 0; w < W; w++) {
                    *dst_ptr_l = *src_ptr_l_l;
                    dst_ptr_l += outputStrides.W;
                    src_ptr_l_l++;
                }
            }
        });
    }
}

inline void blob_copy_4d(vpux::Const::Content& input, vpux::Const::Content& output) {
    const Bit elemSize = getElemTypeSize(input.getStorageElemType());
    VPUX_THROW_WHEN(elemSize.count() < CHAR_BIT, "Unsupported blob 4D transformation for sub byte types {0}",
                    input.getStorageElemType());
    const size_t elemSizeBytes = elemSize.to<Byte>().count();
    switch (elemSizeBytes) {
    case sizeof(float):
        blob_copy_4d_t<float>(input, output, input.getStorageElemType().getContext());
        break;
    case sizeof(uint16_t):
        blob_copy_4d_t<uint16_t>(input, output, input.getStorageElemType().getContext());
        break;
    case sizeof(uint8_t):
        blob_copy_4d_t<uint8_t>(input, output, input.getStorageElemType().getContext());
        break;
    default:
        VPUX_THROW("Unsupported blob 4D transformation for precision {0}", input.getStorageElemType());
    }
}

template <typename StorageType>
void blob_copy_5d_t(vpux::Const::Content& input, vpux::Const::Content& output, mlir::MLIRContext* ctx) {
    const StorageType* srcPtr = reinterpret_cast<const StorageType*>(input.getRawStorageBuf().data());
    StorageType* dstPtr = reinterpret_cast<StorageType*>(output.getRawTempBuf().data());

    auto shapeRef = input.getType().getShape();
    SmallVector<int64_t> shape(shapeRef.begin(), shapeRef.end());

    const size_t N = shape[0];
    const size_t C = shape[1];
    const size_t D = shape[2];
    const size_t H = shape[3];
    const size_t W = shape[4];

    const auto inputStrides = OvLike5DStrides(input.getType());
    const auto outputStrides = OvLike5DStrides(output.getType());
    const auto loopPolicy = LoopExecPolicy::Parallel;
    if (input.getType().getDimsOrder() == DimsOrder::NDHWC) {
        loop_3d(loopPolicy, ctx, N, C, D, [&](size_t n, size_t c, size_t d) {
            StorageType* dst_ptr_l = dstPtr + n * outputStrides.N + c * outputStrides.C + d * outputStrides.D;
            const StorageType* src_ptr_l = srcPtr + n * inputStrides.N + c * inputStrides.C + d * inputStrides.D;
            for (size_t h = 0; h < H; h++) {
                const StorageType* src_ptr_l_l = src_ptr_l + h * inputStrides.H;
                for (size_t w = 0; w < W; w++) {
                    *dst_ptr_l = *src_ptr_l_l;
                    src_ptr_l_l += inputStrides.W;
                    dst_ptr_l++;
                }
            }
        });
    } else {
        loop_3d(loopPolicy, ctx, N, C, D, [&](size_t n, size_t c, size_t d) {
            const StorageType* src_ptr_l = srcPtr + n * inputStrides.N + c * inputStrides.C + d * inputStrides.D;
            StorageType* dst_ptr_l = dstPtr + n * outputStrides.N + c + d * outputStrides.D;
            for (size_t h = 0; h < H; h++) {
                const StorageType* src_ptr_l_l = src_ptr_l + h * inputStrides.H;
                for (size_t w = 0; w < W; w++) {
                    *dst_ptr_l = *src_ptr_l_l;
                    dst_ptr_l += outputStrides.W;
                    src_ptr_l_l++;
                }
            }
        });
    }
}

inline void blob_copy_5d(vpux::Const::Content& input, vpux::Const::Content& output) {
    const auto elemSizeBits = getElemTypeSize(input.getStorageElemType()).count();
    VPUX_THROW_WHEN(elemSizeBits < CHAR_BIT, "Unsupported blob 5D transformation for sub byte types {0}",
                    input.getStorageElemType());
    const size_t elemSizeBytes = elemSizeBits / CHAR_BIT;
    switch (elemSizeBytes) {
    case sizeof(float):
        blob_copy_5d_t<float>(input, output, input.getStorageElemType().getContext());
        break;
    case sizeof(uint16_t):
        blob_copy_5d_t<uint16_t>(input, output, input.getStorageElemType().getContext());
        break;
    case sizeof(uint8_t):
        blob_copy_5d_t<uint8_t>(input, output, input.getStorageElemType().getContext());
        break;
    default:
        VPUX_THROW("Unsupported blob 5D transformation for precision {0}", input.getStorageElemType());
    }
}

};  // namespace

bool Const::details::isOptimizedTransformationSupported(vpux::Const::Content& input, vpux::NDTypeInterface outType,
                                                        const DimsOrder& permOrder) {
    const Bit elemSize = getElemTypeSize(input.getStorageElemType());
    const size_t elemSizeBytes = checked_cast<size_t>(elemSize.to<Byte>().count());
    static const std::unordered_set<size_t> optimizedElemTypeSizes = {sizeof(uint8_t), sizeof(uint16_t), sizeof(float)};
    // Check storage size
    if (optimizedElemTypeSizes.count(elemSizeBytes) == 0) {
        return false;
    }

    const auto createDimsVerifier = [](const vpux::DimsOrder& order1, const vpux::DimsOrder& order2) {
        return [=](const DimsOrder& typeOrder1, const DimsOrder& typeOrder2) {
            return (typeOrder1 == order1 && typeOrder2 == order2) || (typeOrder1 == order2 && typeOrder2 == order1);
        };
    };
    const auto isBetweenNCHW_NHWC = createDimsVerifier(DimsOrder::NHWC, DimsOrder::NCHW);
    const auto isBetweenNCDHW_NDHWC = createDimsVerifier(DimsOrder::NCDHW, DimsOrder::NDHWC);
    const auto inOrder = input.getType().getDimsOrder();
    const auto outOrder = outType.getDimsOrder();
    const auto reorderPerm =
            DimsOrder::fromAffineMap(getPermutationFromOrders(inOrder, outOrder, outType.getContext()));

    return (isBetweenNCHW_NHWC(inOrder, outOrder) || isBetweenNCDHW_NDHWC(inOrder, outOrder)) &&
           permOrder == reorderPerm;
}

// This code is similar to openvino solution,
// \see{https://github.com/openvinotoolkit/openvino/blob/releases/2023/2/src/inference/src/blob_transform.cpp}
void Const::details::memPermuteTransformationOptimized(vpux::Const::Content& input, vpux::Const::Content& output) {
    const size_t numDims = getContentRank(input);
    if (numDims == 4) {
        blob_copy_4d(input, output);
    } else if (numDims == 5) {
        blob_copy_5d(input, output);
    } else {
        VPUX_THROW("Unimplemented blob transformation. Only 4d or 5d supported.");
    }
}
