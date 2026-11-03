//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include "common/utils.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

#include <optional>

using namespace vpux;

namespace {

mlir::Type makeElemType(mlir::MLIRContext& ctx, int64_t bitWidth) {
    return mlir::IntegerType::get(&ctx, static_cast<unsigned>(bitWidth));
}

// Builds the buffer type through `vpux::getMemRefType` so that the strides land in a `vpux::MemRefAttr`.
// `MemRefNDTypeInterface::getStrides()` only reads that attribute - an `mlir::StridedLayoutAttr` would be
// silently ignored and compact strides recomputed instead.
NDTypeInterface makeBufferType(mlir::MLIRContext& ctx, ArrayRef<int64_t> shape, int64_t bitWidth,
                               ArrayRef<int64_t> strideBits = {}) {
    const auto elemType = makeElemType(ctx, bitWidth);

    SmallVector<Bit> strides;
    strides.reserve(strideBits.size());
    for (const auto bits : strideBits) {
        strides.push_back(Bit(bits));
    }

    const auto memRefType = vpux::getMemRefType(ShapeRef(shape), elemType, DimsOrder::fromNumDims(shape.size()),
                                                /*memSpace=*/nullptr, StridesRef(ArrayRef<Bit>(strides)));
    return mlir::cast<NDTypeInterface>(memRefType);
}

// Per-side strides are expressed in BITS and are optional; std::nullopt means the compact layout for `shape`.
// Compact strides are not stored by `getMemRefType`, so passing them explicitly is equivalent to std::nullopt.
struct DmaSplitPartSizesParams {
    SmallVector<int64_t> shape;
    int64_t bitWidth;
    int64_t tileDim;
    int64_t partsNumber;
    SmallVector<int64_t> expectedPartSizes;
    std::optional<SmallVector<int64_t>> inputStrides = std::nullopt;
    std::optional<SmallVector<int64_t>> outputStrides = std::nullopt;
};

ArrayRef<int64_t> asRef(const std::optional<SmallVector<int64_t>>& strides) {
    return strides.has_value() ? ArrayRef<int64_t>(*strides) : ArrayRef<int64_t>{};
}

class DmaSplitPartSizesTest : public MLIR_UnitBase, public testing::WithParamInterface<DmaSplitPartSizesParams> {};

TEST_P(DmaSplitPartSizesTest, ReturnsAlignedPartSizes) {
    mlir::MLIRContext ctx(registry);
    ctx.loadAllAvailableDialects();
    const auto& params = GetParam();
    const auto inputType = makeBufferType(ctx, params.shape, params.bitWidth, asRef(params.inputStrides));
    const auto outputType = makeBufferType(ctx, params.shape, params.bitWidth, asRef(params.outputStrides));

    const auto partSizes = VPUIP::getSplitPartSizes(inputType, outputType, Dim(params.tileDim), params.partsNumber);
    EXPECT_EQ(partSizes, params.expectedPartSizes);
}

INSTANTIATE_TEST_SUITE_P(
        DmaSplitPartSizesCoverage, DmaSplitPartSizesTest,
        testing::Values(
                // clang-format off
                // Sub-byte (i2)
                DmaSplitPartSizesParams{/*shape=*/{3, 3}, /*bitWidth=*/2, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{3}},
                DmaSplitPartSizesParams{/*shape=*/{6, 3}, /*bitWidth=*/2, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{6}},
                DmaSplitPartSizesParams{/*shape=*/{4, 3}, /*bitWidth=*/2, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{4}},
                DmaSplitPartSizesParams{/*shape=*/{8, 3},  /*bitWidth=*/2, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{4, 4}},
                DmaSplitPartSizesParams{/*shape=*/{16, 3}, /*bitWidth=*/2, /*tileDim=*/0, /*partsNumber=*/4, /*expectedPartSizes=*/{4, 4, 4, 4}},
                DmaSplitPartSizesParams{/*shape=*/{20, 3}, /*bitWidth=*/2, /*tileDim=*/0, /*partsNumber=*/4, /*expectedPartSizes=*/{4, 4, 4, 8}},
                DmaSplitPartSizesParams{/*shape=*/{21, 3}, /*bitWidth=*/2, /*tileDim=*/0, /*partsNumber=*/4, /*expectedPartSizes=*/{21}},
                DmaSplitPartSizesParams{/*shape=*/{3, 4}, /*bitWidth=*/2, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{1, 2}},
                DmaSplitPartSizesParams{/*shape=*/{8, 4}, /*bitWidth=*/2, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{4, 4}},
                // input stride 6 bits -> alignment 4, output stride 8 bits -> alignment 1, lcm 4
                DmaSplitPartSizesParams{/*shape=*/{20, 3}, /*bitWidth=*/2, /*tileDim=*/0, /*partsNumber=*/4, /*expectedPartSizes=*/{4, 4, 4, 8},
                     /*inputStrides=*/SmallVector<int64_t>{6, 2}, /*outputStrides=*/SmallVector<int64_t>{8, 2}},

                // Sub-byte (i4)
                DmaSplitPartSizesParams{/*shape=*/{4, 3}, /*bitWidth=*/4, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{2, 2}},
                DmaSplitPartSizesParams{/*shape=*/{8, 3}, /*bitWidth=*/4, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{4, 4}},
                DmaSplitPartSizesParams{/*shape=*/{8, 3}, /*bitWidth=*/4, /*tileDim=*/0, /*partsNumber=*/4, /*expectedPartSizes=*/{2, 2, 2, 2}},
                DmaSplitPartSizesParams{/*shape=*/{10, 3}, /*bitWidth=*/4, /*tileDim=*/0, /*partsNumber=*/4, /*expectedPartSizes=*/{2, 2, 2, 4}},
                DmaSplitPartSizesParams{/*shape=*/{6, 3}, /*bitWidth=*/4, /*tileDim=*/0, /*partsNumber=*/4, /*expectedPartSizes=*/{6}},
                DmaSplitPartSizesParams{/*shape=*/{2, 3, 5}, /*bitWidth=*/4, /*tileDim=*/1, /*partsNumber=*/2, /*expectedPartSizes=*/{3}},
                DmaSplitPartSizesParams{/*shape=*/{2, 3, 5}, /*bitWidth=*/4, /*tileDim=*/2, /*partsNumber=*/3, /*expectedPartSizes=*/{5}},
                // input stride 12 bits -> alignment 2, output stride 16 bits -> alignment 1, lcm 2
                DmaSplitPartSizesParams{/*shape=*/{6, 3}, /*bitWidth=*/4, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{2, 4},
                     /*inputStrides=*/SmallVector<int64_t>{12, 4}, /*outputStrides=*/SmallVector<int64_t>{16, 4}},

                // Byte-aligned (i8)
                DmaSplitPartSizesParams{/*shape=*/{3, 3}, /*bitWidth=*/8, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{1, 2}},

                // Byte-aligned (i16)
                DmaSplitPartSizesParams{/*shape=*/{3, 3}, /*bitWidth=*/16, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{1, 2}},
                DmaSplitPartSizesParams{/*shape=*/{6, 3}, /*bitWidth=*/16, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{3, 3}},

                // Byte-aligned (i32)
                DmaSplitPartSizesParams{/*shape=*/{3, 3}, /*bitWidth=*/32, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{1, 2}},
                DmaSplitPartSizesParams{/*shape=*/{8, 3}, /*bitWidth=*/32, /*tileDim=*/0, /*partsNumber=*/4, /*expectedPartSizes=*/{2, 2, 2, 2}},

                // 3D byte-aligned
                DmaSplitPartSizesParams{/*shape=*/{2, 6, 4}, /*bitWidth=*/8,  /*tileDim=*/1, /*partsNumber=*/3, /*expectedPartSizes=*/{2, 2, 2}},
                DmaSplitPartSizesParams{/*shape=*/{3, 5, 7}, /*bitWidth=*/16, /*tileDim=*/2, /*partsNumber=*/2, /*expectedPartSizes=*/{3, 4}},

                // 3D sub-byte (i4)
                DmaSplitPartSizesParams{/*shape=*/{4, 3, 1}, /*bitWidth=*/4, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{2, 2}},
                DmaSplitPartSizesParams{/*shape=*/{2, 8, 3}, /*bitWidth=*/4, /*tileDim=*/1, /*partsNumber=*/4, /*expectedPartSizes=*/{2, 2, 2, 2}},

                // 3D sub-byte (i2)
                DmaSplitPartSizesParams{/*shape=*/{3, 8, 1},  /*bitWidth=*/2, /*tileDim=*/1, /*partsNumber=*/2, /*expectedPartSizes=*/{4, 4}},
                DmaSplitPartSizesParams{/*shape=*/{3, 12, 1}, /*bitWidth=*/2, /*tileDim=*/1, /*partsNumber=*/2, /*expectedPartSizes=*/{4, 8}},
                DmaSplitPartSizesParams{/*shape=*/{2, 8, 3}, /*bitWidth=*/2, /*tileDim=*/1, /*partsNumber=*/4, /*expectedPartSizes=*/{8},
                     /*inputStrides=*/SmallVector<int64_t>{48, 6, 2}, /*outputStrides=*/SmallVector<int64_t>{64, 8, 2}},
                DmaSplitPartSizesParams{/*shape=*/{3, 16, 3}, /*bitWidth=*/2, /*tileDim=*/1, /*partsNumber=*/3, /*expectedPartSizes=*/{4, 4, 8},
                     /*inputStrides=*/SmallVector<int64_t>{128, 8, 2}, /*outputStrides=*/SmallVector<int64_t>{96, 6, 2}},

                // 4D byte-aligned
                DmaSplitPartSizesParams{/*shape=*/{2, 4, 8, 16}, /*bitWidth=*/8,  /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{1, 1}},
                DmaSplitPartSizesParams{/*shape=*/{2, 4, 8, 16}, /*bitWidth=*/8,  /*tileDim=*/2, /*partsNumber=*/2, /*expectedPartSizes=*/{4, 4}},
                DmaSplitPartSizesParams{/*shape=*/{1, 3, 5, 7},  /*bitWidth=*/32, /*tileDim=*/3, /*partsNumber=*/2, /*expectedPartSizes=*/{3, 4}},

                // 4D sub-byte (i4)
                DmaSplitPartSizesParams{/*shape=*/{8, 3, 1, 1}, /*bitWidth=*/4, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{4, 4}},

                // 4D sub-byte (i2)
                DmaSplitPartSizesParams{/*shape=*/{2, 3, 3, 1}, /*bitWidth=*/2, /*tileDim=*/0, /*partsNumber=*/2, /*expectedPartSizes=*/{2}},
                DmaSplitPartSizesParams{/*shape=*/{2, 1, 3, 1}, /*bitWidth=*/2, /*tileDim=*/2, /*partsNumber=*/2, /*expectedPartSizes=*/{3}},
                DmaSplitPartSizesParams{/*shape=*/{2, 4, 6, 3}, /*bitWidth=*/2, /*tileDim=*/2, /*partsNumber=*/2, /*expectedPartSizes=*/{6}},

                // partsNumber == 1 -> always a single part
                DmaSplitPartSizesParams{/*shape=*/{8, 3}, /*bitWidth=*/4, /*tileDim=*/0, /*partsNumber=*/1, /*expectedPartSizes=*/{8}})
            );
// clang-format on

}  // namespace
