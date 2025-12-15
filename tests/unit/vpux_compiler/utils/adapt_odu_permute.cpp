//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#include <gtest/gtest.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

using namespace vpux;

TEST(MLIR_ODUPermuteTest, returnBestDimOrder) {
    // No alignment single axis tiling
    {
        // initialDimOrder is the original output order of the DPU op
        // oneDims are the dimensions equal to 1; batch is excluded.
        mlir::SmallVector<std::tuple<DimsOrder, mlir::SmallVector<Dim>, DimsOrder>> dimOrderVec = {
                {/*initialDimOrder*/ DimsOrder::fromCode(0x1234), /*oneDims*/ {Dims4D::Act::C, Dims4D::Act::H},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x1342)},
                {/*initialDimOrder*/ DimsOrder::fromCode(0x1432), /*oneDims*/ {Dims4D::Act::C, Dims4D::Act::W},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x1342)},
                {/*initialDimOrder*/ DimsOrder::fromCode(0x1324), /*oneDims*/ {Dims4D::Act::C},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x1342)},
                {/*initialDimOrder*/ DimsOrder::fromCode(0x1243), /*oneDims*/ {Dims4D::Act::C},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x1432)},
                {/*initialDimOrder*/ DimsOrder::fromCode(0x1342), /*oneDims*/ {Dims4D::Act::C},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x1342)},
                {/*initialDimOrder*/ DimsOrder::fromCode(0x1234), /*oneDims*/ {Dims4D::Act::H},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x1243)},
                {/*initialDimOrder*/ DimsOrder::fromCode(0x1423), /*oneDims*/ {Dims4D::Act::H},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x1342)},
                {/*initialDimOrder*/ DimsOrder::fromCode(0x1423), /*oneDims*/ {Dims4D::Act::W},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x1423)},
                {/*initialDimOrder*/ DimsOrder::fromCode(0x1234), /*oneDims*/ {Dims4D::Act::W},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x1423)}};

        for (auto it : dimOrderVec) {
            auto initialDimOrder = std::get<0>(it);
            auto oneDims = std::get<1>(it);
            auto actualOutputDimOrder = vpux::IE::returnBestDimOrder(initialDimOrder, oneDims, false);
            EXPECT_EQ(actualOutputDimOrder, std::get<2>(it));
        }
    }
}
