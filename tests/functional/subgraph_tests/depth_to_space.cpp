// Copyright (C) 2023 - 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>

#include <algorithm>
#include <array>
#include <string_view>
#include <tuple>
#include <vector>

#include <llvm/Support/FormatVariadic.h>

#include <vpu_ov2_layer_test.hpp>
#include "common/functions.h"
#include "openvino/opsets/opset1.hpp"

#include <vpux/utils/core/error.hpp>

/*
      SUBGRAPH

    Conv    Input
      \      /
      D2S   /
        \  /
        Add
         |
       Result
*/

using namespace ov::test;
using namespace ov::opset1;
namespace {

struct DepthToSpaceTestParams {
    ov::Shape convInputShape;
    ov::Shape convWeightsShape;

    ov::Shape addInputShape;

    ov::CoordinateDiff padsBegin;
    ov::CoordinateDiff padsEnd;

    DepthToSpace::DepthToSpaceMode mode;
    int blockSize;
};

class DepthToSpaceTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<DepthToSpaceTestParams> {
public:
    void SetUp() override {
        auto [convInputShape, convWeightsShape, addInputShape, padsBegin, padsEnd, mode, blockSize] = GetParam();

        VPUX_THROW_WHEN(convInputShape.size() != 4, "convInputShape.size() != 4");
        VPUX_THROW_WHEN(convWeightsShape.size() != 4, "convWeightsShape.size() != 4");
        VPUX_THROW_WHEN(addInputShape.size() != 4, "addInputShape.size() != 4");

        VPUX_THROW_WHEN(padsBegin.size() != 2, "padsBegin.size() != 2");
        VPUX_THROW_WHEN(padsEnd.size() != 2, "padsEnd.size() != 2");

        abs_threshold = 0.5f;
        rel_threshold = 1.0f;

        inType = ov::element::f16;
        outType = ov::element::f16;

        const ov::Strides strides{1, 1};
        const ov::Strides dilations{1, 1};

        init_input_shapes(ov::test::static_shapes_to_test_representation({convInputShape, addInputShape}));

        ov::ParameterVector params;
        std::transform(inputDynamicShapes.begin(), inputDynamicShapes.end(), std::back_inserter(params),
                       [&](auto&& shape) {
                           return std::make_shared<ov::op::v0::Parameter>(inType, shape);
                       });

        auto convWeightsShapeElements =
                std::accumulate(convWeightsShape.begin(), convWeightsShape.end(), 1, std::multiplies<size_t>());

        // Generate some weights.
        std::vector<float> weightValues;
        std::generate_n(std::back_inserter(weightValues), convWeightsShapeElements, [i = 0]() mutable {
            return std::sin(i++);
        });

        const auto weights = ov::op::v0::Constant::create(ov::element::f16, convWeightsShape, weightValues);

        // Graph
        const auto convOp =
                std::make_shared<ov::opset1::Convolution>(params[0], weights, strides, padsBegin, padsEnd, dilations);

        const auto d2sOp = std::make_shared<ov::opset1::DepthToSpace>(convOp, mode, blockSize);
        const auto addOp = std::make_shared<ov::opset1::Add>(d2sOp, params[1]);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(addOp)};
        function = std::make_shared<ov::Model>(results, params, "DepthToSpaceTest");
    }

    static std::string getTestCaseName(testing::TestParamInfo<DepthToSpaceTestParams> obj) {
        auto [convInputShape, convWeightsShape, addInputShape, padsBegin, padsEnd, mode, blockSize] = obj.param;

        VPUX_THROW_WHEN(convInputShape.size() != 4, "convInputShape.size() != 4");
        VPUX_THROW_WHEN(convWeightsShape.size() != 4, "convWeightsShape.size() != 4");
        VPUX_THROW_WHEN(addInputShape.size() != 4, "addInputShape.size() != 4");

        VPUX_THROW_WHEN(padsBegin.size() != 2, "padsBegin.size() != 2");
        VPUX_THROW_WHEN(padsEnd.size() != 2, "padsEnd.size() != 2");

        // In C++17, we cannot capture structured bindings because they are not treated as normal variables so we must
        // pass it as an argument. In C++20 we can.
        std::string_view modeStr = [&](auto mode) {
            switch (mode) {
            case ov::opset1::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST:
                return "DEPTH_FIRST";

            case ov::opset1::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST:
                return "BLOCKS_FIRST";
            }
        }(mode);

        constexpr auto shapeFmtStr = "{{{0}, {1}, {2}, {3}}}";

        auto convInputShapeStr = llvm::formatv(shapeFmtStr, convInputShape.at(0), convInputShape.at(1),
                                               convInputShape.at(2), convInputShape.at(3));

        auto convWeightsShapeStr = llvm::formatv(shapeFmtStr, convWeightsShape.at(0), convWeightsShape.at(1),
                                                 convWeightsShape.at(2), convWeightsShape.at(3));

        auto addInputShapeStr = llvm::formatv(shapeFmtStr, addInputShape.at(0), addInputShape.at(1),
                                              addInputShape.at(2), addInputShape.at(3));

        auto paddingStr = llvm::formatv(shapeFmtStr, padsBegin.at(0), padsBegin.at(1), padsEnd.at(0), padsEnd.at(1));

        return llvm::formatv("TestKind{0}"
                             "_convInputShape={1}"
                             "_convWeightsShape={2}"
                             "_addInputShape={3}"
                             "_padding={4}"
                             "_mode={5}"
                             "_blockSize={6}",
                             ov::test::utils::testKind(__FILE__), convInputShapeStr, convWeightsShapeStr,
                             addInputShapeStr, paddingStr, modeStr, blockSize)
                .str();
    }
};

// CONV => tensor<1x32x800x1280xf16>, tensor<12x32x3x3xf16> -> tensor<1x12x800x1280xf16>
// D2S  => tensor<1x12x800x1280xf16> -> tensor<1x3x1600x2560xf16>
// ADD  => tensor<1x3x1600x2560xf16>, tensor<1x3x1600x2560xf16> -> tensor<1x3x1600x2560xf16>

// Parameters
std::vector<DepthToSpaceTestParams> testCases = {
        // DEPTH_FIRST
        {
                /* convInputShape = */ {1, 32, 800, 1280},  // 1600x2560
                /* convWeightsShape = */ {12, 32, 3, 3},
                /* addInputShape = */ {1, 3, 1600, 2560},
                /* padsBegin = */ {1, 1},
                /* padsEnd = */ {1, 1},
                /* mode = */ DepthToSpace::DepthToSpaceMode::DEPTH_FIRST,
                /* blockSize = */ 2,
        },
        {
                /* convInputShape = */ {1, 32, 540, 960},  // 1080x1920
                /* convWeightsShape = */ {12, 32, 3, 3},
                /* addInputShape = */ {1, 3, 1080, 1920},
                /* padsBegin = */ {1, 1},
                /* padsEnd = */ {1, 1},
                /* mode = */ DepthToSpace::DepthToSpaceMode::DEPTH_FIRST,
                /* blockSize = */ 2,
        },
        {
                /* convInputShape = */ {1, 32, 360, 640},  // 720x1280
                /* convWeightsShape = */ {12, 32, 3, 3},
                /* addInputShape = */ {1, 3, 720, 1280},
                /* padsBegin = */ {1, 1},
                /* padsEnd = */ {1, 1},
                /* mode = */ DepthToSpace::DepthToSpaceMode::DEPTH_FIRST,
                /* blockSize = */ 2,
        },

        // BLOCKS_FIRST
        {
                /* convInputShape = */ {1, 32, 256, 256},
                /* convWeightsShape = */ {12, 32, 1, 1},
                /* addInputShape = */ {1, 3, 512, 512},
                /* padsBegin = */ {0, 0},
                /* padsEnd = */ {0, 0},
                /* mode = */ DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
                /* blockSize = */ 2,
        },
};

ov::AnyMap COMMON_CONFIGURATION = {
        {"NPU_COMPILATION_MODE_PARAMS", ""},
        {"PERF_COUNT", "NO"},
};

class DepthToSpaceTest_1C : public DepthToSpaceTestCommon {};
class DepthToSpaceTest_4C : public DepthToSpaceTestCommon {};
class DepthToSpaceTest_6C : public DepthToSpaceTestCommon {};

// NPU4000
TEST_P(DepthToSpaceTest_1C, NPU4000) {
    setDefaultHardwareMode();
    configuration.merge(COMMON_CONFIGURATION);
    configuration[ov::intel_npu::tiles.name()] = 1;
    run(Platform::NPU4000);
}

TEST_P(DepthToSpaceTest_4C, NPU4000) {
    setDefaultHardwareMode();
    configuration.merge(COMMON_CONFIGURATION);
    configuration[ov::intel_npu::tiles.name()] = 4;
    run(Platform::NPU4000);
}

TEST_P(DepthToSpaceTest_6C, NPU4000) {
    setDefaultHardwareMode();
    configuration.merge(COMMON_CONFIGURATION);
    configuration[ov::intel_npu::tiles.name()] = 6;
    run(Platform::NPU4000);
}

// NPU3720
TEST_P(DepthToSpaceTest_1C, NPU3720) {
    setDefaultHardwareMode();
    configuration.merge(COMMON_CONFIGURATION);
    configuration[ov::intel_npu::tiles.name()] = 1;
    run(Platform::NPU3720);
}

TEST_P(DepthToSpaceTest_4C, NPU3720) {
    setDefaultHardwareMode();
    configuration.merge(COMMON_CONFIGURATION);
    configuration[ov::intel_npu::tiles.name()] = 4;
    run(Platform::NPU3720);
}

TEST_P(DepthToSpaceTest_6C, NPU3720) {
    setDefaultHardwareMode();
    configuration.merge(COMMON_CONFIGURATION);
    configuration[ov::intel_npu::tiles.name()] = 6;
    run(Platform::NPU3720);
}

// Instantiate suite
INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpace, DepthToSpaceTest_1C, ::testing::ValuesIn(testCases),
                         DepthToSpaceTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpace, DepthToSpaceTest_4C, ::testing::ValuesIn(testCases),
                         DepthToSpaceTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpace, DepthToSpaceTest_6C, ::testing::ValuesIn(testCases),
                         DepthToSpaceTestCommon::getTestCaseName);

}  // namespace
