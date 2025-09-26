//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/npu_test_env_cfg.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_op/shape_of.hpp"

#include "openvino/op/softmax.hpp"

using namespace ov::test;

namespace LayerTestsDefinitions {

typedef std::tuple<ElementType,           // netPrecision
                   ov::test::InputShape,  // inputShape
                   int64_t,               // axis
                   std::string>
        softmaxLayerTestParamsSet;

class SoftMaxLayerTestDynamicBounds :
        public testing::WithParamInterface<softmaxLayerTestParamsSet>,
        virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<softmaxLayerTestParamsSet>& obj) {
        ElementType inType;
        ov::test::InputShape inShape;
        int64_t axis;
        std::string td;
        std::tie(inType, inShape, axis, td) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << inType << "_";
        result << "IS=" << ov::test::utils::partialShape2str({inShape.first}) << "_";
        result << "TS=";
        for (const auto& shape : inShape.second) {
            result << "(";
            result << ov::test::utils::vec2str(shape);
            result << ")_";
        }
        result << "axis=" << axis << "_";
        result << "device=" << test_utils::TARGET_DEVICE << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        ElementType inType;
        ov::test::InputShape inShape;
        int64_t axis;
        std::string td;
        std::tie(inType, inShape, axis, td) = this->GetParam();
        targetDevice = td;
        configuration["NPU_DYNAMIC_SHAPE_TO_STATIC"] = "YES";
        configuration["NPU_COMPILER_TYPE"] = "MLIR";

        if (inType == ov::element::Type_t::f16) {
            abs_threshold = 0.005;
        }

        init_input_shapes({inShape});

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }

        const auto softMax = std::make_shared<ov::op::v1::Softmax>(params[0], axis);
        auto makeFunction = [](ov::ParameterVector& params, const std::shared_ptr<ov::Node>& lastNode) {
            ov::ResultVector results;

            for (int i = 0; i < lastNode->get_output_size(); i++) {
                results.push_back(std::make_shared<ov::op::v0::Result>(lastNode->output(i)));
            }

            return std::make_shared<ov::Model>(results, params, "SoftMaxLayerTest");
        };
        function = makeFunction(params, softMax);
    }
};

TEST_P(SoftMaxLayerTestDynamicBounds, NPU3720) {
    compile_model();
}

namespace {
const std::vector<ElementType> netPrecisions = {ElementType::f32, ElementType::f16};

const std::vector<int64_t> axis4D = {1, 2, 3};

const std::vector<ov::test::InputShape> inputShapes4D = {
        // ov::Dimension(min, max) -> max = custom bounds
        {{1, 3, ov::Dimension(1, 10), ov::Dimension(1, 14)}, {{5, 6, 7, 8}}},
        {{ov::Dimension(1, 2), 3, ov::Dimension(1, 12), ov::Dimension(1, 16)}, {{5, 3, 7, 8}}},
        // static shape
        {{1, 5, 12, 12}, {{5, 3, 7, 8}}}};

INSTANTIATE_TEST_SUITE_P(precommit_softMaxDynamicTest4D, SoftMaxLayerTestDynamicBounds,
                         ::testing::Combine(testing::ValuesIn(netPrecisions), testing::ValuesIn(inputShapes4D),
                                            testing::ValuesIn(axis4D), testing::Values("NPU.3720")),
                         SoftMaxLayerTestDynamicBounds::getTestCaseName);

}  // namespace
}  // namespace LayerTestsDefinitions
