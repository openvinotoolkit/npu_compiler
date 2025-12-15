// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"
#include "openvino/opsets/opset15.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov::test {

// Opset15 Squeeze test class and parameters
class Squeeze15LayerTest :
        public testing::WithParamInterface<
                std::tuple<std::pair<std::vector<ov::test::InputShape>, std::vector<int>>,  // InputShape and axes
                           ov::element::Type,                                               // Model type
                           ov::test::TargetDevice                                           // Target device name
                           >>,
        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(
            const testing::TestParamInfo<std::tuple<std::pair<std::vector<ov::test::InputShape>, std::vector<int>>,
                                                    ov::element::Type, ov::test::TargetDevice>>& obj) {
        ov::element::Type model_type;
        std::pair<std::vector<ov::test::InputShape>, std::vector<int>> shape_item;
        std::string targetDevice;
        std::tie(shape_item, model_type, targetDevice) = obj.param;

        std::ostringstream result;
        const char separator = '_';
        result << "IS=(";
        for (size_t i = 0lu; i < shape_item.first.size(); i++) {
            result << ov::test::utils::partialShape2str({shape_item.first[i].first})
                   << (i < shape_item.first.size() - 1lu ? "_" : "");
        }
        result << ")_TS=";
        for (size_t i = 0lu; i < shape_item.first.front().second.size(); i++) {
            result << "{";
            for (size_t j = 0lu; j < shape_item.first.size(); j++) {
                result << ov::test::utils::vec2str(shape_item.first[j].second[i])
                       << (j < shape_item.first.size() - 1lu ? "_" : "");
            }
            result << "}_";
        }
        result << "OpType=SQUEEZE15" << separator;
        result << "Axes=" << (shape_item.second.empty() ? "default" : ov::test::utils::vec2str(shape_item.second))
               << separator;
        result << "modelType=" << model_type.to_string() << separator;
        result << "trgDev=" << targetDevice;
        return result.str();
    }

protected:
    void SetUp() override {
        ov::element::Type model_type;
        std::vector<ov::test::InputShape> input_shapes;
        std::vector<int> axes;
        std::pair<std::vector<ov::test::InputShape>, std::vector<int>> shape_item;
        std::tie(shape_item, model_type, targetDevice) = GetParam();
        std::tie(input_shapes, axes) = shape_item;

        init_input_shapes(input_shapes);

        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
        std::shared_ptr<ov::Node> op;

        if (axes.empty()) {
            // Create Squeeze opset15 without axes parameter
            op = std::make_shared<ov::op::v15::Squeeze>(param, false);
        } else {
            // Create Squeeze opset15 with axes parameter
            auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes);
            op = std::make_shared<ov::op::v15::Squeeze>(param, constant, false);
        }

        function = std::make_shared<ov::Model>(op->outputs(), ov::ParameterVector{param}, "Squeeze15");
    }
};

class Squeeze15LayerTestCommon : public Squeeze15LayerTest, virtual public VpuOv2LayerTest {
protected:
    ov::test::utils::SkipCallback skipCompilationCallback = [this](std::stringstream& str) {
        const auto inRank = function->get_parameters().at(0)->get_output_shape(0).size();
        const auto outRank = function->get_results().at(0)->get_input_shape(0).size();
        if (inRank > 4 || outRank > 4) {
            str << "> 4D case is not supported";
        }
    };
};

TEST_P(Squeeze15LayerTestCommon, NPU3720_HW) {
    setSkipCompilationCallback(skipCompilationCallback);
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(Squeeze15LayerTestCommon, NPU4000_HW) {
    setSkipCompilationCallback(skipCompilationCallback);
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(Squeeze15LayerTestCommon, NPU5010_HW) {
    setSkipCompilationCallback(skipCompilationCallback);
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace ov::test

namespace {

using ov::test::Squeeze15LayerTestCommon;

// Test cases for Squeeze opset15
std::vector<std::pair<std::vector<ov::test::InputShape>, std::vector<int>>> squeeze15_axes = {
        // Basic squeeze operations with different axes
        {{{{{1, 1, 2, 1}}, {{1, 1, 2, 1}}}}, {0, 1, 3}},  // Squeeze multiple dimensions
        {{{{{1, 2, 3, 1}}, {{1, 2, 3, 1}}}}, {0, 3}},     // Squeeze first and last dimensions
        {{{{{1, 1, 1, 4}}, {{1, 1, 1, 4}}}}, {0, 1, 2}},  // Squeeze first three dimensions
        {{{{{2, 1, 3, 1}}, {{2, 1, 3, 1}}}}, {1, 3}},     // Squeeze middle dimensions
        {{{{{1, 2, 1, 3}}, {{1, 2, 1, 3}}}}, {0, 2}},     // Squeeze non-consecutive dimensions
};

const std::vector<ov::element::Type> squeeze15ModelTypes = {ov::element::f16};

auto squeeze15ParamConfig =
        testing::Combine(::testing::ValuesIn(squeeze15_axes), ::testing::ValuesIn(squeeze15ModelTypes),
                         ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Squeeze15, Squeeze15LayerTestCommon, squeeze15ParamConfig,
                         Squeeze15LayerTestCommon::getTestCaseName);

}  // namespace
