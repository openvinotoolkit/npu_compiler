//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <limits>
#include <random>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/softmax.hpp"

namespace ov::test {

class AddSoftmaxInnerMaskAwareTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<ov::Shape> {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(targetInputStaticShapes.size() == funcInputs.size(),
                        "Input shapes number does not match with inputs number");
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            const auto elemType = funcInput.get_element_type();
            ov::Tensor tensor;

            if (i == 0) {
                const auto& shape = targetInputStaticShapes[i];
                ov::test::utils::InputGenerateData in_data;
                const auto cols = shape.back();
                in_data.start_from = 0;
                in_data.range = 32 * 2;         // at fp16::lowest() consecutive number are at 32 unit distance.
                in_data.resolution = cols / 4;  // Reduce quantization error when values are processed in FP16.
                tensor = ov::test::utils::create_and_fill_tensor(elemType, targetInputStaticShapes[i], in_data);
            } else {
                // Mask input (boolean): true = masked, false = unmasked
                const auto& shape = targetInputStaticShapes[i];
                tensor = ov::Tensor{elemType, shape};
                const auto cols = shape.back();
                const auto totalSize = ov::shape_size(shape);
                const auto numRows = totalSize / cols;

                auto* data = tensor.data<bool>();
                std::mt19937 gen(42);
                std::uniform_int_distribution<int> rowKindDist(0, 2);
                std::uniform_int_distribution<size_t> colPosDist(0, cols > 1 ? cols - 2 : 0);
                constexpr size_t chunkSize = 8;
                int kind = 0;
                int processedChunks = 0;
                for (size_t row = 0; row < numRows; ++row) {
                    bool* rowPtr = data + row * cols;
                    if ((numRows >= 32) && (processedChunks <= 2)) {
                        // Assign kind per chunk of 8 rows; all rows in a chunk share the same kind
                        if (row % chunkSize == 0) {
                            kind = rowKindDist(gen);
                            if (kind != 0) {
                                kind = rowKindDist(gen);
                            }
                            processedChunks++;
                        }
                    } else if (numRows == 1) {
                        // Single row: force all masked to test mask-aware softmax main path
                        kind = 0;
                    } else {
                        kind = rowKindDist(gen);
                    }
                    if (kind == 0) {
                        // All elements set to true (masked)
                        std::fill(rowPtr, rowPtr + cols, true);
                    } else if (kind == 1) {
                        // Partial: elements before a random split point are true, rest are false
                        const size_t splitPos = colPosDist(gen) + 1;
                        std::fill(rowPtr, rowPtr + splitPos, true);
                        std::fill(rowPtr + splitPos, rowPtr + cols, false);
                    } else {
                        // All false — no masked values on this row
                        std::fill(rowPtr, rowPtr + cols, false);
                    }
                }
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] =
                "enable-softmax-mask-aware=true softmax-mask-aware-threshold=-65104";
    }
    void SetUp() override {
        inType = ov::element::f32;
        outType = ov::element::f16;
        ov::Shape inputShape = GetParam();
        const auto axis = inputShape.size() - 1;

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape, inputShape}));

        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));
        const auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, inputDynamicShapes.at(1));
        const ov::Shape constShape({1});
        std::vector<float> constValuesMin(1, std::numeric_limits<float>::lowest());
        std::vector<float> constValuesZero(1, 0.0f);
        const auto constTensorMin = ov::op::v0::Constant::create(inType, constShape, constValuesMin);
        const auto constTensorZero = ov::op::v0::Constant::create(inType, constShape, constValuesZero);
        const auto select = std::make_shared<ov::op::v1::Select>(mask, constTensorMin, constTensorZero,
                                                                 ov::op::AutoBroadcastType::NUMPY);
        const auto add = std::make_shared<ov::op::v1::Add>(input, select);
        const auto softMax = std::make_shared<ov::op::v8::Softmax>(add, axis);
        const auto convert = std::make_shared<ov::op::v0::Convert>(softMax, ov::element::f16);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(convert)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input, mask}, "AddSoftmaxInnerMaskAware");
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::Shape>& obj) {
        ov::Shape inputShape = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "In1=" << inputShape << sep;
        return result.str();
    };
};

TEST_P(AddSoftmaxInnerMaskAwareTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
TEST_P(AddSoftmaxInnerMaskAwareTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<ov::Shape> inputShapesHW = {
        {1, 1, 1, 32},  {1, 1, 8, 32},  {1, 1, 16, 32}, {1, 1, 32, 32},  {1, 1, 64, 32},  {1, 1, 65, 32},
        {1, 1, 18, 18}, {1, 1, 67, 18}, {1, 1, 1, 16},  {1, 1, 1, 224},  {1, 1, 1, 225},  {1, 16, 5, 5},
        {1, 1, 5, 192}, {1, 1, 17, 5},  {1, 1, 16, 8},  {1, 1, 240, 33}, {1, 1, 241, 257}};

INSTANTIATE_TEST_SUITE_P(smoke_AddSoftmaxInnerMaskAware, AddSoftmaxInnerMaskAwareTestCommon,
                         ::testing::ValuesIn(inputShapesHW), AddSoftmaxInnerMaskAwareTestCommon::getTestCaseName);

const std::vector<ov::Shape> inputShapes8Lines = {{1, 1, 16, 64},  {1, 1, 1, 63},   {1, 1, 8, 65},   {1, 1, 16, 68},
                                                  {1, 1, 32, 127}, {1, 1, 64, 128}, {1, 1, 65, 129}, {1, 1, 18, 255},
                                                  {1, 1, 67, 257}, {1, 1, 19, 192}};
INSTANTIATE_TEST_SUITE_P(smoke_AddSoftmaxInnerMaskAwareBatch8, AddSoftmaxInnerMaskAwareTestCommon,
                         ::testing::ValuesIn(inputShapes8Lines), AddSoftmaxInnerMaskAwareTestCommon::getTestCaseName);
}  // namespace ov::test
