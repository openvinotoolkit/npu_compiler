//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common/print_test_case_name.hpp>
#include <common/random_generator.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/opsets/opset3_decl.hpp>
#include <openvino/opsets/opset4_decl.hpp>
#include <pretty_test_arguments.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/scatter_nd_update.hpp"

namespace {

PRETTY_PARAM(DataShape, ov::test::InputShape);
PRETTY_PARAM(IndicesShape, ov::test::InputShape);
PRETTY_PARAM(InputType, ov::element::Type);
PRETTY_PARAM(IndicesType, ov::element::Type);

using ScatterNDUpdateParams = std::tuple<std::tuple<DataShape, IndicesShape>, InputType, IndicesType>;

}  // namespace

namespace ov::test {

ov::Tensor generateIndices(const ov::Shape& dataShape, const ov::Shape& indicesShape, ov::element::Type_t type) {
    auto tensor = ov::Tensor{type, indicesShape};
    const auto data = tensor.data<int32_t>();
    const auto indexSize = indicesShape.back();

    VPUX_THROW_UNLESS(dataShape.size() <= indexSize, "Provided index size '{0}' is greater than data tensor size '{1}'",
                      indexSize, dataShape.size());

    auto bounds = std::vector<int32_t>(dataShape.begin(), dataShape.begin() + indexSize);

    const auto incrementIndex = [](auto& index, const auto& bounds) {
        for (int32_t i = index.size() - 1; i >= 0; i--) {
            if (index[i] + 1 >= bounds[i]) {
                index[i] = 0;
            } else {
                index[i]++;
                return;
            }
        }
    };

    const auto numElements = ov::shape_size(dataShape);

    // generate all possible indices
    auto allIndices = std::vector<int32_t>(numElements * indexSize);
    auto index = std::vector<int32_t>(indexSize);
    for (auto i = 0; i < numElements; i++) {
        for (auto j = 0; j < index.size(); j++) {
            allIndices[i * index.size() + j] = index[j];
        }
        incrementIndex(index, bounds);
    }

    // randomly take only the required number of indices
    auto requiredIndices = std::vector<int32_t>(numElements);
    std::iota(requiredIndices.begin(), requiredIndices.end(), 0);
    std::shuffle(requiredIndices.begin(), requiredIndices.end(), RandomGenerator::get());

    const auto requiredNumIndices =
            std::accumulate(indicesShape.begin(), indicesShape.end() - 1, int32_t{1}, std::multiplies<>{});
    requiredIndices.resize(requiredNumIndices);
    std::sort(requiredIndices.begin(), requiredIndices.end());

    auto outputPtr = data;
    for (auto i : requiredIndices) {
        const auto index = &allIndices[i * indexSize];
        for (int j = 0; j < indexSize; j++) {
            outputPtr[j] = index[j];
        }
        outputPtr += indexSize;
    }

    return tensor;
}

ov::PartialShape getUpdatesPartialShape(const ov::PartialShape& data, const ov::PartialShape& indices) {
    VPUX_THROW_WHEN(indices.size() == 0, "Indices partial shape is empty");
    const auto indexSizeDimension = *std::prev(indices.end());
    const auto indexSize = indexSizeDimension.get_length();

    auto updatesShape = std::vector<ov::Dimension>(indices.begin(), std::prev(indices.end()));
    updatesShape.insert(updatesShape.end(), std::next(data.begin(), indexSize), data.end());

    return ov::PartialShape(updatesShape);
};

ov::Shape getUpdatesStaticShape(const ov::Shape& data, const ov::Shape& indices) {
    VPUX_THROW_WHEN(indices.size() == 0, "Indices static shape is empty");
    const auto indexSize = *std::prev(indices.end());

    auto updatesShape = ov::Shape{indices.begin(), std::prev(indices.end())};
    updatesShape.insert(updatesShape.end(), std::next(data.begin(), indexSize), data.end());

    return updatesShape;
};

class ScatterNDUpdateCustomLayerTest :
        public testing::WithParamInterface<ScatterNDUpdateParams>,
        public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = function->inputs();
        VPUX_THROW_UNLESS(funcInputs.size() == 3, "Expected to have 3 inputs, got '{0}'", funcInputs.size());

        auto& rg = RandomGenerator::get();
        auto distribution = std::uniform_int_distribution<int>(0, std::numeric_limits<int>::max());

        auto inputTensors = std::array<ov::Tensor, 3>();
        inputTensors[0] = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(), targetShapes[0], 10,
                                                                  0, 1000, distribution(rg));
        inputTensors[1] = generateIndices(targetShapes[0], targetShapes[1], funcInputs[1].get_element_type());
        inputTensors[2] = ov::test::utils::create_and_fill_tensor(funcInputs[2].get_element_type(), targetShapes[2], 10,
                                                                  0, 1000, distribution(rg));

        for (const auto& [funcInput, tensor] : vpux::zip(funcInputs, inputTensors)) {
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        const auto& [inputShapes, inputType, indicesType] = this->GetParam();

        const auto data = std::get<DataShape>(inputShapes).value();
        const auto indices = std::get<IndicesShape>(inputShapes).value();

        const auto appendUpdatesShape = [](const std::vector<ov::Shape>& staticInputsShapes) -> std::vector<ov::Shape> {
            VPUX_THROW_UNLESS(staticInputsShapes.size() == 2, "Unexpected number of inputs");
            const auto& dataShape = staticInputsShapes[0];
            const auto& indicesShape = staticInputsShapes[1];
            auto updatesShape = getUpdatesStaticShape(staticInputsShapes[0], staticInputsShapes[1]);

            return {dataShape, indicesShape, updatesShape};
        };

        llvm::transform(ov::test::utils::combineStaticShapes({data, indices}), std::back_inserter(targetStaticShapes),
                        appendUpdatesShape);

        auto updatesStaticShapes = std::vector<ov::Shape>();
        for (const auto& inputShapes : targetStaticShapes) {
            updatesStaticShapes.push_back(inputShapes.back());
        }

        const auto updatesNetworkShape = getUpdatesPartialShape(data.first, indices.first);
        const auto updates = ov::test::InputShape{updatesNetworkShape, updatesStaticShapes};

        inputDynamicShapes = {ov::test::utils::getBoundedShape(data), ov::test::utils::getBoundedShape(indices),
                              ov::test::utils::getBoundedShape(updates)};

        auto param = std::make_shared<ov::op::v0::Parameter>(inputType.value(), inputDynamicShapes.at(0));
        auto indicesParam = std::make_shared<ov::op::v0::Parameter>(indicesType.value(), inputDynamicShapes.at(1));
        auto updateParam = std::make_shared<ov::op::v0::Parameter>(inputType.value(), inputDynamicShapes.at(2));

        auto scatterNDUpdate = std::make_shared<ov::opset4::ScatterNDUpdate>(param, indicesParam, updateParam);

        auto results = ov::ResultVector();
        for (size_t i = 0; i < scatterNDUpdate->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::opset3::Result>(scatterNDUpdate->output(i)));
        }

        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param, indicesParam, updateParam},
                                               "DynamicScatterNDUpdate");
    }
};

TEST_P(ScatterNDUpdateCustomLayerTest, NPU3720_HW) {
    abs_threshold = std::numeric_limits<float>::epsilon();
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

const std::vector<InputType> inputType = {ov::element::i32, ov::element::f16};
const std::vector<IndicesType> indicesType = {ov::element::i32};

const std::vector<std::tuple<DataShape, IndicesShape>> inputShapes = {
        std::make_tuple(DataShape(generateTestShape(1, 3, 3)), IndicesShape({{-1, -1, -1, 3}, {{1, 2, 3, 3}}})),
        std::make_tuple(DataShape(generateTestShape(1, 20)), IndicesShape({{-1, 2}, {{20, 2}}})),
};

INSTANTIATE_TEST_SUITE_P(smoke_precommit, ScatterNDUpdateCustomLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(inputType),
                                            ::testing::ValuesIn(indicesType)),
                         PrintTestCaseName());

}  // namespace ov::test
