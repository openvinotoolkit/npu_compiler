//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/roi_pooling.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/data_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/roi_pooling.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

using roiPoolingParamsTupleAddLayout = std::tuple<std::vector<InputShape>,  // Input, coords shapes
                                                  ov::Shape,                // Pooled shape {pooled_h, pooled_w}
                                                  float,                    // Spatial scale
                                                  ov::test::utils::ROIPoolingTypes,  // ROIPooling method
                                                  ov::element::Type,                 // Model type
                                                  ov::Layout,                        // Input layout, newly added
                                                  ov::test::TargetDevice>;           // Device name

class ROIPoolingLayerTestAddLayout :
        public testing::WithParamInterface<roiPoolingParamsTupleAddLayout>,
        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<roiPoolingParamsTupleAddLayout>& obj);

protected:
    void SetUp() override;
};

std::string ROIPoolingLayerTestAddLayout::getTestCaseName(
        const testing::TestParamInfo<roiPoolingParamsTupleAddLayout>& obj) {
    std::vector<InputShape> input_shapes;
    ov::Shape pool_shape;
    float spatial_scale;
    ov::test::utils::ROIPoolingTypes pool_method;
    ov::element::Type model_type;
    ov::Layout order;
    std::string target_device;
    std::tie(input_shapes, pool_shape, spatial_scale, pool_method, model_type, order, target_device) = obj.param;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < input_shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({input_shapes[i].first})
               << (i < input_shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < input_shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < input_shapes.size(); j++) {
            result << ov::test::utils::vec2str(input_shapes[j].second[i]) << (j < input_shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "PS=" << ov::test::utils::vec2str(pool_shape) << "_";
    result << "Scale=" << spatial_scale << "_";
    switch (pool_method) {
    case utils::ROIPoolingTypes::ROI_MAX:
        result << "Max_";
        break;
    case utils::ROIPoolingTypes::ROI_BILINEAR:
        result << "Bilinear_";
        break;
    }
    result << "modelType=" << model_type.to_string() << "_";
    result << "Layout=" << order.to_string() << "_";  // newly added
    result << "trgDev=" << test_utils::TARGET_DEVICE;
    return result.str();
}

void ROIPoolingLayerTestAddLayout::SetUp() {
    std::vector<InputShape> input_shapes;
    ov::Shape pool_shape;
    float spatial_scale;
    ov::test::utils::ROIPoolingTypes pool_method;
    ov::element::Type model_type;
    ov::Layout order;
    std::string target_device;
    std::tie(input_shapes, pool_shape, spatial_scale, pool_method, model_type, order, targetDevice) = this->GetParam();

    abs_threshold = 0.08f;

    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]);
    auto coord_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]);
    std::string pool_method_str;
    if (pool_method == ov::test::utils::ROIPoolingTypes::ROI_MAX) {
        pool_method_str = "max";
    } else if (pool_method == ov::test::utils::ROIPoolingTypes::ROI_BILINEAR) {
        pool_method_str = "bilinear";
    } else {
        FAIL() << "Incorrect type of ROIPooling operation";
    }
    auto roi_pooling =
            std::make_shared<ov::op::v0::ROIPooling>(param, coord_param, pool_shape, spatial_scale, pool_method_str);
    function =
            std::make_shared<ov::Model>(roi_pooling->outputs(), ov::ParameterVector{param, coord_param}, "roi_pooling");

    // enable different layouts
    auto preProc = ov::preprocess::PrePostProcessor(function);
    preProc.input(0).tensor().set_layout(order);
    preProc.input(0).model().set_layout("NCHW");
    preProc.output().tensor().set_layout(order);
    preProc.output().model().set_layout("NCHW");
    function = preProc.build();
}

class ROIPoolingLayerTestCommon : public ROIPoolingLayerTestAddLayout, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        ROIPoolingTypes poolMethod = std::get<3>(GetParam());
        float spatialScale = std::get<2>(GetParam());

        inputs.clear();

        const auto is_roi_max_mode = (poolMethod == ROIPoolingTypes::ROI_MAX);

        const int height = is_roi_max_mode ? targetInputStaticShapes.front()[2] / spatialScale : 1;
        const int width = is_roi_max_mode ? targetInputStaticShapes.front()[3] / spatialScale : 1;

        ov::Layout order = std::get<5>(GetParam());
        std::vector<ov::Shape> inShapes = targetInputStaticShapes;
        if (order == "NHWC") {
            // NCHW -> NHWC
            inShapes[0][1] = targetInputStaticShapes[0][2];
            inShapes[0][2] = targetInputStaticShapes[0][3];
            inShapes[0][3] = targetInputStaticShapes[0][1];
        }

        VpuOv2LayerTest::generate_inputs(inShapes);

        const auto& funcInput = function->input(1);
        ov::Tensor tensor{funcInput.get_element_type(), funcInput.get_shape()};
        fill_data_roi(tensor, targetInputStaticShapes.front()[0] - 1, height, width, 1.0f, is_roi_max_mode);
        if (VpuOv2LayerTest::inputs.find(funcInput.get_node()->shared_from_this()) != VpuOv2LayerTest::inputs.end()) {
            VpuOv2LayerTest::inputs[funcInput.get_node()->shared_from_this()] = tensor;
        }
    }
};

TEST_P(ROIPoolingLayerTestCommon, NPU3720_HW) {
    abs_threshold = 0.25;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ROIPoolingLayerTestCommon, NPU4000_HW) {
    abs_threshold = 0.25;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(ROIPoolingLayerTestCommon, NPU5010_SW) {
    abs_threshold = 0.25;
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

const std::vector<ov::Shape> paramShapes = {{{1, 3, 8, 8}}, {{3, 4, 50, 50}}};

const std::vector<ov::Shape> pooledShapes_max = {{{1, 1}}, {{2, 2}}, {{3, 3}}, {{6, 6}}};

const std::vector<ov::Shape> pooledShapes_bilinear = {/*{{1, 1}},*/ {{2, 2}}, {{3, 3}}, {{6, 6}}};

const std::vector<ov::Shape> coordShapes = {{{1, 5}}, /*{{3, 5}}, {{5, 5}}*/};

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<float> spatial_scales = {0.625f, 1.f};

auto inputShapes = [](const std::vector<ov::Shape>& in1, const std::vector<ov::Shape>& in2) {
    std::vector<std::vector<ov::test::InputShape>> res;
    for (const auto& sh1 : in1) {
        for (const auto& sh2 : in2) {
            res.push_back(ov::test::static_shapes_to_test_representation({sh1, sh2}));
        }
    }
    return res;
}(paramShapes, coordShapes);

const auto test_ROIPooling_max = ::testing::Combine(
        ::testing::ValuesIn(inputShapes), ::testing::ValuesIn(pooledShapes_max), ::testing::ValuesIn(spatial_scales),
        ::testing::Values(ROIPoolingTypes::ROI_MAX), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn({ov::Layout("NCHW"), ov::Layout("NHWC")}), ::testing::Values(test_utils::TARGET_DEVICE));

const auto test_ROIPooling_bilinear = ::testing::Combine(
        ::testing::ValuesIn(inputShapes), ::testing::ValuesIn(pooledShapes_bilinear),
        ::testing::Values(spatial_scales[1]), ::testing::Values(ROIPoolingTypes::ROI_BILINEAR),
        ::testing::ValuesIn(modelTypes), ::testing::ValuesIn({ov::Layout("NCHW"), ov::Layout("NHWC")}),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_max, ROIPoolingLayerTestCommon, test_ROIPooling_max,
                         ROIPoolingLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_bilinear, ROIPoolingLayerTestCommon, test_ROIPooling_bilinear,
                         ROIPoolingLayerTestCommon::getTestCaseName);
