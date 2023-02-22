// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/preprocess.hpp"
#include <vpux_layer_test.hpp>
#include "ngraph_functions/preprocess/preprocess_builders.hpp"

using namespace ov::preprocess;

inline std::shared_ptr<ov::Model> create_preprocess_1input(ov::element::Type type, const ov::PartialShape& shape) {
    auto data1 = std::make_shared<ov::op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->output(0).get_tensor().set_names({"input1"});
    std::shared_ptr<ov::op::v0::Result> res;
    auto op1 = std::make_shared<ov::op::v0::Relu>(data1);
    res = std::make_shared<ov::op::v0::Result>(op1);
    res->set_friendly_name("Result1");
    res->output(0).get_tensor().set_names({"Result1"});
    return std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{data1});
}

inline std::shared_ptr<ov::Model> create_dummy_model_1input(ov::element::Type type, const ov::PartialShape& shape) {
    auto data1 = std::make_shared<ov::op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->output(0).get_tensor().set_names({"input1"});
    std::shared_ptr<ov::op::v0::Result> res;
    // (inType == outType) => will be optimized out
    auto op1 = std::make_shared<ov::op::v0::Convert>(data1, type);
    res = std::make_shared<ov::op::v0::Result>(op1);
    res->set_friendly_name("Result1");
    res->output(0).get_tensor().set_names({"Result1"});
    return std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{data1});
}

inline std::shared_ptr<ov::Model> create_preprocess_2inputs(ov::element::Type type, const ov::PartialShape& shape) {
    auto data1 = std::make_shared<ov::op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->output(0).get_tensor().set_names({"input1"});
    auto data2 = std::make_shared<ov::op::v0::Parameter>(type, shape);
    data2->set_friendly_name("input2");
    data2->output(0).get_tensor().set_names({"input2"});
    std::shared_ptr<ov::op::v0::Result> res1, res2;
    auto op1 = std::make_shared<ov::op::v0::Relu>(data1);
    auto op2 = std::make_shared<ov::op::v0::Relu>(data2);
    res1 = std::make_shared<ov::op::v0::Result>(op1);
    res2 = std::make_shared<ov::op::v0::Result>(op2);

    res1->set_friendly_name("Result1");
    res1->output(0).get_tensor().set_names({"Result1"});
    res2->set_friendly_name("Result2");
    res2->output(0).get_tensor().set_names({"Result2"});
    return std::make_shared<ov::Model>(ov::ResultVector{res1, res2}, ov::ParameterVector{data1, data2});
}

inline std::shared_ptr<ov::Model> scale_only() {
    auto function = create_preprocess_1input(ov::element::f32, ov::Shape{1, 3, 24, 24});
    auto p = PrePostProcessor(function);
    p.input().preprocess().scale(2.1f);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> scale_mean() {
    auto function = create_preprocess_1input(ov::element::f32, ov::Shape{1, 3, 24, 24});
    auto p = PrePostProcessor(function);
    p.input().preprocess().scale(2.1f).mean(1.1f);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> scale_vector() {
    auto function = create_preprocess_1input(ov::element::f32, ov::Shape{1, 3, 24, 24});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_layout("NCHW");
    p.input().preprocess().scale({2.2f, 3.3f, 4.4f});
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> convert_element_type_and_mean() {
    auto function = create_preprocess_1input(ov::element::u8, ov::Shape{1, 3, 24, 24});
    auto p = PrePostProcessor(function);
    p.input().preprocess().convert_element_type(ov::element::f32).mean(0.2f).convert_element_type(ov::element::u8);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> tensor_element_type_and_mean() {
    auto function = create_preprocess_1input(ov::element::u8, ov::Shape{1, 3, 12, 12});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_element_type(ov::element::f32);
    p.input().preprocess().mean(0.1f).convert_element_type(ov::element::u8);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> custom_preprocessing() {
    auto function = create_preprocess_1input(ov::element::i32, ov::Shape{3, 4, 10, 20});
    auto p = PrePostProcessor(function);
    p.input().preprocess().custom([](const ov::Output<ov::Node>& node) {
        auto abs = std::make_shared<ov::op::v0::Abs>(node);
        abs->set_friendly_name(node.get_node_shared_ptr()->get_friendly_name() + "/abs");
        return abs;
    });
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> multiple_ops() {
    auto function = create_preprocess_1input(ov::element::u8, ov::Shape{1, 3, 3, 3});
    auto p = PrePostProcessor(function);
    auto p1 = std::move(p);
    p = std::move(p1);
    p.input().tensor().set_element_type(ov::element::f32).set_layout("?CHW");
    p.input()
            .preprocess()
            .mean(1.f)
            .scale(2.f)
            .mean({1.1f, 2.2f, 3.3f})
            .scale({2.f, 3.f, 4.f})
            .custom([](const ov::Output<ov::Node>& node) {
                auto abs = std::make_shared<ov::op::v0::Abs>(node);
                abs->set_friendly_name(node.get_node_shared_ptr()->get_friendly_name() + "/abs");
                return abs;
            });
    p.input().preprocess().convert_element_type(ov::element::u8);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> resize_linear() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 3, 10, 10});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_static_shape(20, 20);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> resize_nearest() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 3, 10, 10});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_static_shape(20, 20);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_NEAREST);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> convert_layout_by_dims() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 30, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().preprocess().convert_layout({0, 3, 1, 2});
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> convert_layout_hwc_to_nchw() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 3, 30, 20});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_layout("HWC").set_element_type(ov::element::u8);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> cvt_color_nv12_cvt_layout_resize() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 3, 10, 10});
    auto p = PrePostProcessor(function);
    p.input()
            .tensor()
            .set_color_format(ColorFormat::NV12_TWO_PLANES)
            .set_element_type(ov::element::u8)
            .set_spatial_static_shape(20, 20);
    p.input()
            .preprocess()
            .convert_color(ColorFormat::RGB)
            .convert_layout()
            .convert_element_type(ov::element::f32)
            .resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> cvt_color_bgrx_to_bgr() {
    auto function = create_preprocess_2inputs(ov::element::f32, ov::PartialShape{1, 160, 160, 3});
    auto p = PrePostProcessor(function);
    p.input(0).tensor().set_color_format(ColorFormat::BGRX);
    p.input(0).preprocess().convert_color(ColorFormat::BGR);
    p.input(1).tensor().set_color_format(ColorFormat::RGBX);
    p.input(1).preprocess().convert_color(ColorFormat::BGR);
    return p.build();
}

inline std::shared_ptr<ov::Model> resize_dynamic() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 3, 20, 20});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_dynamic_shape();
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> cvt_color_nv12_to_rgb_single_plane() {
    auto function = create_preprocess_1input(ov::element::f32, {{1, 20, 20, 3}});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::NV12_SINGLE_PLANE);
    p.input().preprocess().convert_color(ColorFormat::RGB);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> cvt_color_nv12_to_bgr_single_planes() {
    auto function = create_preprocess_1input(ov::element::f32, {{1, 20, 20, 3}});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::NV12_SINGLE_PLANE);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> cvt_color_nv12_to_bgr_two_planes() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 20, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> cvt_color_nv12_to_rgb_two_planes() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 20, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES);
    p.input().preprocess().convert_color(ColorFormat::RGB);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> cvt_color_i420_to_rgb_single_plane() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 20, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::I420_SINGLE_PLANE);
    p.input().preprocess().convert_color(ColorFormat::RGB);
    return p.build();
}

inline std::shared_ptr<ov::Model> cvt_color_i420_to_bgr_three_planes() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 20, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::I420_THREE_PLANES);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    return p.build();
}

inline std::vector<ov::builder::preprocess::preprocess_func> preprocess_functions() {
    return std::vector<ov::builder::preprocess::preprocess_func>{
            ov::builder::preprocess::preprocess_func(scale_only, "scale_only", 0.01f),
            ov::builder::preprocess::preprocess_func(scale_mean, "scale_mean", 0.01f),
            ov::builder::preprocess::preprocess_func(scale_vector, "scale_vector", 0.01f),
            ov::builder::preprocess::preprocess_func(resize_linear, "resize_linear", 0.01f),
            ov::builder::preprocess::preprocess_func(resize_nearest, "resize_nearest", 0.01f),
            ov::builder::preprocess::preprocess_func(convert_layout_by_dims, "convert_layout_by_dims", 0.01f),
            ov::builder::preprocess::preprocess_func(convert_layout_hwc_to_nchw, "convert_layout_hwc_to_nchw", 0.01f),
            ov::builder::preprocess::preprocess_func(cvt_color_bgrx_to_bgr, "cvt_color_bgrx_to_bgr", 0.01f),
            ov::builder::preprocess::preprocess_func(cvt_color_nv12_to_rgb_single_plane,
                                                     "cvt_color_nv12_to_rgb_single_plane", 1.f),
            ov::builder::preprocess::preprocess_func(cvt_color_nv12_to_bgr_single_planes,
                                                     "cvt_color_nv12_to_bgr_single_planes", 1.f),
            ov::builder::preprocess::preprocess_func(cvt_color_nv12_to_bgr_two_planes,
                                                     "cvt_color_nv12_to_bgr_two_planes", 1.f),
            ov::builder::preprocess::preprocess_func(cvt_color_nv12_to_rgb_two_planes,
                                                     "cvt_color_nv12_to_rgb_two_planes", 1.f),
            ov::builder::preprocess::preprocess_func(cvt_color_i420_to_rgb_single_plane,
                                                     "cvt_color_i420_to_rgb_single_plane", 1.f),
            ov::builder::preprocess::preprocess_func(cvt_color_i420_to_bgr_three_planes,
                                                     "cvt_color_i420_to_bgr_three_planes", 1.f),
    };
}

using namespace SubgraphTestsDefinitions;

class VPUXPreProcessTest : virtual public PrePostProcessTest, virtual public VPUXLayerTestsUtils::VPUXLayerTestsCommon {
public:
    void SetUp() override {
        PrePostProcessTest::SetUp();
    }

protected:
    std::map<std::string, std::string> config;
};

TEST_P(VPUXPreProcessTest, CompareWithRefs) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    run();
}

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_PrePostProcess, VPUXPreProcessTest,
                         ::testing::Combine(::testing::ValuesIn(preprocess_functions()),
                                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                         VPUXPreProcessTest::getTestCaseName);
