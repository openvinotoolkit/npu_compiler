
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/preprocess.hpp"
#include <vpu_ov2_layer_test.hpp>
#include "common/utils.hpp"
#include "common_test_utils/subgraph_builders/preprocess_builders.hpp"

#include "openvino/op/relu.hpp"

using namespace ov::test;
using namespace ov::test::utils;
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
    auto function = create_preprocess_1input(ov::element::f16, ov::Shape{1, 3, 24, 24});
    auto p = PrePostProcessor(function);
    p.input().preprocess().convert_element_type(ov::element::f32).mean(0.2f).convert_element_type(ov::element::f16);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> tensor_element_type_and_mean() {
    auto function = create_preprocess_1input(ov::element::f16, ov::Shape{1, 3, 12, 12});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_element_type(ov::element::f32);
    p.input().preprocess().mean(0.1f).convert_element_type(ov::element::f16);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> custom_preprocessing() {
    auto function = create_preprocess_1input(ov::element::f32, ov::Shape{3, 4, 10, 20});
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
    auto function = create_preprocess_1input(ov::element::f32, ov::Shape{1, 3, 3, 3});
    auto p = PrePostProcessor(function);
    auto p1 = std::move(p);
    p = std::move(p1);
    p.input().tensor().set_element_type(ov::element::f32).set_layout("?CHW");
    p.input()
            .preprocess()
            // .mean(1.f) // // [Tracking number: E#75246] - Calling mean twice causes validation errors.
            .scale(2.f)
            .mean({1.1f, 2.2f, 3.3f})
            .scale({2.f, 3.f, 4.f})
            .custom([](const ov::Output<ov::Node>& node) {
                auto abs = std::make_shared<ov::op::v0::Abs>(node);
                abs->set_friendly_name(node.get_node_shared_ptr()->get_friendly_name() + "/abs");
                return abs;
            });

    p.input().preprocess().convert_element_type(ov::element::f16);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> resize_linear_nchw() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 3, 10, 10});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_static_shape(20, 20);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> resize_linear_nhwc() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 10, 10, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_static_shape(20, 20);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NHWC");
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> resize_linear_nchw_model_and_tensor() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 3, 10, 10});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_static_shape(20, 20);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NCHW");
    p.input().tensor().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> resize_linear_nhwc_model_and_tensor() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 10, 10, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_static_shape(20, 20);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NHWC");
    p.input().tensor().set_layout("NHWC");
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> resize_cubic_nchw() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 3, 10, 10});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_static_shape(20, 20);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_CUBIC);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> resize_cubic_nhwc() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 10, 10, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_static_shape(20, 20);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_CUBIC);
    p.input().model().set_layout("NHWC");
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> resize_nearest_nchw() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 3, 10, 10});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_static_shape(20, 20);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_NEAREST);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> resize_nearest_nhwc() {
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 10, 10, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_spatial_static_shape(20, 20);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_NEAREST);
    p.input().model().set_layout("NHWC");
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
            .convert_element_type(ov::element::f32)
            .convert_color(ColorFormat::RGB)
            .convert_layout()
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

inline std::shared_ptr<ov::Model> cvt_color_rgb_to_gray() {
    auto function = create_preprocess_1input(ov::element::f32, {{1, 20, 20, 1}});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::RGB);
    p.input().preprocess().convert_color(ColorFormat::GRAY);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> cvt_color_bgr_to_gray() {
    auto function = create_preprocess_1input(ov::element::f32, {{1, 20, 20, 1}});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::BGR);
    p.input().preprocess().convert_color(ColorFormat::GRAY);
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

inline std::shared_ptr<ov::Model> crop_basic() {
    auto function = create_preprocess_1input(ov::element::f32, ov::Shape{1, 3, 10, 10});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_shape({1, 3, 40, 40});
    p.input().preprocess().crop({0, 0, 5, 10}, {1, 3, 15, 20});
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> m2i_scaling_test(ColorFormat fmt, ov::element::Type modelInType, bool isPlanar) {
    ov::Dimension::value_type inW = 256, inH = 192;
    ov::Dimension::value_type netW = 224, netH = 168;

    // the consuming 'model'
    ov::PartialShape modelInShape;
    if (isPlanar) {
        modelInShape = {1, 3, netH, netW};  // NCHW
    } else {
        modelInShape = {1, netH, netW, 3};  // NHWC
    }
    auto function = create_dummy_model_1input(modelInType, modelInShape);
    auto p = PrePostProcessor(function);

    p.input()
            .tensor()
            .set_color_format(fmt)
            .set_element_type(modelInType)
            .set_layout(isPlanar ? "NCHW" : "NHWC")
            .set_spatial_static_shape(inH, inW);

    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);

    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> m2i_single_plane_test(ColorFormat iFmt, ColorFormat oFmt,
                                                        ov::element::Type modelInType, bool normalize, bool planarOut) {
    ov::Dimension::value_type inW = 256, inH = 192;
    ov::Dimension::value_type netW = 224, netH = 168;

    // the consuming 'model'
    ov::PartialShape modelInShape;
    if (planarOut) {
        modelInShape = {1, 3, netH, netW};  // NCHW
    } else {
        modelInShape = {1, netH, netW, 3};  // NHWC
    }
    auto function = create_dummy_model_1input(modelInType, modelInShape);
    auto p = PrePostProcessor(function);

    p.input()
            .tensor()
            .set_color_format(iFmt)
            .set_element_type(ov::element::u8)  // U8-only NV12/I420 for M2I
            .set_layout("NHWC")                 // not mandatory
            .set_spatial_static_shape(inH, inW);

    p.input().preprocess().convert_color(oFmt).convert_element_type(modelInType).resize(ResizeAlgorithm::RESIZE_LINEAR);

    if (normalize) {
        p.input().preprocess().mean({1.0, 2.0, 3.0}).scale({4.0, 5.0, 6.0});
    }

    if (planarOut) {
        p.input().model().set_layout("NCHW");
    } else {
        p.input().model().set_layout("NHWC");
    }

    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> m2i_basic_test(ColorFormat iFmt, ColorFormat oFmt, ov::element::Type modelInType) {
    ov::Dimension::value_type netW = 224, netH = 168;

    // the consuming 'model'
    auto function = create_dummy_model_1input(modelInType, {1, 3, netH, netW});  // NCHW
    auto p = PrePostProcessor(function);

    p.input()
            .tensor()
            .set_color_format(iFmt)
            .set_element_type(ov::element::u8)  // U8-only NV12/I420 for M2I
            .set_layout("NHWC")                 // not mandatory
            .set_spatial_static_shape(netH, netW);

    p.input().preprocess().convert_color(oFmt).convert_element_type(modelInType);

    p.input().model().set_layout("NCHW");

    function = p.build();
    return function;
}

// csc, [convert], resize, normalise, RGB-PLANAR (u8/fp16) output:
inline std::shared_ptr<ov::Model> m2i_csc_scl_NV12_norm_to_PL_RGB_f16() {
    return m2i_single_plane_test(ColorFormat::NV12_SINGLE_PLANE, ColorFormat::RGB, ov::element::f16, true, true);
}
inline std::shared_ptr<ov::Model> m2i_csc_scl_NV12_norm_to_PL_BGR_f16() {
    return m2i_single_plane_test(ColorFormat::NV12_SINGLE_PLANE, ColorFormat::BGR, ov::element::f16, true, true);
}

inline std::shared_ptr<ov::Model> m2i_csc_scl_I420_norm_to_PL_RGB_f16() {
    return m2i_single_plane_test(ColorFormat::I420_SINGLE_PLANE, ColorFormat::RGB, ov::element::f16, true, true);
}
inline std::shared_ptr<ov::Model> m2i_csc_scl_I420_norm_to_PL_BGR_f16() {
    return m2i_single_plane_test(ColorFormat::I420_SINGLE_PLANE, ColorFormat::BGR, ov::element::f16, true, true);
}

// csc, [convert], resize, RGB-PLANAR (u8/fp16) output:
inline std::shared_ptr<ov::Model> m2i_csc_scl_NV12_to_PL_RGB_f16() {
    return m2i_single_plane_test(ColorFormat::NV12_SINGLE_PLANE, ColorFormat::RGB, ov::element::f16, false, true);
}
inline std::shared_ptr<ov::Model> m2i_csc_scl_NV12_to_PL_BGR_f16() {
    return m2i_single_plane_test(ColorFormat::NV12_SINGLE_PLANE, ColorFormat::BGR, ov::element::f16, false, true);
}

inline std::shared_ptr<ov::Model> m2i_csc_scl_I420_to_PL_RGB_f16() {
    return m2i_single_plane_test(ColorFormat::I420_SINGLE_PLANE, ColorFormat::RGB, ov::element::f16, false, true);
}
inline std::shared_ptr<ov::Model> m2i_csc_scl_I420_to_PL_BGR_f16() {
    return m2i_single_plane_test(ColorFormat::I420_SINGLE_PLANE, ColorFormat::BGR, ov::element::f16, false, true);
}

inline std::shared_ptr<ov::Model> m2i_csc_scl_NV12_to_PL_RGB_ui8() {
    return m2i_single_plane_test(ColorFormat::NV12_SINGLE_PLANE, ColorFormat::RGB, ov::element::u8, false, true);
}
inline std::shared_ptr<ov::Model> m2i_csc_scl_NV12_to_PL_BGR_ui8() {
    return m2i_single_plane_test(ColorFormat::NV12_SINGLE_PLANE, ColorFormat::BGR, ov::element::u8, false, true);
}

inline std::shared_ptr<ov::Model> m2i_csc_scl_I420_to_PL_RGB_ui8() {
    return m2i_single_plane_test(ColorFormat::I420_SINGLE_PLANE, ColorFormat::RGB, ov::element::u8, false, true);
}
inline std::shared_ptr<ov::Model> m2i_csc_scl_I420_to_PL_BGR_ui8() {
    return m2i_single_plane_test(ColorFormat::I420_SINGLE_PLANE, ColorFormat::BGR, ov::element::u8, false, true);
}

// resize (interleaved), permute RGB-PLANAR (u8/fp16) output, normalise:
inline std::shared_ptr<ov::Model> m2i_scl_RGB_norm_to_PL_RGB_f16() {
    return m2i_single_plane_test(ColorFormat::RGB, ColorFormat::RGB, ov::element::f16, true, true);
}

// csc, resize, RGB-INTERLEAVED (u8) output
inline std::shared_ptr<ov::Model> m2i_csc_scl_I420_to_IL_RGB_ui8() {
    return m2i_single_plane_test(ColorFormat::I420_SINGLE_PLANE, ColorFormat::RGB, ov::element::u8, false, false);
}
inline std::shared_ptr<ov::Model> m2i_csc_scl_I420_to_IL_BGR_ui8() {
    return m2i_single_plane_test(ColorFormat::I420_SINGLE_PLANE, ColorFormat::BGR, ov::element::u8, false, false);
}
inline std::shared_ptr<ov::Model> m2i_csc_scl_NV12_to_IL_RGB_ui8() {
    return m2i_single_plane_test(ColorFormat::NV12_SINGLE_PLANE, ColorFormat::RGB, ov::element::u8, false, false);
}
inline std::shared_ptr<ov::Model> m2i_csc_scl_NV12_to_IL_BGR_ui8() {
    return m2i_single_plane_test(ColorFormat::NV12_SINGLE_PLANE, ColorFormat::BGR, ov::element::u8, false, false);
}

// Resize only
inline std::shared_ptr<ov::Model> m2i_scl_PL_RGB_f16() {
    return m2i_scaling_test(ColorFormat::RGB, ov::element::f16, true);
}
inline std::shared_ptr<ov::Model> m2i_scl_PL_RGB_u8() {
    return m2i_scaling_test(ColorFormat::RGB, ov::element::u8, true);
}
inline std::shared_ptr<ov::Model> m2i_scl_IL_RGB_u8() {
    return m2i_scaling_test(ColorFormat::RGB, ov::element::u8, false);
}

// Basic CSC tests (no Resize)
inline std::shared_ptr<ov::Model> m2i_csc_NV12_to_PL_RGB_ui8() {
    return m2i_basic_test(ColorFormat::NV12_SINGLE_PLANE, ColorFormat::RGB, ov::element::u8);
}
inline std::shared_ptr<ov::Model> m2i_csc_NV12_to_PL_BGR_ui8() {
    return m2i_basic_test(ColorFormat::NV12_SINGLE_PLANE, ColorFormat::BGR, ov::element::u8);
}
inline std::shared_ptr<ov::Model> m2i_csc_I420_to_PL_RGB_ui8() {
    return m2i_basic_test(ColorFormat::I420_SINGLE_PLANE, ColorFormat::RGB, ov::element::u8);
}
inline std::shared_ptr<ov::Model> m2i_csc_I420_to_PL_BGR_ui8() {
    return m2i_basic_test(ColorFormat::I420_SINGLE_PLANE, ColorFormat::BGR, ov::element::u8);
}

inline std::shared_ptr<ov::Model> m2i_csc_NV12_to_PL_RGB_f16() {
    return m2i_basic_test(ColorFormat::NV12_SINGLE_PLANE, ColorFormat::RGB, ov::element::f16);
}
inline std::shared_ptr<ov::Model> m2i_csc_NV12_to_PL_BGR_f16() {
    return m2i_basic_test(ColorFormat::NV12_SINGLE_PLANE, ColorFormat::BGR, ov::element::f16);
}
inline std::shared_ptr<ov::Model> m2i_csc_I420_to_PL_RGB_f16() {
    return m2i_basic_test(ColorFormat::I420_SINGLE_PLANE, ColorFormat::RGB, ov::element::f16);
}
inline std::shared_ptr<ov::Model> m2i_csc_I420_to_PL_BGR_f16() {
    return m2i_basic_test(ColorFormat::I420_SINGLE_PLANE, ColorFormat::BGR, ov::element::f16);
}

inline std::vector<ov::builder::preprocess::preprocess_func> preprocess_functions() {
    return std::vector<ov::builder::preprocess::preprocess_func>{
            ov::builder::preprocess::preprocess_func(scale_only, "scale_only", 0.01f),
            ov::builder::preprocess::preprocess_func(scale_mean, "scale_mean", 0.01f),
            ov::builder::preprocess::preprocess_func(scale_vector, "scale_vector", 0.01f),

            ov::builder::preprocess::preprocess_func(crop_basic, "crop_basic", 1.f),

            ov::builder::preprocess::preprocess_func(resize_linear_nchw, "resize_linear_nchw", 0.01f),
            ov::builder::preprocess::preprocess_func(resize_linear_nhwc, "resize_linear_nhwc", 0.01f),
            ov::builder::preprocess::preprocess_func(resize_linear_nchw_model_and_tensor,
                                                     "resize_linear_nchw_model_and_tensor", 0.01f),
            ov::builder::preprocess::preprocess_func(resize_linear_nhwc_model_and_tensor,
                                                     "resize_linear_nhwc_model_and_tensor", 0.01f),
            ov::builder::preprocess::preprocess_func(resize_cubic_nchw, "resize_cubic_nchw", 0.01f),
            ov::builder::preprocess::preprocess_func(resize_cubic_nhwc, "resize_cubic_nhwc", 0.01f),
            ov::builder::preprocess::preprocess_func(resize_nearest_nchw, "resize_nearest_nchw",
                                                     0.01f),  // [Tracking number: E#74951] - Validation error
            ov::builder::preprocess::preprocess_func(resize_nearest_nhwc, "resize_nearest_nhwc",
                                                     0.01f),  // [Tracking number: E#74951] - Validation errorr

            ov::builder::preprocess::preprocess_func(convert_layout_by_dims, "convert_layout_by_dims", 0.01f),
            ov::builder::preprocess::preprocess_func(convert_layout_hwc_to_nchw, "convert_layout_hwc_to_nchw", 0.01f),
            ov::builder::preprocess::preprocess_func(convert_element_type_and_mean, "convert_element_type_and_mean",
                                                     0.01f),
            ov::builder::preprocess::preprocess_func(tensor_element_type_and_mean, "tensor_element_type_and_mean",
                                                     0.01f),

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
            ov::builder::preprocess::preprocess_func(cvt_color_nv12_cvt_layout_resize,
                                                     "cvt_color_nv12_cvt_layout_resize", 1.f),
            ov::builder::preprocess::preprocess_func(cvt_color_rgb_to_gray, "cvt_color_rgb_to_gray", 1.f),
            ov::builder::preprocess::preprocess_func(cvt_color_bgr_to_gray, "cvt_color_bgr_to_gray", 1.f),

            ov::builder::preprocess::preprocess_func(custom_preprocessing, "custom_preprocessing", 0.01f),
            ov::builder::preprocess::preprocess_func(multiple_ops, "multiple_ops", 0.01f),

            // [Tracking number: E#75247]
            // error: C++ exception with description "get_shape was called on a descriptor::Tensor with dynamic shape"
            // thrown in SetUp().
            // ov::builder::preprocess::preprocess_func(resize_dynamic, "resize_dynamic", 0.01f),
    };
}

inline std::vector<ov::builder::preprocess::preprocess_func> preprocess_functions_m2i_u8_single_op() {
    return std::vector<ov::builder::preprocess::preprocess_func>{
            // resize-only:
            ov::builder::preprocess::preprocess_func(m2i_scl_PL_RGB_u8, "m2i_scl_PL_RGB_u8", 1.001f),
            ov::builder::preprocess::preprocess_func(m2i_scl_IL_RGB_u8, "m2i_scl_IL_RGB_u8", 1.001f),

            // csc-only (no resize)
            ov::builder::preprocess::preprocess_func(m2i_csc_NV12_to_PL_RGB_ui8, "m2i_csc_NV12_to_PL_RGB_ui8", 1.f),
            ov::builder::preprocess::preprocess_func(m2i_csc_NV12_to_PL_BGR_ui8, "m2i_csc_NV12_to_PL_BGR_ui8", 1.f),
            ov::builder::preprocess::preprocess_func(m2i_csc_I420_to_PL_RGB_ui8, "m2i_csc_I420_to_PL_RGB_ui8", 1.f),
            ov::builder::preprocess::preprocess_func(m2i_csc_I420_to_PL_BGR_ui8, "m2i_csc_I420_to_PL_BGR_ui8", 1.f),
    };
}

inline std::vector<ov::builder::preprocess::preprocess_func> preprocess_functions_m2i_u8_fused_op() {
    return std::vector<ov::builder::preprocess::preprocess_func>{
            // csc, [convert], resize, RGB-PLANAR (u8/fp16) output
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_NV12_to_PL_RGB_ui8, "m2i_csc_scl_NV12_to_PL_RGB_ui8",
                                                     1.f),
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_NV12_to_PL_BGR_ui8, "m2i_csc_scl_NV12_to_PL_BGR_ui8",
                                                     1.f),
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_I420_to_PL_RGB_ui8, "m2i_csc_scl_I420_to_PL_RGB_ui8",
                                                     1.f),
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_I420_to_PL_BGR_ui8, "m2i_csc_scl_I420_to_PL_BGR_ui8",
                                                     1.f),

            // csc, resize, RGB-INTERLEAVED (u8) output
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_NV12_to_IL_RGB_ui8, "m2i_csc_scl_NV12_to_IL_RGB_ui8",
                                                     1.f),
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_NV12_to_IL_BGR_ui8, "m2i_csc_scl_NV12_to_IL_BGR_ui8",
                                                     1.f),
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_I420_to_IL_RGB_ui8, "m2i_csc_scl_I420_to_IL_RGB_ui8",
                                                     1.f),
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_I420_to_IL_BGR_ui8, "m2i_csc_scl_I420_to_IL_BGR_ui8",
                                                     1.f),
    };
}

inline std::vector<ov::builder::preprocess::preprocess_func> preprocess_functions_m2i_f16_no_csc() {
    return std::vector<ov::builder::preprocess::preprocess_func>{
            // resize (interleaved), permute RGB-PLANAR (u8/fp16) output normalise:
            ov::builder::preprocess::preprocess_func(m2i_scl_RGB_norm_to_PL_RGB_f16, "m2i_scl_RGB_norm_to_PL_RGB_f16",
                                                     0.015f),

            // resize-only:
            ov::builder::preprocess::preprocess_func(m2i_scl_PL_RGB_f16, "m2i_scl_PL_RGB_f16", 0.016f),
    };
}

inline std::vector<ov::builder::preprocess::preprocess_func> preprocess_functions_m2i_f16_csc() {
    return std::vector<ov::builder::preprocess::preprocess_func>{
            // csc, [convert], resize, normalise, RGB-PLANAR (u8/fp16) output:
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_NV12_norm_to_PL_RGB_f16,
                                                     "m2i_csc_scl_NV12_norm_to_PL_RGB_f16", 0.016f),
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_NV12_norm_to_PL_BGR_f16,
                                                     "m2i_csc_scl_NV12_norm_to_PL_BGR_f16", 0.016f),
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_I420_norm_to_PL_RGB_f16,
                                                     "m2i_csc_scl_I420_norm_to_PL_RGB_f16", 0.016f),
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_I420_norm_to_PL_BGR_f16,
                                                     "m2i_csc_scl_I420_norm_to_PL_BGR_f16", 0.016f),

            // csc, [convert], resize, RGB-PLANAR (u8/fp16) output
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_NV12_to_PL_RGB_f16, "m2i_csc_scl_NV12_to_PL_RGB_f16",
                                                     0.016f),
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_NV12_to_PL_BGR_f16, "m2i_csc_scl_NV12_to_PL_BGR_f16",
                                                     0.016f),
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_I420_to_PL_RGB_f16, "m2i_csc_scl_I420_to_PL_RGB_f16",
                                                     0.016f),
            ov::builder::preprocess::preprocess_func(m2i_csc_scl_I420_to_PL_BGR_f16, "m2i_csc_scl_I420_to_PL_BGR_f16",
                                                     0.016f),

            // csc-only (no resize)
            ov::builder::preprocess::preprocess_func(m2i_csc_NV12_to_PL_RGB_f16, "m2i_csc_NV12_to_PL_RGB_f16", 0.016f),
            ov::builder::preprocess::preprocess_func(m2i_csc_NV12_to_PL_BGR_f16, "m2i_csc_NV12_to_PL_BGR_f16", 0.016f),
            ov::builder::preprocess::preprocess_func(m2i_csc_I420_to_PL_RGB_f16, "m2i_csc_I420_to_PL_RGB_f16", 0.016f),
            ov::builder::preprocess::preprocess_func(m2i_csc_I420_to_PL_BGR_f16, "m2i_csc_I420_to_PL_BGR_f16", 0.016f),
    };
}

class PreProcessTestCommon : virtual public PrePostProcessTest, virtual public VpuOv2LayerTest {
public:
    void SetUp() override {
        PrePostProcessTest::SetUp();
    }

    static std::string getTestCaseName(const testing::TestParamInfo<preprocessParamsTuple>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << PrePostProcessTest::getTestCaseName(obj) << sep;

        return result.str();
    }

protected:
    std::map<std::string, std::string> config;
};

TEST_P(PreProcessTestCommon, NPU3720_HW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto test_type = std::get<0>(GetParam());
        if (test_type.m_name == "resize_nearest_nchw" || test_type.m_name == "resize_nearest_nhwc") {
            skip << "[Tracking number: E#74951] - Resize nearest is currently giving an incorrect output";
        }
    });
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

class PreProcessTest_M2I : public PreProcessTestCommon {
public:
    void SetUp() override {
        // TODO: E129229
        configuration[ov::intel_npu::compilation_mode_params.name()] =
                "enable-m2i=true workload-management-enable=false";
        PreProcessTestCommon::SetUp();
    }
};

// u8 single operations should be withing 1 LSB absolute difference wrt openVino
class PreProcessTest_M2I_u8_single_op : public PreProcessTest_M2I {};

TEST_P(PreProcessTest_M2I_u8_single_op, NPU4000_HW) {
    abs_threshold = 1.001;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

// When composing multiple u8 operations also the errors get composed, so the absolute difference increases (test passes
// with an abs_threshold of 5). An additional reason is the different order in which M2I HW processes operations.
// [Tracking number: E#103855]
class PreProcessTest_M2I_u8_fused_op : public PreProcessTest_M2I {};
TEST_P(PreProcessTest_M2I_u8_fused_op, NPU4000_HW) {
    abs_threshold = 1.001;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

// operations with f16 output should only use the relative threshold
class PreProcessTest_M2I_f16_no_csc : public PreProcessTest_M2I {};

TEST_P(PreProcessTest_M2I_f16_no_csc, NPU4000_HW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto test_type = std::get<0>(GetParam());
        if (test_type.m_name == "m2i_scl_RGB_norm_to_PL_RGB_f16") {
            skip << "[Tracking number: E#103858] - High relative error outliers in fused resize + norm f16 test";
        }
    });
    abs_threshold = 1.001;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

// operations with f16 output that contain also a csc operation (single or fused) can't match with openVino at the
// moment. The reason is a difference between usage of clamping in openVino and M2I-HW as described in E#101592
// [Tracking number: E#101592]
class PreProcessTest_M2I_f16_csc : public PreProcessTest_M2I {};
TEST_P(PreProcessTest_M2I_f16_csc, NPU4000_HW) {
    abs_threshold = 1.001;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

INSTANTIATE_TEST_SUITE_P(smoke_precommit_PrePostProcess, PreProcessTestCommon,
                         ::testing::Combine(::testing::ValuesIn(preprocess_functions()),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         PreProcessTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PrePostProcess_M2I_u8_single_op, PreProcessTest_M2I_u8_single_op,
                         ::testing::Combine(::testing::ValuesIn(preprocess_functions_m2i_u8_single_op()),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         PreProcessTest_M2I_u8_single_op::getTestCaseName);

// [Tracking number: E#103855]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_PrePostProcess_M2I_u8_fused_op, PreProcessTest_M2I_u8_fused_op,
                         ::testing::Combine(::testing::ValuesIn(preprocess_functions_m2i_u8_fused_op()),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         PreProcessTest_M2I_u8_fused_op::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PrePostProcess_M2I_f16_no_csc, PreProcessTest_M2I_f16_no_csc,
                         ::testing::Combine(::testing::ValuesIn(preprocess_functions_m2i_f16_no_csc()),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         PreProcessTest_M2I_f16_no_csc::getTestCaseName);

// [Tracking number: E#101592]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_PrePostProcess_M2I_f16_csc, PreProcessTest_M2I_f16_csc,
                         ::testing::Combine(::testing::ValuesIn(preprocess_functions_m2i_f16_csc()),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         PreProcessTest_M2I_f16_csc::getTestCaseName);
