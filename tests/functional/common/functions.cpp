//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functions.h"
#include "common/npu_test_env_cfg.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

std::shared_ptr<ov::Model> buildSingleLayerSoftMaxNetwork() {
    ov::Shape inputShape = {1, 3, 4, 3};
    ov::element::Type model_type = ov::element::f32;
    size_t axis = 1;

    const ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape({inputShape}))};
    params.at(0)->set_friendly_name("Parameter");

    const auto softMax = std::make_shared<ov::op::v1::Softmax>(params.at(0), axis);
    softMax->set_friendly_name("softMax");

    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(softMax)};
    results.at(0)->set_friendly_name("Result");

    auto ov_model = std::make_shared<ov::Model>(results, params, "softMax");

    return ov_model;
}

std::shared_ptr<ov::Model> buildSingleWsFriendlyNetwork(ov::element::Type_t inType, const ov::Shape& inputShape) {
    const auto input1 = std::make_shared<ov::op::v0::Parameter>(inType, inputShape);
    const auto input2 = std::make_shared<ov::op::v0::Parameter>(inType, inputShape);

    size_t totalElements = 1;
    for (const auto dim : inputShape) {
        totalElements *= dim;
    }
    std::vector<float> data(totalElements);
    for (size_t i = 0; i < totalElements; ++i) {
        data[i] = static_cast<float>(i + 1);
    }

    // Note: two separate constant nodes with the same data - this is to
    // make use of the model compression.
    const auto const1 = ov::op::v0::Constant::create(inType, inputShape, data);
    const auto multiply1 = std::make_shared<ov::op::v1::Multiply>(input1, const1);

    const auto const2 = ov::op::v0::Constant::create(inType, inputShape, data);
    const auto multiply2 = std::make_shared<ov::op::v1::Multiply>(input2, const2);

    const auto add = std::make_shared<ov::op::v1::Add>(multiply1, multiply2);

    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{input1, input2},
                                       "WsSameBufferConstantsSubGraphTest");
}

std::shared_ptr<ov::Model> createModelWithLargeSize() {
    auto data = std::make_shared<ov::opset11::Parameter>(ov::element::f16, ov::Shape{4000, 4000});
    auto mul_constant = ov::opset11::Constant::create(ov::element::f16, ov::Shape{1}, {1.5});
    auto mul = std::make_shared<ov::opset11::Multiply>(data, mul_constant);
    auto add_constant = ov::opset11::Constant::create(ov::element::f16, ov::Shape{1}, {0.5});
    auto add = std::make_shared<ov::opset11::Add>(mul, add_constant);
    // Just a sample model here, large iteration to make the model large
    for (int i = 0; i < 1000; i++) {
        add = std::make_shared<ov::opset11::Add>(add, add_constant);
    }
    auto res = std::make_shared<ov::opset11::Result>(add);

    /// Create the OpenVINO model
    return std::make_shared<ov::Model>(ov::ResultVector{std::move(res)}, ov::ParameterVector{std::move(data)});
}

const std::string PlatformEnvironment::PLATFORM = []() -> std::string {
    const auto& var = ov::test::utils::VpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM;
    if (!var.empty()) {
        return var;
    } else {
        std::cerr << "Environment variable is not set: IE_NPU_TESTS_PLATFORM! Exiting..." << std::endl;
        exit(-1);
    }
}();
