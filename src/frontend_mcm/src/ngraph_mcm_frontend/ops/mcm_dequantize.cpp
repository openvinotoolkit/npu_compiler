//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

// clang-format off
#ifdef ENABLE_MCM_COMPILER

#include "ngraph_mcm_frontend/ops/mcm_dequantize.hpp"
#include <memory>

const ngraph::NodeTypeInfo McmDequantize::type_info {"McmDequantize", 0};

McmDequantize::McmDequantize(
        const ngraph::Output<ngraph::Node>& input,
        const ngraph::Output<ngraph::Node>& scale,
        const ngraph::Output<ngraph::Node>& zero_point,
        const ngraph::element::Type& type)
            : Op({input, scale, zero_point}), _type(type) {
    constructor_validate_and_infer_types();
}

void McmDequantize::validate_and_infer_types() {
    const size_t SCALE = 1;
    const size_t ZERO_POINT = 2;

    NODE_VALIDATION_CHECK(this, _type.is_static(), "Output element type must not be dynamic");
    NODE_VALIDATION_CHECK(this, _type.is_real(), "Output element type (", _type, ") must be a floating point type");

    auto resultShape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(
        this,
        ngraph::PartialShape::broadcast_merge_into(
            resultShape,
            get_input_partial_shape(SCALE),
            ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY)),
        "Argument shapes are inconsistent.");
    NODE_VALIDATION_CHECK(this,
        ngraph::PartialShape::broadcast_merge_into(
            resultShape,
            get_input_partial_shape(ZERO_POINT),
            ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY)),
        "Argument shapes are inconsistent.");

    set_output_type(0, _type, resultShape);
}

std::shared_ptr<ngraph::Node> McmDequantize::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<McmDequantize>(new_args.at(0), new_args.at(1), new_args.at(2), _type);
}

#endif
// clang-format on
