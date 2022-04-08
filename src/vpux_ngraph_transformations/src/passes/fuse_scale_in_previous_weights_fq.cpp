//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// clang-format off

#include <ie_common.h>
#include "vpux/passes/fuse_scale_in_previous_weights_fq.hpp"
#include <ngraph/op/constant.hpp>

#include "vpux/quantization_helpers.hpp"

#include <ngraph/op/fake_quantize.hpp>
#include <ngraph/op/clamp.hpp>
#include <legacy/ngraph_ops/scaleshift.hpp>
#include <legacy/ngraph_ops/power.hpp>
#include <vector>
#include <memory>
#include <limits>
#include <legacy/ngraph_ops/convolution_ie.hpp>

namespace vpux {
namespace pass {

/*
 * Pass provide the logic for fusing multiply to FQ_w params
 *
 *  FQ_in   FQ_w
 *   \        /
 *      Conv
 *       |
 *      Add
 *       |
 *     Clamp
 *       |
 *    Multiply
 */

bool FuseScaleAfterClamp::run_on_function(std::shared_ptr<ngraph::Function> f) {
    auto ops = f->get_ordered_ops();
    bool status = false;
    for (const auto& op : ops) {
        if (op->get_type_info() == ngraph::op::PowerIE::get_type_info_static()) {
            auto power_node = std::dynamic_pointer_cast<ngraph::op::PowerIE>(op);
            if (!power_node || power_node->power != 1)
                continue;

            auto node = power_node->input_value(0).get_node_shared_ptr();
            auto clamp_node = std::dynamic_pointer_cast<ngraph::op::v0::Clamp>(node);
            if (clamp_node == nullptr)
                continue;

            auto convolution_node = std::dynamic_pointer_cast<ngraph::op::ConvolutionIE>(clamp_node->input_value(0).get_node_shared_ptr());
            if (convolution_node == nullptr)
                continue;

            const auto weights_fq_node = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(convolution_node->input_value(1).get_node_shared_ptr());
            if (weights_fq_node == nullptr)
                continue;

            auto weights_fq_node3 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(weights_fq_node->input_value(3).get_node_shared_ptr());
            auto weights_fq_node4 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(weights_fq_node->input_value(4).get_node_shared_ptr());
            if (weights_fq_node3 == nullptr || weights_fq_node4 == nullptr)
                continue;

            auto weights_fq_data3 = weights_fq_node3->cast_vector<double>();
            auto weights_fq_data4 = weights_fq_node4->cast_vector<double>();

            auto convolution_biases_node = std::dynamic_pointer_cast<ngraph::op::Constant>(convolution_node->input_value(2).get_node_shared_ptr());
            if (convolution_biases_node == nullptr)
                continue;

            auto convolution_biases_data = (convolution_biases_node)->cast_vector<double>();
            auto scale = power_node->scale;

            const size_t OC = convolution_biases_data.size();
            std::vector<double> new_weights_fq_ol(OC);
            std::vector<double> new_weights_fq_oh(OC);
            for (size_t oc = 0; oc < OC; ++oc) {
                convolution_biases_data[oc] *= scale;
                new_weights_fq_ol[oc] = weights_fq_data3[std::min(weights_fq_data3.size()-1, oc)] * scale;
                new_weights_fq_oh[oc] = weights_fq_data4[std::min(weights_fq_data4.size()-1, oc)] * scale;
            }

            auto new_convolution_biases_node = std::make_shared<ngraph::op::v0::Constant>(ngraph::element::f64, convolution_biases_node->get_shape(), convolution_biases_data.data());
            ngraph::replace_node(convolution_biases_node, new_convolution_biases_node);
            ngraph::replace_node(weights_fq_node3,
                                 std::make_shared<ngraph::op::v0::Constant>(ngraph::element::f64, ngraph::Shape({new_weights_fq_ol.size(),1,1,1}), new_weights_fq_ol.data()));
            ngraph::replace_node(weights_fq_node4,
                                 std::make_shared<ngraph::op::v0::Constant>(ngraph::element::f64, ngraph::Shape({new_weights_fq_oh.size(),1,1,1}), new_weights_fq_oh.data()));
            status = replace_output_update_name(power_node->output(0), power_node->input_value(0));
            IE_ASSERT(status == true);

            auto new_clamp_node = std::make_shared<ngraph::op::v0::Clamp>(clamp_node->input_value(0),
                                                                          clamp_node->get_min() * scale,
                                                                          clamp_node->get_max() * scale);
            ngraph::replace_node(clamp_node, new_clamp_node);
        }
    }

    return status;
}

}  // namespace pass
}  // namespace vpux
