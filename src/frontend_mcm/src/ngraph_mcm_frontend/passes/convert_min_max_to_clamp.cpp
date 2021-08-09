//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

// clang-format on

#include "ngraph_mcm_frontend/passes/convert_min_max_to_clamp.hpp"

#include <memory>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/maximum.hpp>
#include <ngraph/op/minimum.hpp>
#include "ngraph/op/clamp.hpp"

static bool convert_to_clamp(std::shared_ptr<ngraph::Node> min_node, std::shared_ptr<ngraph::Node> max_node,
                             bool min_first) {
    auto max_constant =
            std::dynamic_pointer_cast<ngraph::op::v0::Constant>(max_node->input_value(1).get_node_shared_ptr());
    auto min_constant =
            std::dynamic_pointer_cast<ngraph::op::v0::Constant>(min_node->input_value(1).get_node_shared_ptr());
    if ((max_constant != nullptr) && (min_constant != nullptr)) {
        auto max_value = max_constant->cast_vector<double>();
        auto min_value = min_constant->cast_vector<double>();
        if ((max_value.size() == 1) && (min_value.size() == 1)) {
            const auto clamp = std::make_shared<ngraph::op::v0::Clamp>(
                    min_first ? min_node->input_value(0) : max_node->input_value(0), max_value[0], min_value[0]);
            ngraph::replace_node(min_first ? max_node : min_node, clamp);
            return true;
        }
    }
    return false;
}

// convert min -> max pattern to clamp
bool ConvertMinMaxToClamp::run_on_node(std::shared_ptr<ngraph::Node> node) {
    auto min_node = std::dynamic_pointer_cast<ngraph::op::v1::Minimum>(node);
    auto max_node = std::dynamic_pointer_cast<ngraph::op::v1::Maximum>(node);
    if (min_node != nullptr) {
        auto pre_node = min_node->input_value(0).get_node_shared_ptr();
        auto pre_max_node = std::dynamic_pointer_cast<ngraph::op::v1::Maximum>(pre_node);
        if (pre_max_node != nullptr)
            return convert_to_clamp(min_node, pre_max_node, false);
    } else if (max_node != nullptr) {
        auto pre_node = max_node->input_value(0).get_node_shared_ptr();
        auto pre_min_node = std::dynamic_pointer_cast<ngraph::op::v1::Minimum>(pre_node);
        if (pre_min_node != nullptr)
            return convert_to_clamp(pre_min_node, max_node, true);
    }

    return false;
}
