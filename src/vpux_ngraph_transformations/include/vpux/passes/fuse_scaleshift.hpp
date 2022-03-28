//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <ngraph/pass/pass.hpp>

namespace vpux {
namespace pass {

class FuseScaleShift final : public ngraph::pass::NodePass {
public:
    bool run_on_node(std::shared_ptr<ngraph::Node> node) override;
};

}  // namespace pass
}  // namespace vpux
