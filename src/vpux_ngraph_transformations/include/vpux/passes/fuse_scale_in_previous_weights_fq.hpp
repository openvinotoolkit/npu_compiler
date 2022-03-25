//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <ngraph/pass/pass.hpp>

namespace vpux {
namespace pass {

class FuseScaleAfterClamp final : public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

}  // namespace pass
}  // namespace vpux
