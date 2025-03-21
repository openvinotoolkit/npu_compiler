//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

bool IE::PostOpAttr::isChannelAgnostic() const {
    // TODO: [E#155121] This is a quick and safe solution, but there is a nicer way which requires a bit of refactoring.
    static const std::unordered_map<StringRef, bool> postOps{
            {IE::ReLUOp::getOperationName(), true},      {IE::ClampOp::getOperationName(), true},
            {IE::LeakyReluOp::getOperationName(), true}, {IE::PReluOp::getOperationName(), false},
            {IE::TanhOp::getOperationName(), true},      {IE::SigmoidOp::getOperationName(), true}};

    const auto name = this->getName().strref();
    const auto condition = postOps.find(name);
    VPUX_THROW_WHEN(condition == postOps.end(),
                    "No channel-agnostic condition was defined for post-op: {0}. Ensure the post-op is registered.",
                    name);

    return condition->second;
}
