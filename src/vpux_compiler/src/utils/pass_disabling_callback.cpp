//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/pass_disabling_callback.hpp"
#include "vpux/compiler/utils/npu_action_handler.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassInstrumentation.h>

using namespace llvm;
using namespace vpux;

namespace {
class Breakpoint final : public mlir::tracing::BreakpointBase<Breakpoint> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Breakpoint)

    void print(raw_ostream& os) const override {
        os << "PassDisablingCallback breakpoint";
    }
};

class BreakpointManager final : public mlir::tracing::BreakpointManagerBase<BreakpointManager> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BreakpointManager)

    mlir::tracing::Breakpoint* match(const mlir::tracing::Action& action) const override {
        return mlir::isa<mlir::PassExecutionAction>(&action) ? &_breakpoint : nullptr;
    }

private:
    mutable Breakpoint _breakpoint;
};

std::shared_ptr<Regex> makeRegex(StringRef pattern) {
    if (pattern.empty()) {
        return nullptr;
    }

    std::string error;
    auto regex = std::make_shared<Regex>(pattern);
    VPUX_THROW_UNLESS(regex->isValid(error), "Invalid regular expression '{0}' : {1}", pattern, error);
    return regex;
}
}  // namespace

PassDisablingCallback::PassDisablingCallback(StringRef disabledPasses): _disabledPasses(makeRegex(disabledPasses)) {
}

mlir::tracing::ExecutionContext::Control PassDisablingCallback::operator()(
        const mlir::tracing::ActionActiveStack* actionStack) const {
    const auto passAction = mlir::dyn_cast<mlir::PassExecutionAction>(&actionStack->getAction());
    if (passAction == nullptr || _disabledPasses == nullptr) {
        return NpuActionHandler::Control::Apply;
    }

    const auto passId = passAction->getPass().getArgument();
    const auto passName = passAction->getPass().getName();

    const auto disabled = _disabledPasses->match(passId) || _disabledPasses->match(passName);
    return disabled ? NpuActionHandler::Control::Skip : NpuActionHandler::Control::Apply;
}

std::unique_ptr<mlir::tracing::BreakpointManager> PassDisablingCallback::createBreakpointManager() {
    return std::make_unique<BreakpointManager>();
}
