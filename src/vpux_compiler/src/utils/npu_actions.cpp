//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/npu_actions.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Pass/Pass.h>

namespace vpux {
void NpuCompilerAction::print(llvm::raw_ostream& os) const {
    os << getName();
}

std::string getPrettyName(const mlir::tracing::Action* action) {
    return llvm::TypeSwitch<const mlir::tracing::Action*, std::string>(action)
            .Case([&](const mlir::PassExecutionAction* passAction) {
                const auto& pass = passAction->getPass();
                return pass.getName().str();
            })
            .Case([&](const vpux::NpuCompilerAction* npuAction) {
                return npuAction->getName();
            })
            .Default([&](const mlir::tracing::Action* unknown) {
                std::string actionName;
                llvm::raw_string_ostream stream(actionName);
                unknown->print(stream);
                return actionName;
            });
}

}  // namespace vpux

MLIR_DEFINE_EXPLICIT_TYPE_ID(::vpux::NpuCompilerAction)
