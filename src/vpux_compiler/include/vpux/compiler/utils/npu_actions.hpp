//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/developer_build_utils.hpp"

#include <mlir/IR/Action.h>
#include <mlir/IR/Unit.h>
#include <mlir/Support/TypeID.h>

#include <string>

/** @brief Denotes a particular call-site as an "NPU compiler action".

    This is the main "API" for the user wishing to "mark" a particular call-site
    as being an NPU compiler action. Any NPU compiler action, similarly to any
    other MLIR action (according to an [MLIR action tracing
    framework](https://mlir.llvm.org/docs/ActionTracing/)) can then be observed
    / traced / debugged, etc. according to the action handler registered during
    compilation.

    This macro implicitly takes care of the difference between developer builds
    (when action framework is in effect) and non-developer builds, making the
    cost zero for release builds.
 */
#define NPU_EXECUTE_ACTION(ctx, actionName, irUnits) ::vpux::details::ActionCreator(ctx, irUnits, actionName) << [&]

namespace llvm {
class raw_ostream;
}

namespace vpux {

/** @brief Basic NPU compiler specific action.

    This is an action class that can be used in any NPU compiler specific place.
    As opposed to the MLIR's pre-existing actions, this action covers places
    outside of the framework's focus.
 */
class NpuCompilerAction : public mlir::tracing::ActionImpl<NpuCompilerAction> {
    using Base = mlir::tracing::ActionImpl<NpuCompilerAction>;

    std::string _actionName;

public:
    static constexpr llvm::StringLiteral tag = "npu-compiler-action";

    NpuCompilerAction(llvm::ArrayRef<mlir::IRUnit> irUnits, std::string actionName)
            : Base(irUnits), _actionName(std::move(actionName)) {
    }

    void print(llvm::raw_ostream& os) const override;

    const std::string& getName() const {
        return _actionName;
    }
};

/** @brief Returns a pretty name for the action.

    Returns a nice, human-readable name of the specified action. Supports
    several action types both NPU-specific and "builtin".
 */
std::string getPrettyName(const mlir::tracing::Action* action);

namespace details {
// This is a basic helper class that directly executes a callable passed via
// Type::operator<<().
struct ActionCreatorBase {
    ActionCreatorBase(mlir::MLIRContext*, llvm::ArrayRef<mlir::IRUnit>, const std::string&) {
    }

    template <typename Callable>
    ActionCreatorBase& operator<<(Callable&& callable) && {
        std::forward<Callable>(callable)();
        return *this;
    }
};

// This is the main helper class that wraps a specified callable passed via
// Type::operator<<() into a call to MLIRContext::executeAction(). When an
// action handler is registered for the context, it would see the action
// registered by this class.
struct ActionCreatorWithContextDispatch : ActionCreatorBase {
    mlir::MLIRContext* ctx;
    llvm::ArrayRef<mlir::IRUnit> irUnits;
    std::string actionName;

    ActionCreatorWithContextDispatch(mlir::MLIRContext* ctx, llvm::ArrayRef<mlir::IRUnit> irUnits,
                                     std::string actionName)
            : ActionCreatorBase(ctx, irUnits, actionName),
              ctx(ctx),
              irUnits(irUnits),
              actionName(std::move(actionName)) {
    }

    template <typename Callable>
    ActionCreatorWithContextDispatch& operator<<(Callable&& callable) && {
        ctx->executeAction<NpuCompilerAction>(std::forward<Callable>(callable), irUnits, std::move(actionName));
        return *this;
    }
};

/** @brief A helper class to register new actions conveniently.

    This is the main class that allows one to execute a specified callable that
    is provided via Type::operator<<(). When developer build is enabled, this
    callable is additionally wrapped into a call to
    MLIRContext::executeAction(), thus making it "visible" in the scope of
    Action Tracing framework. For release builds (developer build is disabled),
    the callable is executed directly without any additional runtime overhead.
 */
struct ActionCreator :
        std::conditional_t<vpux::isDeveloperBuild(), ActionCreatorWithContextDispatch, ActionCreatorBase> {
    using Base = std::conditional_t<vpux::isDeveloperBuild(), ActionCreatorWithContextDispatch, ActionCreatorBase>;
    using Base::Base;
    using Base::operator<<;
};
}  // namespace details

}  // namespace vpux

MLIR_DECLARE_EXPLICIT_TYPE_ID(::vpux::NpuCompilerAction)
