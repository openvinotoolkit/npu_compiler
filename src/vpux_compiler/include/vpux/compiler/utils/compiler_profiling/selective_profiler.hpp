//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Regex.h>
#include <mlir/Debug/Observers/ActionProfiler.h>

#include <optional>
#include <string>

namespace vpux::compiler_profiling {
/** @brief This is a base class for profilers with on-demand instrumentation.

    Some profiling environments allow "intrusive" profiling. Such profiling
    allows one to cover only certain sections of code with profiling
    instrumentation, instead of collecting the full program profile. This class
    abstracts away the "selection" criteria for when to turn on the intrusive
    profiling. Users of the class just need to implement the instrumentation
    part.
 */
class SelectiveProfiler : public mlir::tracing::ExecutionContext::Observer {
    std::optional<llvm::Regex> _selection;

public:
    SelectiveProfiler(llvm::StringRef selection);

    void beforeExecute(const mlir::tracing::ActionActiveStack* action, mlir::tracing::Breakpoint* breakpoint,
                       bool willExecute) override;
    void afterExecute(const mlir::tracing::ActionActiveStack* action) override;

private:
    /** @brief This method is called before the action is executed.

        This method can be used to turn on the profiling "right before" the
        associated action is to be executed.

        The depth parameter indicates the "depth" of the current action with
        respect to the position of the action that was initially selected
        according to the profiler's criteria. For example:
        ```
        root action                         // not interesting - ignored
        |- action1                          // interesting - selected; depth = 0
            |- nested_action_lvl1           // already selected; depth = 1
                |- nested_action_lvl2       // already selected; depth = 2
            |- nested_action_lvl1_other     // already selected; depth = 1
        |- action2                          // not interesting - ignored
        ```
        Thus, one can differentiate the action on which the instrumentation
        "started" vs any "nested" action on which the instrumentation is already
        turned on but some additional highlighting is desired.

        @note The associated action is guaranteed to be executed if this method
        is called.
     */
    virtual void profileBeforeExecute(const mlir::tracing::Action* action, size_t depth) = 0;

    /** @brief This method is called after the action is executed.

        This method can be used to turn off the profiling "right after" the
        associated action has been executed.

        The depth parameter indicates the "depth" of the current action with
        respect to the position of the action that was initially selected (see
        profileBeforeExecute() for details). In the case of "after execution",
        the depth parameter can be used to understand how to stop
        instrumentation: completely, partially, etc.
     */
    virtual void profileAfterExecute(const mlir::tracing::Action* action, size_t depth) = 0;
};

}  // namespace vpux::compiler_profiling
