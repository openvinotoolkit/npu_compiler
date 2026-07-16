//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/utils/abstract_tree.hpp"
#include "vpux/compiler/utils/ir_modification.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"

// This file contains weights-separation specific utilities that are related to
// IR traversal and modification.

namespace vpux::VPU {

using CallChainData = std::pair<mlir::func::CallOp, mlir::func::FuncOp>;
using CallChainTree = utils::AbstractTree<CallChainData>;

/** @brief Returns a "call chain" tree constructed from the starting function.

    Returns a weights-separation-specific tree that represents the outlining
    structure. An example of such a tree is:
    ```
    |- {nullptr, main}
       |- {"call foo1", foo1}
          |- {"call foo2", foo2}
       |- {"call foo3", foo3}
    ```
    where "call fooX" is a CallOp operation inside the respective function and
    fooX is a standalone function produced by the outlining.

    @note This tree is the basic data structure used by weights separation to
    construct init and main schedules.
*/
CallChainTree getOutliningRepresentation(mlir::func::FuncOp startFunc);

/** @brief A callable that tells whether a particular FuncOp was already
           visited.
*/
class FuncOpVisitor {
    mlir::DenseSet<mlir::func::FuncOp> _cache;

public:
    // Returns whether the function was already seen.
    bool operator()(mlir::func::FuncOp op) {
        const bool firstOccurrence = _cache.insert(op).second;
        return !firstOccurrence;
    }
};

/// Weights-separation specific argument cache.
using WsArgumentCache = vpux::utils::ArgumentCache<vpux::VPU::ConstArg>;

//! @brief Builds IR for the main function and any outlined functions.
class MainFunctionUpdater : public VPU::CallChainTree::Visitor {
protected:
    Logger _log;
    mlir::DenseMap<mlir::func::FuncOp, WsArgumentCache> _argCaches;
    VPU::FuncOpVisitor _hasSeenThisFunction;
    VPU::IsWorthyToCollect _isWorthy;

    // Appends new function arguments of callee to the caller's arguments.
    // During weights separation, a function's inner constants become inputs.
    // This happens across the call-chain and thus a caller must forward
    // callee's arguments from itself. This function would ensure that new
    // arguments of the callee appear in the caller's arguments.
    virtual void hoistCalleeArgsToCaller(mlir::func::FuncOp callerOp, mlir::func::FuncOp calleeOp);

    // Updates the call-site according to the callee's modified arguments.
    // During weights separation, a function's inner constants become inputs.
    // This helper function would set up the call-site inside the caller to
    // correctly propagate arguments from the caller to the callee.
    virtual void fixCallSite(mlir::OpBuilder& callerBuilder, mlir::func::FuncOp callerOp, mlir::func::FuncOp calleeOp,
                             mlir::func::CallOp oldCall);

    WsArgumentCache& getNonConstArgCache(mlir::func::FuncOp funcOp);

public:
    MainFunctionUpdater(const Logger& log, mlir::ModuleOp moduleOp, VPU::IsWorthyToCollect isWorthy);

    /** @brief Converts the tail transformations of every suitable constant into
               IR form.

        Collects suitable constants within the given function and converts all
        of the "trivial" transformations found at the end of the transformation
        list into respective VPU operations. Adds new arguments that map to the
        original weights on which head transformations are applied.

        For instance,
        ```
        %cst = const.Declare tensor<2x1xf16> = dense_resource<...> : tensor<2x2xf16>,
            [#const.Add<42.0>, #const.SubView<[0, 1], [2, 1]>]
             ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                  head                    tail
        ```
        where head transformation is "non-trivial" and becomes part of the
        "init" schedule (done separately to this process), whereas tail
        transformation is "trivial" and thus becomes VPU.Slice in the current
        function. The input argument with type tensor<2x2xf16> is assumed to be
        "transformed" that is, at runtime '42.0' is already added to the value
        before it gets into this function.

        @note This is where the main algorithm of weights separation happens
        with respect to the "main" modification.
     */
    bool visit(const Node& node) override;

    /** @brief Finishes the conversion of the tail transformations.

        Runs necessary post-modification steps to ensure valid IR for the
        current function and any of its callees. Updates all call-sites at the
        current function level by propagating new arguments from the caller
        downwards. This includes both "unique" arguments only found in the
        callees, as well as any "non-unique" arguments that can also be used by
        the current function.

        For instance,
        ```
        func.func private @bar(%shared_arg: tensor<2x2xf16>, %unique_arg: tensor<2x2xf16>)
                -> tensor<2x2xf16> {
            %add = VPU.Add(%shared_arg, %unique_arg) ...
            return %add : tensor<2x2xf16>
        }

        func.func @foo(%real_input: tensor<2x2xf16>, %cst1: tensor<2x2xf16>, %cst2: tensor<2x2xf16>)
                -> tensor<2x2xf16> {
            %mult = VPU.Multiply(%real_input, %cst1) ...
            %call = func.call @bar(%cst1, %cst2)
            %out = VPU.Add(%mult, %call) ...

            // %cst1 is "shared" as it's used by both @foo and @bar;
            // %cst2 is "unique" for @bar, @foo only forwards it down

            return %out : tensor<2x2xf16>
        }
        ```
     */
    void endVisit(const Node& node) override;

    WsArgumentCache takeArgCache(mlir::func::FuncOp funcOp);
};

}  // namespace vpux::VPU
