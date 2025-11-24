//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DYNAMIC_REWRITER_PASSES
#define VPUX_COMPILER_DYNAMIC_REWRITER_PASSES

include "mlir/Pass/PassBase.td"

//
// DynamicRewriterExecutor
//

def DynamicRewriterExecutor : PassBase<"run-rewriters", "vpux::FunctionPass"> {
    let summary = "Pass that applies a dynamically specified rewriter";
    let description = [{
        This pass allows running various rewriters without creating separate passes for each.
        The rewriter to use is specified through a command-line option.

        ### Rewriter Hierarchy
        - YourRewriterSection (Same concept as "pass")
          - RewriterClass1
          - RewriterClass2

        ### Registration
        Register new rewriters at run-time:
        ```
        void registerYourRewriters(vpux::RewriterRegistry& registry, args...) {
            registry.registerRewriter<YourRewriterClass>("your-rewriter-name", args);
        }
        ```

        Register new rewriters for testing:
        ```
        void registerYourSection(vpux::RewriterRegistry& registry) {
            registry.registerRewriterSet(
                    "your-section-name",
                    [&](args...) {
                        registerYourRewriters(registry, args...);
                    },
                    args...);
        }
        ```

        ### Usage Examples
        1. Apply a single rewriter:
           ```
           vpux-opt ... --run-rewriters="rewriter=copy-sequence" input.mlir
           ```

        2. Apply multiple rewriters (comma-separated):
           ```
           vpux-opt ... --run-rewriters="rewriter=copy-sequence,subview-copy,cmx-to-cmx-copy" input.mlir
           ```

        3. Apply all rewriters in same section:
           ```
           vpux-opt ... --run-rewriters="rewriter=optimize-copies-set" input.mlir
           ```

        4. Apply all registered rewriters:
           ```
           vpux-opt ... --run-rewriters="rewriter=all" input.mlir
           ```
    }];

    let options = [
        Option<
            "rewriterName", "rewriter",
            "std::string", "",
            "Name of the rewriter to run"
        >
    ];

    let constructor = "vpux::createDynamicRewriterExecutorPass()";

    let dependentDialects = [
        "vpux::VPUIP::VPUIPDialect"
    ];
}

#endif
