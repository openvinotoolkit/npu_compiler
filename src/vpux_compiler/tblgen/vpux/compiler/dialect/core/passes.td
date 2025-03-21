//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_CORE_PASSES
#define VPUX_COMPILER_CORE_PASSES

include "mlir/Pass/PassBase.td"

//
// MoveDeclarationsToTop
//

def MoveDeclarationsToTop : PassBase<"move-declarations-to-top", "vpux::FunctionPass"> {
    let summary = "Move all declaration ops to top of parent block";

    let constructor = "vpux::Core::createMoveDeclarationsToTopPass()";
}

//
// PrintDot
//

def PrintDot : PassBase<"print-dot", "vpux::ModulePass"> {
    let summary = "Convert current MLIR graph to Dot graph";

    let description = [{
        Convert current MLIR graph to Dot graph.
    }];

    let options = [
        Option<
            "printOnlyDotInterfaceOpt", "print-only-dot-interfaces",
            "bool", "",
            "Print only ops that have dotInterface implementations"
        >,

        Option<
            "printDeclarationsOpt", "print-declarations",
            "bool", "",
            "Print declare memory operations"
        >,
        Option<
            "printConstOpt", "print-const",
            "bool", "",
            "Print const declare operations"
        >,
        Option<
            "printOnlyTaskAndBarrierOpt", "print-only-task-and-barrier",
            "bool", "",
            "Print only task and barrier operations"
        >,
        Option<
            "printOnlyAsyncExecOpt", "print-only-async-exec",
            "bool", "",
            "Print only async.exec operations"
        >,
        Option<
            "startAfterOpt", "start-after",
            "std::string", "",
            "Start graph generation after this operation"
        >,
        Option<
            "stopBeforeOpt", "stop-before",
            "std::string", "",
            "Stop graph generation after this operation"
        >,
        Option<
            "outputFileOpt", "output",
            "std::string", "",
            "Path to the output file"
        >,
        Option<
            "afterPassOpt", "pass",
            "std::string", "",
            "Print Dot after the pass"
        >
    ];

    let constructor = "vpux::Core::createPrintDotPass()";
}


//
// SetupLocationVerifier
//

def SetupLocationVerifier : PassBase<"setup-location-verifier", "vpux::ModulePass"> {
    let summary = "Set location names verifier state to enabled or disabled";

    let description = [{
        This pass attaches or detaches attribute to mark verifier state
    }];

    let options = [
        Option<
            "mode", "mode",
            "std::string", "",
            "Enable location verification"
        >
    ];

    let constructor = "vpux::Core::createSetupLocationVerifierPass()";
}

#endif
