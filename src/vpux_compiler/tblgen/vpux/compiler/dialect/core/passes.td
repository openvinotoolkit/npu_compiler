//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
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

//
// PackNestedModules
//

def PackNestedModules : PassBase<"pack-nested-modules", "vpux::ModulePass"> {
    let summary = "Move functions into nested modules";

    let description = [{
        This pass collects all functions that sit in the module on which the pass is executed. It then performs an
         analysis to determine which functions can be nested inside sub modules. A function is nested in a submodule if
          1. It does not contain a unit attribute named "do_not_nest"
          2. It is not the network entry point
          3. All its callees can be nested in the *same* submodule

        The third rule ensures that no nested function calls another function outside its own submodule as this is prohibited by
         `ModuleOp`'s `IsolatedFromAbove` property.
        This pass will try to separate nested functions into different submodules as much as possible.

        For example the following pseudo IR

        ```MLIR
            module {
                func f()

                func g()

                func main() {
                    call @f()
                    call @g()
                }
            }
        ```

        would produce

        ```MLIR
            module {
                module @Module0 {
                    func f()
                }

                module @Module1 {
                    func g()
                }

                func main() {
                    Core.NestedCall @Module0::@f()
                    Core.NestedCall @Module1::@g()
                }
            }
        ```

        More involved examples can be found in the pass's LIT test files.
    }];

    let constructor = "vpux::Core::createPackNestedModulesPass()";
}

//
// UnpackNestedModules
//

def UnpackNestedModules : PassBase<"unpack-nested-modules", "vpux::ModulePass"> {
    let summary = "Eliminate any nested module, moving all functions into the main module";
    let constructor = "vpux::Core::createUnpackNestedModulesPass()";
    let options = [
    ];

    let description = [{
        This pass collects functions from any nested modules and moves them
        outside these modules into the main module. Any affected call-sites are
        updated accordingly. Thus:

        ```mlir
        module {
            module @initModule {
                func.func private @init() { ... }
            }
            func.func @main() {
                Core.Call @initModule::@init()
                ...
            }
        }
        ```

        Becomes:

        ```mlir
        module {
            func.func private @init() { ... }
            func.func @main() {
                func.call @init()
                ...
            }
        }
        ````

        Note: This pass additionally replaces all 'Core.Call' operations with
        'func.call'.
    }];
}

//
// AddNetInfoToModule
//

def AddNetInfoToModule : PassBase<"add-netinfo-to-module", "vpux::ModulePass"> {
    let summary = "add net::NetInfoOp to a nested module based on function argument and results inside the module";

    let description = [{
        This pass adds NetInfo to nested ModuleOp for vpux passes that require the inputInfos and outputInfos information.
    }];

    let options = [
        Option<
            "hasTensorSemantics", "has-tensor-semantics",
            "bool", "false",
            "Specifies if outputs come as inputs, to accommodate the usage for VPUIP or IE-VPU"
        >,
    ];

    let constructor = "vpux::Core::createAddNetInfoToModulePass()";
}

#endif
