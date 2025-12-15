//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/llvm_to_binary.hpp"
#include "shave_ld.hpp"
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Program.h>

#if defined(_WIN32) || defined(_WIN64)
#include <llvm/Bitcode/BitcodeWriter.h>

#endif

#include <fstream>

using namespace vpux;

namespace {
std::string getMoviLDArchPath(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return "37xxxx";
    case config::ArchKind::NPU40XX:
        return "40xxxx";
    case config::ArchKind::NPU50XX:
        return "50xxxx";
    default:
        VPUX_THROW("Invalid ArchKind for Movi LLD path resolution");
    }
}
}  // namespace

void vpux::transitivelyCloneFunctions(mlir::ModuleOp dstModuleOp, mlir::ModuleOp srcModuleOp,
                                      mlir::SymbolRefAttr swKernelSymbol) {
    auto llvmFuncOp = srcModuleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(swKernelSymbol);
    VPUX_THROW_UNLESS(llvmFuncOp != nullptr, "llvmFuncOp should be valid");

    llvm::SmallSetVector<mlir::LLVM::LLVMFuncOp, 4> seen;
    llvm::SmallVector<mlir::LLVM::LLVMFuncOp, 4> worklist;
    seen.insert(llvmFuncOp);
    worklist.push_back(llvmFuncOp);

    // We expect all functions to be fully lowered to the llvm dialect.
    // Any symbol uses should be either AddressOf or Call ops.
    while (!worklist.empty()) {
        auto callerOp = worklist.pop_back_val();

        callerOp.walk([&](mlir::SymbolUserOpInterface sOp) {
            mlir::LLVM::LLVMFuncOp callee = nullptr;
            auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(&sOp);
            if (callOp != nullptr && callOp->getCallee()) {
                auto sym = llvm::dyn_cast<mlir::SymbolRefAttr>(callOp->getCallableForCallee());
                if (sym != nullptr) {
                    callee = mlir::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
                            mlir::SymbolTable::lookupNearestSymbolFrom(*callOp, sym));
                }
            }

            if (auto addrOfOp = mlir::dyn_cast<mlir::LLVM::AddressOfOp>(&sOp)) {
                auto symNameAttr = mlir::StringAttr::get(dstModuleOp.getContext(), addrOfOp->getGlobalName());
                callee = mlir::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
                        mlir::SymbolTable::lookupNearestSymbolFrom(*addrOfOp, symNameAttr));
            }

            if (callee && seen.insert(callee)) {
                worklist.push_back(callee);
            }
        });
    }

    for (auto funcOp : seen) {
        dstModuleOp.getBody()->push_back(funcOp.clone());
    }
}

static void addDenormalFlags(llvm::Module& module) {
    // Set the denormal math behavior in order to enable proper lowering of intrinsics.
    StringRef denormalAttrName = "denormal-fp-math";
    for (auto& F : module) {
        if (F.empty() || F.hasFnAttribute(denormalAttrName)) {
            // Skip any functions which don't have a body or
            // already have the attribute specified.
            continue;
        }
        F.addFnAttr(denormalAttrName, llvm::DenormalMode::getPreserveSign().str());
    }
}

std::unique_ptr<llvm::Module> vpux::translateToLLVMIR(mlir::ModuleOp moduleOp, mlir::SymbolRefAttr swKernelSymbol,
                                                      llvm::LLVMContext& llvmContext) {
    auto llvmFuncOp = moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(swKernelSymbol);

    VPUX_THROW_UNLESS(llvmFuncOp != nullptr, "llvmFuncOp should be valid");

    // We create a temporary module in which we clone the llvmFuncOp and then translate it
    // to LLVM IR.
    auto moduleBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());
    auto tmpModuleOp = moduleBuilder.create<mlir::ModuleOp>(moduleOp.getLoc(), llvm::StringRef("TempModule"));

    // Transitively clone the function and its dependencies into
    // the temporary module.
    transitivelyCloneFunctions(tmpModuleOp, moduleOp, swKernelSymbol);

    // Translate the LLVM dialect module to the LLVM IR module. The translation
    // is inspired from MLIR Toy example chapter 6 (https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/).
    // Note: mlir::registerBuiltinDialectTranslation() and
    // mlir::registerLLVMDialectTranslation() are called in init.cpp,
    // in function vpux::registerCommonInterfaces().

    // Convert the module to LLVM IR.
    auto llvmModule = mlir::translateModuleToLLVMIR(tmpModuleOp, llvmContext);
    if (!llvmModule) {
        VPUX_THROW("Failed to emit LLVM IR\n");
    }
    addDenormalFlags(*llvmModule);
    tmpModuleOp.erase();
    return llvmModule;
}

llvm::StringRef vpux::getShaveKernelLDScript() {
    return SHAVE_LD_SCRIPT;
}

void vpux::lowerLLVMToBinary(mlir::ModuleOp moduleOp, std::unique_ptr<llvm::Module> llvmModule,
                             mlir::SymbolRefAttr swKernelSymbol, vpux::Logger log) {
    auto llvmFuncOp = moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(swKernelSymbol);
    VPUX_THROW_UNLESS(llvmFuncOp != nullptr, "llvmFuncOp should be valid");

    const auto arch = config::getArch(moduleOp);
    VPUX_THROW_UNLESS(arch != config::ArchKind::UNKNOWN, "Could not identify arch");

    auto llvmFuncOpNameStr = llvmFuncOp.getName().str();
    ShaveBinaryResources& sbr = ShaveBinaryResourcesCache::getCache(moduleOp->getContext());

    auto archArgument = sbr.getSwKernelArchString(arch);

    // We write llvmModule to file sw_layer.ll.
    std::error_code llFileEC;
    llvm::raw_fd_ostream llFile("sw_layer.ll", llFileEC);
    llFile << *llvmModule;

    llvm::SmallVector<std::optional<StringRef>> redirects = {
            std::nullopt,  // stdin(0)
            std::nullopt,  // stdout(1)
            std::nullopt   // stderr(2)
    };

    std::string errMsg;

    // We compile with moviCompile the sw_layer.ll to sw_layer.s (SHAVE assembly).
    auto mvToolsEnvVar = std::getenv("MV_TOOLS_DIR");
    auto mvToolsVersionEnvVar = std::getenv("MV_TOOLS_VERSION");

    VPUX_THROW_UNLESS(mvToolsEnvVar && mvToolsVersionEnvVar,
                      "Error: Environment variable 'MV_TOOLS_DIR' or 'MV_TOOLS_VERSION' is not set.");

    auto mvToolsDirStrWoNull = std::string(mvToolsEnvVar);
    auto mvToolsVersionStrWoNull = std::string(mvToolsVersionEnvVar);

    VPUX_THROW_UNLESS(!mvToolsDirStrWoNull.empty() && !mvToolsVersionStrWoNull.empty(),
                      "Error: Environment variable 'MV_TOOLS_DIR' or 'MV_TOOLS_VERSION' are empty.");

    auto mvToolsPathCompleteStr = mvToolsDirStrWoNull + "/" + mvToolsVersionStrWoNull;

    const auto prgMCStr = std::string(mvToolsPathCompleteStr) + "/linux64/bin/moviCompile";
    const auto prgMC = StringRef(prgMCStr);
    const auto mcpuStr = vpux::SmallString("-mcpu=") + archArgument;
    const auto mcpu = mcpuStr.str();

    llvm::SmallVector<llvm::StringRef> runArgsMC = {prgMC,         // Movicompile tool
                                                    mcpu,          // CPU
                                                    "-S",          // Only run preprocess and compilation steps
                                                    "-o",          // Write output to:
                                                    "sw_layer.s",  // file sw_layer.s
                                                    "-x",          // Treat subsequent input files as having:
                                                    "ir",          // type ir
                                                    "-O3",         // optimize code
                                                    "-mllvm",      // Next option is for llvm
                                                    "-enable-loop-flatten",  // Enable the loop flatten optimization
                                                    "sw_layer.ll"};          // Input file

    const auto procErrMC = llvm::sys::ExecuteAndWait(prgMC, runArgsMC, /*Env=*/std::nullopt, redirects,
                                                     /*SecondsToWait*/ 100, /*MemoryLimit=*/0, &errMsg);
    VPUX_THROW_UNLESS(procErrMC == 0, "Call to moviCompile failed");

    // We run moviAsm from MoviTools to obtain from sw_layer.s a file sw_layer.o.
    std::string prgAsmStr = mvToolsPathCompleteStr + "/linux64/bin/moviAsm";
    llvm::StringRef prgAsm = prgAsmStr;
    //
    llvm::SmallVector<llvm::StringRef> runArgsAsm = {prgAsm, "sw_layer.s", "--cv", archArgument, "--noSPrefixing"};

    const auto procErrAsm = llvm::sys::ExecuteAndWait(prgAsm, runArgsAsm, /*Env=*/std::nullopt, redirects,
                                                      /*SecondsToWait*/ 100, /*MemoryLimit=*/0, &errMsg);
    VPUX_THROW_UNLESS(procErrAsm == 0, "Call to moviAsm failed");

    std::string elfPathFileNameStr = llvmFuncOpNameStr + "/a.out";

    // Create the folder associated with this shave kernel (e.g. generated_Cos0).
    llvm::sys::fs::create_directory(llvmFuncOpNameStr);

    // We run the linker to obtain the ELF file a.out from sw_layers.o
    //   (we include 4 libraries as dependencies to link the
    //   external __coss function, which returns cos applied on
    //   the float input value, for which it does check if it
    //   is in range 0..pi, etc)

    auto moviLibArchPath = getMoviLDArchPath(arch);

    std::string prgLdStr = mvToolsPathCompleteStr + "/linux64/sparc-myriad-rtems-6.3.0/bin/sparc-myriad-rtems-ld";
    std::string mLibMStr = mvToolsPathCompleteStr + "/common/moviCompile/lib/" + moviLibArchPath + "/mlibm.a";
    std::string mLibCrtStr = mvToolsPathCompleteStr + "/common/moviCompile/lib/" + moviLibArchPath + "/mlibcrt.a";
    std::string mLibCLGPLStr = mvToolsPathCompleteStr + "/common/moviCompile/lib/" + moviLibArchPath + "/mlibc_lgpl.a";
    std::string mLibCStr = mvToolsPathCompleteStr + "/common/moviCompile/lib/" + moviLibArchPath + "/mlibc.a";
    llvm::StringRef prgLd = prgLdStr;

    std::string linkerStr = SHAVE_LD_SCRIPT;
    std::ofstream ldScriptFile("shave_kernel.ld");
    if (!ldScriptFile.is_open()) {
        throw std::runtime_error("Error: Could not open file shave_kernel.ld.");
    }
    ldScriptFile << linkerStr;
    ldScriptFile.close();
    std::string scriptStr = std::string("--script=shave_kernel.ld");

    llvm::SmallVector<llvm::StringRef> runArgsLd = {prgLd,
                                                    llvm::StringRef(scriptStr),
                                                    "-entry",
                                                    llvmFuncOpNameStr,
                                                    "--strip-debug",
                                                    "--discard-all",
                                                    "-zmax-page-size=16",
                                                    "-EL",
                                                    "-O9",
                                                    "--gc-sections",
                                                    "sw_layer.o",
                                                    "--start-group",
                                                    llvm::StringRef(mLibMStr),
                                                    llvm::StringRef(mLibCrtStr),
                                                    llvm::StringRef(mLibCLGPLStr),
                                                    llvm::StringRef(mLibCStr),
                                                    "--end-group",
                                                    "--output",
                                                    llvm::StringRef(elfPathFileNameStr)};

    const auto procErrLd = llvm::sys::ExecuteAndWait(prgLd, runArgsLd, /*Env=*/std::nullopt, redirects,
                                                     /*SecondsToWait*/ 100, /*MemoryLimit=*/0, &errMsg);
    VPUX_THROW_UNLESS(procErrLd == 0, "Call to sparc-myriad-rtems-ld failed");

    // We create file FileList.in containing each ELF file and
    //   folder (name, more exactly key in ShaveBinaryResources dictionary),
    //   associated to the current sw layer.
    std::ofstream fOut("FileList.in", std::ios::app);
    if (fOut.is_open()) {  // Make sure file opened before writing
        if (!(fOut << llvmFuncOpNameStr + "/a.out\n")) {
            log.trace("[SCG] Write to FileList.in failed");
        }

        if (!(fOut << llvmFuncOpNameStr + "\n")) {
            log.trace("[SCG] Write to FileList.in failed");
        }
    } else {
        log.trace("[SCG] Cannot open file FileList.in");
    }
}
