//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/llvm_to_binary.hpp"
#include "shave_ld.hpp"
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/utils/core/developer_build_utils.hpp"
#include "vpux/utils/core/small_string.hpp"

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>

#if defined(_WIN32) || defined(_WIN64)
#include <llvm/Bitcode/BitcodeWriter.h>

#endif

#include <fstream>
#include <sstream>

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

// RAII helper owning a unique temporary directory holding the intermediate artifacts (LLVM IR,
// assembly, object file, linker script, ELF) produced while compiling a single SHAVE kernel. The
// directory is always removed on scope exit (success, failure, or thrown exception), unless
// IE_NPU_SCG_KEEP_COMPILATION_ARTIFACTS is set, in which case the artifacts are preserved: copied to
// <IE_NPU_SCG_ARTIFACTS_PATH>/<kernelName>/ (overwriting any previous run) if that path is set,
// otherwise simply left under the OS temp dir. Both env vars are honored in developer builds only;
// in non-developer builds they're ignored and the directory is always removed.
class ScopedTempDirectory {
public:
    ScopedTempDirectory(llvm::StringRef prefix, vpux::Logger log): _kernelName(prefix), _log(log) {
        if (isDeveloperBuild()) {
            _keep = std::getenv("IE_NPU_SCG_KEEP_COMPILATION_ARTIFACTS") != nullptr;
            if (const char* destDir = std::getenv("IE_NPU_SCG_ARTIFACTS_PATH")) {
                _destDir = destDir;
            }
        }
        const auto ec = llvm::sys::fs::createUniqueDirectory(prefix, _path);
        VPUX_THROW_UNLESS(!ec, "Failed to create temporary directory for SHAVE kernel compilation: {0}", ec.message());
    }
    ScopedTempDirectory(const ScopedTempDirectory&) = delete;
    ScopedTempDirectory& operator=(const ScopedTempDirectory&) = delete;
    ~ScopedTempDirectory() {
        if (_keep) {
            if (_destDir.empty()) {
                // No destination requested: leave the artifacts where they are, under the OS temp dir.
                return;
            }
            // If preservation is requested and a destination directory is set, we copy the artifacts
            // into a per-kernel subdirectory of that destination before removing the temporary directory.
            preserveArtifacts();
        }
        if (const auto ec = llvm::sys::fs::remove_directories(_path)) {
            _log.warning("Could not remove SHAVE kernel compilation temporary directory '{0}': {1}", _path,
                         ec.message());
        }
    }

    std::string path(llvm::StringRef fileName) const {
        llvm::SmallString<256> full(_path);
        llvm::sys::path::append(full, fileName);
        return full.str().str();
    }

    llvm::StringRef directory() const {
        return _path;
    }

    bool keepsFiles() const {
        return _keep;
    }

    llvm::StringRef destinationDirectory() const {
        return _destDir;
    }

private:
    // Copies the temporary directory's artifacts into _destDir/_kernelName, wiping that
    // subdirectory first so a previous run's stale files can't linger. Best-effort: failures are
    // only logged as warnings, never thrown, so they can't mask the actual compilation result.
    void preserveArtifacts() const {
        llvm::SmallString<256> destSubDir(_destDir);
        llvm::sys::path::append(destSubDir, _kernelName);

        if (const auto ec = llvm::sys::fs::remove_directories(destSubDir)) {
            _log.warning("Could not clean up previous contents of directory '{0}' before preserving SHAVE kernel "
                         "compilation artifacts: {1}",
                         destSubDir, ec.message());
        }

        if (const auto ec = llvm::sys::fs::create_directories(destSubDir)) {
            _log.warning("Could not create directory '{0}' to preserve SHAVE kernel compilation artifacts: {1}",
                         destSubDir, ec.message());
            return;
        }

        std::error_code ec;
        for (llvm::sys::fs::directory_iterator it(_path, ec), end; it != end && !ec; it.increment(ec)) {
            llvm::SmallString<256> destFile(destSubDir);
            llvm::sys::path::append(destFile, llvm::sys::path::filename(it->path()));

            // rename() is a cheap move when the destination is on the same filesystem as the OS temp
            // dir; fall back to a copy when it isn't (e.g. cross-device destination).
            if (llvm::sys::fs::rename(it->path(), destFile)) {
                if (const auto copyEc = llvm::sys::fs::copy_file(it->path(), destFile)) {
                    _log.warning("Could not preserve SHAVE kernel compilation artifact '{0}' to '{1}': {2}", it->path(),
                                 destFile, copyEc.message());
                }
            }
        }
        if (ec) {
            _log.warning("Error while iterating SHAVE kernel compilation artifacts in '{0}': {1}", _path, ec.message());
        }
    }

    llvm::SmallString<128> _path;
    std::string _kernelName;
    bool _keep = false;
    std::string _destDir;
    vpux::Logger _log;
};
}  // namespace

void vpux::transitivelyCloneFunctions(mlir::ModuleOp dstModuleOp, mlir::ModuleOp srcModuleOp,
                                      mlir::SymbolRefAttr swKernelSymbol) {
    auto llvmFuncOp = srcModuleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(swKernelSymbol);
    VPUX_THROW_UNLESS(llvmFuncOp != nullptr, "llvmFuncOp should be valid");

    llvm::SmallSetVector<mlir::Operation*, 4> seen;
    llvm::SmallVector<mlir::Operation*, 4> worklist;
    seen.insert(llvmFuncOp);
    worklist.push_back(llvmFuncOp);

    // We expect all functions to be fully lowered to the llvm dialect.
    // Any symbol uses should be either AddressOf or Call ops.
    while (!worklist.empty()) {
        auto currentOp = worklist.pop_back_val();

        currentOp->walk([&](mlir::SymbolUserOpInterface sOp) {
            mlir::Operation* referencedSymbol = nullptr;
            auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(&sOp);
            if (callOp != nullptr && callOp->getCallee()) {
                auto sym = llvm::dyn_cast<mlir::SymbolRefAttr>(callOp->getCallableForCallee());
                if (sym != nullptr) {
                    referencedSymbol = mlir::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
                            mlir::SymbolTable::lookupNearestSymbolFrom(*callOp, sym));
                }
            }

            if (auto addrOfOp = mlir::dyn_cast<mlir::LLVM::AddressOfOp>(&sOp)) {
                auto symNameAttr = mlir::StringAttr::get(dstModuleOp.getContext(), addrOfOp->getGlobalName());
                auto* symbolOp = mlir::SymbolTable::lookupNearestSymbolFrom(*addrOfOp, symNameAttr);

                if (auto funcOp = mlir::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(symbolOp)) {
                    referencedSymbol = funcOp;
                } else if (auto globalOp = mlir::dyn_cast_or_null<mlir::LLVM::GlobalOp>(symbolOp)) {
                    // Adding a work-around to also include verification of globalOp
                    referencedSymbol = globalOp;
                }
            }

            if (referencedSymbol && seen.insert(referencedSymbol)) {
                worklist.push_back(referencedSymbol);
            }
        });
    }

    for (auto funcOp : seen) {
        dstModuleOp.getBody()->push_back(funcOp->clone());
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

static void stripIncompatibleAttributes(llvm::Module& module) {
    // Remove attributes which movicompile can't yet handle.
    // TODO: E#197224 remove these workarounds when movicompile reaches llvm 20.x
    for (auto& func : module) {
        // The captures attribute was introduced in llvm 20.x.
        func.removeRetAttr(llvm::Attribute::Captures);
        for (auto& arg : func.args()) {
            arg.removeAttr(llvm::Attribute::Captures);
        }

        for (auto& basicBlock : func) {
            for (auto& inst : basicBlock) {
                if (auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(&inst)) {
                    // GEP no wrap flags were introduced in llvm 19.x.
                    gep->setNoWrapFlags(llvm::GEPNoWrapFlags::none());
                }
            }
        }
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
    stripIncompatibleAttributes(*llvmModule);
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
    auto& sbr = getShaveBinaryResources(moduleOp->getContext());

    auto archArgument = sbr.getSwKernelArchString(arch);

    // All intermediate compilation artifacts are kept in a dedicated, uniquely-named temporary
    // directory so that concurrent kernel compilations never clash and stale files are never left
    // behind on disk, regardless of whether compilation succeeds or throws.
    ScopedTempDirectory tempDir(llvmFuncOpNameStr, log);
    if (tempDir.keepsFiles()) {
        if (!tempDir.destinationDirectory().empty()) {
            log.trace("IE_NPU_SCG_KEEP_COMPILATION_ARTIFACTS is set, SHAVE kernel compilation artifacts will be "
                      "copied to '{0}/{1}'",
                      tempDir.destinationDirectory(), llvmFuncOpNameStr);
        } else {
            log.trace("IE_NPU_SCG_KEEP_COMPILATION_ARTIFACTS is set, preserving SHAVE kernel temporary directory "
                      "'{0}'",
                      tempDir.directory());
        }
    }

    const auto llFilePath = tempDir.path("sw_layer.ll");
    const auto sFilePath = tempDir.path("sw_layer.s");
    const auto oFilePath = tempDir.path("sw_layer.o");
    const auto ldScriptPath = tempDir.path("shave_kernel.ld");
    const auto elfPath = tempDir.path("a.out");

    // We write llvmModule to file sw_layer.ll.
    std::error_code llFileEC;
    llvm::raw_fd_ostream llFile(llFilePath, llFileEC);
    VPUX_THROW_UNLESS(!llFileEC, "Could not open file '{0}' for writing LLVM IR: {1}", llFilePath, llFileEC.message());
    llFile << *llvmModule;

    // Runs an external compilation tool, redirecting its stdout/stderr into files inside the
    // temporary directory so that failures can be diagnosed: on a non-zero exit code the captured
    // output (plus any launch-level error reported by ExecuteAndWait itself) is included in the
    // thrown exception instead of being silently discarded.
    const auto runToolAndCheck = [&](llvm::StringRef toolName, llvm::StringRef program,
                                     llvm::ArrayRef<llvm::StringRef> args) {
        const auto stdoutPath = tempDir.path(toolName.str() + ".stdout.log");
        const auto stderrPath = tempDir.path(toolName.str() + ".stderr.log");
        const llvm::SmallVector<std::optional<StringRef>> redirects = {std::nullopt,                  // stdin(0)
                                                                       llvm::StringRef(stdoutPath),   // stdout(1)
                                                                       llvm::StringRef(stderrPath)};  // stderr(2)

        std::string errMsg;
        const auto procErr = llvm::sys::ExecuteAndWait(program, args, /*Env=*/std::nullopt, redirects,
                                                       /*SecondsToWait*/ 100, /*MemoryLimit=*/0, &errMsg);
        if (procErr != 0) {
            const auto readOutput = [](const std::string& path) {
                std::ifstream file(path);
                std::stringstream buffer;
                buffer << file.rdbuf();
                return buffer.str();
            };
            VPUX_THROW("Call to {0} failed (exit code {1}): {2}\n--- stdout ---\n{3}--- stderr ---\n{4}", toolName,
                       procErr, errMsg, readOutput(stdoutPath), readOutput(stderrPath));
        }
    };

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
    // concatenation ("+") below of two SmallString objects generates a Twine
    // Twines should not be stored into variable on the stack as they are destroyed immediately after the expression
    // see Twine's documentation
    // hence use concatenation result (via .str()) immediately
    const auto mcpu = (vpux::SmallString("-mcpu=") + archArgument).str();

    llvm::SmallVector<llvm::StringRef> runArgsMC = {
            prgMC,                       // Movicompile tool
            mcpu,                        // CPU
            "-S",                        // Only run preprocess and compilation steps
            "-o",                        // Write output to:
            llvm::StringRef(sFilePath),  // file sw_layer.s
            "-x",                        // Treat subsequent input files as having:
            "ir",                        // type ir
            "-O3",                       // optimize code
            "-mllvm",                    // Next option is for llvm
            "-enable-loop-flatten",      // Enable the loop flatten optimization
            // Preemption flags
            "-mshave-preemption-checks=restore", "-mshave-low-impact-preemption", "-mshave-preemption-max-loop-depth=1",
            llvm::StringRef(llFilePath)};  // Input file

    runToolAndCheck("moviCompile", prgMC, runArgsMC);

    // We run moviAsm from MoviTools to obtain from sw_layer.s a file sw_layer.o.
    std::string prgAsmStr = mvToolsPathCompleteStr + "/linux64/bin/moviAsm";
    llvm::StringRef prgAsm = prgAsmStr;
    //
    llvm::SmallVector<llvm::StringRef> runArgsAsm = {prgAsm, llvm::StringRef(sFilePath), "--cv", archArgument,
                                                     "--noSPrefixing"};

    runToolAndCheck("moviAsm", prgAsm, runArgsAsm);

    // We run the linker to obtain the ELF file a.out from sw_layer.o
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
    std::ofstream ldScriptFile(ldScriptPath);
    if (!ldScriptFile.is_open()) {
        throw std::runtime_error("Error: Could not open file " + ldScriptPath + ".");
    }
    ldScriptFile << linkerStr;
    ldScriptFile.close();
    std::string scriptStr = std::string("--script=") + ldScriptPath;

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
                                                    llvm::StringRef(oFilePath),
                                                    "--start-group",
                                                    llvm::StringRef(mLibMStr),
                                                    llvm::StringRef(mLibCrtStr),
                                                    llvm::StringRef(mLibCLGPLStr),
                                                    llvm::StringRef(mLibCStr),
                                                    "--end-group",
                                                    "--output",
                                                    llvm::StringRef(elfPath)};

    runToolAndCheck("moviLLD", prgLd, runArgsLd);

    // Read the ELF file into a buffer and add it to the ShaveBinaryResources.
    auto elfBufferOrErr = llvm::MemoryBuffer::getFile(elfPath, /*IsText=*/false, /*RequiresNullTerminator=*/false);
    VPUX_THROW_UNLESS(elfBufferOrErr, "Could not read compiled SHAVE ELF file '{0}': {1}", elfPath,
                      elfBufferOrErr.getError().message());
    const auto& elfBuffer = *elfBufferOrErr;
    const auto elfBinary =
            llvm::ArrayRef(reinterpret_cast<const uint8_t*>(elfBuffer->getBufferStart()), elfBuffer->getBufferSize());

    sbr.addCompiledElf(llvmFuncOpNameStr, elfBinary, arch, /*overwrite=*/true);
}
