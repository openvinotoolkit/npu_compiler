//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/dot_printer.hpp"

#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/dot_graph_writer.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include "vpux/utils/core/error.hpp"

#include <llvm/Support/Regex.h>

#include <memory>
#include <string>

namespace vpux::Core {
#define GEN_PASS_DECL_PRINTDOT
#define GEN_PASS_DEF_PRINTDOT
#include "vpux/compiler/dialect/core/passes.hpp.inc"
}  // namespace vpux::Core

using namespace vpux;

namespace {

//
// PrintDotPass
//

class PrintDotPass final : public Core::impl::PrintDotBase<PrintDotPass> {
public:
    PrintDotPass() = default;
    PrintDotPass(StringRef fileName, const GraphWriterParams& writerParams)
            : _fileName(fileName.str()), _writerParams(writerParams) {
    }

public:
    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;

public:
    void processModule(mlir::ModuleOp module);
    void processFunc(mlir::func::FuncOp func);

public:
    bool checkPass(mlir::Pass* pass) const;

private:
    void safeRunOnModule() final {
        processModule(getOperation());
    }

    std::string composeFileName(std::string funcName) {
        std::string extension = ".";
        if (auto pos = _fileName.find(extension)) {
            std::string newName = _fileName;
            newName.erase(pos, extension.size());
            newName = newName.insert(pos, ("_" + funcName + extension));
            return newName;
        }
        return _fileName + "_" + funcName;
    }

private:
    std::string _fileName;
    GraphWriterParams _writerParams;
    std::shared_ptr<llvm::Regex> _passNameFilter;
};

mlir::LogicalResult PrintDotPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }

    if (printOnlyDotInterfaceOpt.hasValue()) {
        _writerParams.printOnlyDotInterfaces = printOnlyDotInterfaceOpt.getValue();
    }
    if (printDeclarationsOpt.hasValue()) {
        _writerParams.printDeclarations = printDeclarationsOpt.getValue();
    }
    if (printConstOpt.hasValue()) {
        _writerParams.printConst = printConstOpt.getValue();
    }
    if (printOnlyTaskAndBarrierOpt.hasValue()) {
        _writerParams.printOnlyTaskAndBarrier = printOnlyTaskAndBarrierOpt.getValue();
    }
    if (printOnlyAsyncExecOpt.hasValue()) {
        _writerParams.printOnlyAsyncExec = printOnlyAsyncExecOpt.getValue();
    }
    if (startAfterOpt.hasValue()) {
        _writerParams.startAfter = startAfterOpt.getValue();
    }
    if (stopBeforeOpt.hasValue()) {
        _writerParams.stopBefore = stopBeforeOpt.getValue();
    }
    if (outputFileOpt.hasValue()) {
        _fileName = outputFileOpt.getValue();
    }
    if (afterPassOpt.hasValue()) {
        _passNameFilter = std::make_shared<llvm::Regex>(afterPassOpt.getValue(), llvm::Regex::IgnoreCase);

        std::string regexErr;
        if (!_passNameFilter->isValid(regexErr)) {
            VPUX_THROW("Invalid regular expression '{0}' : {1}", afterPassOpt.getValue(), regexErr);
        }
    }

    return mlir::success();
}

bool PrintDotPass::checkPass(mlir::Pass* pass) const {
    VPUX_THROW_WHEN(_passNameFilter == nullptr, "Pass name filter was not specified");
    return _passNameFilter->match(pass->getName()) || _passNameFilter->match(pass->getArgument());
}

void PrintDotPass::processModule(mlir::ModuleOp module) {
    module.walk([this](mlir::func::FuncOp func) {
        processFunc(func);
    });
}

void PrintDotPass::processFunc(mlir::func::FuncOp func) {
    VPUX_THROW_WHEN(_fileName.empty(), "Output file name for PrintDot was not provided");
    VPUX_THROW_WHEN(mlir::failed(writeDotGraph(func, composeFileName(std::string(func.getName())), _writerParams)),
                    "Could not create Dot File");
}

//
// PrintDotInstrumentation
//

class PrintDotInstrumentation final : public mlir::PassInstrumentation {
public:
    void addPrinter(std::unique_ptr<PrintDotPass> printer) {
        _printers.push_back(std::move(printer));
    }

public:
    void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override;
    void runAfterPassFailed(mlir::Pass* pass, mlir::Operation* op) override;

private:
    std::vector<std::unique_ptr<PrintDotPass>> _printers;
};

void PrintDotInstrumentation::runAfterPass(mlir::Pass* pass, mlir::Operation* op) {
    for (auto& printer : _printers) {
        if (!printer->checkPass(pass)) {
            continue;
        }

        llvm::outs() << " Generating Dot after " << pass->getName() << "\n";

        if (auto module = mlir::dyn_cast<mlir::ModuleOp>(op)) {
            printer->processModule(module);
        } else if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
            printer->processFunc(func);
        }
    }
}

void PrintDotInstrumentation::runAfterPassFailed(mlir::Pass* pass, mlir::Operation* op) {
    PrintDotInstrumentation::runAfterPass(pass, op);
}

}  // namespace

//
// addDotPrinter
//

void vpux::addDotPrinter(mlir::PassManager& pm, StringRef options) {
    auto instr = std::make_unique<PrintDotInstrumentation>();

    std::stringstream ss(options.data());
    std::string split;
    while (std::getline(ss, split, ',')) {
        std::unique_ptr<PrintDotPass> printer = std::make_unique<PrintDotPass>();
        VPUX_THROW_WHEN(mlir::failed(printer->initializeOptions(split, nullptr)), "Failed to initialize options");

        instr->addPrinter(std::move(printer));
    }

    pm.addInstrumentation(std::move(instr));
}

//
// createPrintDotPass
//

std::unique_ptr<mlir::Pass> vpux::Core::createPrintDotPass(StringRef fileName, StringRef startAfter,
                                                           StringRef stopBefore, bool printOnlyDotInterFaces,
                                                           bool printConst, bool printDeclarations) {
    GraphWriterParams writerParams;
    writerParams.printConst = printConst;
    writerParams.printDeclarations = printDeclarations;
    writerParams.printOnlyDotInterfaces = printOnlyDotInterFaces;
    writerParams.startAfter = startAfter.str();
    writerParams.stopBefore = stopBefore.str();

    return std::make_unique<PrintDotPass>(fileName, writerParams);
}
