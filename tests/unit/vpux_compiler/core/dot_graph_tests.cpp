//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/utils/dot_printer.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>
#include <fstream>

namespace DotGraphTests {

class CustomTestPass : public mlir::PassWrapper<CustomTestPass, vpux::ModulePass> {
public:
    ::llvm::StringRef getName() const override {
        return "TestPass";
    }
    void safeRunOnModule() final {
    }
};

class CustomTestPass2 : public mlir::PassWrapper<CustomTestPass, vpux::ModulePass> {
public:
    ::llvm::StringRef getName() const override {
        return "TestPass2";
    }
    void safeRunOnModule() final {
    }
};

}  // namespace DotGraphTests

namespace {

constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%arg0: memref<1x512xf32>, %arg1: memref<1x512xf32>) -> memref<1x512xf32> {
                %0 = memref.alloc() : memref<1x512xf32>
                %1 = IERT.SoftMax {axisInd = 1 : i32, test = 2 : i8} inputs(%arg0 : memref<1x512xf32>) outputs(%0 : memref<1x512xf32>) -> memref<1x512xf32>
                %2 = VPUIP.Copy inputs(%1 : memref<1x512xf32>) outputs(%arg1 : memref<1x512xf32>) -> memref<1x512xf32>
                memref.dealloc %0 : memref<1x512xf32>
                return %2 : memref<1x512xf32>
            }
        }
    )";

constexpr llvm::StringLiteral multipleFuncIR = R"(
 module @TwoFunctions {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }
    func.func @foo1(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %0 = IE.Convolution(%arg, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        return %0 : tensor<1x48x60x60xf32>
    }
    func.func @foo2(%arg: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %0 = IE.SoftMax(%arg) {axisInd = 3} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %0 : tensor<1x48x60x60xf32>
    }
    func.func @main(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %0 = call @foo1(%arg) : (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>
        %1 = call @foo2(%0) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        return %1 : tensor<1x48x60x60xf32>
    }
}
    )";

void CheckDotFile(const std::string fileName, std::string patternToFind) {
    std::ifstream output_file(fileName);
    ASSERT_TRUE(output_file.good());
    std::string str;
    auto pos = std::string::npos;
    while (std::getline(output_file, str)) {
        pos = str.find(patternToFind);
        if (pos != std::string::npos) {
            break;
        }
    }

    ASSERT_TRUE(pos != std::string::npos);
}

}  // namespace
using MLIR_DotGraph = MLIR_UnitBase;

TEST_F(MLIR_DotGraph, GenerateViaPass) {
    mlir::MLIRContext ctx(registry);

    const std::string fileName = "output.dot";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(vpux::Core::createPrintDotPass(fileName));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));
    const std::string expectedOutputFileName = "output_main.dot";

    CheckDotFile(expectedOutputFileName, std::string("IERT.SoftMax"));
    std::remove(expectedOutputFileName.c_str());
}

TEST_F(MLIR_DotGraph, GenerateViaEnvVar) {
    mlir::MLIRContext ctx(registry);

    const std::string fileName = "output.dot";
    const std::string fileName2 = "output2.dot";

    const std::string options = "output=" + fileName + " pass=TestPass,output=" + fileName2 + " pass=TestPass2";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    vpux::addDotPrinter(pm, options);
    pm.addPass(std::make_unique<DotGraphTests::CustomTestPass>());
    pm.addPass(std::make_unique<DotGraphTests::CustomTestPass2>());

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    const std::string expectedOutputFileName = "output_main.dot";
    const std::string expectedOutputFileName2 = "output2_main.dot";

    CheckDotFile(expectedOutputFileName, std::string("IERT.SoftMax"));
    CheckDotFile(expectedOutputFileName2, std::string("IERT.SoftMax"));
    std::remove(expectedOutputFileName.c_str());
    std::remove(expectedOutputFileName2.c_str());
}

TEST_F(MLIR_DotGraph, GenerateViaPass_TwoFunctions) {
    mlir::MLIRContext ctx(registry);

    const std::string fileName = "output.dot";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(multipleFuncIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);

    mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(vpux::Core::createPrintDotPass(fileName));

    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    const std::string expectedOutputFileMain = "output_main.dot";
    const std::string expectedOutputFileFoo1 = "output_foo1.dot";
    const std::string expectedOutputFileFoo2 = "output_foo2.dot";

    CheckDotFile(expectedOutputFileMain, std::string("@foo1"));
    CheckDotFile(expectedOutputFileMain, std::string("@foo2"));
    CheckDotFile(expectedOutputFileFoo1, std::string("IE.Convolution"));
    CheckDotFile(expectedOutputFileFoo2, std::string("IE.SoftMax"));

    std::remove(expectedOutputFileMain.c_str());
    std::remove(expectedOutputFileFoo1.c_str());
    std::remove(expectedOutputFileFoo2.c_str());
}
