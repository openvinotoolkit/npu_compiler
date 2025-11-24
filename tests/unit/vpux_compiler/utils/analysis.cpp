//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include "common/utils.hpp"

#include <mlir/Parser/Parser.h>

using namespace vpux;

class GetTopModuleOp : public MLIR_UnitBase {
public:
    void SetUp() override {
        testing::Test::SetUp();
        ctxt = std::make_unique<mlir::MLIRContext>(registry);
    }

    mlir::SymbolRefAttr getNestedSymbol(ArrayRef<StringRef> strings) {
        assert(!strings.empty() && "getNestedSymbol() expects at least 1 string!");
        SmallVector<mlir::FlatSymbolRefAttr> nested;
        llvm::transform(strings.drop_front(), std::back_inserter(nested), [&](StringRef s) {
            return mlir::FlatSymbolRefAttr::get(ctxt.get(), s);
        });
        return mlir::SymbolRefAttr::get(ctxt.get(), strings[0], nested);
    }

    std::unique_ptr<mlir::MLIRContext> ctxt;
};

TEST_F(GetTopModuleOp, TopModuleExists) {
    constexpr StringLiteral IR = R"(
        module @top {
            module @nested_1 {
                func.func private @test() -> ()
            }

            module @nested_2 {
                module @sub {
                    func.func private @test() -> ()
                }
            }

            func.func private @test() -> ()
        }
    )";

    auto actualTopModuleOp = mlir::parseSourceString<mlir::ModuleOp>(IR, ctxt.get());
    ASSERT_TRUE(actualTopModuleOp.get() != nullptr);

    // All ops must have the same top level module, namely actualTopModuleOp.
    actualTopModuleOp.get().walk([&](mlir::Operation* op) {
        auto topModuleOp = getTopParentOpOfType<mlir::ModuleOp>(op);
        EXPECT_EQ(topModuleOp, actualTopModuleOp.get());
    });
}

TEST_F(GetTopModuleOp, TopModuleDoesNotExist) {
    constexpr StringLiteral IR = R"(
        func.func private @test() -> () {
            return
        }
    )";

    auto actualTopModuleOp = mlir::parseSourceString(IR, ctxt.get());
    ASSERT_TRUE(actualTopModuleOp.get() != nullptr);

    // All ops must have no top level module.
    actualTopModuleOp.get()->walk([&](mlir::Operation* op) {
        auto topModuleOp = getTopParentOpOfType<mlir::ModuleOp>(op);
        EXPECT_EQ(topModuleOp, nullptr);
    });
}
