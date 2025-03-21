//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/hash.hpp"
#include "vpux/compiler/utils/logging.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTFUNCARGSTODECLARATIONS
#define GEN_PASS_DEF_CONVERTFUNCARGSTODECLARATIONS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// BufferDeclarationChain
//  Represents a chain of view-like operations that lead to a buffer declaration for CallOp argument
//  For instance, main function:
//      %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !type1
//      %1 = VPUIP.GenericReshape inputs(%0 : !type1) -> !type2
//      %2 = VPUIP.SubView %1 [0, 0, 0, 0] [i, j, k, f] : !type2 to !type3
//
//      %3 = func.call @foo1(%2, %n, %m)
//    For %2 argument of CallOp the structure will be:
//      viewOps = [VPUIP.SubView, VPUIP.GenericReshape]
//      declOp = VPURT.DeclareBuffer // (%0)

struct BufferDeclarationChain {
    SmallVector<mlir::Operation*> viewOps;
    VPURT::DeclareBufferOp declBuffOp;
};

using ArgToBufferDeclarationMap = DenseMap<int64_t, BufferDeclarationChain>;

bool operator==(const BufferDeclarationChain& lhs, const BufferDeclarationChain& rhs) {
    if (lhs.viewOps.size() != rhs.viewOps.size()) {
        return false;
    }

    for (auto p : llvm::zip(lhs.viewOps, rhs.viewOps)) {
        if (hashOperation(std::get<0>(p)) != hashOperation(std::get<1>(p))) {
            return false;
        }
    }

    return hashOperation(lhs.declBuffOp) == hashOperation(rhs.declBuffOp);
}

bool operator!=(const BufferDeclarationChain& lhs, const BufferDeclarationChain& rhs) {
    return !(lhs == rhs);
}
//
// ConvertFuncArgsToDeclarationsPass
//

class ConvertFuncArgsToDeclarationsPass final :
        public VPUIP::impl::ConvertFuncArgsToDeclarationsBase<ConvertFuncArgsToDeclarationsPass> {
public:
    explicit ConvertFuncArgsToDeclarationsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ConvertFuncArgsToDeclarationsPass::safeRunOnModule() {
    auto moduleOp = getOperation();

    const auto buildNewDecl = [](mlir::OpBuilder& argBuilder, mlir::Value val, VPURT::BufferSection section,
                                 int64_t sectionIndex) {
        return argBuilder.create<VPURT::DeclareBufferOp>(val.getLoc(), val.getType(), section, sectionIndex, 0)
                .getResult();
    };

    const auto isPureViewLike = [](mlir::Operation* op) {
        return mlir::isa<mlir::ViewLikeOpInterface>(op) && !mlir::isa<VPUIP::LayerOpInterface>(op);
    };

    const auto replaceArgs = [&](mlir::func::FuncOp funcOp, auto buildDecl) {
        VPUX_THROW_WHEN(funcOp.isExternal(), "It is assumed that the method is run only for functions with body");
        VPUX_THROW_UNLESS(funcOp.getNumArguments() >= funcOp.getNumResults(), "Function '{0}' is not bufferized",
                          funcOp);

        const auto numInputs = funcOp.getNumArguments() - funcOp.getNumResults();

        auto returnOp = *funcOp.getOps<mlir::func::ReturnOp>().begin();
        auto& firstOp = *funcOp.getOps().begin();

        auto argBuilder = mlir::OpBuilder::atBlockBegin(&funcOp.getBody().front());

        size_t argIndex = 0;
        const auto replaceUse = [&](mlir::ValueRange args, VPURT::BufferSection section) {
            for (auto p : args | indexed) {
                auto val = p.value();

                if (val.getUses().empty()) {
                    continue;
                }

                argBuilder.setInsertionPoint(&firstOp);
                auto newArg = buildDecl(argBuilder, val, section, p.index());
                if (hasCompileMethodDebatch(moduleOp)) {
                    auto argDeclareOp = newArg.template getDefiningOp<VPURT::DeclareBufferOp>();
                    if (argDeclareOp != nullptr) {
                        VPURT::BufferSection declarationSection = argDeclareOp.getSection();
                        if (declarationSection == VPURT::BufferSection::DDR) {
                            argDeclareOp.setSection(VPURT::BufferSection::FunctionInput);

                            VPUX_THROW_WHEN(argDeclareOp.getSectionIndexAttr(),
                                            "Declaration {0} must not carry any section index attribute");
                            argDeclareOp.setSectionIndexAttr(
                                    getIntArrayAttr(argDeclareOp.getContext(), SmallVector<size_t>{argIndex}));
                            extendOpLoc(argDeclareOp, StringLiteral("func_{0}_arg_{1}"), funcOp.getName(), argIndex);
                        }
                    }
                }
                _log.trace("Replace all uses of '{0}' with '{1}', argIndex: {2}", val,
                           newArg.template getDefiningOp<VPURT::DeclareBufferOp>(), argIndex);
                val.replaceAllUsesExcept(newArg, llvm::SmallPtrSet<mlir::Operation*, 1>{returnOp});
                argIndex++;
            }
        };

        replaceUse(funcOp.getArguments().take_front(numInputs), VPURT::BufferSection::NetworkInput);
        replaceUse(funcOp.getArguments().drop_front(numInputs), VPURT::BufferSection::NetworkOutput);
    };

    mlir::func::FuncOp netFunc;
    vpux::IE::CNNNetworkOp cnnOp;
    vpux::IE::CNNNetworkOp::getFromModule(moduleOp, cnnOp, netFunc);

    replaceArgs(netFunc, buildNewDecl);
    SmallVector<mlir::func::FuncOp> funcOps;
    funcOps.push_back(netFunc);

    const auto getDeclaration = [&](mlir::func::CallOp callOp, mlir::Value val) {
        auto funcBlockArg = mlir::cast<mlir::BlockArgument>(val);

        auto callOperand = callOp.getOperand(funcBlockArg.getArgNumber());
        auto producerOp = callOperand.getDefiningOp();
        SmallVector<mlir::Operation*> viewOps;
        while (producerOp != nullptr && isPureViewLike(producerOp)) {
            viewOps.push_back(producerOp);
            producerOp = mlir::cast<mlir::ViewLikeOpInterface>(producerOp).getViewSource().getDefiningOp();
        }

        auto originDeclOp = mlir::dyn_cast_or_null<VPURT::DeclareBufferOp>(producerOp);
        VPUX_THROW_WHEN(originDeclOp == nullptr, "Could not find declare buffer op for CallOp '{0}' operand '{1}'",
                        callOp, callOperand);
        return BufferDeclarationChain{std::move(viewOps), originDeclOp};
    };

    mlir::DenseMap<mlir::func::FuncOp, ArgToBufferDeclarationMap> funcToDeclChains;

    while (!funcOps.empty()) {
        SmallVector<mlir::func::FuncOp> nextLevelFuncOps;
        for (auto& funcOp : funcOps) {
            funcOp.walk([&](mlir::func::CallOp callOp) {
                auto func = getCalledFunction(callOp);

                if (funcToDeclChains.contains(func)) {
                    for (const auto& arg : func.getArguments()) {
                        if (!funcToDeclChains[func].contains(arg.getArgNumber())) {
                            continue;
                        }

                        auto declChain = getDeclaration(callOp, arg);
                        if (hasCompileMethodDebatch(moduleOp)) {
                            // Limitation in having a same declaration chain for all repeating calls
                            // wasn't completely overcomed here by using the additional FunctionInpu/Output section.
                            // A scenario with isPureViewLike operations propagation hasn't been solved yet.
                            // We throw an exception here only when a declChain contains isPureViewLike operations and
                            // support different non-pure-view buffer declarations as function arguments
                            if (funcToDeclChains[func][arg.getArgNumber()] != declChain) {
                                VPUX_THROW_WHEN(declChain.viewOps.size() != 0 ||
                                                        funcToDeclChains[func][arg.getArgNumber()].viewOps.size() != 0,
                                                "Declaration chain for argument at idx '{0}' of CallOp with callee "
                                                "'{1}' is not "
                                                "equal "
                                                "to the chain of the first call of the same function.",
                                                arg.getArgNumber(), func.getName());

                                _log.info("Declaration chain for argument at idx '{0}' of CallOp with callee '{1}' is "
                                          "not "
                                          "equal "
                                          "to the chain of the first call of the same function.",
                                          arg.getArgNumber(), func.getName());
                            }
                        } else {
                            // Default way for declaration chains processing
                            VPUX_THROW_UNLESS(funcToDeclChains[func][arg.getArgNumber()] == declChain,
                                              "Declaration chain for argument at idx '{0}' of CallOp with callee '{1}' "
                                              "is not equal "
                                              "to the chain of the first call of the same function.",
                                              arg.getArgNumber(), func.getName());
                        }
                    }
                }

                ArgToBufferDeclarationMap declChainsMap;
                const auto buildDeclaration = [&](mlir::OpBuilder& argBuilder, mlir::Value val, VPURT::BufferSection,
                                                  int64_t) {
                    auto originDeclChain = getDeclaration(callOp, val);
                    auto funcBlockArg = mlir::cast<mlir::BlockArgument>(val);
                    declChainsMap[funcBlockArg.getArgNumber()] = originDeclChain;

                    // clang-format off
                    // Clone full chain of pure view like ops to get correct offset in the "child" function after ConvertViewOpsToDeclarations
                    // For example:
                    // main:
                    // %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x10x10x12xf16, @DDR>
                    // %1 = VPUIP.SubView %0 [0, 1] [1, 8] : memref<1x10f16, @DDR> to memref<1x8xf16, {order = #NC, strides = [10, 1]}, @DDR>
                    //
                    // foo1:
                    // original argument is %arg1: memref<1x8xf16, {order = #NC, strides = [10, 1]}, @DDR>
                    // when clone only DeclareBufferOp:
                    // %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x10x10x12xf16, @DDR>
                    // %1 = VPUIP.SubView %0 [0, 2] [1, 6] : memref<1x8xf16, {order = #NC, strides = [10, 1]}, @DDR> to memref<1x6xf16, {order = #NC, strides = [10, 1]}, @DDR>
                    //
                    // After ConvertViewOpsToDeclarations offset is 2, but shoud be 3 = 1("main" offset) + 2("foo1" offset)
                    // clang-format on

                    auto resValue = argBuilder.clone(*originDeclChain.declBuffOp.getOperation())->getResult(0);
                    for (auto& currViewOp : originDeclChain.viewOps | reversed) {
                        mlir::IRMapping mapper;
                        mapper.map(mlir::cast<mlir::ViewLikeOpInterface>(currViewOp).getViewSource(), resValue);
                        resValue = argBuilder.clone(*currViewOp, mapper)->getResult(0);
                    }

                    return resValue;
                };

                replaceArgs(func, buildDeclaration);
                funcToDeclChains[func] = std::move(declChainsMap);
                nextLevelFuncOps.push_back(func);
            });
        }

        funcOps = std::move(nextLevelFuncOps);
    }
}

}  // namespace

//
// createConvertFuncArgsToDeclarationsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertFuncArgsToDeclarationsPass(Logger log) {
    return std::make_unique<ConvertFuncArgsToDeclarationsPass>(log);
}
