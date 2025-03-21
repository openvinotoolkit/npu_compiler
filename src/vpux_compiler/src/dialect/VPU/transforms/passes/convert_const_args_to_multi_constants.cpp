//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_CONVERTCONSTARGSTOMULTICONSTANTS
#define GEN_PASS_DEF_CONVERTCONSTARGSTOMULTICONSTANTS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

namespace {

using namespace vpux;

// ConvertConstArgsToMultiConstants
//
// The outliner might produce outlined functions where all 'const.Declare's inside the function
// body are put into the main function's body. Then all these constants are provided as
// arguments at the call sites. This pass removes all these constants inside main's body and puts
// them in a global scope. Occurances in the outlined functions are then replaced with 'const.MultiDeclare's.
//
// The logic is the following:
//
// Assume there is an outlined function f with multiple call sites and where some operands are network inputs and
// other operands are 'const.Declare's. In this schematic operands prefixed with c stem from 'const.Declare'.
//
//      func @main(...)
//          ...
//          f(x, c1, y, c2, c3)
//          ...
//          f(x, c4, y, c5, c6)
//          ...
//          f(z,  z, y, c5, c6)
//          ...
//
// This pass then concludes that operands 3 and 4 and be safely removed because they come from 'const.Declare'
// at *all* call sites.
//
// In this example the pass
// 1. Creates 'const.Rodata's for c2, c3, c5 and c6
// 2. Creates 'const.RodataBundle's representing [c2, c5, c5] and [c3, c6, c6]
// 3. Removes operands 3 and 4 from all call sites and changes the function signature of f accordingly
// 4. Replaces usages of operands 3 and 4 in f's body with 'const.MultiDeclare' using the newly created bundles
//
class ConvertConstArgsToMultiConstants final :
        public VPU::impl::ConvertConstArgsToMultiConstantsBase<ConvertConstArgsToMultiConstants> {
public:
    using Base = ConvertConstArgsToMultiConstantsBase;

    explicit ConvertConstArgsToMultiConstants(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

protected:
    void safeRunOnModule() final;

private:
    llvm::BitVector getIsCandidateBitVector(MutableArrayRef<mlir::func::CallOp> callOps);
    mlir::SymbolRefAttr getNestedSymbol(StringRef root, StringRef leaf);
    bool checkPrecondition(mlir::func::FuncOp netFunc);
    Const::RodataBundleOp buildRodataBundleOp(Const::DataOp dataOp, Const::BundleDataOp bundleDataOp,
                                              ArrayRef<mlir::func::CallOp> callOps, size_t operandIndex);
    Const::MultiDeclareOp buildMultiDeclareOp(mlir::func::FuncOp funcOp, Const::RodataBundleOp rodataBundleOp,
                                              Const::TransformAttrInterfaceArrayAttr transformationsAttr);

    // keep track of the created 'const.Rodata' and 'const.RodataBundle' ops
    mlir::IRMapping _declareOpToRodataOp;
    size_t _bundleCount = 0;

    static constexpr llvm::StringLiteral DATA_SYMBOL = "Data";
    static constexpr llvm::StringLiteral BUNDLE_DATA_SYMBOL = "BundleData";
    static constexpr llvm::StringLiteral RODATA_PREFIX = "rodata_";
    static constexpr llvm::StringLiteral RODATA_BUNDLE_PREFIX = "bundle_";
};

}  // namespace

void ConvertConstArgsToMultiConstants::safeRunOnModule() {
    auto moduleOp = getOperation();
    IE::CNNNetworkOp netInfo;
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

    // prints and error and returns false if precondition is not met
    if (!checkPrecondition(netFunc)) {
        signalPassFailure();
        return;
    }

    // find useful ops
    auto funcOps = moduleOp.getBodyRegion().getOps<mlir::func::FuncOp>();
    VPUX_THROW_WHEN(funcOps.empty(), "ModuleOp does not contain any 'func.Func' ops, which is unexpected");

    // put 'const.Data' and 'const.BundleData' ops right before the first 'func.Func' op
    auto firstFuncOp = *funcOps.begin();
    mlir::OpBuilder builder(moduleOp.getBodyRegion());
    builder.setInsertionPoint(firstFuncOp);

    // create 'const.Data' and 'const.BundleData'
    //
    // const.Data @Data {
    // }
    //
    // const.BundleData @BundleData {
    // }
    //
    auto dataOp = builder.create<Const::DataOp>(appendLoc(netFunc.getLoc(), "data"), DATA_SYMBOL);
    auto bundleDataOp = builder.create<Const::BundleDataOp>(appendLoc(netFunc.getLoc(), "bundle"), BUNDLE_DATA_SYMBOL);

    for (auto funcOp : funcOps) {
        auto callOps = getCallSites(funcOp, netFunc);

        // ignore if the function has no uses
        if (callOps.empty()) {
            continue;
        }

        // a bit vector whose entry at i is true iff. the operands at index i stem from 'const.Declare' for all call
        // sites
        auto isCandidate = getIsCandidateBitVector(callOps);

        // nothing to do, early exit
        if (isCandidate.none()) {
            continue;
        }

        auto& bodyBlock = funcOp.getFunctionBody().front();

        // create 'const.RodataBundle' and 'const.MultiDeclare' for every argument that is a candidate of a function
        for (size_t operandIndex = 0; operandIndex < funcOp.getFunctionType().getNumInputs(); ++operandIndex) {
            if (!isCandidate[operandIndex]) {
                continue;
            }

            auto rodataBundleOp = buildRodataBundleOp(dataOp, bundleDataOp, callOps, operandIndex);
            auto declareOp = mlir::cast<Const::DeclareOp>(callOps.front()->getOperand(operandIndex).getDefiningOp());
            auto multiDeclareOp =
                    buildMultiDeclareOp(funcOp, rodataBundleOp, declareOp.getContentAttr().getTransformationsAttr());

            bodyBlock.getArgument(operandIndex).replaceAllUsesWith(multiDeclareOp);
        }

        // remove candidate inputs at call sites
        for (auto callOp : callOps) {
            callOp->eraseOperands(isCandidate);
        }

        // update function type and remove unused arguments
        auto newFunctionType = funcOp.getFunctionType().getWithoutArgsAndResults(isCandidate, llvm::BitVector{});
        funcOp.setFunctionType(newFunctionType);
        bodyBlock.eraseArguments(isCandidate);
    }

    // delete the constants which we converted to 'const.Rodata' ops
    for (auto pair : _declareOpToRodataOp.getOperationMap()) {
        pair.getFirst()->erase();
    }
}

llvm::BitVector ConvertConstArgsToMultiConstants::getIsCandidateBitVector(MutableArrayRef<mlir::func::CallOp> callOps) {
    size_t operandCount = callOps.front()->getNumOperands();
    size_t callCount = callOps.size();
    llvm::BitVector isCandidate(operandCount, true);

    // Determine for each argument if it is "constant". This is the case iff. for each
    // call site that arguments value is the direct result of a 'const.Declare' op.
    for (size_t operandIndex = 0; operandIndex < operandCount; ++operandIndex) {
        for (size_t callIndex = 0; callIndex < callCount; ++callIndex) {
            bool isDeclare = mlir::isa_and_nonnull<Const::DeclareOp>(
                    callOps[callIndex].getOperand(operandIndex).getDefiningOp());
            isCandidate[operandIndex] = isCandidate[operandIndex] && isDeclare;
        }
    }

    // No candidate is allowed to have shared operands with non-candidates. For example, if on all
    // call sites for operand 0 we have candidates [c0, c1] and for operand 1 we have [c0, x],
    // then operand 0 will not be considered a candidate anymore because it shares a constant (c0)
    // with an operand that is not a candidate (because it has non-constant input x).
    mlir::DenseSet<mlir::Value> nonCandidateSet;

    for (size_t operandIndex = 0; operandIndex < operandCount; ++operandIndex) {
        for (size_t callIndex = 0; callIndex < callCount; ++callIndex) {
            if (!isCandidate[operandIndex]) {
                nonCandidateSet.insert(callOps[callIndex].getOperand(operandIndex));
            }
        }
    }

    for (size_t operandIndex = 0; operandIndex < operandCount; ++operandIndex) {
        for (size_t callIndex = 0; callIndex < callCount; ++callIndex) {
            auto operand = callOps[callIndex].getOperand(operandIndex);
            isCandidate[operandIndex] = isCandidate[operandIndex] && !nonCandidateSet.contains(operand);
        }
    }

    // Check if the constants associated with one operand all have the same transformations and underlying
    // type. If not, return an empty candidate vector to signal failure, as this is unhandled at the moment.
    for (size_t operandIndex = 0; operandIndex < operandCount; ++operandIndex) {
        if (!isCandidate[operandIndex]) {
            continue;
        }

        auto declareOp = callOps.front().getOperand(operandIndex).getDefiningOp<Const::DeclareOp>();
        auto firstType = declareOp.getContentAttr().getBaseContent().getType();
        auto firstTrans = declareOp.getContentAttr().getTransformationsAttr();

        for (size_t callIndex = 1; callIndex < callCount; ++callIndex) {
            auto thisDeclareOp = callOps[callIndex].getOperand(operandIndex).getDefiningOp<Const::DeclareOp>();
            auto thisType = thisDeclareOp.getContentAttr().getBaseContent().getType();
            auto thisTrans = thisDeclareOp.getContentAttr().getTransformationsAttr();

            // at the moment this should not happen and means something likely broke
            auto callOp = callOps[callIndex];
            VPUX_THROW_WHEN(thisType != firstType || thisTrans != firstTrans,
                            "A possible bundle would contain 'const.Declare' ops with differing base content types or "
                            "transformations for 'func.Func' {0}. This is unexpected and might indicate "
                            "a problem with outlining!",
                            callOp.getCallee());
        }
    }

    return isCandidate;
}

// returns a mlir::SymbolRefAttr representing @root::@leaf
mlir::SymbolRefAttr ConvertConstArgsToMultiConstants::getNestedSymbol(StringRef root, StringRef leaf) {
    auto leafAttr = mlir::SymbolRefAttr::get(mlir::StringAttr::get(&getContext(), leaf));
    return mlir::SymbolRefAttr::get(mlir::StringAttr::get(&getContext(), root), {leafAttr});
}

bool ConvertConstArgsToMultiConstants::checkPrecondition(mlir::func::FuncOp netFunc) {
    auto moduleOp = getOperation();

    if (!moduleOp.getOps<Const::DataOp>().empty()) {
        moduleOp.emitError("IR contains unexpected op 'const.Data'");
        return false;
    }

    if (!moduleOp.getOps<Const::BundleDataOp>().empty()) {
        moduleOp.emitError("IR contains unexpected op 'const.BundleData'");
        return false;
    }

    for (auto funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
        if (funcOp == netFunc) {
            continue;
        }

        if (!funcOp.getOps<mlir::func::CallOp>().empty()) {
            funcOp->emitOpError(
                    formatv("{0} contains disallowed 'func.Call' op outside of net func", funcOp.getSymName()));
            return false;
        }
    }

    return true;
}

Const::RodataBundleOp ConvertConstArgsToMultiConstants::buildRodataBundleOp(Const::DataOp dataOp,
                                                                            Const::BundleDataOp bundleDataOp,
                                                                            ArrayRef<mlir::func::CallOp> callOps,
                                                                            size_t operandIndex) {
    auto dataBlock = &dataOp.getBody().front();
    auto bundleDataBlock = &bundleDataOp.getBody().front();

    mlir::OpBuilder dataBodyBuilder(dataBlock, dataBlock->end());
    mlir::OpBuilder bundleDataBodyBuilder(bundleDataBlock, bundleDataBlock->end());

    SmallVector<mlir::SymbolRefAttr> rodataSymbols;
    for (auto callOp : callOps) {
        // A 'const.Declare' might be used as an input to multiple arguments or even multiple functions. This is why we
        // perform some deduplication logic so that 'const.RodataBundle' can nicely reuse symbols.
        auto declareOp = callOp.getOperand(operandIndex).getDefiningOp<Const::DeclareOp>();
        auto rodataOp =
                mlir::dyn_cast_or_null<Const::RodataOp>(_declareOpToRodataOp.getOperationMap().lookup(declareOp));

        // create 'const.Rodata' op from a 'const.Declare' op, if it doesn't already exist
        if (rodataOp == nullptr) {
            std::string symName = formatv("{0}{1}", RODATA_PREFIX, _declareOpToRodataOp.getOperationMap().size());
            auto content = declareOp.getContentAttr().getBaseContent();
            rodataOp = dataBodyBuilder.create<Const::RodataOp>(dataOp->getLoc(), symName, content);
            _declareOpToRodataOp.map(declareOp, rodataOp);
        }

        rodataSymbols.push_back(getNestedSymbol(DATA_SYMBOL, rodataOp.getSymName()));
    }

    // create RodataBundleOp
    std::string symName = formatv("{0}{1}", RODATA_BUNDLE_PREFIX, _bundleCount++);
    auto bundleType = mlir::func::CallOp(callOps.front())  // create copy to circumvent const reference
                              .getOperand(operandIndex)
                              .getDefiningOp<Const::DeclareOp>()
                              .getContentAttr()
                              .getBaseContent()
                              .getType();
    auto rodataBundleOp = bundleDataBodyBuilder.create<Const::RodataBundleOp>(bundleDataOp->getLoc(), symName,
                                                                              rodataSymbols, bundleType);
    return rodataBundleOp;
}

Const::MultiDeclareOp ConvertConstArgsToMultiConstants::buildMultiDeclareOp(
        mlir::func::FuncOp funcOp, Const::RodataBundleOp rodataBundleOp,
        Const::TransformAttrInterfaceArrayAttr transformationsAttr) {
    // builder that inserts at the beginning of funcOp's body
    auto bodyBlock = &funcOp.getFunctionBody().front();
    mlir::OpBuilder funcBuilder(bodyBlock, bodyBlock->begin());

    // create MultiContentSymbolAttr
    auto bundleDataOp = rodataBundleOp->getParentOfType<Const::BundleDataOp>();
    auto symbol = getNestedSymbol(bundleDataOp.getSymName(), rodataBundleOp.getSymName());
    auto multiContentSymbolAttr = Const::MultiContentSymbolAttr::get(
            &getContext(), symbol, rodataBundleOp.getBundleType(), transformationsAttr);

    // create 'const.MultiDeclare'
    auto finalType = Const::inferFinalType(rodataBundleOp.getBundleType(), transformationsAttr);
    auto multiDeclareOp = funcBuilder.create<Const::MultiDeclareOp>(funcOp.getLoc(), finalType, multiContentSymbolAttr);

    return multiDeclareOp;
}

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertConstArgsToMultiConstantsPass(Logger log) {
    return std::make_unique<ConvertConstArgsToMultiConstants>(log);
}
