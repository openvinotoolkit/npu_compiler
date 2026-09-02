//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/utils.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/SmallBitVector.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Math/Transforms/Approximation.h>
#include <mlir/Dialect/Math/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Pass/AnalysisManager.h>

// TODO: E66812, it should be sufficient to have warnings disabled for 3-rd parties
// in CMake but it does not work for early versions of MSVC 2019
#ifdef _MSC_VER
#pragma warning(push)
#endif
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

namespace vpux {
#define GEN_PASS_DECL_CONVERTAFFINE2LLVM
#define GEN_PASS_DEF_CONVERTAFFINE2LLVM
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

class ConvertAffine2LLVMPass final : public impl::ConvertAffine2LLVMBase<ConvertAffine2LLVMPass> {
public:
    using ArgIndices = SmallVector<size_t>;
    using SwKernelUses = SmallVector<vpux::VPUIP::SwKernelOp>;
    explicit ConvertAffine2LLVMPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

    // Produces a bit vector (allUseMask) with (number of memref inputs + number of memref
    // outputs) bits. The nth bit is set iff for all invocations of funcOp via SW kernels
    // the corresponding input or output does not alias the other input or output memrefs.
    void getArgMemRefNoAliasMask(mlir::func::FuncOp funcOp, llvm::SmallBitVector& allUseMask, SwKernelUses& funcUses);

    // Produces a vector containing all argument indices for which we can add llvm.noalias for
    // the function signature of funcOp converted to the LLVM dialect.
    void getLLVMArgNoAliasMask(mlir::func::FuncOp funcOp, ArgIndices& noaliasIndices,
                               mlir::TypeConverter::SignatureConversion& sigConv, SwKernelUses& funcUses);

    // Given some argument indices for funcOp, add llvm.noalias attributes for the
    // arguments that correspond to those indices.
    void addNoAliasLLVMAttributes(mlir::LLVM::LLVMFuncOp funcOp, ArgIndices& noAliasArgIdx);

    // For each memref argument whose descriptor fields (offset, sizes, strides) are
    // statically known from the pre-conversion memref type, replace uses of those
    // LLVM block arguments with integer constants at the function entry point. This
    // recovers static information about the layout that would otherwise be lost when
    // a memref argument is expanded into the descriptor form.
    void replaceStaticMemRefDescriptorArgs(mlir::LLVM::LLVMFuncOp newFuncOp, llvm::ArrayRef<mlir::Type> origArgTypes,
                                           mlir::TypeConverter::SignatureConversion& sigConv);

    // Rewrite function index arguments to i64 and insert entry-block casts back to
    // index, then RAUW argument uses with the cast results.
    void convertIndexArgsToI64(mlir::func::FuncOp funcOp);
};

void ConvertAffine2LLVMPass::getArgMemRefNoAliasMask(mlir::func::FuncOp funcOp, llvm::SmallBitVector& allUseMask,
                                                     SwKernelUses& funcUses) {
    unsigned memrefArgCount = llvm::count_if(funcOp.getArguments(), [](const mlir::Value val) {
        return mlir::isa<mlir::MemRefType>(val.getType());
    });

    allUseMask = llvm::SmallBitVector(memrefArgCount, true);

    for (auto swKern : funcUses) {
        auto parentFunc = swKern->getParentOfType<mlir::func::FuncOp>();
        auto& aliasesInfo = getChildAnalysis<AliasesInfo>(parentFunc);
        // Count the number of times each root was seen into a map while
        // going over all inputs/outputs. If a memref has a root that was
        // seen more than once then we know that it will alias with
        // another memref.
        llvm::DenseMap<mlir::Value, unsigned> rootBufferCount;
        llvm::SmallVector<AliasesInfoBase::ValuesVector> argRoots;

        // Calling convention: the SwKernelOp's inputs followed by its outputs correspond
        // one-to-one, in order, to the leading memref arguments of the kernel function.
        SmallVector<mlir::Value> memrefFuncArgs;
        for (auto funcArg : funcOp.getArguments()) {
            if (mlir::isa<mlir::MemRefType>(funcArg.getType())) {
                memrefFuncArgs.push_back(funcArg);
            }
        }
        VPUX_THROW_UNLESS(memrefFuncArgs.size() == swKern.getInputs().size() + swKern.getOutputs().size(),
                          "SwKernelOp input/output operand count does not match the number of memref arguments of "
                          "kernel function '{0}'",
                          funcOp.getName());

        auto addArgInfo = [&](mlir::Value callArg, mlir::Value calleeArg) {
            // If the corresponding funcOp argument is unused, treat it as having no roots.
            // This allows other arguments that do alias it to still have noalias.
            if (calleeArg.use_empty()) {
                argRoots.emplace_back();
                return;
            }
            // Record the set of roots for this argument.
            argRoots.emplace_back(aliasesInfo.getRoots(callArg));
            // Bump the counter for every root of this argument.
            for (auto buf : argRoots.back()) {
                rootBufferCount.try_emplace(buf, 0).first->second++;
            }
        };

        // Record roots for every input/output memrefs. Note that all
        // inputs and outputs for a SwKernelOp are known to be memrefs.
        size_t argIdx = 0;
        for (auto input : swKern.getInputs()) {
            addArgInfo(input, memrefFuncArgs[argIdx++]);
        }
        for (auto output : swKern.getOutputs()) {
            addArgInfo(output, memrefFuncArgs[argIdx++]);
        }

        // Construct the bitmask for this invocation and &-it to
        // the overall bitmask. At each location (i.e. for each memref),
        // if one of our roots was seen more than once then we alias
        // with another memref in this SW kernel call.
        llvm::SmallBitVector mask(argRoots.size(), false);
        for (unsigned i = 0, e = argRoots.size(); i < e; ++i) {
            mask[i] = !llvm::any_of(argRoots[i], [&](mlir::Value root) {
                return rootBufferCount[root] > 1;
            });
        }
        allUseMask = allUseMask & mask;
    }
}

void ConvertAffine2LLVMPass::getLLVMArgNoAliasMask(mlir::func::FuncOp funcOp, ArgIndices& noaliasIndices,
                                                   mlir::TypeConverter::SignatureConversion& sigConv,
                                                   SwKernelUses& funcUses) {
    llvm::SmallBitVector allUseMask;
    getArgMemRefNoAliasMask(funcOp, allUseMask, funcUses);

    uint64_t memrefArgCount = 0;
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
        if (!mlir::isa<mlir::MemRefType>(funcOp.getArgument(i).getType())) {
            continue;
        }
        // If this is a memref but we can't add noalias to it skip it.
        if (!allUseMask[memrefArgCount++]) {
            continue;
        }
        if (auto argConvRes = sigConv.getInputMapping(i)) {
            // First argument from the converted memref is the unaligned
            // pointer which won't be used and we can skip it. The second
            // argument is what gets used and should get the noalias
            // attribute.
            noaliasIndices.push_back(argConvRes->inputNo + 1);
        }
    }
}

void ConvertAffine2LLVMPass::addNoAliasLLVMAttributes(mlir::LLVM::LLVMFuncOp funcOp, ArgIndices& noAliasArgIdx) {
    mlir::OpBuilder builder(funcOp);
    for (auto index : noAliasArgIdx) {
        funcOp.setArgAttr(static_cast<unsigned>(index), mlir::LLVM::LLVMDialect::getNoAliasAttrName(),
                          builder.getUnitAttr());
    }
}

void ConvertAffine2LLVMPass::replaceStaticMemRefDescriptorArgs(mlir::LLVM::LLVMFuncOp newFuncOp,
                                                               llvm::ArrayRef<mlir::Type> origArgTypes,
                                                               mlir::TypeConverter::SignatureConversion& sigConv) {
    auto& entryBlock = newFuncOp.getFunctionBody().front();
    mlir::OpBuilder builder(&entryBlock, entryBlock.begin());
    auto* ctx = newFuncOp.getContext();
    auto i32Ty = mlir::IntegerType::get(ctx, 32);
    auto loc = newFuncOp.getLoc();

    auto makeConst = [&](int64_t val) -> mlir::Value {
        return builder.create<mlir::LLVM::ConstantOp>(loc, i32Ty, mlir::IntegerAttr::get(i32Ty, val));
    };

    for (unsigned i = 0, e = origArgTypes.size(); i < e; ++i) {
        auto memrefType = mlir::dyn_cast<mlir::MemRefType>(origArgTypes[i]);
        if (!memrefType) {
            continue;
        }

        auto mapping = sigConv.getInputMapping(i);
        VPUX_THROW_WHEN(!mapping, "Expected valid LLVM mapping for memref arg {0}", i);

        // The MLIR memref descriptor layout for a rank-N memref, starting at inputNo:
        //   +0  allocated ptr
        //   +1  aligned ptr
        //   +2  offset  (i32)
        //   +3 .. +3+N-1         sizes[0..N-1]  (i32)
        //   +3+N .. +3+2N-1      strides[0..N-1] (i32)
        const unsigned base = mapping->inputNo;
        const int64_t rank = memrefType.getRank();
        VPUX_THROW_WHEN(mapping->size != static_cast<size_t>(2 * rank + 3),
                        "Unexpected memref descriptor mapping for arg {0}: expected {1} arguments, got {2}", i,
                        static_cast<size_t>(2 * rank + 3), mapping->size);

        // Offsets within the descriptor relative to `base` (see layout comment above).
        constexpr unsigned OFFSET_INDEX = 2;
        constexpr unsigned SIZES_INDEX = 3;

        auto offsetArgIdx = [base]() {
            return base + OFFSET_INDEX;
        };
        auto sizeArgIdx = [base](int64_t dim) {
            return base + SIZES_INDEX + static_cast<unsigned>(dim);
        };
        auto strideArgIdx = [base, rank](int64_t dim) {
            return base + SIZES_INDEX + static_cast<unsigned>(rank + dim);
        };

        auto [staticStrides, staticOffset] = memrefType.getStridesAndOffset();

        if (staticOffset != mlir::ShapedType::kDynamic) {
            newFuncOp.getArgument(offsetArgIdx()).replaceAllUsesWith(makeConst(staticOffset));
        }

        auto shape = memrefType.getShape();
        for (int64_t dim = 0; dim < rank; ++dim) {
            if (shape[dim] != mlir::ShapedType::kDynamic) {
                newFuncOp.getArgument(sizeArgIdx(dim)).replaceAllUsesWith(makeConst(shape[dim]));
            }
            if (staticStrides[dim] != mlir::ShapedType::kDynamic) {
                newFuncOp.getArgument(strideArgIdx(dim)).replaceAllUsesWith(makeConst(staticStrides[dim]));
            }
        }
    }
}

void ConvertAffine2LLVMPass::convertIndexArgsToI64(mlir::func::FuncOp funcOp) {
    VPUX_THROW_WHEN(funcOp.isExternal(), "Expected non-external function '{0}'", funcOp.getName());

    auto oldFuncType = funcOp.getFunctionType();
    auto* ctx = funcOp.getContext();
    auto i64Ty = mlir::IntegerType::get(ctx, 64);
    auto loc = funcOp.getLoc();

    llvm::SmallVector<unsigned> indexArgPositions;
    indexArgPositions.reserve(oldFuncType.getNumInputs());
    for (unsigned i = 0, e = oldFuncType.getNumInputs(); i < e; ++i) {
        if (mlir::isa<mlir::IndexType>(oldFuncType.getInput(i))) {
            indexArgPositions.push_back(i);
        }
    }

    if (indexArgPositions.empty()) {
        return;
    }

    llvm::SmallVector<mlir::Type> newInputTypes(oldFuncType.getInputs().begin(), oldFuncType.getInputs().end());
    for (auto indexArgPos : indexArgPositions) {
        newInputTypes[indexArgPos] = i64Ty;
    }

    auto newFuncType = mlir::FunctionType::get(ctx, newInputTypes, oldFuncType.getResults());
    funcOp.setType(newFuncType);

    auto& entryBlock = funcOp.getBody().front();
    mlir::OpBuilder builder(&entryBlock, entryBlock.begin());
    for (auto indexArgPos : indexArgPositions) {
        auto arg = entryBlock.getArgument(indexArgPos);
        arg.setType(i64Ty);
        auto cast = builder.create<mlir::index::CastSOp>(loc, builder.getIndexType(), arg);
        arg.replaceAllUsesExcept(cast.getResult(), cast.getOperation());
    }
}

void ConvertAffine2LLVMPass::safeRunOnModule() {
    auto& ctx = getContext();

    // LLVMConversionTarget defines LLVM as single Legal dialect by default
    mlir::LLVMConversionTarget target(ctx);

    // We want to completely lower to LLVM, so we use a `FullConversion`. This
    // ensures that only legal operations will remain after the conversion.
    auto module = getOperation();

    mlir::LowerToLLVMOptions options(&ctx);
    options.overrideIndexBitwidth(32);
    mlir::LLVMTypeConverter typeConverter(&ctx, options);

    auto vpuSwmoduleOp = module.lookupSymbol<mlir::ModuleOp>("VPU.SW");

    // Create a cache of uses of every FuncOp by SwKernelOps.
    llvm::DenseMap<mlir::func::FuncOp, SwKernelUses> funcUseMap;
    module.walk([&](vpux::VPUIP::SwKernelOp swKernelOp) {
        auto kernelFunc = module.lookupSymbol<mlir::func::FuncOp>(swKernelOp.getKernelFunctionAttr());
        funcUseMap[kernelFunc].push_back(swKernelOp);
    });

    // After adding support for Quantize/Dequantize per channel/axis for SCG
    // the scale constants are stored in memref.global, which is outside of
    // func.func modules - this ensures that they are also lowered to LLVM
    // E204326 [ShaveCodeGen] Remove global ops/Move them to capsule arguments
    mlir::RewritePatternSet patterns(&ctx);
    mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    // Collect GlobalOps first to avoid iterator invalidation
    llvm::SmallVector<mlir::memref::GlobalOp> globalOps;
    module.walk([&](mlir::memref::GlobalOp globalOp) {
        globalOps.push_back(globalOp);
    });

    for (auto globalOp : globalOps) {
        if (failed(applyFullConversion(globalOp, target, frozenPatterns))) {
            signalPassFailure();
        }
    }

    for (auto funcOp :
         llvm::make_early_inc_range(vpuSwmoduleOp.getOperation()->getRegion(0).getOps<mlir::func::FuncOp>())) {
        if (funcOp.getBlocks().size() == 0 && !funcOp->hasAttr(ShaveCodeGen::IntrinsicAttrName)) {
            // Ignore functions which were not generated by ShaveCodeGen.
            continue;
        }

        mlir::TypeConverter::SignatureConversion sigConv(funcOp.getNumArguments());
        llvm::SmallVector<mlir::Type> origArgTypes;
        ArgIndices noaliasIndices;
        if (!funcOp.isExternal()) {
            convertIndexArgsToI64(funcOp);
            typeConverter.convertFunctionSignature(funcOp.getFunctionType(), false, false, sigConv);
            origArgTypes.assign(funcOp.getArgumentTypes().begin(), funcOp.getArgumentTypes().end());
            getLLVMArgNoAliasMask(funcOp, noaliasIndices, sigConv, funcUseMap[funcOp]);
        }
        auto funcName = funcOp.getSymName().str();

        // Executing the population of patterns for each loop iteration since
        //   the patterns variable is altered by applyFullConversion(...std::move(patterns)).
        mlir::RewritePatternSet patterns(&ctx);

        mlir::populateAffineToStdConversionPatterns(patterns);
        mlir::populateSCFToControlFlowConversionPatterns(patterns);
        mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
        mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
        mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        mlir::index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);

        if (failed(applyFullConversion(funcOp, target, std::move(patterns)))) {
            signalPassFailure();
            return;
        }

        auto sym = mlir::FlatSymbolRefAttr::get(&ctx, funcName);
        auto newFuncOp = vpuSwmoduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(sym);
        if (newFuncOp == nullptr) {
            signalPassFailure();
            return;
        }

        if (!newFuncOp.isExternal()) {
            // Set noalias attributes on memrefs. This informs code generation
            // that there are no dependencies between loads and stores to
            // different memrefs. This at the moment enables vectorization without
            // loop versioning/runtime checks. It should also enable other
            // optimizations as well.
            addNoAliasLLVMAttributes(newFuncOp, noaliasIndices);

            // Replace LLVM descriptor arguments whose values are statically known from
            // the original memref types with integer constants. This exposes offset=0
            // and contiguous strides to the optimizer for no-layout memrefs, recovering
            // information that is otherwise lost during the conversion.
            replaceStaticMemRefDescriptorArgs(newFuncOp, origArgTypes, sigConv);
        }
    }
}

}  // namespace

//
// createConvertAffine2LLVMPass
//

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createConvertAffine2LLVMPass(Logger log) {
    return std::make_unique<ConvertAffine2LLVMPass>(log);
}
