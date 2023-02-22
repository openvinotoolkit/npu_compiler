//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUIP/sw_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"

#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>

using namespace vpux;

namespace {

/// Insert an allocation and deallocation for the given MemRefType.
static mlir::Value insertAllocAndDealloc(mlir::MemRefType type, mlir::Location loc, mlir::PatternRewriter& rewriter,
                                         bool isUnit = false) {
    auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);
    if (isUnit) {
        alloc = rewriter.create<mlir::memref::AllocOp>(loc, mlir::MemRefType::get({1}, type.getElementType()));
    }

    // Make sure to allocate at the beginning of the block.
    auto* parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());

    // Make sure to deallocate this alloc at the end of the block.
    auto dealloc = rewriter.create<mlir::memref::DeallocOp>(loc, alloc);
    dealloc->moveBefore(&parentBlock->back());
    return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, a range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for the iteration. It returns a value to store at the
/// current index of the iteration.
using LoopIterationFn = llvm::function_ref<mlir::Value(mlir::OpBuilder& rewriter, mlir::ValueRange memRefOperands,
                                                       mlir::ValueRange loopIvs)>;

static void lowerOpToLoops(mlir::Operation* op, mlir::ValueRange operands, mlir::PatternRewriter& rewriter,
                           LoopIterationFn processIteration) {
    auto memRefType = (*op->result_type_begin()).cast<mlir::MemRefType>();
    auto loc = op->getLoc();

    // Create a nest of affine loops, with one loop per dimension of the shape.
    // The buildAffineLoopNest function takes a callback that is used to construct
    // the body of the innermost loop given a builder, a location and a range of
    // loop induction variables.
    SmallVector<int64_t, 4> lowerBounds(memRefType.getRank(), /*Value=*/0);
    SmallVector<int64_t, 4> steps(memRefType.getRank(), /*Value=*/1);
    buildAffineLoopNest(rewriter, loc, lowerBounds, memRefType.getShape(), steps,
                        [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
                            // Call the processing function with the rewriter, the memref operands,
                            // and the loop induction variables. This function will return the value
                            // to store at the current index.
                            mlir::Value valueToStore = processIteration(nestedBuilder, operands, ivs);

                            VPUX_THROW_UNLESS(operands.size() >= 2, "Need to have 2 operands");

                            nestedBuilder.create<mlir::AffineStoreOp>(loc, valueToStore, operands[1], ivs);
                        });

    // Replace this operation with the generated alloc.
    // rewriter.replaceOp(op, alloc);
    rewriter.replaceOp(op, operands[1]);  // We return the container in which we stored the computed values.
}

struct UnaryOpLoweringCos : public mlir::ConversionPattern {
    UnaryOpLoweringCos(mlir::MLIRContext* ctx): mlir::ConversionPattern(IERT::CosOp::getOperationName(), 1, ctx) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(op, operands, rewriter,
                       [loc](mlir::OpBuilder& builder, mlir::ValueRange memRefOperands, mlir::ValueRange loopIvs) {
                           // Generate an adaptor for the remapped operands of the UnaryOp. This
                           // allows for using the nice named accessors that are generated by the
                           // ODS.
                           IERT::CosOp::Adaptor unaryAdaptor(memRefOperands);

                           // Generate load for the element of 'lhs' at the inner loop.
                           auto loadedOpnd = builder.create<mlir::AffineLoadOp>(loc, unaryAdaptor.input(), loopIvs);

                           auto cosOp = builder.create<mlir::math::CosOp>(loc, loadedOpnd);

                           return cosOp;
                       });

        return mlir::success();
    }
};

struct UnaryOpLoweringHSwish : public mlir::ConversionPattern {
    UnaryOpLoweringHSwish(mlir::MLIRContext* ctx): mlir::ConversionPattern(IERT::HSwishOp::getOperationName(), 1, ctx) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();
        lowerOpToLoops(
                op, operands, rewriter,
                [loc, op](mlir::OpBuilder& builder, mlir::ValueRange memRefOperands, mlir::ValueRange loopIvs) {
                    // Generate an adaptor for the remapped operands of the UnaryOp. This
                    // allows for using the nice named accessors that are generated by the
                    // ODS.
                    IERT::HSwishOp::Adaptor unaryAdaptor(memRefOperands);

                    // Generate load for the element.
                    auto loadedOpnd = builder.create<mlir::AffineLoadOp>(loc, unaryAdaptor.input(), loopIvs);

                    // IMPORTANT: HSwish(x) = x * min(max(x+3, 0), 6) / 6
                    float f;
                    mlir::MLIRContext* ctx = op->getContext();

                    auto memRefOperands0Type = unaryAdaptor.input().getType();

                    auto memRefOperands0TypeMemref = memRefOperands0Type.dyn_cast_or_null<mlir::MemRefType>();
                    VPUX_THROW_UNLESS(memRefOperands0TypeMemref != nullptr, "Abnormal situation encountered");

                    mlir::arith::ConstantFloatOp valOp1;
                    bool typeIsF32 = memRefOperands0TypeMemref.getElementType().isF32();
                    bool typeIsF16 = memRefOperands0TypeMemref.getElementType().isF16();

                    if (typeIsF32) {
                        f = 3.0;
                        valOp1 = builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(f),
                                                                              mlir::FloatType::getF32(ctx));
                    } else if (typeIsF16) {
                        llvm::APInt data(16, 0x4200);  // 0x4200 is value for 3.0 for f16

                        llvm::APFloat value = llvm::APFloat(llvm::APFloat::IEEEhalf(), data);

                        valOp1 = builder.create<mlir::arith::ConstantFloatOp>(loc, value, mlir::FloatType::getF16(ctx));
                    }

                    auto addFOp = builder.create<mlir::arith::AddFOp>(loc, loadedOpnd, valOp1->getResult(0));

                    mlir::arith::ConstantFloatOp valOp2;
                    if (typeIsF32) {
                        f = 0.0;
                        valOp2 = builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(f),
                                                                              mlir::FloatType::getF32(ctx));
                    } else if (typeIsF16) {
                        llvm::APInt data(16, 0x0);
                        llvm::APFloat value = llvm::APFloat(llvm::APFloat::IEEEhalf(), data);
                        valOp2 = builder.create<mlir::arith::ConstantFloatOp>(loc, value, mlir::FloatType::getF16(ctx));
                    }

                    auto maxFOp = builder.create<mlir::arith::MaxFOp>(loc, addFOp, valOp2->getResult(0));

                    mlir::arith::ConstantFloatOp valOp3;
                    if (typeIsF32) {
                        f = 6.0;
                        valOp3 = builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(f),
                                                                              mlir::FloatType::getF32(ctx));
                    } else if (typeIsF16) {
                        llvm::APInt data(16, 0x4600);  // 0x4600 is value for 6.0 for f16
                        llvm::APFloat value = llvm::APFloat(llvm::APFloat::IEEEhalf(), data);
                        valOp3 = builder.create<mlir::arith::ConstantFloatOp>(loc, value, mlir::FloatType::getF16(ctx));
                    }

                    auto minFOp = builder.create<mlir::arith::MinFOp>(loc, maxFOp, valOp3->getResult(0));

                    auto divFOp = builder.create<mlir::arith::DivFOp>(loc, minFOp, valOp3->getResult(0));

                    auto mulFOp = builder.create<mlir::arith::MulFOp>(loc, divFOp, loadedOpnd);

                    return mulFOp;
                });

        return mlir::success();
    }
};

struct UnaryOpLoweringSoftMax : public mlir::ConversionPattern {
    UnaryOpLoweringSoftMax(mlir::MLIRContext* ctx)
            : mlir::ConversionPattern(IERT::SoftMaxOp::getOperationName(), 1, ctx) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op, ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        // SoftMax is defined for example at https://slaystudy.com/implementation-of-softmax-activation-function-in-c-c/

        auto loc = op->getLoc();

        auto memRefType = (*op->result_type_begin()).cast<mlir::MemRefType>();

        auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter, true);

        SmallVector<int64_t, 4> lowerBounds(memRefType.getRank(), /*Value=*/0);
        SmallVector<int64_t, 4> steps(memRefType.getRank(), /*Value=*/1);
        buildAffineLoopNest(rewriter, loc, lowerBounds, memRefType.getShape(), steps,
                            [&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
                                // Call the processing function with the rewriter, the memref operands,
                                // and the loop induction variables. This function will return the value
                                // to store at the current index.

                                VPUX_THROW_UNLESS(operands.size() >= 2, "Need to have 2 operands");

                                auto loadedVal = nestedBuilder.create<mlir::AffineLoadOp>(loc, operands[0], ivs);
                                auto expTerm = nestedBuilder.create<mlir::math::ExpOp>(loc, loadedVal);

                                mlir::Value zeroIndex = nestedBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);

                                auto loadedVal2 = nestedBuilder.create<mlir::AffineLoadOp>(loc, alloc, zeroIndex);

                                auto addOp = nestedBuilder.create<mlir::arith::AddFOp>(loc, loadedVal2, expTerm);

                                nestedBuilder.create<mlir::AffineStoreOp>(loc, addOp, alloc, zeroIndex);
                            });

        lowerOpToLoops(
                op, operands, rewriter,
                [alloc, loc](mlir::OpBuilder& builder, mlir::ValueRange memRefOperands, mlir::ValueRange loopIvs) {
                    // Generate an adaptor for the remapped operands of the UnaryOp. This
                    // allows for using the nice named accessors that are generated by the
                    // ODS.
                    IERT::SoftMaxOp::Adaptor unaryAdaptor(memRefOperands);

                    auto loadedOpnd = builder.create<mlir::AffineLoadOp>(loc, unaryAdaptor.input(), loopIvs);

                    // Following
                    // https://slaystudy.com/implementation-of-softmax-activation-function-in-c-c/,
                    //   softmax[i] = exp(input[i]) / sum, where sum = \sum_k exp(input[k])

                    auto expOp = builder.create<mlir::math::ExpOp>(loc, loadedOpnd);

                    mlir::Value zeroIndex = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
                    auto acc = builder.create<mlir::AffineLoadOp>(loc, alloc, zeroIndex);

                    auto binOp1 = builder.create<mlir::arith::DivFOp>(loc, expOp, acc->getResult(0));

                    return binOp1;
                });

        return mlir::success();
    }
};

//
// ConvertSWLayers2AffinePass
//

class ConvertSWLayers2AffinePass final : public ConvertSWLayers2AffineBase<ConvertSWLayers2AffinePass> {
public:
    explicit ConvertSWLayers2AffinePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ConvertSWLayers2AffinePass::safeRunOnModule() {
    auto& ctx = getContext();
    auto module = getOperation();
    const auto arch = VPU::getArch(module);
    if (arch != VPU::ArchKind::VPUX37XX) {
        _log.trace("ConvertSWLayers2AffinePass enabled only for VPUX37XX device. Got: {0}", arch);
        return;
    }

    vpux::BufferizeTypeConverter typeConverter;

    mlir::ConversionTarget target(ctx);

    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to a combination of the
    // `Affine`, `Arithmetic`, `MemRef`, and `Standard` dialects.
    target.addLegalDialect<mlir::AffineDialect, mlir::arith::ArithmeticDialect, mlir::memref::MemRefDialect,
                           mlir::math::MathDialect, mlir::StandardOpsDialect>();

    // To avoid getting strange legalization errors with operations such as IERT::CosOp, IERT::AsinOp, etc
    target.addIllegalDialect<vpux::IERT::IERTDialect>();

    target.addLegalOp<mlir::memref::AllocOp>();
    target.addLegalOp<VPUIP::CopyOp>();
    vpux::populateBufferizeMaterializationLegality(target);

    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<UnaryOpLoweringHSwish>(&ctx);
    patterns.add<UnaryOpLoweringCos>(&ctx);
    patterns.add<UnaryOpLoweringSoftMax>(&ctx);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertLayers2AffinePass
//

std::unique_ptr<mlir::Pass> vpux::createConvertSWLayers2AffinePass(Logger log) {
    return std::make_unique<ConvertSWLayers2AffinePass>(log);
}
