//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTVPUIPCOPYTOSWCOPY
#define GEN_PASS_DEF_CONVERTVPUIPCOPYTOSWCOPY
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

mlir::FailureOr<bool> verifyDimsByteAligned(mlir::Location location, vpux::NDTypeInterface origOpInterface) {
    auto memShape = to_small_vector(origOpInterface.getMemShape());
    auto memStrides = to_small_vector(origOpInterface.getMemStrides());
    const auto elemSize = origOpInterface.getElemTypeSize().count();

    llvm::SmallVector<int64_t> inputBitShape(memShape.size());
    inputBitShape.back() = memShape.back() * elemSize;

    // Calculate tensor shape in bit
    // EX: in_shape = 1x2x3x3xu4
    //     strides = [36, 18, 6, 1]
    //                V
    //     shape_in_bit = (1*2*3*3*4) x (2*3*3*4) x (3*3*4) x (3*4) =>   72x72x36x12
    //     strides_in_bit = [36*4, 18*4, 6*4, 1*4] =>    [144, 72, 24, 4]
    for (int idx = memShape.size() - 2; idx >= 0; --idx) {
        inputBitShape[idx] = inputBitShape[idx + 1] * memShape[idx];
    }

    // Check if strides in bits are greater than dims in bits
    //     strides_in_bit[0] > shape_in_bit[1]   ==> is not contiguous
    //           (144)			    (72)
    bool stridesGreaterThanDims = false;
    for (size_t idx = 0; idx < memShape.size() - 1; ++idx) {
        if (inputBitShape[idx + 1] < memStrides[idx].count()) {
            stridesGreaterThanDims = true;
            break;
        }
    }
    // Check innermost dim alignment
    int64_t innerMostDim = memShape.pop_back_val();
    int64_t alignedDim = (innerMostDim * elemSize) % CHAR_BIT;
    if (stridesGreaterThanDims && (alignedDim != 0)) {
        return errorAt(location, "Strides > dimensions && innermost dim ({0}) is not aligned!", innerMostDim);
    }
    return true;
}

mlir::FailureOr<bool> verifyStridesByteAligned(mlir::Location location, vpux::NDTypeInterface origOpInterface) {
    // Based on DMA transactions and concepts of sizes and strides from dma_transaction_utils
    auto memShape = to_small_vector(origOpInterface.getMemShape());
    auto memStrides = to_small_vector(origOpInterface.getMemStrides());

    // Extend shape and strides to accommodate for element type size and batch stride
    const auto elemSize = origOpInterface.getElemTypeSize().count();
    memShape.push_back(elemSize);
    memStrides.insert(memStrides.begin(), memStrides.front() * memShape.front());
    auto innerMostIndex = memShape.size() - 1;

    llvm::SmallVector<Bit> reducedBitDims;
    llvm::SmallVector<Bit> reducedBitStrides;

    const auto alignToByteBoundary = [&](Bit val) {
        return alignMemSize(Bit(val), Byte(1));
    };

    // Iterate over dim/stride pairs and push cases of non compact strides
    int64_t accumulatedSize = 1;
    auto previousStrideInBits = Bit(1);
    for (int64_t dim = innerMostIndex; dim >= 0; --dim) {
        auto currentSize = memShape[dim];
        auto currentStrideInBits = memStrides[dim];

        accumulatedSize *= currentSize;
        // Found non-compact stride
        if (checked_cast<int64_t>(currentSize) * previousStrideInBits < currentStrideInBits) {
            reducedBitDims.push_back(Bit(accumulatedSize));
            reducedBitStrides.push_back(currentStrideInBits);
            accumulatedSize = elemSize;
        }

        previousStrideInBits = currentStrideInBits;
    }

    // Flush out remaining accumulated sizes.
    // Also handle scalar cases of 1 byte transfers.
    if (accumulatedSize > elemSize || reducedBitDims.empty()) {
        reducedBitDims.emplace_back(accumulatedSize);
        reducedBitStrides.push_back(memStrides.front());
    }

    if (std::any_of(std::begin(reducedBitStrides), std::prev(std::end(reducedBitStrides)), [&](auto bitStride) {
            return bitStride.count() % CHAR_BIT != 0;
        })) {
        return errorAt(location, "Non byte aligned bitStride!");
    }

    // Align all strides to byte size
    auto reducedStrides = to_small_vector(reducedBitStrides | vpux::transformed(alignToByteBoundary) |
                                          vpux::transformed([](auto bitStride) {
                                              return checked_cast<size_t>(bitStride.template to<Byte>().count());
                                          }));

    // Validate byte alignment for strides
    for (auto idx : irange(reducedBitStrides.size() - 1)) {
        if (reducedBitStrides[idx].count() % CHAR_BIT) {
            return errorAt(location, "Non byte aligned inner stride {0}", reducedBitStrides[idx]);
        }
    }
    return true;
};

SmallVector<mlir::Type> convertToUnrankedTypes(mlir::Value operand) {
    SmallVector<mlir::Type> result;
    if (auto type = mlir::dyn_cast_or_null<mlir::MemRefType>(operand.getType())) {
        result.emplace_back(mlir::UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace()));
    } else if (auto type = mlir::dyn_cast_or_null<VPUIP::BoundedBufferType>(operand.getType())) {
        const auto dataNDType = mlir::cast<vpux::NDTypeInterface>(type.getData());
        result.emplace_back(mlir::UnrankedMemRefType::get(dataNDType.getElementType(), dataNDType.getMemSpace()));
        const auto shapeNDType = mlir::cast<vpux::NDTypeInterface>(type.getDynamicShape());
        result.emplace_back(mlir::UnrankedMemRefType::get(shapeNDType.getElementType(), shapeNDType.getMemSpace()));
    } else if (auto type = mlir::dyn_cast_or_null<VPUIP::DistributedBufferType>(operand.getType())) {
        result.emplace_back(mlir::UnrankedMemRefType::get(type.getElementType(), type.getMemSpace()));
    } else {
        VPUX_THROW_UNLESS(!result.empty(),
                          "This type is not supported by createBuiltInFunction."
                          "Got: {0}",
                          operand.getType());
    }
    return result;
};

//
// ConvertVPUIPCopyToSWCopy
//

class ConvertVPUIPCopyToSWCopy final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    ConvertVPUIPCopyToSWCopy(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertVPUIPCopyToSWCopy::matchAndRewrite(VPUIP::CopyOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    auto origOpInterface = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    mlir::Location location = origOp.getLoc();
    const auto ctx = origOp.getContext();

    mlir::OperandRange operandsBuff(origOp.getInputs().begin(), origOp.getInputs().end());
    mlir::OperandRange outputsBuff(origOp.getOutputs().begin(), origOp.getOutputs().end());

    auto module = origOp->getParentOfType<mlir::ModuleOp>();

    SmallString builtInFunctionName{VPUIP::SW_KERNEL_NAME_PREFIX};
    auto nonNamespaceOpName = origOp->getName().getStringRef().slice(origOp->getName().getDialectNamespace().size() + 1,
                                                                     mlir::StringRef::npos);
    builtInFunctionName.append(nonNamespaceOpName);

    // Get operands/results types for builtin function
    SmallVector<mlir::Type> types;
    for (const auto& operand : operandsBuff) {
        types.append(convertToUnrankedTypes(operand));
    };
    for (const auto& result : outputsBuff) {
        types.append(convertToUnrankedTypes(result));
    };

    // Prepare bitOffsets attribute for SW.Copy
    size_t shapeSize = origOpInterface.getMemShape().size();
    SmallVector<int64_t> offsets(shapeSize, 0);
    mlir::ArrayAttr outBitOffsets, inBitOffsets;
    outBitOffsets = inBitOffsets = getIntArrayAttr(ctx, offsets);

    auto typeSize = origOpInterface.getElemTypeSize().count();
    // Transform SubView offsets to byte position of bits
    const auto transformOffsetToByte = [&](SmallVector<int64_t> offsetVec) {
        SmallVector<int64_t> newOffsets(0);
        for (auto element : offsetVec) {
            newOffsets.push_back((element * typeSize) / CHAR_BIT);
        }
        return newOffsets;
    };

    auto reverseIntArrayAttr = [&](const DimsOrder& inOrder, mlir::ArrayAttr arrayAttr) {
        if (inOrder.empty() || arrayAttr.empty()) {
            return arrayAttr;
        }
        const auto& origPerm = inOrder.toPermutation();
        const auto origArray = parseIntArrayAttr<int64_t>(arrayAttr);
        SmallVector<int64_t> permArray(arrayAttr.size());
        for (const auto srcInd : irange(origPerm.size())) {
            const auto dstInd = origPerm[srcInd].ind();
            const auto revSrcInd = origPerm.size() - 1 - srcInd;
            const auto revDstInd = dstInd;
            permArray[revDstInd] = origArray[revSrcInd];
        }

        return getIntArrayAttr(ctx, permArray);
    };

    // Verify if parameters are SubViews
    auto dimsOrder = origOpInterface.getDimsOrder();
    if (auto maybeSubViewInput = origOp->getOperand(0).getDefiningOp<VPUIP::SubViewOp>()) {
        SmallVector<int64_t> inOffsetVec = parseIntArrayAttr<int64_t>(maybeSubViewInput.getStaticOffsetsAttr());
        auto inBitOffsetsAttr = getIntArrayAttr(ctx, transformOffsetToByte(std::move(inOffsetVec)));
        inBitOffsets = reverseIntArrayAttr(dimsOrder, inBitOffsetsAttr);
    }
    if (auto maybeSubViewOutput = origOp->getOperand(1).getDefiningOp<VPUIP::SubViewOp>()) {
        SmallVector<int64_t> outOffsetVec = parseIntArrayAttr<int64_t>(maybeSubViewOutput.getStaticOffsetsAttr());
        auto outBitOffsetsAttr = getIntArrayAttr(ctx, transformOffsetToByte(std::move(outOffsetVec)));
        outBitOffsets = reverseIntArrayAttr(dimsOrder, outBitOffsetsAttr);
    }

    // Create SWKernelOp type Copy
    VPUIP::createRuntimeKernelDefinition(module, _log.nest());

    const int64_t tileIndex = 0;
    vpux::VPUIP::KernelInfo kernelInfo(SmallVector<mlir::Attribute>{inBitOffsets, outBitOffsets}, SmallString("copy"),
                                       SmallString("copy.cpp"), SmallString("copy"));
    auto builtInFunction = vpux::VPUIP::createBuiltInFunction(module, builtInFunctionName, types, kernelInfo.entryName,
                                                              kernelInfo.sourceFileName, kernelInfo.layerName, _log);

    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(location, operandsBuff, outputsBuff, builtInFunction,
                                                         getIntAttr(ctx, tileIndex));

    vpux::VPUIP::initSwKernel(swKernelOp, operandsBuff, outputsBuff, kernelInfo.args, _log.nest(),
                              /*swKernelRunOp=*/nullptr);

    _log.trace("Replace origin op {0} with new outputs from SW Kernel Copy", location);
    rewriter.replaceOp(origOp, swKernelOp);

    return mlir::success();
}

mlir::BlockArgument getRootBlockArgument(mlir::Value val, const AliasesInfo& aliasesInfo) {
    // There could be pure view ops between swKernelOp's operands and BlockArgument values
    auto rootBuffers = aliasesInfo.getRoots(val);
    if (rootBuffers.size() != 1) {
        return nullptr;
    }
    return mlir::dyn_cast<mlir::BlockArgument>(*rootBuffers.begin());
}

//
// ConvertVPUIPCopyToSWCopyPass
//

class ConvertVPUIPCopyToSWCopyPass final :
        public VPUIP::impl::ConvertVPUIPCopyToSWCopyBase<ConvertVPUIPCopyToSWCopyPass> {
public:
    explicit ConvertVPUIPCopyToSWCopyPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ConvertVPUIPCopyToSWCopyPass::safeRunOnModule() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<VPUIP::SwKernelOp>();
    target.addDynamicallyLegalOp<VPUIP::CopyOp>([&](VPUIP::CopyOp copyOp) {
        auto origOpInterfaceInput = mlir::cast<NDTypeInterface>(copyOp.getInput().getType());
        auto origOpInterfaceOutput = mlir::cast<NDTypeInterface>(copyOp.getOutput().getType());

        mlir::Type origOpType = origOpInterfaceInput.getElementType();
        mlir::Location location = copyOp.getLoc();
        const auto ctx = copyOp.getContext();

        if (origOpType != vpux::getSInt4Type(ctx) && origOpType != vpux::getUInt4Type(ctx)) {
            return true;
        }

        auto stridesByteAlignedInput = mlir::succeeded(verifyStridesByteAligned(location, origOpInterfaceInput));
        auto stridesByteAlignedOutput = mlir::succeeded(verifyStridesByteAligned(location, origOpInterfaceOutput));

        // Check if stride values are byte aligned
        if (stridesByteAlignedInput && stridesByteAlignedOutput) {
            auto dimsByteAlignedInput = mlir::succeeded(verifyDimsByteAligned(location, origOpInterfaceInput));
            auto dimsByteAlignedOutput = mlir::succeeded(verifyDimsByteAligned(location, origOpInterfaceOutput));
            // Verify byte alignment for dimensions regarding to strides
            if (dimsByteAlignedInput && dimsByteAlignedOutput) {
                return true;
            }
        }
        auto parentFunc = copyOp->getParentOfType<mlir::func::FuncOp>();
        auto& aliasesInfo = getChildAnalysis<AliasesInfo>(parentFunc);

        // Tracking number [E#160558]
        VPUX_THROW_UNLESS(!getRootBlockArgument(copyOp.getInput(), aliasesInfo),
                          "Got Input Network I/O block argument when converting VPUIP.Copy to SW.Copy."
                          "Direct access to Network I/O arguments is not supported!");
        VPUX_THROW_UNLESS(!getRootBlockArgument(copyOp.getOutput(), aliasesInfo),
                          "Got Output Network I/O block argument when converting VPUIP.Copy to SW.Copy."
                          "Direct access to Network I/O arguments is not supported!");
        return false;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConvertVPUIPCopyToSWCopy>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertVPUIPCopyToSWCopyPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertVPUIPCopyToSWCopyPass(Logger log) {
    return std::make_unique<ConvertVPUIPCopyToSWCopyPass>(log);
}
