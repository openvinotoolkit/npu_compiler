//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/codec_factory.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_COMPRESSWEIGHTSBTC
#define GEN_PASS_DEF_COMPRESSWEIGHTSBTC
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// CompressWeightsBTCPass
//

class CompressWeightsBTCPass final : public VPUIP::impl::CompressWeightsBTCBase<CompressWeightsBTCPass> {
public:
    explicit CompressWeightsBTCPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// NNDMAOpConverter
//

class NNDMAOpConverter final : public mlir::OpRewritePattern<VPUIP::NNDMAOp> {
public:
    NNDMAOpConverter(mlir::MLIRContext* ctx, const ICodec::CompressionAlgorithm& algo, VPU::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<VPUIP::NNDMAOp>(ctx), _log(log), _codec(vpux::makeCodec(algo, arch)) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::NNDMAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    ICodec::CompressionMode getCompressionMode(Const::DeclareOp constOp) const;

    Logger _log;
    const std::unique_ptr<ICodec> _codec;
    mlir::FailureOr<std::vector<uint8_t>> compressDataFromDeclareOp(Const::DeclareOp constOp,
                                                                    ICodec::CompressionMode compressionMode) const;
};

ICodec::CompressionMode NNDMAOpConverter::getCompressionMode(Const::DeclareOp constOp) const {
    if (!_codec->supportsFP16compression()) {
        return ICodec::CompressionMode::UINT8;
    }

    const auto inputElementType = constOp.getType().cast<vpux::NDTypeInterface>().getElementType();
    return inputElementType.isF16() ? ICodec::CompressionMode::FP16 : ICodec::CompressionMode::UINT8;
}

mlir::FailureOr<std::vector<uint8_t>> NNDMAOpConverter::compressDataFromDeclareOp(
        Const::DeclareOp constOp, ICodec::CompressionMode compressionMode) const {
    const auto content = constOp.getContent();
    const Byte totalInputSize = getTotalSize(constOp);
    std::vector<uint8_t> origData(checked_cast<size_t>(totalInputSize.count()));
    content.copyTo(MutableArrayRef(reinterpret_cast<char*>(origData.data()), origData.size()));

    return _codec->compress(origData, compressionMode, _log);
}

mlir::LogicalResult NNDMAOpConverter::matchAndRewrite(VPUIP::NNDMAOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto loc = origOp->getLoc();
    auto input = origOp.getInput();
    auto output = origOp.getOutputBuff();
    const auto inputType = input.getType().cast<vpux::NDTypeInterface>();
    const auto outputType = output.getType().cast<vpux::NDTypeInterface>();

    auto inConstOp = input.getDefiningOp<Const::DeclareOp>();
    if (inConstOp == nullptr) {
        return mlir::failure();
    }

    auto outBufferOp = output.getDefiningOp<VPURT::DeclareBufferOp>();
    if (outBufferOp == nullptr) {
        return mlir::failure();
    }

    if (outputType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
        _log.nest().trace("CompressedDMA only support CONST2CMX");
        return mlir::failure();
    }

    _log.trace("Check if can change to compressed DMA, operation - '{0}'", loc);

    const auto originInShape = inputType.getShape().raw();
    const auto originOutShape = outputType.getShape().raw();

    const auto strideInReqs = StrideReqs::compact(originInShape.size());
    const auto strideOutReqs = StrideReqs::compact(originOutShape.size());

    if (!strideInReqs.checkStrides(input) || !strideOutReqs.checkStrides(output)) {
        _log.nest().trace("Strides check failed");
        return mlir::failure();
    }

    if (outputType.isa<VPUIP::DistributedBufferType>()) {
        const auto distributedType = outputType.dyn_cast<VPUIP::DistributedBufferType>();
        const auto distributionAttr = distributedType.getDistribution();
        const auto distributionMode = distributionAttr.getMode().getValue();
        if (distributionMode != VPU::DistributionMode::DUPLICATED) {
            _log.nest().trace("Only DUPLICATE Distributed mode supported, mode - '{0}'",
                              VPU::stringifyDistributionMode(distributionMode));
            return mlir::failure();
        }
    }

    const Byte totalInputSize = getTotalSize(origOp.getInput());
    constexpr Byte MIN_INPUT_SIZE = 4_KB;
    if (totalInputSize < MIN_INPUT_SIZE) {
        _log.nest().trace("Size smaller than minimal '{0}' < '{1}'", totalInputSize.count(), MIN_INPUT_SIZE.count());
        return mlir::failure();
    }

    auto compressionMode = getCompressionMode(inConstOp);
    _log.trace("Compress constant '{0}', type - '{1}', compression mode: {2}", inConstOp->getLoc(), inputType,
               ICodec::compressionModeToStr(compressionMode));

    const auto compressedDataOrFailure = compressDataFromDeclareOp(inConstOp, compressionMode);

    if (mlir::failed(compressedDataOrFailure)) {
        return mlir::failure();
    }
    const auto compressedData = compressedDataOrFailure.value();

    const auto ctx = rewriter.getContext();
    auto u8Type = getUInt8Type(ctx);
    auto f16Type = mlir::FloatType::getF16(ctx);
    auto newDstType = outputType;
    mlir::MemRefType newSrcType;
    mlir::DenseElementsAttr newSrcContentAttr;

    if (compressionMode == ICodec::CompressionMode::UINT8) {
        const Shape newDstShape{totalInputSize.count(), 1, 1, 1};
        newDstType = mlir::isa<VPUIP::DistributedBufferType>(newDstType)
                             ? VPU::changeShapeElemTypeForDuplicatedDistributedBuffers(newDstType, newDstShape, u8Type)
                             : newDstType.changeShapeElemType(newDstShape, u8Type);

        const Shape compressedDataShape{checked_cast<int64_t>(compressedData.size()), 1, 1, 1};
        newSrcType = getMemRefType(compressedDataShape, u8Type, DimsOrder::NCHW, inputType.getMemSpace(),
                                   /*strides=*/StridesRef(), getSwizzlingSchemeAttr(inputType));
        newSrcType = mlir::cast<mlir::MemRefType>(
                vpux::setCompressionState(newSrcType, VPUIP::CompressionState::CompiletimeCompressed));
        const auto newSrcStorageType = mlir::RankedTensorType::get(compressedDataShape.raw(), u8Type);
        newSrcContentAttr = Const::createConstContent(newSrcStorageType, ArrayRef(compressedData));
    } else if (compressionMode == ICodec::CompressionMode::FP16) {
        unsigned f16TypeSizeBytes = f16Type.getWidth() / CHAR_BIT;
        const Shape newDstShape{totalInputSize.count() / f16TypeSizeBytes, 1, 1, 1};
        newDstType = mlir::isa<VPUIP::DistributedBufferType>(newDstType)
                             ? VPU::changeShapeElemTypeForDuplicatedDistributedBuffers(newDstType, newDstShape, f16Type)
                             : newDstType.changeShapeElemType(newDstShape, f16Type);

        const Shape compressedDataShape{checked_cast<int64_t>(compressedData.size() / f16TypeSizeBytes), 1, 1, 1};
        newSrcType = getMemRefType(compressedDataShape, f16Type, DimsOrder::NCHW, inputType.getMemSpace(),
                                   /*strides=*/StridesRef(), getSwizzlingSchemeAttr(inputType));
        newSrcType = mlir::cast<mlir::MemRefType>(
                vpux::setCompressionState(newSrcType, VPUIP::CompressionState::CompiletimeCompressed));
        const auto newSrcStorageType = mlir::RankedTensorType::get(compressedDataShape.raw(), f16Type);
        newSrcContentAttr = Const::createConstContent(
                newSrcStorageType, ArrayRef<vpux::type::float16>(
                                           const_cast<vpux::type::float16*>(
                                                   reinterpret_cast<const vpux::type::float16*>(compressedData.data())),
                                           compressedData.size() / f16TypeSizeBytes));
    } else {
        VPUX_THROW("Unsupported compression mode");
    }

    newDstType = newDstType.changeDimsOrder(DimsOrder::NCHW);

    rewriter.setInsertionPointAfter(outBufferOp);
    auto newDstBufferOp = rewriter.create<VPURT::DeclareBufferOp>(
            outBufferOp->getLoc(), newDstType, outBufferOp.getSectionAttr(), outBufferOp.getSectionIndexAttr(),
            outBufferOp.getByteOffsetAttr(), outBufferOp.getSwizzlingKeyAttr());

    rewriter.setInsertionPointAfter(inConstOp);
    auto newSrcConstOp = rewriter.create<Const::DeclareOp>(inConstOp->getLoc(), newSrcType,
                                                           Const::ContentAttr::get(newSrcContentAttr));

    rewriter.setInsertionPoint(origOp);
    rewriter.create<VPUIP::DecompressDMAOp>(loc, newSrcConstOp.getOutput(), /*act_compression_size_entry*/ nullptr,
                                            /*act_compression_sparsity_map*/ nullptr, newDstBufferOp.getBuffer(),
                                            origOp.getPortAttr(), origOp.getIsOutOfOrderAttr(),
                                            origOp.getIsCriticalAttr(),
                                            /*dmaHwpId=*/nullptr,
                                            /*profilingMetadata=*/nullptr);
    rewriter.replaceOp(origOp, {outBufferOp.getBuffer()});

    const auto uncompressed = totalInputSize.count();
    const auto compressed = compressedData.size();
    _log.trace("Compressed weights for {0}: {1} / {2} ({3})", loc, compressed, uncompressed,
               (double)compressed / uncompressed);

    return mlir::success();
}

void CompressWeightsBTCPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);
    const auto algo = ICodec::CompressionAlgorithm::BITCOMPACTOR_CODEC;

    _log.trace("VPUIP CompressWeightsBTCPass");
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<NNDMAOpConverter>(&ctx, algo, arch, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createCompressWeightsBTCPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createCompressWeightsBTCPass(Logger log) {
    return std::make_unique<CompressWeightsBTCPass>(log);
}
