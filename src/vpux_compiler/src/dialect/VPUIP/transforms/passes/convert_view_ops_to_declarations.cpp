//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/swizzling_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/core/IR/strided_dmas_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/utils/attributes.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTVIEWOPSTODECLARATIONS
#define GEN_PASS_DEF_CONVERTVIEWOPSTODECLARATIONS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// ViewLikeRewrite
//

class ViewLikeRewrite final : public mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface> {
public:
    ViewLikeRewrite(mlir::MLIRContext* ctx, const AliasesInfo* aliasInfo, Logger log)
            : mlir::OpInterfaceRewritePattern<mlir::ViewLikeOpInterface>(ctx), _aliasInfo(aliasInfo), _log(log) {
        VPUX_THROW_UNLESS(_aliasInfo != nullptr, "Got NULL pointer for AliasesInfo in ViewLikeRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::ViewLikeOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Byte calculateOffset(mlir::Value val) const;
    Shape calculateDimOffsets(mlir::Value val) const;

private:
    const AliasesInfo* _aliasInfo = nullptr;
    Logger _log;
};

Byte ViewLikeRewrite::calculateOffset(mlir::Value val) const {
    Byte offset(0);

    if (auto source = _aliasInfo->getSource(val)) {
        offset = calculateOffset(source);
    }

    if (auto declareOp = mlir::dyn_cast_or_null<VPURT::DeclareBufferOp>(val.getDefiningOp())) {
        offset += Byte(declareOp.getByteOffset());
    }

    if (auto subViewOp = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(val.getDefiningOp())) {
        // When explicit output shapes and offsets are set on a SubViewOp with an OVERLAPPED
        // distributed input, the per-cluster byte offset is computed via
        // perClusterBufferOffsetAttrName during unrolling instead.
        auto inputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(subViewOp.getSource().getType());
        bool isOverlappedWithExplicitAttrs =
                subViewOp.getExplicitOutputShapes().has_value() && subViewOp.getExplicitOutputOffsets().has_value() &&
                inputDistType != nullptr &&
                (inputDistType.getDistribution().getMode().getValue() == VPU::DistributionMode::OVERLAPPED);
        if (!isOverlappedWithExplicitAttrs) {
            offset += subViewOp.getByteOffset();
        }
    }
    if (auto extractFlatViewOp = mlir::dyn_cast_or_null<VPUIP::ExtractFlatSliceOp>(val.getDefiningOp())) {
        offset += extractFlatViewOp.getByteOffset();
    }

    return offset;
}

/*
    Below function only handles a subset of ViewLikeOps which can actually
    occur between dynamic strides DMA and function argument. This is enforced
    by a separate legalization pass.
    Legal ops:
    SubView - doesn't affect input strides so is legal
    PermuteCast - doesn't affect input strides so it is legal
    GenericReshape which expands/contracts original tensor by unit dimensions
        it does affect original tensor strides but original shape can be recovered
        if only unit dimensions are affected
*/
Shape ViewLikeRewrite::calculateDimOffsets(mlir::Value val) const {
    Shape viewOffsets{};
    if (auto source = _aliasInfo->getSource(val)) {
        viewOffsets = calculateDimOffsets(source);
    }

    return llvm::TypeSwitch<mlir::Operation*, Shape>(val.getDefiningOp())
            .Case<VPUIP::SubViewOp>([&](VPUIP::SubViewOp subViewOp) {
                auto addendOffsetsVec = parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets());
                if (viewOffsets.empty()) {
                    return Shape(addendOffsetsVec);
                }
                std::transform(viewOffsets.begin(), viewOffsets.end(), addendOffsetsVec.begin(), viewOffsets.begin(),
                               std::plus<>{});
                return viewOffsets;
            })
            .Case<VPUIP::GenericReshapeOp>([&](VPUIP::GenericReshapeOp genericReshapeOp) {
                if (viewOffsets.empty()) {
                    return viewOffsets;
                }
                auto inType = mlir::cast<NDTypeInterface>(genericReshapeOp.getInput().getType());
                auto outType = mlir::cast<NDTypeInterface>(genericReshapeOp.getOutput().getType());
                auto inShape = inType.getShape();
                auto inStride = inType.getMemStrides();
                auto inShapeStrideZip = llvm::zip_equal(inShape, inStride);
                llvm::SmallVector<std::tuple<int64_t, MemSize<MemType::Bit>>> inShapeStride(inShapeStrideZip.begin(),
                                                                                            inShapeStrideZip.end());
                auto outStrides = outType.getMemStrides();
                auto outShape = outType.getShape();
                auto outShapeStrideZip = llvm::zip_equal(outShape, outStrides);
                llvm::SmallVector<std::tuple<int64_t, MemSize<MemType::Bit>>> outShapeStride(outShapeStrideZip.begin(),
                                                                                             outShapeStrideZip.end());

                SmallVector<int64_t> reshapedOffsets(outShape.size(), 0);
                size_t outPosIdx = 0;
                for (size_t idx = 0; idx < inShape.size(); idx++) {
                    auto outPos =
                            std::find(outShapeStride.begin() + outPosIdx, outShapeStride.end(), inShapeStride[idx]);
                    if (outPos != outShapeStride.end()) {
                        outPosIdx = std::distance(outShapeStride.begin(), outPos);
                        reshapedOffsets[outPosIdx] = viewOffsets[Dim(idx)];
                    }
                }

                // Final dim tiling fixup. In case of final dim tiling above algorithm
                // will get confused as strides on final dimension are not provided by
                // getMemStrides. Below code manualy checks if final in dim is 1 and if
                // there is tiling on that dimension.
                if (viewOffsets[Dim(0)] != 0 && inShape[Dim(0)] == 1) {
                    reshapedOffsets[0] = viewOffsets[Dim(0)];
                }

                return Shape(reshapedOffsets);
            })
            .Case<VPUIP::PermuteCastOp>([&](VPUIP::PermuteCastOp permuteCastOp) {
                if (viewOffsets.empty()) {
                    return viewOffsets;
                }
                auto inType = mlir::cast<NDTypeInterface>(permuteCastOp.getSource().getType());
                auto outType = mlir::cast<NDTypeInterface>(permuteCastOp.getResult().getType());
                // When the logical shape is unchanged, PermuteCast only alters the memory
                // layout. The logical dimension offsets (NCHW order) are identical before
                // and after, so pass them through unchanged.
                if (inType.getShape() == outType.getShape()) {
                    return viewOffsets;
                }
                // When the logical shape changes (e.g., NCHW 1x1x4x3 -> NHWC 1x3x1x4),
                // NCHW dimension indices are remapped between input and output.
                // Convert via memory positions: input logical -> memory -> output logical.
                auto srcOrder = inType.getDimsOrder();
                auto dstOrder = DimsOrder::fromAffineMap(permuteCastOp.getDstOrder());
                return dstOrder.toLogicalOrder(srcOrder.toMemoryOrder(viewOffsets));
            })
            .Default([&](mlir::Operation*) {
                return viewOffsets;
            });
}

mlir::LogicalResult ViewLikeRewrite::matchAndRewrite(mlir::ViewLikeOpInterface origOp,
                                                     mlir::PatternRewriter& rewriter) const {
    if (!mlir::isa<Core::ReinterpretCastOp, VPUIP::GenericReshapeOp, VPUIP::SubViewOp, VPUIP::PermuteCastOp,
                   VPUIP::QuantizeCastOp, VPUIP::DistributedCastOp, VPUIP::NonDistributedCastOp, VPUIP::ShapeCastOp,
                   VPUIP::StubOp, VPUIP::ViewOp, VPUIP::ExtractFlatSliceOp>(origOp.getOperation())) {
        return matchFailed(rewriter, origOp, "Unknown view-like operation '{0}'", origOp->getName());
    }

    _log.trace("Found view-like Operation '{0}'", origOp->getLoc());

    const auto origVal = mlir::isa<VPUIP::NonDistributedCastOp>(origOp) ? origOp->getOperand(0) : origOp->getResult(0);
    const Byte offset = calculateOffset(origVal);

    const auto rootVal = _aliasInfo->getRoot(origVal);

    auto declareOp = rootVal.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_WHEN(declareOp == nullptr, "Unsupported source owner: '{0}'", rootVal);

    _log.nest().trace("It aliases internal buffer produced by '{0}'", declareOp->getLoc());

    auto section = declareOp.getSection();
    auto sectionIndex = declareOp.getSectionIndex();
    // TODO:#114687 -- section index is missed for CMX for some reason
    if (!sectionIndex.has_value()) {
        const auto outType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
        auto memSpaceIndex = outType.getMemSpace().getIndex();
        if (memSpaceIndex.has_value()) {
            sectionIndex = getIntArrayAttr(rewriter, ArrayRef({memSpaceIndex.value()}));
        }
    }

    const auto outType = origOp->getResult(0).getType();
    auto swizzlingScheme = VPUIP::getSwizzlingSchemeAttr(outType);
    mlir::IntegerAttr swizzlingKey;
    if (swizzlingScheme && swizzlingScheme.getKey().getInt() != 0) {
        swizzlingKey = swizzlingScheme.getKey();
    }

    mlir::ArrayAttr sectionIndexAttr = sectionIndex.has_value() ? sectionIndex.value() : nullptr;
    auto newDeclareOp = rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, outType, section, sectionIndexAttr,
                                                                            offset.count(), swizzlingKey);

    // When replacing a SubViewOp that slices a DistributedBuffer on the same axis as the
    // distribution tiling axis, attach per-cluster buffer offsets (shape-level) so that
    // UnrollDistributedOps can compute the correct per-cluster byte offsets.
    // The per-cluster buffer offset = subViewStaticOffsets + output perClusterMemoryShapeOffsets,
    // giving each cluster's actual offset in the input buffer's coordinate space.
    if (auto subViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(origOp.getOperation())) {
        auto outputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(subViewOp.getResult().getType());
        auto inputDistType = mlir::dyn_cast<VPUIP::DistributedBufferType>(subViewOp.getSource().getType());
        if (outputDistType != nullptr && inputDistType != nullptr && subViewOp.getExplicitOutputOffsets().has_value() &&
            (inputDistType.getDistribution().getMode().getValue() == VPU::DistributionMode::OVERLAPPED)) {
            const auto subViewOffsets = parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets());
            const auto outputPerClusterOffsets = outputDistType.getPerClusterMemoryShapeOffsets();
            const auto inputPerClusterOffsets = inputDistType.getPerClusterMemoryShapeOffsets();

            SmallVector<Shape> perClusterBufferOffsets;
            perClusterBufferOffsets.reserve(outputPerClusterOffsets.size());
            for (size_t cluster = 0; cluster < outputPerClusterOffsets.size(); ++cluster) {
                Shape combined(subViewOffsets);
                for (size_t dim = 0; dim < combined.size(); ++dim) {
                    combined[Dim(dim)] += outputPerClusterOffsets[cluster][Dim(dim)];
                    combined[Dim(dim)] -= inputPerClusterOffsets[cluster][Dim(dim)];
                }
                perClusterBufferOffsets.push_back(std::move(combined));
            }

            for (size_t cluster = 0; cluster < perClusterBufferOffsets.size(); ++cluster) {
                for (size_t dim = 0; dim < perClusterBufferOffsets[cluster].size(); ++dim) {
                    VPUX_THROW_WHEN(perClusterBufferOffsets[cluster][Dim(dim)] < 0,
                                    "Negative per-cluster buffer offset at cluster {0}, dim {1}: {2}. "
                                    "Check subview offsets and distribution attributes of the source and result "
                                    "buffers of the SubViewOp at {3}: {4}.",
                                    cluster, dim, perClusterBufferOffsets[cluster][Dim(dim)], subViewOp.getLoc(),
                                    subViewOp);
                }
            }

            newDeclareOp->setAttr(vpux::perClusterBufferOffsetAttrName,
                                  vpux::getIntArrayOfArray(rewriter.getContext(), perClusterBufferOffsets));
        }
    }

    if ((section == VPURT::BufferSection::NetworkInput || section == VPURT::BufferSection::NetworkOutput) &&
        vpux::net::isArgStrided(origOp->getParentOfType<mlir::ModuleOp>(),
                                declareOp.getNonEmptySectionIndex().front())) {
        auto dimOffsets = calculateDimOffsets(origVal);
        if (!dimOffsets.empty()) {
            newDeclareOp->setAttr(vpux::viewOffsetsAttrName, getIntArrayAttr(getContext(), dimOffsets));
        }
    }

    return mlir::success();
}

//
// ConvertViewOpsToDeclarationsPass
//

class ConvertViewOpsToDeclarationsPass final :
        public VPUIP::impl::ConvertViewOpsToDeclarationsBase<ConvertViewOpsToDeclarationsPass> {
public:
    explicit ConvertViewOpsToDeclarationsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertViewOpsToDeclarationsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto& aliasInfo = getAnalysis<AliasesInfo>();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<mlir::async::AsyncDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp, mlir::func::CallOp>();
    // The logic for ConcatView has been moved to BreakDataFlow pass
    // Leave ConcatView illegal here for sanity check
    target.addIllegalOp<Core::ReinterpretCastOp, VPUIP::GenericReshapeOp, VPUIP::SubViewOp, VPUIP::ConcatViewOp,
                        VPUIP::PermuteCastOp, VPUIP::QuantizeCastOp, VPUIP::DistributedCastOp,
                        VPUIP::NonDistributedCastOp, VPUIP::ShapeCastOp, VPUIP::StubOp, VPUIP::ViewOp,
                        VPUIP::ExtractFlatSliceOp>();
    target.addLegalOp<VPUIP::SwKernelOp>();
    target.markOpRecursivelyLegal<VPUIP::SwKernelOp>([&](mlir::Operation*) {
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ViewLikeRewrite>(&ctx, &aliasInfo, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertViewOpsToDeclarationsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertViewOpsToDeclarationsPass(Logger log) {
    return std::make_unique<ConvertViewOpsToDeclarationsPass>(log);
}
