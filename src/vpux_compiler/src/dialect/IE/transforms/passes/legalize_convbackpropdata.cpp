//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/transposed_convolution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/core/strides.hpp>

#include <variant>

namespace vpux::IE {
#define GEN_PASS_DECL_LEGALIZECONVBACKPROPDATA
#define GEN_PASS_DEF_LEGALIZECONVBACKPROPDATA
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Returns the filter permutation order for different ranks (non-group convolution)
DimsOrder getNewFilterDimsOrderForConv(const int64_t rank) {
    switch (rank) {
    case 3:
        return DimsOrder::HCW;
    case 4:
        return DimsOrder::IOYX;
    case 5:
        return DimsOrder::OGIYX;
    default:
        VPUX_THROW("Unsupported filter rank in getNewFilterDimsOrderForConv: rank={0}", rank);
    }
}

// Returns the filter permutation order for different ranks (group convolution)
DimsOrder getNewFilterDimsOrderForGroupConv(const int64_t rank) {
    switch (rank) {
    case 4:
        return DimsOrder::OYIX;
    case 5:
        return DimsOrder::GIOYX;
    case 6:
        return DimsOrder::GIOZYX;
    default:
        VPUX_THROW("Unsupported filter rank in getNewFilterDimsOrderForGroupConv: rank={0}", rank);
    }
}

std::pair<SmallVector<mlir::ArrayAttr>, SmallVector<mlir::ArrayAttr>> createPads(mlir::MLIRContext* ctx, bool use3x3) {
    SmallVector<mlir::ArrayAttr, 4> padsBegin, padsEnd;
    if (use3x3) {
        padsBegin.assign(4, getIntArrayAttr(ctx, ov::CoordinateDiff{1, 1}));
        padsEnd.assign(4, getIntArrayAttr(ctx, ov::CoordinateDiff{1, 1}));
    } else {
        padsBegin = {getIntArrayAttr(ctx, ov::CoordinateDiff{1, 1}), getIntArrayAttr(ctx, ov::CoordinateDiff{1, 0}),
                     getIntArrayAttr(ctx, ov::CoordinateDiff{0, 1}), getIntArrayAttr(ctx, ov::CoordinateDiff{0, 0})};
        padsEnd = {getIntArrayAttr(ctx, ov::CoordinateDiff{0, 0}), getIntArrayAttr(ctx, ov::CoordinateDiff{0, 1}),
                   getIntArrayAttr(ctx, ov::CoordinateDiff{1, 0}), getIntArrayAttr(ctx, ov::CoordinateDiff{1, 1})};
    }
    return std::make_pair(padsBegin, padsEnd);
}

bool isConv2x2or3x3Feasible(vpux::NDTypeInterface& filterType, ArrayRef<int64_t> filterShape,
                            ArrayRef<int64_t> orgStrides, ArrayRef<int64_t> orgPadsBegin, ArrayRef<int64_t> orgPadsEnd,
                            ArrayRef<int64_t> orgOutputPadding,
                            mlir::TypedValue<mlir::RankedTensorType> outputShapeAttr, vpux::config::ArchKind arch) {
    bool isKernelSplitFeasible = false;
    // Only for 4D filters with 4x4 kernel, stride 2, pad 1, no output padding, and no output shape attr
    if (filterType.getRank() == 4 && filterShape[Dims4D::Filter::KY.ind()] == 4 &&
        filterShape[Dims4D::Filter::KX.ind()] == 4 && orgStrides[0] == 2 && orgStrides[1] == 2 &&
        orgPadsBegin[0] == 1 && orgPadsBegin[1] == 1 && orgPadsEnd[0] == 1 && orgPadsEnd[1] == 1 &&
        orgOutputPadding[0] == 0 && orgOutputPadding[1] == 0 && outputShapeAttr == nullptr) {
        isKernelSplitFeasible = true;
    }
    // since DMA writes on MTL and LNL are taking longer for small tensors, use old convTranspose logic
    if (isKernelSplitFeasible && arch <= config::ArchKind::NPU40XX &&
        filterShape[Dims4D::Filter::OC.ind()] == filterShape[Dims4D::Filter::IC.ind()] &&
        filterShape[Dims4D::Filter::OC.ind()] > 1 && filterShape[Dims4D::Filter::OC.ind()] <= 32) {
        isKernelSplitFeasible = false;
    }
    return isKernelSplitFeasible;
}

// Helper to fill 2x2 split weights buffer from original weights
void fill2x2SplitWeights(std::vector<std::vector<char>>& weightsSplit, const char* weightsBuf, int N, int C,
                         int kernelHeight, int kernelWidth, int splitKY, int splitKX, int splitKernelSize,
                         const Byte& elemSize) {
    constexpr int numSplits = 4;
    for (int splitIdx = 0; splitIdx < numSplits; ++splitIdx) {
        for (int outChIdx = 0; outChIdx < N; ++outChIdx) {
            for (int inChIdx = 0; inChIdx < C; ++inChIdx) {
                auto base = (outChIdx * C + inChIdx) * kernelHeight * kernelWidth;
                for (int splitY = 0; splitY < splitKY; ++splitY) {
                    for (int splitX = 0; splitX < splitKX; ++splitX) {
                        int outIdx =
                                (outChIdx * C + inChIdx) * splitKernelSize + splitY * (kernelWidth / splitKX) + splitX;
                        // Indices after 180-degree flip
                        int idx0 = (kernelHeight - 1 - splitKY * splitY) * kernelWidth +
                                   (kernelWidth - 1 - splitKX * splitX);  // top-left
                        int idx1 = (kernelHeight - 1 - splitKY * splitY) * kernelWidth +
                                   (kernelWidth - 1 - (splitKX * splitX + 1));  // top-right
                        int idx2 = (kernelHeight - 1 - (splitKY * splitY + 1)) * kernelWidth +
                                   (kernelWidth - 1 - splitKX * splitX);  // bottom-left
                        int idx3 = (kernelHeight - 1 - (splitKY * splitY + 1)) * kernelWidth +
                                   (kernelWidth - 1 - (splitKX * splitX + 1));  // bottom-right

                        std::array<int, 4> idxs = {idx0, idx1, idx2, idx3};
                        auto outRawInd = checked_cast<size_t>(outIdx * elemSize.count());
                        auto inRawInd = checked_cast<size_t>((base + idxs[splitIdx]) * elemSize.count());
                        std::copy_n(weightsBuf + inRawInd, checked_cast<size_t>(elemSize.count()),
                                    weightsSplit[splitIdx].data() + outRawInd);
                    }
                }
            }
        }
    }
}

// Helper to fill 3x3 split weights buffer from 2x2 split buffer
void fill3x3SplitWeights(std::vector<std::vector<char>>& weightsSet, const std::vector<std::vector<char>>& weightsSplit,
                         int N, int C, int splitKernelSize, int kernelShiftSize, int splitKY, int splitKX, int shiftKX,
                         const Byte& elemSize) {
    constexpr std::pair<int, int> offsets[4] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    for (int splitIdx = 0; splitIdx < 4; ++splitIdx) {
        for (int outChIdx = 0; outChIdx < N; ++outChIdx) {
            for (int inChIdx = 0; inChIdx < C; ++inChIdx) {
                int srcBase = ((outChIdx * C + inChIdx) * splitKernelSize);  // 2x2
                int dstBase = ((outChIdx * C + inChIdx) * kernelShiftSize);  // 3x3
                auto [yOffset, xOffset] = offsets[splitIdx];
                for (int splitY = 0; splitY < splitKY; ++splitY) {
                    for (int splitX = 0; splitX < splitKX; ++splitX) {
                        int srcIdx = srcBase + splitY * splitKX + splitX;
                        int dst_idx = dstBase + (splitY + yOffset) * shiftKX + (splitX + xOffset);
                        auto inRawInd = checked_cast<size_t>(srcIdx * elemSize.count());
                        auto outRawInd = checked_cast<size_t>(dst_idx * elemSize.count());
                        std::copy_n(weightsSplit[splitIdx].data() + inRawInd, checked_cast<size_t>(elemSize.count()),
                                    weightsSet[splitIdx].data() + outRawInd);
                    }
                }
            }
        }
    }
}

template <typename T>
mlir::DenseElementsAttr createContent(mlir::RankedTensorType dataStorageType, const std::vector<char>& weightsSet) {
    return Const::createConstContent(
            dataStorageType, ArrayRef<T>(reinterpret_cast<const T*>(weightsSet.data()), weightsSet.size() / sizeof(T)));
}

// Helper to create split filter attributes for different element types
SmallVector<mlir::DenseElementsAttr> createSplitFilterAttrs(mlir::Type elemType,
                                                            const SmallVector<int64_t>& splitFilterShape,
                                                            const std::vector<std::vector<char>>& weightsSet) {
    SmallVector<mlir::DenseElementsAttr> splitFilterAttrs;
    mlir::RankedTensorType dataStorageType = mlir::RankedTensorType::get(splitFilterShape, elemType);
    splitFilterAttrs.reserve(weightsSet.size());
    if (elemType.isSignedInteger(8)) {
        for (size_t i = 0; i < weightsSet.size(); ++i) {
            splitFilterAttrs.push_back(createContent<int8_t>(dataStorageType, weightsSet[i]));
        }
    } else if (elemType.isUnsignedInteger(8)) {
        for (size_t i = 0; i < weightsSet.size(); ++i) {
            splitFilterAttrs.push_back(createContent<uint8_t>(dataStorageType, weightsSet[i]));
        }
    } else if (elemType.isUnsignedInteger(16)) {
        for (size_t i = 0; i < weightsSet.size(); ++i) {
            splitFilterAttrs.push_back(createContent<uint16_t>(dataStorageType, weightsSet[i]));
        }
    } else if (elemType.isF16()) {
        for (size_t i = 0; i < weightsSet.size(); ++i) {
            splitFilterAttrs.push_back(createContent<vpux::type::float16>(dataStorageType, weightsSet[i]));
        }
    } else if (elemType.isF32()) {
        for (size_t i = 0; i < weightsSet.size(); ++i) {
            splitFilterAttrs.push_back(createContent<float>(dataStorageType, weightsSet[i]));
        }
    } else {
        VPUX_THROW("Unsupported element type in assign splitFilter attrs: {0}", elemType);
    }
    return splitFilterAttrs;
}

//
// ConvBackpropDataToTransConvConversion
//

class ConvBackpropDataToTransConvConversion final : public mlir::OpRewritePattern<IE::ConvolutionBackpropDataOp> {
public:
    ConvBackpropDataToTransConvConversion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionBackpropDataOp>(ctx), _log(log) {
        setDebugName("ConvBackpropDataToTransConvConversion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionBackpropDataOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvBackpropDataToTransConvConversion::matchAndRewrite(IE::ConvolutionBackpropDataOp origOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::ConvolutionBackpropDataOp Operation '{0}'", origOp->getLoc());
    const auto arch = config::getArch(origOp);

    // Supported filter pattern:
    //     Const::DeclareOp
    //         |
    //   (IE::FakeQuantizeOp)
    //         |
    //   (IE::ConvertOp)
    //         |
    // IE::ConvolutionBackpropDataOp

    auto filterTensor = origOp.getFilter();
    auto filterConvertOp = filterTensor.getDefiningOp<IE::ConvertOp>();
    if (filterConvertOp != nullptr) {
        filterTensor = filterConvertOp.getInput();
    }

    auto filterFqOp = filterTensor.getDefiningOp<IE::FakeQuantizeOp>();
    if (filterFqOp != nullptr) {
        filterTensor = filterFqOp.getInput();
    }

    mlir::Value newFilter;
    auto filterOp = filterTensor.getDefiningOp<Const::DeclareOp>();
    if (filterOp == nullptr) {
        const auto filterTensorType = mlir::cast<vpux::NDTypeInterface>(filterTensor.getType());
        const auto filterRank = filterTensorType.getRank();
        auto permutation = to_small_vector(filterTensorType.getDimsOrder().toPermutation() | transformed([](Dim dim) {
                                               return checked_cast<uint32_t>(dim.ind());
                                           }));

        SmallVector<int64_t> axes;
        if (filterRank == 5) {
            axes = SmallVector<int64_t>{Dims5D::Filter::KZ.ind(), Dims5D::Filter::KY.ind(), Dims5D::Filter::KX.ind()};
            std::swap(permutation[Dims5D::Filter::OC.ind()], permutation[Dims5D::Filter::IC.ind()]);
        } else if (filterRank == 4) {
            axes = SmallVector<int64_t>{Dims4D::Filter::KY.ind(), Dims4D::Filter::KX.ind()};
            std::swap(permutation[Dims4D::Filter::OC.ind()], permutation[Dims4D::Filter::IC.ind()]);
        } else if (filterRank == 3) {
            axes = SmallVector<int64_t>{2};
            std::swap(permutation[0], permutation[1]);
        } else {
            VPUX_THROW("Only support 3D, 4D and 5D filter shape rank, but got {0}", filterRank);
        }

        // Create transposeOp
        auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permutation, getContext()));
        auto transposeOp = rewriter.create<IE::TransposeOp>(appendLoc(origOp->getLoc(), "_transpose"), filterTensor,
                                                            /*order=*/nullptr, orderAttr);

        // Create reverseOp
        const auto axesAttr = getIntArrayAttr(getContext(), axes);
        IE::ReverseModeAttr modeAttr = IE::ReverseModeAttr::get(getContext(), IE::ReverseMode::INDEX);
        auto reverseOp = rewriter.create<IE::ReverseOp>(appendLoc(origOp->getLoc(), "_reverse"),
                                                        transposeOp.getOutput(), nullptr, axesAttr, modeAttr);

        newFilter = reverseOp.getOutput();

        // Replace Op with transposedConv
        rewriter.replaceOpWithNewOp<IE::TransposedConvolutionOp>(
                origOp, origOp.getInput(), newFilter, origOp.getOutputShape(), /*bias*/ nullptr,
                origOp.getStridesAttr(), origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
                origOp.getSpatialOutputPaddingAttr(), /*postOp=*/nullptr, /*clamp=*/nullptr,
                /*outputPadding=*/nullptr,
                /*inputPadding=*/nullptr);

        return mlir::success();
    }

    // Reverse IC and OC dimensions in filter constant:
    //   [IC, OC, X] -> [OC, IC, X]
    //   [IC, OC, Y, X] -> [OC, IC, Y, X]
    //   [IC, OC, Z, Y, X] -> [OC, IC, Z, Y, X]
    auto filterType = mlir::cast<vpux::NDTypeInterface>(filterOp.getType());
    auto newDimsOrder = getNewFilterDimsOrderForConv(filterType.getRank());
    auto filterDimOC = Dim(1);
    auto filterShape = to_small_vector(filterType.getShape());

    auto orgStrides = parseIntArrayAttr<int64_t>(origOp.getStridesAttr());
    auto orgPadsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBeginAttr());
    auto orgPadsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEndAttr());
    auto orgOutputPadding = parseIntArrayAttr<int64_t>(origOp.getSpatialOutputPaddingAttr());

    vpux::Const::ContentAttr contentAttr;

    if (isConv2x2or3x3Feasible(filterType, filterShape, orgStrides, orgPadsBegin, orgPadsEnd, orgOutputPadding,
                               origOp.getOutputShape(), arch)) {
        return mlir::failure();
    }

    contentAttr = filterOp.transformContentAttr().reverse(filterDimOC).transpose(newDimsOrder).get();
    std::swap(filterShape[Dims4D::Filter::OC.ind()], filterShape[Dims4D::Filter::IC.ind()]);

    auto newFilterType = filterType.changeShape(ShapeRef(filterShape));

    auto newFilterConstant =
            rewriter.create<Const::DeclareOp>(takeOpLoc(filterOp, "new_filter"), newFilterType, std::move(contentAttr));
    newFilter = newFilterConstant.getOutput();

    const auto transposeFqInput = [&](mlir::Value fqInput, StringRef locSuffix) -> mlir::Value {
        auto fqInputType = mlir::cast<vpux::NDTypeInterface>(fqInput.getType());
        if (fqInputType.getNumElements() == 1) {
            return fqInput;
        }

        auto permutation = to_small_vector(fqInputType.getDimsOrder().toPermutation() | transformed([](Dim dim) {
                                               return checked_cast<uint32_t>(dim.ind());
                                           }));
        std::swap(permutation[Dims4D::Filter::OC.ind()], permutation[Dims4D::Filter::IC.ind()]);
        auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permutation, getContext()));
        auto transposeOp = rewriter.create<IE::TransposeOp>(
                takeOpLoc(filterOp, StringLiteral("transpose_{0}"), locSuffix), fqInput,
                /*order=*/nullptr, orderAttr);
        return transposeOp.getOutput();
    };

    if (filterFqOp != nullptr) {
        // In case the filter is quantized per-axis, make sure the axes are also correct for the new filter
        auto inputLow = transposeFqInput(filterFqOp.getInputLow(), "input_low");
        auto inputHigh = transposeFqInput(filterFqOp.getInputHigh(), "input_high");
        auto outputLow = transposeFqInput(filterFqOp.getOutputLow(), "output_low");
        auto outputHigh = transposeFqInput(filterFqOp.getOutputHigh(), "output_high");
        newFilter = rewriter.createOrFold<IE::FakeQuantizeOp>(
                takeOpLoc(filterFqOp, "fq_in"), newFilter, inputLow, inputHigh, outputLow, outputHigh,
                filterFqOp.getLevelsAttr(), filterFqOp.getLowFpTypeAttr(), filterFqOp.getAutoBroadcastAttr());
    }

    if (filterConvertOp != nullptr) {
        newFilter = rewriter.createOrFold<IE::ConvertOp>(takeOpLoc(filterFqOp, "filter_cvt_in"), newFilter,
                                                         filterConvertOp.getDstElemTypeAttr());
    }
    rewriter.replaceOpWithNewOp<IE::TransposedConvolutionOp>(
            origOp, origOp.getInput(), newFilter, origOp.getOutputShape(), /*bias*/ nullptr, origOp.getStridesAttr(),
            origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
            origOp.getSpatialOutputPaddingAttr(), /*postOp=*/nullptr, /*clamp=*/nullptr, /*outputPadding=*/nullptr,
            /*inputPadding=*/nullptr);

    return mlir::success();
}

//
// ConvBackpropDataToMultipleConvConversion
//

class ConvBackpropDataToMultipleConvConversion final : public mlir::OpRewritePattern<IE::ConvolutionBackpropDataOp> {
public:
    ConvBackpropDataToMultipleConvConversion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionBackpropDataOp>(ctx), _log(log) {
        setDebugName("ConvBackpropDataToMultipleConvConversion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionBackpropDataOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvBackpropDataToMultipleConvConversion::matchAndRewrite(IE::ConvolutionBackpropDataOp origOp,
                                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::ConvolutionBackpropDataOp Operation '{0}'", origOp->getLoc());
    const auto arch = config::getArch(origOp);

    // Supported filter pattern:
    //     Const::DeclareOp
    //         |
    //   (IE::FakeQuantizeOp)
    //         |
    //   (IE::ConvertOp)
    //         |
    // IE::ConvolutionBackpropDataOp

    auto filterTensor = origOp.getFilter();
    auto filterConvertOp = filterTensor.getDefiningOp<IE::ConvertOp>();
    if (filterConvertOp != nullptr) {
        filterTensor = filterConvertOp.getInput();
    }

    auto filterFqOp = filterTensor.getDefiningOp<IE::FakeQuantizeOp>();
    if (filterFqOp != nullptr) {
        filterTensor = filterFqOp.getInput();
    }

    auto filterOp = filterTensor.getDefiningOp<Const::DeclareOp>();
    if (filterOp == nullptr) {
        // Filter is not a constant, cannot proceed
        return mlir::failure();
    }

    auto filterType = mlir::cast<vpux::NDTypeInterface>(filterOp.getType());
    auto filterShape = to_small_vector(filterType.getShape());

    auto orgStrides = parseIntArrayAttr<int64_t>(origOp.getStridesAttr());
    auto orgPadsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBeginAttr());
    auto orgPadsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEndAttr());
    auto orgOutputPadding = parseIntArrayAttr<int64_t>(origOp.getSpatialOutputPaddingAttr());

    if (!isConv2x2or3x3Feasible(filterType, filterShape, orgStrides, orgPadsBegin, orgPadsEnd, orgOutputPadding,
                                origOp.getOutputShape(), arch)) {
        return mlir::failure();
    }

    //
    // Example flow (for 4x4 kernel split into four 2x2 kernels):
    //
    //   Input
    //     |
    //   [Split 4x4 kernel into 4 x 2x2 or 3x3 kernels]
    //     |
    //   +-------------------+-------------------+-------------------+-------------------+
    //   | Conv (2x2 or 3x3, offset0) | Conv (2x2 or 3x3, offset1) | Conv (2x2 or 3x3, offset2) | Conv (2x2 or 3x3,
    //   offset3) |
    //   +-------------------+-------------------+-------------------+-------------------+
    //     |
    //   [Concat along channel axis]
    //     |
    //   [DepthToSpace (BLOCKS_FIRST, block size 2)]
    //     |
    //   Output (spatially interleaved, matches transposed convolution)
    //

    // Optimization: For 4x4 transposed convolution, the kernel is split into four 2x2 or 3x3 kernels.
    // Each 2x2 or 3x3 split kernel is used in a separate standard convolution operation. After all convolutions are
    // performed, their outputs are concatenated and a depth2space operation is applied to interleave the data
    // correctly.
    //
    // A new constant is created here to represent the split kernel weights after splitting the original filter
    // (e.g., reordering, transposing, and reshaping). This ensures that each 2x2 or 3x3 split kernel used in the

    const auto filterContent = filterOp.getContent();
    const auto elemType = filterOp.getContentAttr().getBaseContent().getElementType();

    // ConvolutionBackpropData contains filter in [IC, OC, KY, KX] format
    auto IC = filterShape[Dims4D::Filter::OC.ind()];
    auto OC = filterShape[Dims4D::Filter::IC.ind()];
    auto kernelHeight = filterShape[Dims4D::Filter::KY.ind()];
    auto kernelWidth = filterShape[Dims4D::Filter::KX.ind()];

    constexpr int numSplits = 4;
    constexpr int splitKY = 2, splitKX = 2;
    constexpr int shiftKY = 3, shiftKX = 3;
    const int splitKernelSize = splitKY * splitKX;
    const int kernelShiftSize = shiftKY * shiftKX;
    const Byte elemSize = getElemTypeSize(filterContent.getStorageElemType());
    const Bit elemBitSize = getElemTypeSize(filterContent.getStorageElemType());

    // allocate and fill 2x2 split buffer
    std::vector<std::vector<char>> weightsSplit(numSplits);
    const auto splitKernelType = filterType.changeShape(ShapeRef({IC, OC, splitKY, splitKX}));
    const auto weightsBuf = filterContent.getRawStorageBuf();

    // Compute the buffer size in bytes for each split kernel.
    // This accounts for the element type (e.g., int8, uint8, float16, float32) and the number of elements in the split
    // kernel.
    const auto splitRawByteSize = elemBitSize * splitKernelType.getNumElements();
    for (int i = 0; i < numSplits; ++i) {
        weightsSplit[i].resize(Byte(splitRawByteSize).count());
    }
    fill2x2SplitWeights(weightsSplit, weightsBuf.data(), IC, OC, kernelHeight, kernelWidth, splitKY, splitKX,
                        splitKernelSize, elemSize);

    auto outputshape = getShape(origOp.getOutput());
    bool use3x3 = false;
    constexpr int MEM_LIMIT = 16384;

    // Decide between 2x2 or 3x3 filter shape:
    // - For large tensors, 3x3 is chosen to avoid performance issues with slicing in 2x2 logic.
    // - For smaller tensors, 2x2 is used for efficiency.
    // Please follow E-164658 for more detailed information.
    if ((arch > config::ArchKind::NPU40XX) &&
        ((outputshape[Dims4D::Act::H] * outputshape[Dims4D::Act::W]) >= MEM_LIMIT)) {
        use3x3 = true;
    }

    // If use3x3, allocate and fill 3x3 buffer using 2x2 split buffer
    std::vector<std::vector<char>> weightsSet(numSplits);
    SmallVector<int64_t> splitFilterShape;
    if (use3x3) {
        const auto shiftKernelType = filterType.changeShape(ShapeRef({IC, OC, shiftKY, shiftKX}));
        const auto shiftRawByteSize = elemBitSize * shiftKernelType.getNumElements();
        for (int i = 0; i < numSplits; ++i) {
            weightsSet[i].resize(Byte(shiftRawByteSize).count());
        }
        fill3x3SplitWeights(weightsSet, weightsSplit, IC, OC, splitKernelSize, kernelShiftSize, splitKY, splitKX,
                            shiftKX, elemSize);
        splitFilterShape = SmallVector<int64_t>{IC, OC, shiftKY, shiftKX};
    } else {
        weightsSet = std::move(weightsSplit);
        splitFilterShape = SmallVector<int64_t>{IC, OC, splitKY, splitKX};
    }

    auto splitFilterAttrs = createSplitFilterAttrs(elemType, splitFilterShape, weightsSet);

    std::vector<Const::ContentAttr> wContentAttrs(numSplits);
    std::vector<Const::ContentSetup> contentAttrSetups;
    contentAttrSetups.reserve(numSplits);

    mlir::RankedTensorType dataStorageType = mlir::RankedTensorType::get(splitFilterShape, elemType);
    for (int i = 0; i < numSplits; ++i) {
        contentAttrSetups.emplace_back(dataStorageType);
    }

    auto castElem = [&](mlir::Type type) {
        for (auto& setup : contentAttrSetups) {
            // Always cast to ensure correct element type and quantization parameters
            setup = setup.castElemType(type);
        }
    };

    // Ensure the filter content setup matches the float16/float32 type used for computation.
    // This guarantees that any required transformation (such as casting) is applied to the new weights.
    castElem(filterType.getElementType());

    // generate contentAttr using the correct buffer objects
    for (int i = 0; i < numSplits; ++i) {
        wContentAttrs[i] = Const::ContentAttr::get(splitFilterAttrs[i], contentAttrSetups[i]);
    }

    auto splitFilterType = filterType.changeShape(ShapeRef(use3x3 ? SmallVector<int64_t>{OC, IC, shiftKY, shiftKX}
                                                                  : SmallVector<int64_t>{OC, IC, splitKY, splitKX}));

    // Reverse IC and OC dimensions in filter constant:
    //   [IC, OC, Y, X] -> [OC, IC, Y, X]
    auto newDimsOrder = getNewFilterDimsOrderForConv(filterType.getRank());
    for (auto& attr : wContentAttrs) {
        attr = attr.transform().transpose(newDimsOrder).get();
    }

    // Declare constant ops
    std::vector<mlir::Value> weightsConstOps;
    for (int i = 0; i < numSplits; ++i) {
        auto loc = takeOpLoc(filterOp, "newFilterDeclare_" + std::to_string(i + 1));
        auto declOp = rewriter.create<Const::DeclareOp>(loc, splitFilterType, std::move(wContentAttrs[i]));
        weightsConstOps.push_back(declOp.getOutput());
    }

    const auto transposeFqInput = [&](mlir::Value fqInput, StringRef locSuffix) -> mlir::Value {
        auto fqInputType = mlir::cast<vpux::NDTypeInterface>(fqInput.getType());

        auto constOp = fqInput.getDefiningOp<Const::DeclareOp>();
        if (constOp != nullptr) {
            auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getContentAttr().getBaseContent());
            if (denseAttr && denseAttr.isSplat()) {
                return fqInput;
            }
        }

        auto permutation = to_small_vector(fqInputType.getDimsOrder().toPermutation() | transformed([](Dim dim) {
                                               return checked_cast<uint32_t>(dim.ind());
                                           }));
        std::swap(permutation[Dims4D::Filter::OC.ind()], permutation[Dims4D::Filter::IC.ind()]);
        auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permutation, getContext()));
        auto transposeOp = rewriter.create<IE::TransposeOp>(
                takeOpLoc(filterOp, StringLiteral("transpose_{0}"), locSuffix), fqInput,
                /*order=*/nullptr, orderAttr);
        return transposeOp.getOutput();
    };

    if (filterFqOp != nullptr) {
        // In case the filter is quantized per-axis, make sure the axes are also correct for the new filter
        auto inputLow = transposeFqInput(filterFqOp.getInputLow(), "input_low");
        auto inputHigh = transposeFqInput(filterFqOp.getInputHigh(), "input_high");
        auto outputLow = transposeFqInput(filterFqOp.getOutputLow(), "output_low");
        auto outputHigh = transposeFqInput(filterFqOp.getOutputHigh(), "output_high");
        for (int i = 0; i < numSplits; ++i) {
            auto loc = takeOpLoc(filterFqOp, "fq_in_w" + std::to_string(i + 1));
            weightsConstOps[i] = rewriter.createOrFold<IE::FakeQuantizeOp>(
                    loc, weightsConstOps[i], inputLow, inputHigh, outputLow, outputHigh, filterFqOp.getLevelsAttr(),
                    filterFqOp.getLowFpTypeAttr(), filterFqOp.getAutoBroadcastAttr());
        }
    }

    if (filterConvertOp != nullptr) {
        for (int i = 0; i < numSplits; ++i) {
            auto loc = takeOpLoc(filterFqOp, "filter_cvt_in_w" + std::to_string(i + 1));
            weightsConstOps[i] =
                    rewriter.createOrFold<IE::ConvertOp>(loc, weightsConstOps[i], filterConvertOp.getDstElemTypeAttr());
        }
    }

    SmallVector<mlir::Value> convOutputs;
    auto newStrides = getIntArrayAttr(getContext(), ov::Strides{1, 1});

    auto [padsBegin, padsEnd] = createPads(getContext(), use3x3);

    convOutputs.reserve(numSplits);

    for (int splitIdx = 0; splitIdx < numSplits; ++splitIdx) {
        std::string suffix = "_split_" + std::to_string(splitIdx + 1);
        auto convOp = rewriter.create<IE::ConvolutionOp>(
                appendLoc(origOp->getLoc(), "Convolution" + suffix), origOp.getInput(), weightsConstOps[splitIdx],
                /*bias=*/nullptr, newStrides, padsBegin[splitIdx], padsEnd[splitIdx], origOp.getDilationsAttr(),
                /*postOp=*/nullptr,
                /*clamp=*/nullptr,
                /*static_scale=*/nullptr,
                /*outputPadding=*/nullptr,
                /*inputPadding=*/nullptr);
        convOutputs.push_back(convOp.getOutput());
    }

    vpux::IE::AddOp biasAddOp = nullptr;
    bool isBiasAdd = false;

    // Lambda for bias fusion logic
    auto fuseBiasAdd = [&](vpux::IE::AddOp addOp) {
        auto biasInput1 = addOp.getInput1().getDefiningOp<Const::DeclareOp>();
        auto biasInput2 = addOp.getInput2().getDefiningOp<Const::DeclareOp>();
        if (biasInput1 == nullptr && biasInput2 == nullptr) {
            _log.info("There is no constant bias that could be fused into Convolution");
            return false;
        }
        for (auto splitIdx = 0; splitIdx < numSplits; ++splitIdx) {
            std::string suffix = "+Bias_Add_" + std::to_string(splitIdx + 1);
            auto op = rewriter.create<IE::AddOp>(appendLoc(origOp->getLoc(), suffix), convOutputs[splitIdx],
                                                 ((biasInput1 == nullptr) ? biasInput2 : biasInput1),
                                                 addOp.getAutoBroadcastAttr(), nullptr, nullptr, nullptr, nullptr);
            convOutputs[splitIdx] = op.getOutput();
        }
        return true;
    };

    // Bias fusion logic:
    // Only handle the case where the output of ConvolutionBackpropDataOp is used by a single AddOp.
    // If there are multiple users, bias fusion is not performed.
    if (origOp.getResult().hasOneUse()) {
        // Only one user, check if it's AddOp
        auto user = *origOp.getResult().user_begin();
        biasAddOp = mlir::dyn_cast<IE::AddOp>(user);
        if (biasAddOp != nullptr) {
            isBiasAdd = fuseBiasAdd(biasAddOp);
        }
    }

    // Each convolution produces a partial output for a different spatial offset. After concatenation, the DepthToSpace
    // operation rearranges the channels into the correct spatial grid, reconstructing the final output as if produced
    // by a single transposed convolution.

    // Block size for DepthToSpaceOp. For 2x2 or 3x3 kernel split, this is always 2 (spatial upscaling factor).
    constexpr int spatialBlockSize = 2;
    auto combined = rewriter.create<IE::ConcatOp>(appendLoc(origOp->getLoc(), "Concat_Convs"), convOutputs, 1);
    mlir::Operation* opToReplace = (biasAddOp != nullptr && isBiasAdd) ? biasAddOp : origOp;

    auto interleave = rewriter.replaceOpWithNewOp<IE::DepthToSpaceOp>(
            opToReplace, combined.getOutput(), spatialBlockSize, IE::DepthToSpaceMode::BLOCKS_FIRST);
    extendOpLoc(interleave, "Depth2Space");

    return mlir::success();
}

//
// ConvBackpropDataToGroupTransConvConversion
//

class ConvBackpropDataToGroupTransConvConversion final :
        public mlir::OpRewritePattern<IE::GroupConvolutionBackpropDataOp> {
public:
    ConvBackpropDataToGroupTransConvConversion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionBackpropDataOp>(ctx), _log(log) {
        setDebugName("ConvBackpropDataToGroupTransConvConversion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionBackpropDataOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvBackpropDataToGroupTransConvConversion::matchAndRewrite(
        IE::GroupConvolutionBackpropDataOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::GroupConvolutionBackpropDataOp Operation '{0}'", origOp->getLoc());

    // Supported filter pattern:
    //     Const::DeclareOp
    //         |
    //   (IE::FakeQuantizeOp)
    //         |
    //   (IE::ConvertOp)
    //         |
    // IE::GroupConvolutionBackpropDataOp

    auto filterTensor = origOp.getFilter();
    auto filterConvertOp = filterTensor.getDefiningOp<IE::ConvertOp>();
    if (filterConvertOp != nullptr) {
        filterTensor = filterConvertOp.getInput();
    }

    auto filterFqOp = filterTensor.getDefiningOp<IE::FakeQuantizeOp>();
    if (filterFqOp != nullptr) {
        filterTensor = filterFqOp.getInput();
    }

    auto filterOp = filterTensor.getDefiningOp<Const::DeclareOp>();
    if (filterOp == nullptr) {
        auto filterTensorType = mlir::cast<NDTypeInterface>(filterTensor.getType());
        auto permutation = to_small_vector(filterTensorType.getDimsOrder().toPermutation() | transformed([](Dim dim) {
                                               return checked_cast<uint32_t>(dim.ind());
                                           }));
        std::swap(permutation[IE::GROUP_TRANSPOSED_CONV_C_IN_DIM_INDEX],
                  permutation[IE::GROUP_TRANSPOSED_CONV_C_OUT_DIM_INDEX]);
        auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permutation, getContext()));
        auto transposeOp = rewriter.create<IE::TransposeOp>(appendLoc(origOp->getLoc(), "_transpose"), filterTensor,
                                                            /*order=*/nullptr, orderAttr);

        const auto rank = filterTensorType.getRank();
        const auto axes = SmallVector<int64_t>{rank - 2, rank - 1};
        const auto axesAttr = getIntArrayAttr(getContext(), axes);
        IE::ReverseModeAttr modeAttr = IE::ReverseModeAttr::get(getContext(), IE::ReverseMode::INDEX);
        auto reverseOp = rewriter.create<IE::ReverseOp>(appendLoc(origOp->getLoc(), "_reverse"),
                                                        transposeOp.getOutput(), nullptr, axesAttr, modeAttr);

        auto newFilter = reverseOp.getOutput();

        rewriter.replaceOpWithNewOp<IE::GroupTransposedConvolutionOp>(
                origOp, origOp.getInput(), newFilter, origOp.getOutputShape(), origOp.getStridesAttr(),
                origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
                origOp.getSpatialOutputPaddingAttr(), /*postOp=*/nullptr, /*clamp=*/nullptr, /*outputPadding=*/nullptr,
                /*inputPadding=*/nullptr);

        return mlir::success();
    }

    // Reverse IC and OC dimensions in filter constant for correct group transposed convolution:
    //   [GROUPS, IC, OC, X] -> [GROUPS, OC, IC, X]
    //   [GROUPS, IC, OC, Y, X] -> [GROUPS, OC, IC, Y, X]
    //   [GROUPS, IC, OC, Z, Y, X] -> [GROUPS, OC, IC, Z, Y, X]
    auto filterType = mlir::cast<vpux::NDTypeInterface>(filterOp.getType());
    auto newDimsOrder = getNewFilterDimsOrderForGroupConv(filterType.getRank());
    auto filterDimOC = Dim(2);
    auto contentAttr = filterOp.transformContentAttr().reverse(filterDimOC).transpose(newDimsOrder).get();

    auto filterShape = to_small_vector(filterType.getShape());
    std::swap(filterShape[IE::GROUP_TRANSPOSED_CONV_C_IN_DIM_INDEX],
              filterShape[IE::GROUP_TRANSPOSED_CONV_C_OUT_DIM_INDEX]);
    auto newFilterType = filterType.changeShape(ShapeRef(filterShape));

    auto newFilterConstant =
            rewriter.create<Const::DeclareOp>(takeOpLoc(filterOp, "new_filter"), newFilterType, std::move(contentAttr));
    auto newFilter = newFilterConstant.getOutput();

    const auto transposeFqInput = [&](mlir::Value fqInput, StringLiteral locSuffix) -> mlir::Value {
        auto fqInputType = mlir::cast<vpux::NDTypeInterface>(fqInput.getType());
        if (fqInputType.getNumElements() == 1) {
            return fqInput;
        }

        auto permutation = to_small_vector(fqInputType.getDimsOrder().toPermutation() | transformed([](Dim dim) {
                                               return checked_cast<uint32_t>(dim.ind());
                                           }));
        std::swap(permutation[IE::GROUP_TRANSPOSED_CONV_C_IN_DIM_INDEX],
                  permutation[IE::GROUP_TRANSPOSED_CONV_C_OUT_DIM_INDEX]);
        auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permutation, getContext()));
        auto transposeOp = rewriter.create<IE::TransposeOp>(
                takeOpLoc(filterOp, StringLiteral("transpose_{0}"), locSuffix), fqInput,
                /*order=*/nullptr, orderAttr);
        return transposeOp.getOutput();
    };

    if (filterFqOp != nullptr) {
        // In case the filter is quantized per-axis, make sure the axes are also correct for the new filter
        auto inputLow = transposeFqInput(filterFqOp.getInputLow(), "input_low");
        auto inputHigh = transposeFqInput(filterFqOp.getInputHigh(), "input_high");
        auto outputLow = transposeFqInput(filterFqOp.getOutputLow(), "output_low");
        auto outputHigh = transposeFqInput(filterFqOp.getOutputHigh(), "output_high");
        newFilter = rewriter.createOrFold<IE::FakeQuantizeOp>(
                takeOpLoc(filterFqOp, "in_fq"), newFilter, inputLow, inputHigh, outputLow, outputHigh,
                filterFqOp.getLevelsAttr(), filterFqOp.getLowFpTypeAttr(), filterFqOp.getAutoBroadcastAttr());
    }

    if (filterConvertOp != nullptr) {
        newFilter = rewriter.createOrFold<IE::ConvertOp>(takeOpLoc(filterConvertOp, "filter_cvt_in"), newFilter,
                                                         filterConvertOp.getDstElemTypeAttr());
    }

    rewriter.replaceOpWithNewOp<IE::GroupTransposedConvolutionOp>(
            origOp, origOp.getInput(), newFilter, origOp.getOutputShape(), origOp.getStridesAttr(),
            origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
            origOp.getSpatialOutputPaddingAttr(), /*postOp=*/nullptr, /*clamp=*/nullptr, /*outputPadding=*/nullptr,
            /*inputPadding=*/nullptr);

    return mlir::success();
}

//
// LegalizeConvBackpropDataPass
//

class LegalizeConvBackpropDataPass final : public IE::impl::LegalizeConvBackpropDataBase<LegalizeConvBackpropDataPass> {
public:
    explicit LegalizeConvBackpropDataPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void LegalizeConvBackpropDataPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvBackpropDataToMultipleConvConversion>(&ctx, _log);
    patterns.add<ConvBackpropDataToTransConvConversion>(&ctx, _log);
    patterns.add<ConvBackpropDataToGroupTransConvConversion>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createLegalizeConvBackpropDataPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createLegalizeConvBackpropDataPass(Logger log) {
    return std::make_unique<LegalizeConvBackpropDataPass>(log);
}
