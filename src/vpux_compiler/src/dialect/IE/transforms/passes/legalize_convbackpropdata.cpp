//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_dequantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/transposed_convolution_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/core/strides.hpp>

#include <cstring>
#include <variant>

namespace vpux::IE {
#define GEN_PASS_DECL_LEGALIZECONVBACKPROPDATA
#define GEN_PASS_DEF_LEGALIZECONVBACKPROPDATA
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//  Returns the filter permutation order for different ranks (non-group convolution)
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

//  Returns the filter permutation order for different ranks (group convolution)
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

// Fills the whole buffer with repeated copies of the raw bytes of `value`, truncated/reinterpreted to
// the storage element width `elemSize`. Used to pre-initialize a split-kernel buffer so that any
// position later left untouched (e.g. the 5 unused positions in a 3x3 split kernel) already holds the
// correct raw zero-point encoding instead of raw 0.
void fillRawScalar(std::vector<char>& buf, const int64_t value, const Byte& elemSize) {
    const auto width = checked_cast<size_t>(elemSize.count());
    VPUX_THROW_WHEN(width == 0 || width > sizeof(int64_t), "Unsupported element size for raw scalar fill: {0}", width);
    VPUX_THROW_WHEN(buf.size() % width != 0, "Buffer size {0} is not a multiple of element size {1}", buf.size(),
                    width);

    // Narrow `value` to the storage width via static_cast (not checked_cast, since zero-points may be
    // negative for signed types) and memcpy its native representation, so the result is correct on both
    // little- and big-endian hosts.
    switch (width) {
    case sizeof(uint8_t): {
        const auto narrowed = static_cast<uint8_t>(value);
        for (size_t offset = 0; offset < buf.size(); offset += width) {
            std::memcpy(buf.data() + offset, &narrowed, width);
        }
        break;
    }
    case sizeof(uint16_t): {
        const auto narrowed = static_cast<uint16_t>(value);
        for (size_t offset = 0; offset < buf.size(); offset += width) {
            std::memcpy(buf.data() + offset, &narrowed, width);
        }
        break;
    }
    case sizeof(uint32_t): {
        const auto narrowed = static_cast<uint32_t>(value);
        for (size_t offset = 0; offset < buf.size(); offset += width) {
            std::memcpy(buf.data() + offset, &narrowed, width);
        }
        break;
    }
    case sizeof(uint64_t): {
        const auto narrowed = static_cast<uint64_t>(value);
        for (size_t offset = 0; offset < buf.size(); offset += width) {
            std::memcpy(buf.data() + offset, &narrowed, width);
        }
        break;
    }
    default:
        VPUX_THROW("Unsupported element size for raw scalar fill: {0}", width);
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

// Helper for transposing a DynamicDequantize scale/zp input when the filter tensor is transposed
mlir::Value transposeDynDeqParam(mlir::PatternRewriter& rewriter, mlir::Value param, mlir::Location baseLoc,
                                 StringRef locSuffix, mlir::MLIRContext* ctx, size_t dimIndex1, size_t dimIndex2) {
    auto paramType = mlir::cast<vpux::NDTypeInterface>(param.getType());

    // Early return for scalar tensors (no transpose needed)
    if (paramType.getNumElements() == 1) {
        return param;
    }

    if (auto constOp = param.getDefiningOp<Const::DeclareOp>()) {
        auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getContentAttr().getBaseContent());
        if (denseAttr && denseAttr.isSplat()) {
            return param;
        }
    }

    auto permutation = to_small_vector(paramType.getDimsOrder().toPermutation() | transformed([](Dim dim) {
                                           return checked_cast<uint32_t>(dim.ind());
                                       }));
    std::swap(permutation[dimIndex1], permutation[dimIndex2]);
    auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permutation, ctx));
    auto transposeOp = IE::TransposeOp::create(rewriter, appendLoc(baseLoc, locSuffix), param,
                                               /*order=*/nullptr, orderAttr);
    return transposeOp.getOutput();
}

// This class encapsulates the logic for matching, validating, and transforming filter tensors used in convolution and
// transposed convolution operations.
// It handles complex filter subgraph that include DeclareOp, FakeQuantizeOp/DequantizeOp and ConvertOp
class FilterPattern {
public:
    explicit FilterPattern(Logger log): _log(std::move(log)) {
    }

    // Matching function that verifies if the pattern is usable for the pass
    mlir::LogicalResult match(mlir::Value inputFilterTensor) {
        // Supported filter pattern:
        //          Const::DeclareOp
        //                |
        //          (IE::DynamicDequantizeOp)
        //                |
        //          (IE::ConvertOp)
        //                |
        //   IE::(Group)ConvolutionBackpropDataOp
        _filterTensor = inputFilterTensor;
        filterConvertOp = _filterTensor.getDefiningOp<IE::ConvertOp>();
        if (filterConvertOp != nullptr) {
            _filterTensor = filterConvertOp.getInput();
        }

        filterDqOrFqOp = _filterTensor.getDefiningOp();
        if (mlir::isa_and_present<IE::DynamicDequantizeOp>(filterDqOrFqOp)) {
            if (auto typedValue =
                        mlir::dyn_cast<mlir::TypedValue<mlir::RankedTensorType>>(filterDqOrFqOp->getOperand(0))) {
                _filterTensor = typedValue;
            }
        }

        filterOp = _filterTensor.getDefiningOp();

        return mlir::success();
    }

    // Creates a single transformed filter tensor with dimension swapping and optional reversal
    mlir::Value getNewFilter(mlir::PatternRewriter& rewriter, int32_t inChannelDimIndex, int32_t outChannelDimIndex,
                             std::optional<Dim> filterDimToReverse) {
        auto updatedConstant = getUpdatedConstant(rewriter, inChannelDimIndex, outChannelDimIndex, filterDimToReverse);
        return finalizeFilterSubgraph(rewriter, updatedConstant, inChannelDimIndex, outChannelDimIndex, std::nullopt);
    }

    // Creates multiple transformed filter tensors for split convolution operations
    std::vector<mlir::Value> getMultipleNewFilters(
            mlir::PatternRewriter& rewriter,
            std::pair<vpux::NDTypeInterface, std::vector<vpux::Const::ContentAttr>&> splitInfo,
            int32_t inChannelDimIndex, int32_t outChannelDimIndex) {
        auto& [splitFilterType, wContentAttrs] = splitInfo;
        auto numSplits = wContentAttrs.size();
        std::vector<mlir::Value> newFilters;
        newFilters.reserve(numSplits);

        for (decltype(numSplits) i = 0; i < numSplits; ++i) {
            // Same pattern as getNewFilter
            auto updatedConstant = getUpdatedConstant(rewriter, inChannelDimIndex, outChannelDimIndex, std::nullopt,
                                                      splitFilterType, wContentAttrs[i]);
            auto newFilter =
                    finalizeFilterSubgraph(rewriter, updatedConstant, inChannelDimIndex, outChannelDimIndex, i);
            newFilters.push_back(newFilter);
        }
        return newFilters;
    }

    // Helper functions for the pattern
    bool isFilterConst() const {
        return mlir::isa_and_present<Const::DeclareOp>(filterOp);
    }

    mlir::Operation* getFilterOp() const {
        return filterOp;
    }

    mlir::Value getFilterTensor() const {
        return _filterTensor;
    }

    // Checks whether the filter is dequantized at runtime via IE::DynamicDequantizeOp (i.e. stored as an integer type
    // and rescaled with a scale/zero-point pair downstream)
    bool hasDynamicDequantize() const {
        return mlir::isa_and_present<IE::DynamicDequantizeOp>(filterDqOrFqOp);
    }

    // Returns the zero-point used by the filter's IE::DynamicDequantizeOp, if it is known at compile
    // time to be a single, per-tensor value (can also be a constant splat or no zero-point at all)
    std::optional<int64_t> getConstantZeroPoint() const {
        auto dynDeqOp = mlir::dyn_cast_if_present<IE::DynamicDequantizeOp>(filterDqOrFqOp);
        if (dynDeqOp == nullptr) {
            return std::nullopt;
        }
        auto zp = dynDeqOp.getZp();
        if (zp == nullptr) {
            return 0;
        }
        auto splatValue = Const::getSplatValue<int64_t>(zp);
        if (mlir::failed(splatValue)) {
            return std::nullopt;
        }
        return *splatValue;
    }

private:
    // Creates a new constant filter with transformed dimensions and attributes
    Const::DeclareOp getUpdatedConstant(mlir::PatternRewriter& rewriter, int32_t inChannelDimIndex,
                                        int32_t outChannelDimIndex, std::optional<Dim> filterDimToReverse,
                                        std::optional<vpux::NDTypeInterface> splitFilterType = std::nullopt,
                                        std::optional<vpux::Const::ContentAttr> splitContentAttr = std::nullopt) {
        auto constFilterOp = mlir::dyn_cast_if_present<Const::DeclareOp>(getFilterOp());
        VPUX_THROW_WHEN(constFilterOp == nullptr, "Filter must be Const::DeclareOp");
        bool isSplit = splitFilterType.has_value() && splitContentAttr.has_value();
        auto contentAttr = isSplit ? *splitContentAttr : constFilterOp.transformContentAttr().get();
        auto newFilterType = isSplit ? *splitFilterType : mlir::cast<vpux::NDTypeInterface>(constFilterOp.getType());
        auto newDimsOrder = (inChannelDimIndex == IE::GROUP_TRANSPOSED_CONV_C_IN_DIM_INDEX)
                                    ? getNewFilterDimsOrderForGroupConv(newFilterType.getRank())
                                    : getNewFilterDimsOrderForConv(newFilterType.getRank());
        auto filterShape = to_small_vector(newFilterType.getShape());

        if (filterDimToReverse.has_value()) {
            contentAttr = contentAttr.transform().reverse(filterDimToReverse.value()).transpose(newDimsOrder).get();
        } else {
            contentAttr = contentAttr.transform().transpose(newDimsOrder).get();
        }

        if (!isSplit) {
            std::swap(filterShape[inChannelDimIndex], filterShape[outChannelDimIndex]);
            newFilterType = newFilterType.changeShape(ShapeRef(filterShape));
        }

        auto loc = filterDqOrFqOp ? filterDqOrFqOp->getLoc() : constFilterOp.getLoc();
        return Const::DeclareOp::create(rewriter, loc, newFilterType, std::move(contentAttr));
    }

    // Recreates the complete filter subgraph with the transformed constant
    mlir::Value finalizeFilterSubgraph(mlir::PatternRewriter& rewriter, Const::DeclareOp newConst,
                                       int32_t inChannelDimIndex, int32_t outChannelDimIndex,
                                       std::optional<size_t> splitIndex = std::nullopt) {
        auto constFilterOp = mlir::dyn_cast_if_present<Const::DeclareOp>(filterOp);
        VPUX_THROW_WHEN(constFilterOp == nullptr, "Filter must be Const::DeclareOp");
        mlir::Value currentFilter = newConst.getOutput();
        std::string suffix = splitIndex.has_value() ? ("_w" + std::to_string(*splitIndex + 1)) : "";

        if (auto filterDynDeqOp = mlir::dyn_cast_if_present<IE::DynamicDequantizeOp>(filterDqOrFqOp)) {
            auto transposeParam = [&](mlir::Value input, const std::string& name) {
                return transposeDynDeqParam(rewriter, input, filterDynDeqOp.getLoc(), name + suffix,
                                            rewriter.getContext(), outChannelDimIndex, inChannelDimIndex);
            };

            auto scale = transposeParam(filterDynDeqOp.getScale(), "transpose_scale");
            auto zp = filterDynDeqOp.getZp() != nullptr ? transposeParam(filterDynDeqOp.getZp(), "transpose_zp")
                                                        : nullptr;

            auto loc = takeOpLoc(filterDynDeqOp, "dyn_dq_in" + suffix);
            auto newDynDeqOp = rewriter.createOrFold<IE::DynamicDequantizeOp>(loc, currentFilter, scale, zp,
                                                                              filterDynDeqOp.getDstElemType());
            // newDynDeqOp is a Value; need to get the defining op for attributes
            if (auto newDynDeqOpDef = newDynDeqOp.getDefiningOp<IE::DynamicDequantizeOp>()) {
                if (filterDynDeqOp->hasAttr(IE::WEIGHTS_IMPORT_DYN_DEQUANT_ATTR)) {
                    newDynDeqOpDef->setAttr(IE::WEIGHTS_IMPORT_DYN_DEQUANT_ATTR,
                                            mlir::UnitAttr::get(rewriter.getContext()));
                }
            }
            currentFilter = newDynDeqOp;
        }

        if (filterConvertOp != nullptr) {
            auto loc = takeOpLoc(filterConvertOp, "filter_cvt_in" + suffix);
            currentFilter =
                    rewriter.createOrFold<IE::ConvertOp>(loc, currentFilter, filterConvertOp.getDstElemTypeAttr());
        }
        return currentFilter;
    }

private:
    Logger _log;
    mlir::Value _filterTensor = nullptr;
    mlir::Operation* filterOp = nullptr;
    mlir::Operation* filterDqOrFqOp = nullptr;
    IE::ConvertOp filterConvertOp = nullptr;
};

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

    // Checking for the pattern's eligibility
    FilterPattern pattern(_log);
    if (mlir::failed(pattern.match(origOp.getFilter()))) {
        return mlir::failure();
    }

    if (!pattern.isFilterConst()) {
        auto filterTensor = pattern.getFilterTensor();
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
        auto transposeOp = IE::TransposeOp::create(rewriter, appendLoc(origOp->getLoc(), "transpose"), filterTensor,
                                                   /*order=*/nullptr, orderAttr);

        // Create reverseOp
        const auto axesAttr = getIntArrayAttr(getContext(), axes);
        IE::ReverseModeAttr modeAttr = IE::ReverseModeAttr::get(getContext(), IE::ReverseMode::INDEX);
        auto reverseOp = IE::ReverseOp::create(rewriter, appendLoc(origOp->getLoc(), "reverse"),
                                               transposeOp.getOutput(), nullptr, axesAttr, modeAttr);

        // Replace Op with transposedConv
        rewriter.replaceOpWithNewOp<IE::TransposedConvolutionOp>(
                origOp, origOp.getInput(), reverseOp.getOutput(), origOp.getOutputShape(), /*bias*/ nullptr,
                origOp.getStridesAttr(), origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
                origOp.getSpatialOutputPaddingAttr(), /*postOp=*/nullptr, /*clamp=*/nullptr,
                /*outputPadding=*/nullptr,
                /*inputPadding=*/nullptr);

        return mlir::success();
    }

    // If the filter is constant
    //  Reverse IC and OC dimensions in filter constant:
    //    [IC, OC, X] -> [OC, IC, X]
    //    [IC, OC, Y, X] -> [OC, IC, Y, X]
    //    [IC, OC, Z, Y, X] -> [OC, IC, Z, Y, X]
    auto filterOp = mlir::dyn_cast_if_present<Const::DeclareOp>(pattern.getFilterOp());
    if (filterOp == nullptr) {
        _log.trace("Expected filterOp to be a Const::DeclareOp");
        return mlir::failure();
    }
    auto filterType = mlir::cast<vpux::NDTypeInterface>(filterOp.getType());
    auto filterShape = to_small_vector(filterType.getShape());

    auto orgStrides = parseIntArrayAttr<int64_t>(origOp.getStridesAttr());
    auto orgPadsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBeginAttr());
    auto orgPadsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEndAttr());
    auto orgOutputPadding = parseIntArrayAttr<int64_t>(origOp.getSpatialOutputPaddingAttr());

    if (isConv2x2or3x3Feasible(filterType, filterShape, orgStrides, orgPadsBegin, orgPadsEnd, orgOutputPadding,
                               origOp.getOutputShape(), arch)) {
        return mlir::failure();
    }

    // The new filter is created here using the old filter as input
    auto newFilter = pattern.getNewFilter(rewriter, Dims4D::Filter::IC.ind(), Dims4D::Filter::OC.ind(), Dim(1));

    rewriter.replaceOpWithNewOp<IE::TransposedConvolutionOp>(
            origOp, origOp.getInput(), newFilter, origOp.getOutputShape(), /*bias*/ nullptr, origOp.getStridesAttr(),
            origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
            origOp.getSpatialOutputPaddingAttr(),
            /*postOp=*/nullptr, /*clamp=*/nullptr, /*outputPadding=*/nullptr,
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

    // Checking for the pattern's eligibility
    FilterPattern pattern(_log);
    if (mlir::failed(pattern.match(origOp.getFilter()))) {
        return mlir::failure();
    }

    if (!pattern.isFilterConst()) {
        return mlir::failure();
    }

    auto filterOp = mlir::dyn_cast_if_present<Const::DeclareOp>(pattern.getFilterOp());
    if (filterOp == nullptr) {
        _log.trace("Expected filterOp to be a Const::DeclareOp");
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

    // fill3x3SplitWeights repacks each 2x2 split kernel into a 3x3 buffer, leaving 5 of the 9 raw
    // storage positions untouched. For plain float filters, leaving those bytes as raw 0 is already
    // correct (0.0f). For quantized filters going through IE::DynamicDequantizeOp
    // ((raw - zero_point) * scale), raw 0 only decodes to 0.0 when zero_point == 0; for any asymmetric
    // (nonzero zero-point) scheme, those padding positions would otherwise decode to a spurious
    // nonzero weight, corrupting every 3x3 split convolution. When the zero-point is known at compile
    // time to be a single per-tensor value (constant splat, or no zero-point at all), pre-fill those
    // padding bytes with the *raw* zero-point value instead of 0, so they decode to exactly 0.0 after
    // dequantization - this avoids adding a runtime mask-multiply op entirely, and still benefits from
    // the 3x3 kernel-split performance optimization.
    // When the zero-point is non-constant or varies per-channel, the padding positions cannot be safely
    // pre-filled with a single raw value, so fall back to the 2x2 kernel split instead, which has no
    // unused padding positions and therefore doesn't need any zeroing at all.
    const auto constZeroPoint = pattern.hasDynamicDequantize() ? pattern.getConstantZeroPoint() : std::nullopt;
    const bool bakeZeroPointIntoPadding = constZeroPoint.has_value();
    if (use3x3 && pattern.hasDynamicDequantize() && !bakeZeroPointIntoPadding) {
        use3x3 = false;
    }

    // If use3x3, allocate and fill 3x3 buffer using 2x2 split buffer
    std::vector<std::vector<char>> weightsSet(numSplits);
    SmallVector<int64_t> splitFilterShape;
    if (use3x3) {
        const auto shiftKernelType = filterType.changeShape(ShapeRef({IC, OC, shiftKY, shiftKX}));
        const auto shiftRawByteSize = elemBitSize * shiftKernelType.getNumElements();
        for (int i = 0; i < numSplits; ++i) {
            weightsSet[i].resize(Byte(shiftRawByteSize).count());
            if (bakeZeroPointIntoPadding) {
                fillRawScalar(weightsSet[i], *constZeroPoint, elemSize);
            }
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
        contentAttrSetups.emplace_back(splitFilterAttrs[i], dataStorageType);
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

    // The new filter vector is created here using the old filter vector and atributes as inputs
    auto weightsConstOps =
            pattern.getMultipleNewFilters(rewriter, std::make_pair(splitFilterType, std::ref(wContentAttrs)),
                                          Dims4D::Filter::IC.ind(), Dims4D::Filter::OC.ind());

    if (weightsConstOps.empty()) {
        return mlir::failure();
    }

    SmallVector<mlir::Value> convOutputs;
    auto newStrides = getIntArrayAttr(getContext(), ov::Strides{1, 1});

    auto [padsBegin, padsEnd] = createPads(getContext(), use3x3);

    convOutputs.reserve(numSplits);

    for (int splitIdx = 0; splitIdx < numSplits; ++splitIdx) {
        std::string suffix = "_split_" + std::to_string(splitIdx + 1);
        auto convOp = IE::ConvolutionOp::create(rewriter, appendLoc(origOp->getLoc(), "Convolution" + suffix),
                                                origOp.getInput(), weightsConstOps[splitIdx], newStrides,
                                                padsBegin[splitIdx], padsEnd[splitIdx], origOp.getDilationsAttr());
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
            auto op = IE::AddOp::create(rewriter, appendLoc(origOp->getLoc(), suffix), convOutputs[splitIdx],
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
    auto combined = IE::ConcatOp::create(rewriter, appendLoc(origOp->getLoc(), "Concat_Convs"), convOutputs, 1);
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

    // Checking for the pattern's eligibility
    FilterPattern pattern(_log);
    if (mlir::failed(pattern.match(origOp.getFilter()))) {
        return mlir::failure();
    }

    if (!pattern.isFilterConst()) {
        auto filterTensor = origOp.getFilter();
        auto filterTensorType = mlir::cast<NDTypeInterface>(filterTensor.getType());
        auto permutation = to_small_vector(filterTensorType.getDimsOrder().toPermutation() | transformed([](Dim dim) {
                                               return checked_cast<uint32_t>(dim.ind());
                                           }));
        std::swap(permutation[IE::GROUP_TRANSPOSED_CONV_C_IN_DIM_INDEX],
                  permutation[IE::GROUP_TRANSPOSED_CONV_C_OUT_DIM_INDEX]);
        auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(permutation, getContext()));
        auto transposeOp = IE::TransposeOp::create(rewriter, appendLoc(origOp->getLoc(), "transpose"), filterTensor,
                                                   /*order=*/nullptr, orderAttr);

        const auto rank = filterTensorType.getRank();
        const auto axes = SmallVector<int64_t>{rank - 2, rank - 1};  // height and width are flipped
        const auto axesAttr = getIntArrayAttr(getContext(), axes);
        IE::ReverseModeAttr modeAttr = IE::ReverseModeAttr::get(getContext(), IE::ReverseMode::INDEX);
        auto reverseOp = IE::ReverseOp::create(rewriter, appendLoc(origOp->getLoc(), "reverse"),
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
    // We get it's type, rank dim and attribute
    auto filterDimOC = Dim(2);  // Dimension to reverse for group convolution

    // Here we create the new filter using the old filter and it's attributes as input
    auto newFilter = pattern.getNewFilter(rewriter, IE::GROUP_TRANSPOSED_CONV_C_IN_DIM_INDEX,
                                          IE::GROUP_TRANSPOSED_CONV_C_OUT_DIM_INDEX,
                                          filterDimOC);  // With reverse dimension

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
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createLegalizeConvBackpropDataPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createLegalizeConvBackpropDataPass(Logger log) {
    return std::make_unique<LegalizeConvBackpropDataPass>(log);
}
