//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/IE/transposed_convolution_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/core/strides.hpp>

#include <variant>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTCONVBACKPROPDATATOCONVORTRANSPOSEDCONV
#define GEN_PASS_DEF_CONVERTCONVBACKPROPDATATOCONVORTRANSPOSEDCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvBackpropDataToConvOrTransConvConversion
//

class ConvBackpropDataToConvOrTransConvConversion final : public mlir::OpRewritePattern<IE::ConvolutionBackpropDataOp> {
public:
    ConvBackpropDataToConvOrTransConvConversion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionBackpropDataOp>(ctx), _log(log) {
        setDebugName("ConvBackpropDataToConvOrTransConvConversion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionBackpropDataOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

inline std::pair<llvm::SmallVector<mlir::ArrayAttr, 4>, llvm::SmallVector<mlir::ArrayAttr, 4>> createPads(
        mlir::MLIRContext* ctx, bool use3x3) {
    llvm::SmallVector<mlir::ArrayAttr, 4> padsBegin;
    llvm::SmallVector<mlir::ArrayAttr, 4> padsEnd;

    if (use3x3) {
        padsBegin = {getIntArrayAttr(ctx, ov::CoordinateDiff{1, 1}), getIntArrayAttr(ctx, ov::CoordinateDiff{1, 1}),
                     getIntArrayAttr(ctx, ov::CoordinateDiff{1, 1}), getIntArrayAttr(ctx, ov::CoordinateDiff{1, 1})};
        padsEnd = {getIntArrayAttr(ctx, ov::CoordinateDiff{1, 1}), getIntArrayAttr(ctx, ov::CoordinateDiff{1, 1}),
                   getIntArrayAttr(ctx, ov::CoordinateDiff{1, 1}), getIntArrayAttr(ctx, ov::CoordinateDiff{1, 1})};
    } else {
        padsBegin = {getIntArrayAttr(ctx, ov::CoordinateDiff{1, 1}), getIntArrayAttr(ctx, ov::CoordinateDiff{1, 0}),
                     getIntArrayAttr(ctx, ov::CoordinateDiff{0, 1}), getIntArrayAttr(ctx, ov::CoordinateDiff{0, 0})};
        padsEnd = {getIntArrayAttr(ctx, ov::CoordinateDiff{0, 0}), getIntArrayAttr(ctx, ov::CoordinateDiff{0, 1}),
                   getIntArrayAttr(ctx, ov::CoordinateDiff{1, 0}), getIntArrayAttr(ctx, ov::CoordinateDiff{1, 1})};
    }

    return {std::move(padsBegin), std::move(padsEnd)};
}

template <typename T, typename SplitArray>
inline void assignWeights(SplitArray& weightsSplits, int out_idx, const T* weightsPtr, int base, int idx0, int idx1,
                          int idx2, int idx3) {
    weightsSplits[0][out_idx] = weightsPtr[base + idx0];
    weightsSplits[1][out_idx] = weightsPtr[base + idx1];
    weightsSplits[2][out_idx] = weightsPtr[base + idx2];
    weightsSplits[3][out_idx] = weightsPtr[base + idx3];
}

mlir::LogicalResult ConvBackpropDataToConvOrTransConvConversion::matchAndRewrite(
        IE::ConvolutionBackpropDataOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::ConvolutionBackpropDataOp Operation '{0}'", origOp->getLoc());
    const auto arch = config::getArch(origOp);

    // Support filter pattern:
    //     Const::DeclareOp
    //             |
    //    (IE::FakeQuantizeOp)
    //             |
    //     (IE::ConvertOp)
    //             |
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
    const auto getNewFilterDimsOrder = [](const int64_t rank) {
        SmallVector<vpux::Dim> permutation{};
        if (rank == 3) {
            permutation = {Dim(1), Dim(0), Dim(2)};
        } else if (rank == 4) {
            permutation = {Dim(1), Dim(0), Dim(2), Dim(3)};
        } else if (rank == 5) {
            permutation = {Dim(1), Dim(0), Dim(2), Dim(3), Dim(4)};
        }
        return DimsOrder::fromPermutation(permutation);
    };

    auto filterType = mlir::cast<vpux::NDTypeInterface>(filterOp.getType());
    auto newDimsOrder = getNewFilterDimsOrder(filterType.getRank());
    auto filterDimOC = Dim(1);
    auto filterShape = to_small_vector(filterType.getShape());

    auto orgStrides = parseIntArrayAttr<int64_t>(origOp.getStridesAttr());
    auto orgPadsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBeginAttr());
    auto orgPadsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEndAttr());
    auto orgOutputPadding = parseIntArrayAttr<int64_t>(origOp.getSpatialOutputPaddingAttr());
    bool isKernelSplitFeasible = false;

    vpux::Const::ContentAttr contentAttr;
    // Optimization for Transpose convolution with a 4x4 kernel size, stride of 2, pad of 1
    if (filterType.getRank() == 4) {
        if ((filterShape[Dims4D::Filter::KY.ind()] == 4 && filterShape[Dims4D::Filter::KX.ind()] == 4) &&
            (orgStrides[0] == 2 && orgStrides[1] == 2) && (orgPadsBegin[0] == 1 && orgPadsBegin[1] == 1) &&
            (orgPadsEnd[0] == 1 && orgPadsEnd[1] == 1) && (orgOutputPadding[0] == 0 && orgOutputPadding[1] == 0) &&
            origOp.getOutputShape() == nullptr) {
            // since DMA writes on MTL and LNL are taking longer, using old logic
            isKernelSplitFeasible = true;
        }
    }

    // since DMA writes on MTL and LNL are taking longer for small tensors, using old logic
    if (isKernelSplitFeasible && arch <= config::ArchKind::NPU40XX) {
        if ((filterShape[Dims4D::Filter::OC.ind()] == filterShape[Dims4D::Filter::IC.ind()]) &&
            (filterShape[Dims4D::Filter::OC.ind()] > 1 && filterShape[Dims4D::Filter::OC.ind()] <= 32)) {
            isKernelSplitFeasible = false;
        }
    }

    if (isKernelSplitFeasible) {
        contentAttr = filterOp.transformContentAttr().get();
    } else {
        contentAttr = filterOp.transformContentAttr().reverse(filterDimOC).transpose(newDimsOrder).get();
        std::swap(filterShape[Dims4D::Filter::OC.ind()], filterShape[Dims4D::Filter::IC.ind()]);
    }

    auto newFilterType = filterType.changeShape(ShapeRef(filterShape));

    auto newFilterConstant =
            rewriter.create<Const::DeclareOp>(takeOpLoc(filterOp, "new_filter"), newFilterType, std::move(contentAttr));
    newFilter = newFilterConstant.getOutput();
    // optimized way of implementing 4x4 transposed convolution: split 4x4 kernel into four 2x2/3x3 kernels and perform
    // standard convolutions, followed by concat and depth2space (to interleave data)
    if (isKernelSplitFeasible) {
        const auto newFilterContent = newFilterConstant.getContent();
        const auto elemType = filterOp.getContentAttr().getBaseContent().getElementType();
        union {
            const int8_t* int8WeightsPtr;
            const uint8_t* uint8WeightsPtr;
            const uint16_t* uint16WeightsPtr;
            const vpux::type::float16* fp16WeightsPtr;
            const float* fp32WeightsPtr;
        };

        bool isSint8 = false, isUint8 = false, isUint16 = false, isF16 = false;
        if (elemType.isSignedInteger(8)) {
            int8WeightsPtr = reinterpret_cast<const int8_t*>(newFilterContent.getRawStorageBuf().data());
            isSint8 = true;
        } else if (elemType.isUnsignedInteger(8)) {
            uint8WeightsPtr = reinterpret_cast<const uint8_t*>(newFilterContent.getRawStorageBuf().data());
            isUint8 = true;
        } else if (elemType.isUnsignedInteger(16)) {
            uint16WeightsPtr = reinterpret_cast<const uint16_t*>(newFilterContent.getRawStorageBuf().data());
            isUint16 = true;
        } else if (elemType.isF16()) {
            fp16WeightsPtr = reinterpret_cast<const vpux::type::float16*>(newFilterContent.getRawStorageBuf().data());
            isF16 = true;
        } else {
            fp32WeightsPtr = reinterpret_cast<const float*>(newFilterContent.getRawStorageBuf().data());
        }

        auto N = filterShape[Dims4D::Filter::OC.ind()];
        auto C = filterShape[Dims4D::Filter::IC.ind()];
        auto H = filterShape[Dims4D::Filter::KY.ind()];
        auto W = filterShape[Dims4D::Filter::KX.ind()];

        auto splitKY = 2;
        auto splitKX = 2;
        auto splitKernelSize = splitKY * splitKX;

        const int numSplits = 4;

        auto shiftKY = 3;
        auto shiftKX = 3;
        auto kernelShiftSize = shiftKY * shiftKX;

        std::vector<std::vector<int8_t>> weightsSplitsI8(numSplits, std::vector<int8_t>(N * C * splitKernelSize, 0));
        std::vector<std::vector<uint8_t>> weightsSplitsU8(numSplits, std::vector<uint8_t>(N * C * splitKernelSize, 0));
        std::vector<std::vector<uint16_t>> weightsSplitsU16(numSplits,
                                                            std::vector<uint16_t>(N * C * splitKernelSize, 0));
        std::vector<std::vector<vpux::type::float16>> weightsSplitsFp16(
                numSplits, std::vector<vpux::type::float16>(N * C * splitKernelSize, 0.0f));
        std::vector<std::vector<float>> weightsSplitsFp(numSplits, std::vector<float>(N * C * splitKernelSize, 0.0f));

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                auto base = (n * C + c) * H * W;
                for (int i = 0; i < splitKY; ++i) {
                    for (int j = 0; j < splitKX; ++j) {
                        auto out_idx = (n * C + c) * splitKernelSize + i * (W / splitKX) + j;

                        // Indices after 180-degree flip
                        auto idx0 = (H - 1 - 2 * i) * W + (W - 1 - 2 * j);              // top-left
                        auto idx1 = (H - 1 - 2 * i) * W + (W - 1 - (2 * j + 1));        // top-right
                        auto idx2 = (H - 1 - (2 * i + 1)) * W + (W - 1 - 2 * j);        // bottom-left
                        auto idx3 = (H - 1 - (2 * i + 1)) * W + (W - 1 - (2 * j + 1));  // bottom-right

                        if (isSint8) {
                            assignWeights<int8_t>(weightsSplitsI8, out_idx, int8WeightsPtr, base, idx0, idx1, idx2,
                                                  idx3);
                        } else if (isUint8) {
                            assignWeights<uint8_t>(weightsSplitsU8, out_idx, uint8WeightsPtr, base, idx0, idx1, idx2,
                                                   idx3);
                        } else if (isUint16) {
                            assignWeights<uint16_t>(weightsSplitsU16, out_idx, uint16WeightsPtr, base, idx0, idx1, idx2,
                                                    idx3);
                        } else if (isF16) {
                            assignWeights<vpux::type::float16>(weightsSplitsFp16, out_idx, fp16WeightsPtr, base, idx0,
                                                               idx1, idx2, idx3);
                        } else {
                            assignWeights<float>(weightsSplitsFp, out_idx, fp32WeightsPtr, base, idx0, idx1, idx2,
                                                 idx3);
                        }
                    }
                }
            }
        }

        // Offsets per 2x2 split to map into 3x3 grid
        const std::pair<int, int> offsets[4] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

        std::vector<std::vector<int8_t>> weightsSplitsShiftI8(numSplits,
                                                              std::vector<int8_t>(N * C * kernelShiftSize, 0));
        std::vector<std::vector<uint8_t>> weightsSplitsShiftU8(numSplits,
                                                               std::vector<uint8_t>(N * C * kernelShiftSize, 0));
        std::vector<std::vector<uint16_t>> weightsSplitsShiftU16(numSplits,
                                                                 std::vector<uint16_t>(N * C * kernelShiftSize, 0));
        std::vector<std::vector<vpux::type::float16>> weightsSplitsShiftFp16(
                numSplits, std::vector<vpux::type::float16>(N * C * kernelShiftSize, 0.0f));
        std::vector<std::vector<float>> weightsSplitsShiftFp(numSplits,
                                                             std::vector<float>(N * C * kernelShiftSize, 0.0f));

        for (auto k = 0; k < numSplits; ++k) {
            for (auto n = 0; n < N; ++n) {
                for (auto c = 0; c < C; ++c) {
                    auto src_base = ((n * C + c) * splitKernelSize);  // 2x2
                    auto dst_base = ((n * C + c) * kernelShiftSize);  // 3x3

                    auto [i_off, j_off] = offsets[k];  //{0,0},{0,1},{1,0},{1,1}

                    for (auto i = 0; i < splitKY; ++i) {
                        for (auto j = 0; j < splitKX; ++j) {
                            auto src_idx = src_base + i * splitKX + j;
                            auto dst_idx = dst_base + (i + i_off) * shiftKX + (j + j_off);

                            if (isSint8) {
                                weightsSplitsShiftI8[k][dst_idx] = weightsSplitsI8[k][src_idx];
                            } else if (isUint8) {
                                weightsSplitsShiftU8[k][dst_idx] = weightsSplitsU8[k][src_idx];
                            } else if (isUint16) {
                                weightsSplitsShiftU16[k][dst_idx] = weightsSplitsU16[k][src_idx];
                            } else if (isF16) {
                                weightsSplitsShiftFp16[k][dst_idx] = weightsSplitsFp16[k][src_idx];
                            } else {
                                weightsSplitsShiftFp[k][dst_idx] = weightsSplitsFp[k][src_idx];
                            }
                        }
                    }
                }
            }
        }

        auto outputshape = getShape(origOp.getOutput());
        bool use2x2 = false, use3x3 = false;
        constexpr int MEM_LIMIT = 16384;

        // Determine if 2x2 or 3x3 filter shape is needed
        // proposed solution for convTranspose with 4x4 kernel is to split 4x4 kernel into 4 2x2 kernels are do
        // convolution, slice certain rows/cols and followed by interleaving the final data.
        // Reason for using 3x3 is: with 2x2 logic, tensors with larger size are taking longer time when slicing col/row
        // are added as part of conv layer, hence based on MEM_LIMIT size 2x2 or 3x3 kernel logic is chosen
        if ((arch > config::ArchKind::NPU40XX) &&
            ((outputshape[Dims4D::Act::H] * outputshape[Dims4D::Act::W]) >= MEM_LIMIT)) {
            use3x3 = true;
        } else {
            use2x2 = true;
        }

        const auto filterContext = newFilterType.getContext();
        mlir::RankedTensorType dataStorageType;
        mlir::DenseElementsAttr splitFilterAttr1;
        mlir::DenseElementsAttr splitFilterAttr2;
        mlir::DenseElementsAttr splitFilterAttr3;
        mlir::DenseElementsAttr splitFilterAttr4;
        SmallVector<int64_t> splitFilterShape;
        if (use3x3) {
            splitFilterShape = {N, C, shiftKY, shiftKX};
        } else {
            splitFilterShape = {N, C, splitKY, splitKX};
        }
        if (isSint8) {
            auto splitFiltertype = mlir::IntegerType::get(filterContext, 8, mlir::IntegerType::Signed);
            dataStorageType = mlir::RankedTensorType::get(splitFilterShape, splitFiltertype);
            const auto& weightsSet = use3x3 ? weightsSplitsShiftI8 : weightsSplitsI8;
            splitFilterAttr1 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[0]));
            splitFilterAttr2 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[1]));
            splitFilterAttr3 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[2]));
            splitFilterAttr4 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[3]));
        } else if (isUint8) {
            auto splitFiltertype = mlir::IntegerType::get(filterContext, 8, mlir::IntegerType::Unsigned);
            dataStorageType = mlir::RankedTensorType::get(splitFilterShape, splitFiltertype);
            const auto& weightsSet = use3x3 ? weightsSplitsShiftU8 : weightsSplitsU8;
            splitFilterAttr1 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[0]));
            splitFilterAttr2 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[1]));
            splitFilterAttr3 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[2]));
            splitFilterAttr4 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[3]));
        } else if (isUint16) {
            auto splitFiltertype = mlir::IntegerType::get(filterContext, 16, mlir::IntegerType::Unsigned);
            dataStorageType = mlir::RankedTensorType::get(splitFilterShape, splitFiltertype);
            const auto& weightsSet = use3x3 ? weightsSplitsShiftU16 : weightsSplitsU16;
            splitFilterAttr1 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[0]));
            splitFilterAttr2 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[1]));
            splitFilterAttr3 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[2]));
            splitFilterAttr4 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[3]));
        } else if (isF16) {
            auto splitFiltertype = mlir::Float16Type::get(filterContext);
            dataStorageType = mlir::RankedTensorType::get(splitFilterShape, splitFiltertype);
            const auto& weightsSet = use3x3 ? weightsSplitsShiftFp16 : weightsSplitsFp16;
            splitFilterAttr1 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[0]));
            splitFilterAttr2 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[1]));
            splitFilterAttr3 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[2]));
            splitFilterAttr4 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[3]));
        } else {  // default to float32
            auto splitFiltertype = mlir::Float32Type::get(filterContext);
            dataStorageType = mlir::RankedTensorType::get(splitFilterShape, splitFiltertype);
            const auto& weightsSet = use3x3 ? weightsSplitsShiftFp : weightsSplitsFp;
            splitFilterAttr1 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[0]));
            splitFilterAttr2 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[1]));
            splitFilterAttr3 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[2]));
            splitFilterAttr4 = Const::createConstContent(dataStorageType, ArrayRef(weightsSet[3]));
        }

        std::vector<Const::ContentAttr> wContentAttrs(numSplits);
        std::vector<Const::ContentSetup> contentAttrSetups;
        contentAttrSetups.reserve(numSplits);

        for (int i = 0; i < numSplits; ++i) {
            contentAttrSetups.emplace_back(dataStorageType);
        }

        auto castElem = [&](mlir::Type type) {
            for (auto& setup : contentAttrSetups) {
                setup = setup.castElemType(type);
            }
        };

        if (const auto qElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(newFilterType.getElementType())) {
            castElem(qElemType);
        } else if (mlir::isa<mlir::Float16Type>(newFilterType.getElementType())) {
            castElem(mlir::Float16Type::get(filterContext));
        } else if (mlir::isa<mlir::Float32Type>(newFilterType.getElementType())) {
            castElem(mlir::Float32Type::get(filterContext));
        }

        // generate contentAttr
        wContentAttrs[0] = Const::ContentAttr::get(splitFilterAttr1, contentAttrSetups[0]);
        wContentAttrs[1] = Const::ContentAttr::get(splitFilterAttr2, contentAttrSetups[1]);
        wContentAttrs[2] = Const::ContentAttr::get(splitFilterAttr3, contentAttrSetups[2]);
        wContentAttrs[3] = Const::ContentAttr::get(splitFilterAttr4, contentAttrSetups[3]);

        vpux::NDTypeInterface splitFilterType;
        if (use3x3) {
            splitFilterType = newFilterType.changeShape(ShapeRef({C, N, shiftKY, shiftKX}));
        }
        if (use2x2) {
            splitFilterType = newFilterType.changeShape(ShapeRef({C, N, splitKY, splitKX}));
        }

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
                weightsConstOps[i] = rewriter.createOrFold<IE::ConvertOp>(loc, weightsConstOps[i],
                                                                          filterConvertOp.getDstElemTypeAttr());
            }
        }

        SmallVector<mlir::Value> nceOps;
        auto newStrides = getIntArrayAttr(getContext(), ov::Strides{1, 1});

        auto [padsBegin, padsEnd] = createPads(getContext(), use3x3);

        nceOps.reserve(numSplits);

        for (int i = 0; i < numSplits; ++i) {
            std::string suffix = "_split_" + std::to_string(i + 1);
            auto convOp = rewriter.create<IE::ConvolutionOp>(
                    appendLoc(origOp->getLoc(), "Convolution" + suffix), origOp.getInput(), weightsConstOps[i],
                    /*bias=*/nullptr, newStrides, padsBegin[i], padsEnd[i], origOp.getDilationsAttr(),
                    /*postOp=*/nullptr,
                    /*clamp=*/nullptr,
                    /*static_scale=*/nullptr,
                    /*outputPadding=*/nullptr,
                    /*inputPadding=*/nullptr);
            nceOps.push_back(convOp.getOutput());
        }

        vpux::IE::AddOp addOp = nullptr;
        if (!(origOp.getResult().getUsers().empty())) {
            addOp = mlir::dyn_cast_or_null<IE::AddOp>(*origOp.getResult().getUsers().begin());
        }

        if (addOp != nullptr) {
            auto input1 = addOp.getInput1().getDefiningOp<Const::DeclareOp>();
            auto input2 = addOp.getInput2().getDefiningOp<Const::DeclareOp>();

            for (auto i = 0; i < numSplits; ++i) {
                std::string suffix = "+Bias_Add_" + std::to_string(i + 1);

                auto op = rewriter.create<IE::AddOp>(appendLoc(origOp->getLoc(), suffix), nceOps[i],
                                                     ((input1 == nullptr) ? input2 : input1),
                                                     addOp.getAutoBroadcastAttr(), nullptr, nullptr, nullptr, nullptr);

                nceOps[i] = op.getOutput();
            }
        }

        auto combined = rewriter.create<IE::ConcatOp>(appendLoc(origOp->getLoc(), "Concat_Convs"), nceOps, 1);
        auto interleave = rewriter.replaceOpWithNewOp<IE::DepthToSpaceOp>(origOp, combined.getOutput(), 2,
                                                                          IE::DepthToSpaceMode::BLOCKS_FIRST);
        extendOpLoc(interleave, "Depth2Space");

        const auto adjustedInputShapeAttr = getIntArrayAttr(
                origOp->getContext(), Shape({outputshape[Dims4D::Act::N], outputshape[Dims4D::Act::C],
                                             outputshape[Dims4D::Act::H], outputshape[Dims4D::Act::W]}));
        if (addOp != nullptr) {
            auto finalReshape = rewriter.replaceOpWithNewOp<IE::ReshapeOp>(addOp, interleave.getOutput(), nullptr,
                                                                           false, adjustedInputShapeAttr);
            extendOpLoc(finalReshape, "Reshape");
        }
    } else {
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
                origOp, origOp.getInput(), newFilter, origOp.getOutputShape(), /*bias*/ nullptr,
                origOp.getStridesAttr(), origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
                origOp.getSpatialOutputPaddingAttr(), /*postOp=*/nullptr, /*clamp=*/nullptr, /*outputPadding=*/nullptr,
                /*inputPadding=*/nullptr);
    }

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

    // Support filter pattern:
    //     Const::DeclareOp
    //             |
    //    (IE::FakeQuantizeOp)
    //             |
    //     (IE::ConvertOp)
    //             |
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

    // Reverse IC and OC dimensions in filter constant:
    //   [GROUPS, IC, OC, X] -> [GROUPS, OC, IC, X]
    //   [GROUPS, IC, OC, Y, X] -> [GROUPS, OC, IC, Y, X]
    //   [GROUPS, IC, OC, Z, Y, X] -> [GROUPS, OC, IC, Z, Y, X]
    const auto getNewFilterDimsOrder = [](const int64_t rank) {
        SmallVector<vpux::Dim> permutation{};
        if (rank == 4) {
            permutation = {Dim(0), Dim(2), Dim(1), Dim(3)};
        } else if (rank == 5) {
            permutation = {Dim(0), Dim(2), Dim(1), Dim(3), Dim(4)};
        } else if (rank == 6) {
            permutation = {Dim(0), Dim(2), Dim(1), Dim(3), Dim(4), Dim(5)};
        }
        return DimsOrder::fromPermutation(permutation);
    };
    auto filterType = mlir::cast<vpux::NDTypeInterface>(filterOp.getType());
    auto newDimsOrder = getNewFilterDimsOrder(filterType.getRank());
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
// ConvertConvBackpropDataToConvOrTransposedConvPass
//

class ConvertConvBackpropDataToConvOrTransposedConvPass final :
        public IE::impl::ConvertConvBackpropDataToConvOrTransposedConvBase<
                ConvertConvBackpropDataToConvOrTransposedConvPass> {
public:
    explicit ConvertConvBackpropDataToConvOrTransposedConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertConvBackpropDataToConvOrTransposedConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::ConvolutionBackpropDataOp>();
    target.addIllegalOp<IE::GroupConvolutionBackpropDataOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<IE::GroupTransposedConvolutionOp>();
    target.addLegalOp<IE::TransposeOp>();
    target.addLegalOp<IE::TransposedConvolutionOp>();
    target.addLegalOp<IE::ConvertOp>();
    target.addLegalOp<IE::ReverseOp>();
    target.addLegalOp<IE::ConvolutionOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::DepthToSpaceOp>();
    target.addLegalOp<IE::AddOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvBackpropDataToConvOrTransConvConversion>(&ctx, _log);
    patterns.add<ConvBackpropDataToGroupTransConvConversion>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertConvBackpropDataToConvOrTransposedConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertConvBackpropDataToConvOrTransposedConvPass(Logger log) {
    return std::make_unique<ConvertConvBackpropDataToConvOrTransposedConvPass>(log);
}
