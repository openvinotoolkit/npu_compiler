//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <cstdint>

namespace vpux::VPU {
#define GEN_PASS_DECL_ADJUSTBLOCKSIZEFORSCFTILING
#define GEN_PASS_DEF_ADJUSTBLOCKSIZEFORSCFTILING
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

constexpr size_t MAX_TILES_PER_DYNAMIC_DIMENSION = 3;  // START, MIDDLE, END

using DimToBlockSizesMap = llvm::DenseMap<int64_t, llvm::DenseMap<int64_t, int64_t>>;
struct SliceOpData {
    SmallVector<size_t> dynamicDims;
    std::array<mlir::Value, MAX_NUM_DIMS> blockPositionValues;
    llvm::DenseMap<size_t, vpux::Shape> caseToShapeMap;

    SliceOpData(): dynamicDims() {
        blockPositionValues.fill(nullptr);
    }

    void setBlockPositionVar(size_t dimIdx, mlir::Value positionVar) {
        if (dimIdx >= MAX_NUM_DIMS) {
            VPUX_THROW("Dimension index {0} exceeds maximum supported dimensions {1}", dimIdx, MAX_NUM_DIMS);
        }
        blockPositionValues[dimIdx] = positionVar;
        dynamicDims.push_back(dimIdx);
    }

    bool hasAdjustedDynDims() const {
        return !dynamicDims.empty();
    }
};

struct IndexOperandConversionInfo {
    llvm::DenseMap<int64_t, SmallVector<int64_t>> operandStaticValues;

    bool shouldConvert() const {
        return !operandStaticValues.empty();
    }
};

//
// AdjustBlockSizeForScfTilingPass
//
class AdjustBlockSizeForScfTilingPass final :
        public VPU::impl::AdjustBlockSizeForScfTilingBase<AdjustBlockSizeForScfTilingPass> {
public:
    explicit AdjustBlockSizeForScfTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

    struct AdjustedIndexInfo {
        mlir::OpFoldResult index;
        mlir::Operation* indexDefiningOp = nullptr;
    };

    mlir::LogicalResult adjustOutputBlockIdxAndSize(mlir::scf::ForOp forOp,
                                                    llvm::DenseMap<mlir::scf::ForOp, int64_t>& forOpToDim);
    mlir::LogicalResult calculateInputBlockPosAndShapeInformation(
            mlir::scf::ForOp forOp, llvm::DenseMap<mlir::Operation*, SliceOpData>& mapSliceOpToData,
            llvm::DenseMap<mlir::scf::ForOp, int64_t>& forOpToDim);
    mlir::LogicalResult generateBlockAwareFuncOps(mlir::func::CallOp callOp,
                                                  llvm::DenseMap<mlir::Operation*, SliceOpData>& mapSliceOpToData,
                                                  const llvm::DenseMap<mlir::scf::ForOp, int64_t>& forOpToDimInfo,
                                                  const IndexOperandConversionInfo& conversionInfo);
    mlir::scf::IfOp getTensorBlockPosition(mlir::scf::ForOp forOp, mlir::OpBuilder builder, mlir::Value offsetVal,
                                           mlir::Operation* insertionPoint, mlir::Value currentBlkSize,
                                           mlir::Value upperBound);
    mlir::LogicalResult processIndexTypeArgsInCallOp(mlir::func::CallOp callOp,
                                                     const IndexOperandConversionInfo& conversionInfo);
    mlir::LogicalResult processIndexTypeArgsInCallOp(mlir::OpBuilder& builder, mlir::func::FuncOp funcOp,
                                                     mlir::func::CallOp callOp,
                                                     const IndexOperandConversionInfo& conversionInfo,
                                                     const llvm::DenseMap<mlir::scf::ForOp, int64_t>& forOpToDimInfo,
                                                     const llvm::DenseMap<int64_t, size_t>& dimToEncodingIndex,
                                                     ArrayRef<TilePosition> caseValues);

    std::pair<mlir::IntegerAttr, mlir::Value> createConstOpOutsideForOp(int64_t val) {
        if (_constOpCache.contains(val)) {
            return _constOpCache[val];
        }

        mlir::OpBuilder builder(_firstForOp);
        auto constAttr = builder.getIntegerAttr(builder.getIndexType(), val);
        auto constOp = builder.create<mlir::arith::ConstantOp>(takeOpLoc(_firstForOp, "const_val"), constAttr);
        _constOpCache[val] = {constAttr, constOp};
        return _constOpCache[val];
    }

    mlir::tensor::DimOp createDimOp(mlir::OpBuilder builder, mlir::scf::ForOp forOp, mlir::Value tensor, size_t idx) {
        mlir::OpBuilder localBuilder(forOp);
        auto dimOp = forOp.isDefinedOutsideOfLoop(tensor)
                             ? localBuilder.create<mlir::tensor::DimOp>(takeOpLoc(forOp, "dim"), tensor, idx)
                             : builder.create<mlir::tensor::DimOp>(takeOpLoc(forOp, "dim"), tensor, idx);
        return dimOp;
    }

    mlir::arith::ConstantIndexOp getCstIndexOp(int64_t val) {
        if (_constIndexOpCache.contains(val)) {
            return _constIndexOpCache[val];
        }

        mlir::OpBuilder builder(_firstForOp);
        auto cstOp = builder.create<mlir::arith::ConstantIndexOp>(
                takeOpLoc(_firstForOp, "const_" + std::to_string(val)), val);
        _constIndexOpCache[val] = cstOp;
        return cstOp;
    }

    void buildMapOfBlockSizesForSliceOps(
            mlir::scf::ForOp forOp,
            llvm::DenseMap<mlir::Operation*, DimToBlockSizesMap>& mapSliceOpToPerDimensionBlockSizes);

    OpChainAnalysis _opChainAnalysis;
    mlir::ModuleOp _moduleOp;
    mlir::scf::ForOp _firstForOp;
    mlir::func::FuncOp _mainFuncOp;
    llvm::DenseMap<int64_t, mlir::arith::ConstantIndexOp> _constIndexOpCache;
    llvm::DenseMap<int64_t, std::pair<mlir::IntegerAttr, mlir::Value>> _constOpCache;
};

mlir::Operation* getOffsetInsertionPosition(mlir::Operation* op) {
    mlir::Operation* firstNonArithOp = op;
    if (auto forOp = op->getParentOfType<mlir::scf::ForOp>()) {
        for (auto& op : forOp.getBody()->getOperations()) {
            if (mlir::isa<mlir::arith::ArithDialect>(op.getDialect()) ||
                mlir::isa<mlir::affine::AffineDialect>(op.getDialect()) || mlir::isa<mlir::scf::IfOp>(op)) {
                continue;
            } else {
                firstNonArithOp = &op;
                break;
            }
        }
    }
    return firstNonArithOp;
}

// Helper function that returns the scf::ForOp on which the `offsetVal` transitively depends.
// Returns `nullptr` if it does not depend on any `scf::ForOp`.
mlir::scf::ForOp getEnclosingForOp(mlir::Value offsetVal, OpChainAnalysis& opChainAnalysis) {
    auto getForOpFromBlockArg = [](mlir::Value val) -> mlir::scf::ForOp {
        if (auto blockOperand = mlir::dyn_cast<mlir::BlockArgument>(val)) {
            auto parentOp = blockOperand.getOwner()->getParentOp();
            return mlir::dyn_cast_or_null<mlir::scf::ForOp>(parentOp);
        }
        return {};
    };

    if (auto forOp = getForOpFromBlockArg(offsetVal)) {
        return forOp;
    }

    auto opChain = opChainAnalysis.collectParentOpsChain(offsetVal);
    for (auto op : opChain) {
        for (auto operand : op->getOperands()) {
            if (auto forOp = getForOpFromBlockArg(operand)) {
                return forOp;
            }
        }
    }
    return nullptr;
}

// Get the top-most scf.for op in the nested loops
mlir::scf::ForOp getOutermostForOp(mlir::scf::ForOp forOp) {
    auto currentForOp = forOp;
    auto topForOp = forOp;
    while (auto parentForOp = currentForOp->getParentOfType<mlir::scf::ForOp>()) {
        topForOp = parentForOp;
        currentForOp = parentForOp;
    }

    return topForOp;
}

/**
 * @brief Generates conditional logic to determine tile position for SCF tiling operations.
 *
 * This function creates a conditional structure using SCF if operations to handle different scenarios
 * when processing tensor slice insertions within nested SCF for loops. It determines the appropriate tile
 * position based on the current offset value and block sizes.
 *
 * Tensor Block Division Diagram:
 * ┌─────────────┬──────────────────────────────────┬─────────────┐
 * │    START    │             MIDDLE               │     END     │
 * │   (index=0) │        (index > 0, sum < bound)  │ (last tile) │
 * └─────────────┴──────────────────────────────────┴─────────────┘
 *   startBlkSize        middleBlkSize                 lastBlkSize
 *
 * The generated logic handles three main cases:
 * 1. When offset is zero: StartBlkSize should have at least step size elements. yields 2
 * 2. When offset + step < upper_bound: Middle tile. Yields 0
 * 3. When offset + step >= upper_bound: Last tile. Yields 1
 *
 * @return mlir::scf::IfOp The root conditional operation containing the tile position logic,
 *         or nullptr if the parent SCF for operation cannot be found
 */
mlir::scf::IfOp AdjustBlockSizeForScfTilingPass::getTensorBlockPosition(mlir::scf::ForOp, mlir::OpBuilder builder,
                                                                        mlir::Value offset,
                                                                        mlir::Operation* insertionPoint,
                                                                        mlir::Value currentBlkSize,
                                                                        mlir::Value upperBound) {
    builder.setInsertionPoint(insertionPoint);
    auto zeroConst = getCstIndexOp(static_cast<int64_t>(TilePosition::MIDDLE));
    auto oneConst = getCstIndexOp(static_cast<int64_t>(TilePosition::END));
    auto twoConst = getCstIndexOp(static_cast<int64_t>(TilePosition::START));
    auto threeConst = getCstIndexOp(static_cast<int64_t>(TilePosition::FULLBLK));
    auto indexIsZero = builder.create<mlir::arith::CmpIOp>(takeOpLoc(insertionPoint, "cmp_index_zero"),
                                                           mlir::arith::CmpIPredicate::eq, offset, zeroConst);

    auto ifIndexZero = builder.create<mlir::scf::IfOp>(takeOpLoc(insertionPoint, "if_offset_zero"),
                                                       llvm::ArrayRef<mlir::Type>{builder.getIndexType()}, indexIsZero,
                                                       /*withElseRegion=*/true);
    // if index == 0 :
    //    assert if num_elements < first_blk_size
    //    if num_elements == total_elements
    //       yield 3
    //    else:
    //       yield 2
    {
        mlir::OpBuilder thenBuilder = ifIndexZero.getThenBodyBuilder();
        auto checkForFullTile = thenBuilder.create<mlir::arith::CmpIOp>(
                takeOpLoc(ifIndexZero, "full_tile_check"), mlir::arith::CmpIPredicate::eq, currentBlkSize, upperBound);
        auto selectIndex = thenBuilder.create<mlir::arith::SelectOp>(takeOpLoc(ifIndexZero, "select_full_tile"),
                                                                     checkForFullTile, threeConst, twoConst);
        thenBuilder.create<mlir::scf::YieldOp>(takeOpLoc(selectIndex, "yield"),
                                               mlir::ValueRange{selectIndex->getResult(0)});
    }
    // Else block: index != 0
    // sum = index + current_block_size
    // if sum < upper_bound :
    //    yield 0
    // else:
    //    yield 1
    {
        mlir::OpBuilder elseBuilder = ifIndexZero.getElseBodyBuilder();
        auto sum = elseBuilder.create<mlir::arith::AddIOp>(takeOpLoc(ifIndexZero, "else_add"), offset, currentBlkSize);
        auto sumLessThanBound = elseBuilder.create<mlir::arith::CmpIOp>(
                takeOpLoc(ifIndexZero, "pos_check"), mlir::arith::CmpIPredicate::slt, sum, upperBound);
        auto selectIndex = elseBuilder.create<mlir::arith::SelectOp>(takeOpLoc(ifIndexZero, "select_tile_pos"),
                                                                     sumLessThanBound, zeroConst, oneConst);
        elseBuilder.create<mlir::scf::YieldOp>(takeOpLoc(ifIndexZero, "yield"), selectIndex->getResult(0));
    }
    return ifIndexZero;
}

// Special condition where the output is divided equally on all tiles and shape is known at compile time.
bool checkForStaticShape(const SmallVector<mlir::OpFoldResult, 4>& mixedSizes) {
    return llvm::all_of(mixedSizes, [&](mlir::OpFoldResult size) {
        if (auto sizeVal = mlir::dyn_cast<mlir::Value>(size)) {
            return mlir::isa<mlir::arith::ConstantOp>(sizeVal.getDefiningOp());
        }
        return true;
    });
}

/**
 * @brief Adjusts indices into dynamic tensors within scf.for loops based on the current index and fixed step size.
 * When the remaining elements in the current iteration are fewer than the step size, this method backtracks the index
 * to ensure extraction of a static-shaped tensor.
 *
 * For example, given a dynamic tensor of shape <1x1x32x?xfp16> with bounds [1, 1, 32, 1000] and step size 100,
 * processing an input tensor of <1x1x32x250xfp16> would normally have the last iteration at index 200,
 * but only 50 elements would remain. This method adjusts the index to 150 (extracting a slice
 * from offset 150 with size 100), resulting in a static-shaped tensor <1x1x32x100xfp16>.
 */
mlir::LogicalResult AdjustBlockSizeForScfTilingPass::adjustOutputBlockIdxAndSize(
        mlir::scf::ForOp parentForOp, llvm::DenseMap<mlir::scf::ForOp, int64_t>& forOpToDim) {
    OpChainAnalysis opChainAnalysis;
    for (auto insertSliceOp : make_early_inc_range(parentForOp.getOps<mlir::tensor::InsertSliceOp>())) {
        if (checkForStaticShape(insertSliceOp.getMixedSizes())) {
            _log.trace("Block shape on insertSliceOp is known at compile time. Continue");
            continue;
        }

        applyIndexBacktracking(insertSliceOp, {});

        mlir::OpBuilder builder(insertSliceOp);
        SmallVector<mlir::OpFoldResult> newOffsets = {}, newSizes = {};
        for (auto [idx, size] : llvm::enumerate(insertSliceOp.getMixedSizes())) {
            auto val = mlir::dyn_cast_or_null<mlir::Value>(size);
            if (val == nullptr || mlir::isa<mlir::arith::ConstantOp>(val.getDefiningOp())) {
                newSizes.push_back(size);
                newOffsets.push_back(insertSliceOp.getMixedOffsets()[idx]);
                continue;
            }

            auto offset = insertSliceOp.getMixedOffsets()[idx];
            newOffsets.push_back(offset);
            auto forOp = getEnclosingForOp(val, _opChainAnalysis);
            VPUX_THROW_UNLESS(forOp != nullptr, "Expected parent scf.for operation for slice size but got none");
            auto blockSize = forOp.getStep();
            newSizes.push_back(blockSize);

            // Insert into forOpToDim and check for conflicts
            auto insertResult = forOpToDim.insert({forOp, static_cast<int64_t>(idx)});
            if (!insertResult.second) {
                // ForOp already exists, check if it's mapped to the same dimension
                auto existingDim = insertResult.first->second;
                if (existingDim != static_cast<int64_t>(idx)) {
                    VPUX_THROW("Multiple dimensions found for the same scf.for loop. Dimension {0} vs {1}", existingDim,
                               idx);
                }
            }

            // Add runtime check to ensure block size is not greater than dimension size
            mlir::tensor::DimOp dimOp;
            if (mlir::failed(
                        getTensorDimOpFromIndex(builder, insertSliceOp.getDest(), checked_cast<int64_t>(idx), dimOp))) {
                return mlir::failure();
            }
            addCheckForBlockSize(builder, dimOp, blockSize, _mainFuncOp,
                                 "Not enough elements to backtrack in scf.for loop for Output tensor");
        }

        // Create a new insert_slice operation with adjusted offsets and sizes
        builder.setInsertionPoint(insertSliceOp);
        auto newInsertSliceOp = builder.create<mlir::tensor::InsertSliceOp>(
                insertSliceOp.getLoc(), insertSliceOp.getSource(), insertSliceOp.getDest(), newOffsets, newSizes,
                insertSliceOp.getMixedStrides());
        newInsertSliceOp->setAttrs(insertSliceOp->getAttrs());
        extendOpLoc(newInsertSliceOp, "insert_slice_adjusted");

        // Replace the old insert_slice operation with the new one
        insertSliceOp.getResult().replaceAllUsesExcept(newInsertSliceOp.getResult(), newInsertSliceOp);
        insertSliceOp.erase();
    }

    return mlir::success();
}

bool getTensorBlockSizes(mlir::Value val, llvm::DenseMap<int64_t, int64_t>& mapToDimBlockSizes,
                         OpChainAnalysis& opChainAnalysis, Logger& log) {
    auto forOp = getEnclosingForOp(val, opChainAnalysis);
    if (forOp == nullptr) {
        log.trace("Failed to find parent scf.for operation for slice size");
        return false;
    }

    auto upperBound = opChainAnalysis.getIntegerFromValue(forOp.getUpperBound(), true);
    auto stepSize = opChainAnalysis.getIntegerFromValue(forOp.getStep());
    if (!upperBound.has_value() || !stepSize.has_value()) {
        log.trace("Failed to get integer value for step size and upper bound");
        return false;
    }

    auto upperBoundInt = upperBound.value();
    auto stepInt = stepSize.value();

    SmallVector<int64_t> valueMap;
    SmallVector<TilePosition> positionVec;
    int64_t numBlocks = (upperBoundInt + stepInt - 1) / stepInt;
    valueMap.push_back(0);
    switch (numBlocks) {
    case 1:
        positionVec = {TilePosition::FULLBLK};
        break;
    case 2:
        positionVec = {TilePosition::START, TilePosition::END};
        valueMap.push_back(stepInt);
        break;
    default:
        positionVec = {TilePosition::START, TilePosition::MIDDLE, TilePosition::END};
        valueMap.push_back(stepInt);
        valueMap.push_back(upperBoundInt - stepInt);
    }

    ValueRangeMap mapToValues;
    mapToValues.insert_or_assign(forOp.getInductionVar(), std::move(valueMap));
    auto blockSizesVal = opChainAnalysis.getOpFoldResultValue(val, mapToValues, OpChainAnalysis::MODE::ALL_VALUES);
    assert(blockSizesVal.has_value() && "Failed to get block sizes from operation chain");
    auto blockSizes = blockSizesVal.value();

    for (auto [pos, blkSize] : llvm::zip(positionVec, blockSizes)) {
        mapToDimBlockSizes[static_cast<int64_t>(pos)] = blkSize;
    }

    ValueRangeMap emptyMapToValues;
    auto newBlockSizeVals =
            opChainAnalysis.getOpFoldResultValue(val, emptyMapToValues, OpChainAnalysis::MODE::ALL_VALUES);
    for (auto blkSize : newBlockSizeVals.value()) {
        log.trace("Block size value after removing block arg dependency: {0}", blkSize);
    }

    return true;
}

void getMapForPositionAndBlockSizes(
        SmallVector<std::pair<vpux::Dim, llvm::DenseMap<int64_t, int64_t>>>& dynDimsToTilePositionAndSizeVec,
        mlir::tensor::ExtractSliceOp sliceOp, SliceOpData& sliceOpData) {
    auto numDynDims = dynDimsToTilePositionAndSizeVec.size();
    auto numCombos = static_cast<int64_t>(1LL << (NUMBITS * numDynDims));
    const auto mask = static_cast<int64_t>((1LL << NUMBITS) - 1);

    auto baseShape = Shape(getShape(sliceOp.getResult()));
    for (int64_t combo = 0; combo < numCombos; ++combo) {
        bool supportedCase = true;
        SmallVector<std::pair<vpux::Dim, TilePosition>> dimPositions;
        auto newShape = baseShape;
        for (size_t j = 0; j < numDynDims; ++j) {
            auto denseShift = checked_cast<int>(NUMBITS * (numDynDims - j - 1));
            auto pos = static_cast<TilePosition>((combo >> denseShift) & mask);
            auto dim = dynDimsToTilePositionAndSizeVec[j].first;
            auto& tensorPosAndSizeMap = dynDimsToTilePositionAndSizeVec[j].second;

            dimPositions.push_back({dim, pos});
            if (!tensorPosAndSizeMap.contains(static_cast<int64_t>(pos))) {
                supportedCase = false;
            } else {
                newShape[dim] = tensorPosAndSizeMap[static_cast<int64_t>(pos)];
            }
        }

        if (supportedCase) {
            sliceOpData.caseToShapeMap[getEncodedPointPosition(dimPositions)] = std::move(newShape);
        }
    }
}

/**
 * @brief Calculates block position information and input tensor slice shape information for SCF tiling.
 *
 * This function processes ExtractSliceOp operations within an SCF for loop to determine tile positions
 * and compute shape information for dynamic tensor slicing. It performs the following key operations:
 *
 * 1. **Tile Position Detection**: For each dynamic dimension in ExtractSliceOp, generates conditional
 *    logic to determine the tile position (START, MIDDLE, END, or FULLBLK) based on loop indices
 * 2. **Shape Computation**: Calculates static shapes for each tile position and builds mapping from
 *    case values to tensor shapes for function specialization
 * 3. **Operation Replacement**: Replaces original ExtractSliceOp with adjusted versions that have
 *    static sizes, inserting CastOp operations to maintain dynamic type compatibility
 *
 * @param forOp The SCF for loop containing the slice operations to process
 * @param mapSliceOpToPerDimensionBlockSizes Input/Output map containing block size information per dimension for each
 * slice operation
 * @param dynDimsTensorBlockId Output vector of dynamic dimension to tensor block ID mappings for function generation
 * @param mapSliceOpToTileSizes Output map from slice operations to their supported tile shapes indexed by case values
 * @return mlir::success() if all slice operations were successfully processed, mlir::failure() otherwise
 */

mlir::LogicalResult AdjustBlockSizeForScfTilingPass::calculateInputBlockPosAndShapeInformation(
        mlir::scf::ForOp forOp, llvm::DenseMap<mlir::Operation*, SliceOpData>& mapSliceOpToData,
        llvm::DenseMap<mlir::scf::ForOp, int64_t>& forOpToDim) {
    llvm::DenseMap<mlir::Value, AdjustedIndexInfo> mapToAdjustedIdx;
    auto outermostForOp = getOutermostForOp(forOp);
    for (auto sliceOp : make_early_inc_range(forOp.getOps<mlir::tensor::ExtractSliceOp>())) {
        if (checkForStaticShape(sliceOp.getMixedSizes())) {
            _log.trace("Block shape on insertSliceOp is known at compile time. Continue");
            continue;
        }

        mlir::OpBuilder builder(sliceOp);
        auto insertionPoint = getOffsetInsertionPosition(sliceOp);
        llvm::DenseMap<int64_t, int64_t> mapToBlockPosAndSize;
        SmallVector<mlir::OpFoldResult> newOffsets = {}, newSizes = {};
        SmallVector<std::pair<vpux::Dim, llvm::DenseMap<int64_t, int64_t>>> dynDimsToTilePositionAndSizeVec;
        SliceOpData sliceOpData;

        for (auto [idx, size] : llvm::enumerate(sliceOp.getMixedSizes())) {
            mapToBlockPosAndSize.clear();
            auto val = mlir::dyn_cast_or_null<mlir::Value>(size);
            if (val == nullptr || mlir::isa<mlir::arith::ConstantOp>(val.getDefiningOp())) {
                newSizes.push_back(size);
                newOffsets.push_back(sliceOp.getMixedOffsets()[idx]);
                continue;
            }

            builder.setInsertionPoint(insertionPoint);

            // Get block sizes for each tile position along this dynamic dimension
            llvm::DenseMap<int64_t, int64_t> mapPosToBlockSizes;
            if (!getTensorBlockSizes(val, mapPosToBlockSizes, _opChainAnalysis, _log)) {
                return errorAt(sliceOp, "Failed to get block sizes for each tile position for dim {0}", idx);
            }
            dynDimsToTilePositionAndSizeVec.push_back({vpux::Dim(idx), mapPosToBlockSizes});

            if (auto forOp = getEnclosingForOp(val, _opChainAnalysis)) {
                forOpToDim.insert({forOp, static_cast<int64_t>(idx)});
            }

            if (mapPosToBlockSizes.size() == 1) {
                auto staticShape = mapPosToBlockSizes[static_cast<int64_t>(TilePosition::FULLBLK)];
                auto [shapeAttr, blockSize] = createConstOpOutsideForOp(staticShape);
                newSizes.push_back(shapeAttr);
            } else {
                auto shapeForStartBlk = mapPosToBlockSizes[static_cast<int64_t>(TilePosition::START)];
                auto [shapeAttr, blockSize] = createConstOpOutsideForOp(shapeForStartBlk);
                newSizes.push_back(shapeAttr);
            }

            auto offset = sliceOp.getMixedOffsets()[idx];
            auto offsetValue = mlir::dyn_cast_or_null<mlir::Value>(offset);
            assert(offsetValue != nullptr && "Expected a non-const offsetValue");

            auto newIndex = mapToAdjustedIdx.contains(offsetValue) ? mapToAdjustedIdx[offsetValue].index : offsetValue;
            newOffsets.push_back(newIndex);

            // Check if the index is already adjusted
            if (mapToAdjustedIdx.contains(offsetValue)) {
                auto adjustedIdxOp = mapToAdjustedIdx[offsetValue].indexDefiningOp;
                sliceOpData.setBlockPositionVar(idx, adjustedIdxOp->getResult(0));
                continue;
            }

            // Get the upperbound
            auto src = sliceOp.getSource();
            mlir::tensor::DimOp srcDimOp = createDimOp(builder, outermostForOp, src, idx);
            auto ifIndexZero = getTensorBlockPosition(forOp, builder, offsetValue, insertionPoint, val, srcDimOp);
            if (ifIndexZero == nullptr) {
                return errorAt(sliceOp,
                               "Failed to create tile position and backtrack index for insert_slice operation");
            }

            mapToAdjustedIdx[offsetValue] = {newIndex, ifIndexZero};
            sliceOpData.setBlockPositionVar(idx, ifIndexZero->getResult(0));
            insertionPoint = ifIndexZero;
        }

        // ExtractSliceOp now has static shape. However, for funcOps that have padding attributes, multiple funcOps
        // will be created to process different slices of the input tensor. Insert a CastOp to keep the dynamic
        // dimensions, as different funcOps might require different input shapes
        builder.setInsertionPoint(sliceOp);
        auto newSliceOp = builder.create<mlir::tensor::ExtractSliceOp>(
                appendLoc(sliceOp->getLoc(), "adjusted_input_slice"), sliceOp.getSource(), newOffsets, newSizes,
                sliceOp.getMixedStrides());
        auto newResultType = removeBoundsAttr(newSliceOp.getResultType());
        newSliceOp.getResult().setType(newResultType);

        // ExtractSliceOp -> CastOp -> xxxOp require CastOp to change the input type
        // ExtractSliceOp -> xxxOp require castOp to get the dynamic shape back from static shape
        SmallVector<mlir::OpOperand*> requireCastOp;
        for (auto& use : make_early_inc_range(sliceOp->getUses())) {
            if (mlir::isa<mlir::tensor::CastOp>(use.getOwner())) {
                auto castOp = mlir::cast<mlir::tensor::CastOp>(use.getOwner());
                auto newCastOp = builder.create<mlir::tensor::CastOp>(
                        newSliceOp->getLoc(), castOp.getResult().getType(), newSliceOp.getResult());
                castOp.getResult().replaceAllUsesWith(newCastOp);
                castOp.erase();
            } else {
                requireCastOp.push_back(&use);
            }
        }
        for (auto* opOperand : requireCastOp) {
            auto newCastOp = builder.create<mlir::tensor::CastOp>(newSliceOp->getLoc(), sliceOp->getResult(0).getType(),
                                                                  newSliceOp->getResult(0));
            opOperand->set(newCastOp.getResult());
        }

        sliceOp.erase();

        getMapForPositionAndBlockSizes(dynDimsToTilePositionAndSizeVec, newSliceOp, sliceOpData);
        mapSliceOpToData[newSliceOp.getOperation()] = std::move(sliceOpData);
    }

    return mlir::success();
}

void cleanUnusedFuncAndCallOps(mlir::func::CallOp callOp, mlir::ModuleOp moduleOp) {
    auto oldFuncName = callOp.getCallee().str();
    auto originalFuncOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(oldFuncName);

    if (callOp.use_empty()) {
        callOp->erase();
    }

    if (originalFuncOp.use_empty()) {
        originalFuncOp->erase();
    }
}

std::string generateUniqueFunctionSuffix(std::string const& funcName, ArrayRef<TilePosition> caseValues,
                                         ArrayRef<vpux::Dim> dynDims) {
    SmallVector<std::string> dimStr = {"N", "C", "H", "W"};

    // Add dynamic dimensions
    std::string suffix = "_dims_";
    for (auto dim : dynDims) {
        suffix += dimStr[dim.ind()];
    }

    // Add case values
    suffix += "_cases_";
    for (auto caseVal : caseValues) {
        suffix += std::to_string(static_cast<int64_t>(caseVal));
    }

    return funcName + suffix;
}

mlir::FailureOr<mlir::func::CallOp> createNewFuncOp(mlir::ModuleOp moduleOp, mlir::OpBuilder& builder,
                                                    const SmallVector<vpux::Dim>& dynDims,
                                                    const SmallVector<TilePosition>& caseValue,
                                                    mlir::func::CallOp callOp) {
    // Generate unique function name
    assert(callOp.getResultTypes().size() == 1 && "Expected single output function");
    auto oldFuncName = callOp.getCallee().str();
    auto originalFuncOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(oldFuncName);
    assert(originalFuncOp && "Expected function operation to be present");

    auto effectiveDynDimsOrFail = VPU::remapDimsThroughInputPermuteCast(originalFuncOp, dynDims);
    if (mlir::failed(effectiveDynDimsOrFail)) {
        return mlir::failure();
    }
    auto effectiveDynDims = effectiveDynDimsOrFail.value();
    std::string newFuncName = generateUniqueFunctionSuffix(oldFuncName, caseValue, effectiveDynDims);

    if (moduleOp.lookupSymbol<mlir::func::FuncOp>(newFuncName) != nullptr) {
        // Clone the call operation with new function
        auto newCallOp = builder.create<mlir::func::CallOp>(callOp.getLoc(), newFuncName, callOp.getResultTypes(),
                                                            callOp.getOperands());
        assert(newCallOp != nullptr && "Failed to create new call operation");
        return newCallOp;
    }

    // Clone funcOp
    auto moduleBuilder = mlir::OpBuilder(moduleOp);
    moduleBuilder.setInsertionPointAfter(originalFuncOp);
    auto clonedOp = moduleBuilder.clone(*originalFuncOp.getOperation());
    assert(clonedOp && "Failed to clone function operation");
    auto newFuncOp = mlir::dyn_cast<mlir::func::FuncOp>(clonedOp);
    assert(newFuncOp && "Cloned operation is not a function operation");
    newFuncOp.setSymName(newFuncName);

    // Fix the padding attributes
    newFuncOp.walk([&](mlir::Operation* op) {
        if (!isNceOpWithPadAttr(op)) {
            return;
        }
        auto padAttr = op->getAttrOfType<VPU::PaddingAttr>("pad");
        assert(padAttr != nullptr && "Expected padding attribute to be present");
        std::pair<int64_t, int64_t> hPad = {padAttr.getTop().getInt(), padAttr.getBottom().getInt()};
        std::pair<int64_t, int64_t> wPad = {padAttr.getLeft().getInt(), padAttr.getRight().getInt()};

        // Determine which padding fields to modify based on dimension
        // Handle padding for all dynamic dimensions
        for (size_t idx = 0; idx < effectiveDynDims.size(); ++idx) {
            auto [keepStart, keepEnd] = getTilePaddingMask(caseValue[idx]);
            if (effectiveDynDims[idx] == Dims4D::Act::H) {
                if (!keepStart) {
                    hPad.first = 0;
                }
                if (!keepEnd) {
                    hPad.second = 0;
                }
            } else if (effectiveDynDims[idx] == Dims4D::Act::W) {
                if (!keepStart) {
                    wPad.first = 0;
                }
                if (!keepEnd) {
                    wPad.second = 0;
                }
            }
        }

        auto newPadAttr = VPU::PaddingAttr::get(
                builder.getContext(), builder.getI64IntegerAttr(wPad.first), builder.getI64IntegerAttr(wPad.second),
                builder.getI64IntegerAttr(hPad.first), builder.getI64IntegerAttr(hPad.second));
        op->setAttr("pad", newPadAttr);
        vpux::inferReturnTypes(op, vpux::InferShapedTypeMode::SHAPE);
    });

    // Clone the call operation with new function
    auto newCallOp = builder.create<mlir::func::CallOp>(callOp.getLoc(), newFuncName, callOp.getResultTypes(),
                                                        callOp.getOperands());
    assert(newCallOp != nullptr && "Failed to create new call operation");
    return newCallOp;
}

void propagateTypeInCallOp(mlir::OpBuilder builder, mlir::ModuleOp moduleOp, mlir::func::CallOp newCallOp) {
    SmallVector<mlir::Type> newInputTypes;
    auto calleeName = newCallOp.getCallee();
    auto funcOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(calleeName);

    for (auto callOpOperand : newCallOp.getOperands()) {
        newInputTypes.push_back(callOpOperand.getType());
    }
    auto newFuncType = builder.getFunctionType(newInputTypes, funcOp.getFunctionType().getResults());
    funcOp.setType(newFuncType);

    for (size_t funcOpArgIdx = 0; funcOpArgIdx < funcOp.getNumArguments(); ++funcOpArgIdx) {
        auto funcOpArg = funcOp.getArgument(funcOpArgIdx);
        auto argType = funcOpArg.getType();
        if (mlir::isa<mlir::IndexType>(argType)) {
            continue;
        }
        if (auto argNdType = mlir::dyn_cast<NDTypeInterface>(argType)) {
            if (argNdType.getShape().isDynamic()) {
                funcOpArg.setType(newInputTypes[funcOpArgIdx]);
            }
        }
    }

    funcOp.walk([&](mlir::InferTypeOpInterface op) {
        if (llvm::any_of(op->getResultTypes(), [](mlir::Type type) {
                return !mlir::isa<vpux::NDTypeInterface>(type);
            })) {
            return;
        }
        vpux::inferReturnTypes(op, vpux::InferShapedTypeMode::SHAPE);
    });

    auto returnOps = funcOp.getOps<mlir::func::ReturnOp>();
    VPUX_THROW_UNLESS(llvm::hasSingleElement(returnOps), "expected single ReturnOp in '{0}'", funcOp.getName());
    auto singleReturnOp = *returnOps.begin();
    auto retTypes = singleReturnOp.getOperandTypes();
    VPUX_THROW_UNLESS(retTypes.size() == newCallOp.getNumResults(),
                      "propagateTypeInCallOp: inferred {0} result type(s) but call op has {1} result(s)",
                      retTypes.size(), newCallOp.getNumResults());
    funcOp.setType(builder.getFunctionType(funcOp.getFunctionType().getInputs(), retTypes));
    for (auto [callResult, newType] : llvm::zip(newCallOp.getResults(), retTypes)) {
        callResult.setType(newType);
    }
}

llvm::DenseMap<int64_t, size_t> buildDimToEncodingIndexMap(
        const llvm::DenseMap<mlir::scf::ForOp, int64_t>& forOpToDim) {
    auto sortedDims = llvm::to_vector(llvm::make_second_range(forOpToDim));
    llvm::sort(sortedDims);

    llvm::DenseMap<int64_t, size_t> dimToEncodingIndex;
    for (auto [encodingIdx, dim] : llvm::enumerate(sortedDims)) {
        dimToEncodingIndex[dim] = encodingIdx;
    }
    return dimToEncodingIndex;
}

// Map tile position to static value index
size_t getStaticValueIndexForTilePosition(TilePosition tilePosition, size_t numStaticValues) {
    if (tilePosition == TilePosition::START) {
        return 0;
    } else if (tilePosition == TilePosition::MIDDLE) {
        return (numStaticValues > 1) ? 1 : 0;
    } else if (tilePosition == TilePosition::END) {
        return (numStaticValues > 1) ? numStaticValues - 1 : 0;
    } else {  // TilePosition::FULLBLK or default
        return 0;
    }
}

void pruneCallOpOperands(mlir::OpBuilder& builder, mlir::func::CallOp callOp, ArrayRef<size_t> indicesToRemove) {
    llvm::DenseSet<size_t> removedSet(indicesToRemove.begin(), indicesToRemove.end());
    SmallVector<mlir::Value> newOperands;
    newOperands.reserve(callOp.getNumOperands() - indicesToRemove.size());

    for (auto [idx, operand] : llvm::enumerate(callOp.getOperands())) {
        if (!removedSet.contains(idx)) {
            newOperands.push_back(operand);
        }
    }

    builder.setInsertionPoint(callOp);
    auto newCallOp = builder.create<mlir::func::CallOp>(callOp.getLoc(), callOp.getCallee(), callOp.getResultTypes(),
                                                        newOperands);
    callOp.replaceAllUsesWith(newCallOp.getResults());
    callOp.erase();
}

mlir::LogicalResult pruneFuncOpOperands(mlir::OpBuilder& builder, mlir::func::FuncOp funcOp) {
    SmallVector<mlir::Type> newArgTypes;
    SmallVector<unsigned> argsToErase;
    for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
        if (arg.use_empty()) {
            argsToErase.push_back(checked_cast<unsigned>(idx));
            continue;
        }

        newArgTypes.push_back(arg.getType());
    }

    for (auto idx : llvm::reverse(argsToErase)) {
        auto result = funcOp.eraseArgument(idx);
        if (mlir::failed(result)) {
            return mlir::failure();
        }
    }

    auto newFuncType = builder.getFunctionType(newArgTypes, funcOp.getFunctionType().getResults());
    funcOp.setFunctionType(newFuncType);
    return mlir::success();
}

// Helper to inline static values into function and update call site
mlir::LogicalResult updateFunctionAndCallOp(mlir::OpBuilder& builder, mlir::func::FuncOp funcOp,
                                            mlir::func::CallOp callOp,
                                            const llvm::DenseMap<size_t, int64_t>& operandToStaticValue) {
    if (operandToStaticValue.empty()) {
        return mlir::success();
    }

    SmallVector<size_t> sortedOperandIndices = llvm::to_vector(llvm::make_first_range(operandToStaticValue));
    llvm::sort(sortedOperandIndices);

    mlir::IRMapping mapper;
    constexpr size_t maxExtractSliceOps = 8;
    llvm::SmallSetVector<mlir::tensor::ExtractSliceOp, maxExtractSliceOps> extractSliceOpsToUpdate;
    for (auto operandIdx : sortedOperandIndices) {
        if (operandIdx >= funcOp.getNumArguments()) {
            continue;
        }

        auto funcArg = funcOp.getArgument(operandIdx);
        {
            builder.setInsertionPointToStart(&funcOp.getBody().front());
            auto staticAttr = builder.getIndexAttr(operandToStaticValue.at(operandIdx));
            auto constOp = builder.create<mlir::arith::ConstantIndexOp>(funcOp.getLoc(), staticAttr.getInt());
            mapper.map(funcArg, constOp.getResult());
        }

        for (auto& use : llvm::make_early_inc_range(funcArg.getUses())) {
            if (auto extractSliceOp = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(use.getOwner())) {
                extractSliceOpsToUpdate.insert(extractSliceOp);
            }
        }
    }

    for (auto extractSliceOp : llvm::make_early_inc_range(extractSliceOpsToUpdate)) {
        builder.setInsertionPoint(extractSliceOp);
        auto newOp = builder.clone(*extractSliceOp, mapper);
        extractSliceOp.getResult().replaceAllUsesWith(newOp->getResult(0));
        extractSliceOp->erase();
    }

    pruneCallOpOperands(builder, callOp, sortedOperandIndices);
    return pruneFuncOpOperands(builder, funcOp);
}

bool isValidOperandIndex(size_t operandIdx, mlir::func::CallOp callOp, mlir::func::FuncOp funcOp) {
    return operandIdx < callOp.getNumOperands() && operandIdx < funcOp.getNumArguments();
}

// Rebuild call operands to match function signature preserving operand order.
// This is needed when the function has already had index args pruned and newly
// generated call ops still carry the original operand list.
static mlir::FailureOr<SmallVector<mlir::Value>> alignCallOperandsToFuncSignature(mlir::func::CallOp callOp,
                                                                                  mlir::func::FuncOp funcOp) {
    SmallVector<mlir::Value> alignedOperands;
    alignedOperands.reserve(funcOp.getNumArguments());

    auto funcArgIter = funcOp.getArguments().begin();
    auto callOpOperandIter = callOp.getOperands().begin();

    while (funcArgIter != funcOp.getArguments().end()) {
        bool found = false;
        for (auto operandIter = callOpOperandIter; operandIter != callOp.getOperands().end(); ++operandIter) {
            if (funcArgIter->getType() != (*operandIter).getType()) {
                continue;
            }

            alignedOperands.push_back(*operandIter);
            callOpOperandIter = std::next(operandIter);
            found = true;
            break;
        }

        if (!found) {
            return mlir::failure();
        }
        funcArgIter = std::next(funcArgIter);
    }

    return alignedOperands;
}

mlir::LogicalResult AdjustBlockSizeForScfTilingPass::processIndexTypeArgsInCallOp(
        mlir::OpBuilder& builder, mlir::func::FuncOp funcOp, mlir::func::CallOp callOp,
        const IndexOperandConversionInfo& conversionInfo,
        const llvm::DenseMap<mlir::scf::ForOp, int64_t>& forOpToDimInfo,
        const llvm::DenseMap<int64_t, size_t>& dimToEncodingIndex, ArrayRef<TilePosition> caseValues) {
    llvm::DenseMap<size_t, int64_t> operandToStaticValue;
    for (auto operandIdx : llvm::make_first_range(conversionInfo.operandStaticValues)) {
        if (!isValidOperandIndex(operandIdx, callOp, funcOp)) {
            return mlir::failure();
        }

        auto forOp = getEnclosingForOp(callOp->getOperand(checked_cast<unsigned int>(operandIdx)), _opChainAnalysis);
        if ((forOp == nullptr) || !forOpToDimInfo.contains(forOp)) {
            return mlir::failure();
        }

        int64_t dimension = forOpToDimInfo.at(forOp);
        if (!dimToEncodingIndex.contains(dimension)) {
            return mlir::failure();
        }

        size_t encodingIndex = dimToEncodingIndex.at(dimension);
        TilePosition tilePosition = caseValues[encodingIndex];

        const auto& staticValues = conversionInfo.operandStaticValues.at(operandIdx);
        size_t staticValueIndex = getStaticValueIndexForTilePosition(tilePosition, staticValues.size());

        if (staticValueIndex < staticValues.size()) {
            operandToStaticValue[operandIdx] = staticValues[staticValueIndex];
        }
    }

    return updateFunctionAndCallOp(builder, funcOp, callOp, operandToStaticValue);
}

mlir::LogicalResult AdjustBlockSizeForScfTilingPass::generateBlockAwareFuncOps(
        mlir::func::CallOp callOp, llvm::DenseMap<mlir::Operation*, SliceOpData>& mapSliceOpToData,
        const llvm::DenseMap<mlir::scf::ForOp, int64_t>& forOpToDimInfo,
        const IndexOperandConversionInfo& conversionInfo) {
    auto getInputSliceOps = [&](mlir::func::CallOp callOp) {
        SmallVector<std::pair<int64_t, mlir::Operation*>> sliceOps;
        for (auto& operand : callOp->getOpOperands()) {
            auto operandIdx = operand.getOperandNumber();
            auto user = operand.get().getDefiningOp();

            if (auto castOp = mlir::dyn_cast_or_null<mlir::tensor::CastOp>(user)) {
                user = castOp.getSource().getDefiningOp();
            }

            if (mlir::isa<mlir::tensor::ExtractSliceOp>(user)) {
                sliceOps.push_back({operandIdx, user});
            }
        }
        return sliceOps;
    };

    auto firstInputSliceOpIt = mapSliceOpToData.begin();
    auto& firstInputSliceOpData = firstInputSliceOpIt->second;

    mlir::OpBuilder builder(callOp);
    SmallVector<mlir::Value> values;
    SmallVector<vpux::Dim> dynDims;
    for (auto [dim, tensorBlockId] : llvm::enumerate(firstInputSliceOpData.blockPositionValues)) {
        if (tensorBlockId == nullptr) {
            continue;
        }

        values.push_back(tensorBlockId);
        dynDims.push_back(Dim(dim));
    }
    auto shiftAmounts = llvm::to_vector(llvm::map_range(dynDims, [this](vpux::Dim dim) {
        return createConstOpOutsideForOp(getTilePositionShift(dim)).second;
    }));
    auto encodedIdx = encodePointPosition(builder, callOp.getLoc(), values, shiftAmounts);

    // Although the number of cases is 4^N, not all cases are supported
    // E#183027 tracks the support for missing cases
    auto mapOfCaseVals = firstInputSliceOpData.caseToShapeMap;
    SmallVector<int64_t> validCaseValues;
    for (auto& [caseVal, shape] : mapOfCaseVals) {
        validCaseValues.push_back(checked_cast<int64_t>(caseVal));
    }
    sort(validCaseValues);

    auto getShapeAttrs = [this](const Shape& blockShape, SmallVector<mlir::OpFoldResult>& newShapeValues) {
        newShapeValues.reserve(blockShape.size());
        for (int64_t dim : blockShape) {
            newShapeValues.push_back(createConstOpOutsideForOp(dim).first);
        }
    };

    auto getShapeMapForSliceOp = [&](mlir::Operation* sliceOp) {
        if (mapSliceOpToData.contains(sliceOp)) {
            return mapSliceOpToData[sliceOp].caseToShapeMap;
        }
        VPUX_THROW("Failed to get position to block sizes map for sliceOp");
        return llvm::DenseMap<size_t, Shape>{};
    };

    // Each case block might require a different input slice shape. Since the funcOp requires a dynamic input
    // shape, the following pattern is used to extract static shaped tensor from input and call funcOp with dynamic
    // shape Example:
    //      case {
    //          %slice = tensor.extract_slice [1, 0, 0, 0] [1, 100, 100, 100] [1, 1, 1, 1] -> tensor<1x100x100x100xfp16>
    //          %cast = tensor.cast %slice to tensor<1x100x?x?xfp16> // CastOp to convert back to dynamic shape
    //          %res = func.call @func(%cast) // funcOp with adjusted padding attribute
    //      }
    auto constructBlock = [&](mlir::OpBuilder& builder, mlir::Block& block, int64_t caseIndex,
                              const SmallVector<vpux::Dim>& dynDims, SmallVector<TilePosition>& dynDimCaseValues,
                              mlir::func::CallOp callOp,
                              bool defaultCase = false) -> std::pair<mlir::func::CallOp, mlir::func::FuncOp> {
        auto extractSliceOps = getInputSliceOps(callOp);
        auto newCallOpOrFail = createNewFuncOp(_moduleOp, builder, dynDims, dynDimCaseValues, callOp);
        if (mlir::failed(newCallOpOrFail)) {
            return std::make_pair(mlir::func::CallOp(), mlir::func::FuncOp());
        }
        auto newCallOp = newCallOpOrFail.value();
        builder.create<mlir::scf::YieldOp>(newCallOp.getLoc(), newCallOp.getResults());
        builder.setInsertionPointToStart(&block);
        auto locString = defaultCase ? "_default_case" : ("_case_" + std::to_string(caseIndex));

        for (auto [operandIdx, sliceOp] : extractSliceOps) {
            auto extractSliceOp = mlir::cast<mlir::tensor::ExtractSliceOp>(sliceOp);
            SmallVector<mlir::OpFoldResult> newShape;
            auto shapeMap = getShapeMapForSliceOp(sliceOp);
            getShapeAttrs(shapeMap[caseIndex], newShape);

            if (defaultCase) {
                auto falseAttr = builder.create<mlir::arith::ConstantOp>(takeOpLoc(callOp, "bool_" + locString),
                                                                         builder.getBoolAttr(false));
                builder.create<mlir::cf::AssertOp>(takeOpLoc(callOp, "assert_valid_shape" + locString), falseAttr,
                                                   builder.getStringAttr("Unsupported case"));
            }

            // Create new extract slice op with adjusted shape and remove bounds attribute
            // Add CastOp to convert back to dynamic shape
            auto newSliceOp = builder.create<mlir::tensor::ExtractSliceOp>(
                    takeOpLoc(newCallOp, "adjusted_input_slice_" + locString), extractSliceOp.getSource(),
                    extractSliceOp.getMixedOffsets(), newShape, extractSliceOp.getMixedStrides());
            auto newSliceOpResultType = newSliceOp->getResult(0).getType();
            assert(mlir::isa<mlir::RankedTensorType>(newSliceOpResultType) &&
                   "Expected RankedTensorType for extract slice result");
            auto newReturnType = removeBoundsAttr(mlir::cast<mlir::RankedTensorType>(newSliceOpResultType));
            newSliceOp.getResult().setType(newReturnType);

            auto adjustedCastOpType = mlir::cast<vpux::NDTypeInterface>(newCallOp->getOperand(operandIdx).getType());
            auto boundedType = mlir::dyn_cast<vpux::Core::BoundedTensorType>(adjustedCastOpType);
            if (boundedType != nullptr) {
                ArrayRef<int64_t> shape = newSliceOp.getResult().getType().getShape();
                const auto tensorAttr =
                        vpux::getTensorAttr(adjustedCastOpType.getContext(), adjustedCastOpType.getDimsOrder(),
                                            adjustedCastOpType.getMemSpace(), BoundsRef(shape));
                adjustedCastOpType = mlir::RankedTensorType::get(adjustedCastOpType.getShape(),
                                                                 adjustedCastOpType.getElementType(), tensorAttr);
            }
            auto newCastOp = builder.create<mlir::tensor::CastOp>(newCallOp.getLoc(), adjustedCastOpType,
                                                                  newSliceOp.getResult());
            newCallOp->setOperand(operandIdx, newCastOp.getResult());
        }

        auto funcOp = _moduleOp.lookupSymbol<mlir::func::FuncOp>(newCallOp.getCallee());
        VPUX_THROW_WHEN(funcOp == nullptr, "Failed to lookup function {0}", newCallOp.getCallee());

        // If funcOp has been modified (fewer args than callOp operands), align
        // operands to function signature instead of taking a raw prefix.
        if (funcOp.getNumArguments() < newCallOp.getNumOperands()) {
            auto alignedOperands = alignCallOperandsToFuncSignature(newCallOp, funcOp);
            if (mlir::failed(alignedOperands)) {
                return std::make_pair(mlir::func::CallOp(), mlir::func::FuncOp());
            }

            auto updatedCallOp = builder.create<mlir::func::CallOp>(
                    newCallOp.getLoc(), funcOp.getName(), newCallOp.getResultTypes(), alignedOperands.value());
            newCallOp.replaceAllUsesWith(updatedCallOp.getResults());
            newCallOp.erase();
            newCallOp = updatedCallOp;
        }
        propagateTypeInCallOp(builder, _moduleOp, newCallOp);
        return {newCallOp, funcOp};
    };

    auto dimToEncodingIndex = buildDimToEncodingIndexMap(forOpToDimInfo);
    llvm::DenseSet<llvm::StringRef> processedFunctions;

    auto switchOp = builder.create<mlir::scf::IndexSwitchOp>(callOp.getLoc(), callOp.getResultTypes(), encodedIdx,
                                                             validCaseValues, validCaseValues.size());
    for (auto caseIndex : switchOp.getCases()) {
        auto it = llvm::find(validCaseValues, caseIndex);
        auto index = std::distance(validCaseValues.begin(), it);
        auto& region = switchOp.getCaseRegions()[index];
        auto& block = region.empty() ? region.emplaceBlock() : region.front();
        mlir::OpBuilder caseBuilder = mlir::OpBuilder::atBlockBegin(&block);
        auto caseValues = decodePointPosition(caseIndex, dynDims);
        auto [newCallOp, funcOp] = constructBlock(caseBuilder, block, caseIndex, dynDims, caseValues, callOp);
        if (!newCallOp || !funcOp) {
            return mlir::failure();
        }
        if (!conversionInfo.shouldConvert() || !funcOp || processedFunctions.contains(funcOp.getName())) {
            continue;
        }
        processedFunctions.insert(funcOp.getName());
        auto result = processIndexTypeArgsInCallOp(caseBuilder, funcOp, newCallOp, conversionInfo, forOpToDimInfo,
                                                   dimToEncodingIndex, caseValues);
        if (mlir::failed(result)) {
            return mlir::failure();
        }
    }

    auto& defRegion = switchOp.getDefaultRegion();
    auto& defBlock = defRegion.empty() ? defRegion.emplaceBlock() : defRegion.front();
    mlir::OpBuilder defBuilder = mlir::OpBuilder::atBlockBegin(&defBlock);
    auto defCaseValues = decodePointPosition(validCaseValues.front(), dynDims);
    auto [defaultCallOp, defaultFuncOp] =
            constructBlock(defBuilder, defBlock, validCaseValues.front(), dynDims, defCaseValues, callOp, true);
    if (!defaultCallOp || !defaultFuncOp) {
        return mlir::failure();
    }

    auto& defaultRegion = switchOp.getDefaultRegion();
    auto defaultYieldOp = mlir::cast<mlir::scf::YieldOp>(defaultRegion.front().back());
    auto referenceTypes = defaultYieldOp.getOperandTypes();

    for (auto& caseRegion : switchOp.getCaseRegions()) {
        auto caseYieldOp = mlir::cast<mlir::scf::YieldOp>(caseRegion.front().back());
        VPUX_THROW_UNLESS(llvm::equal(caseYieldOp.getOperandTypes(), referenceTypes),
                          "generateBlockAwareFuncOps: scf.index_switch cases yield different types – "
                          "all cases must produce the same output tile type");
    }

    for (auto [result, yieldOperand] : llvm::zip(switchOp.getResults(), defaultYieldOp.getOperands())) {
        result.setType(yieldOperand.getType());
    }

    callOp.replaceAllUsesWith(switchOp.getResults());
    cleanUnusedFuncAndCallOps(callOp, _moduleOp);

    return mlir::success();
}

bool checkForSliceOpWithDynamicArgOffset(mlir::Value operand) {
    return llvm::all_of(operand.getUses(), [&](const auto& use) {
        auto* owner = use.getOwner();
        auto extractSliceOp = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(owner);
        if (extractSliceOp == nullptr) {
            return false;
        }

        // Check if operand appears in offsets
        bool inOffsets = llvm::any_of(extractSliceOp.getMixedOffsets(), [&](const auto& offset) {
            if (auto val = mlir::dyn_cast_if_present<mlir::Value>(offset)) {
                return val == operand;
            }
            return false;
        });

        // Check if operand appears in sizes
        bool inSizes = llvm::any_of(extractSliceOp.getMixedSizes(), [&](const auto& size) {
            if (auto val = mlir::dyn_cast_if_present<mlir::Value>(size)) {
                return val == operand;
            }
            return false;
        });

        return inOffsets || inSizes;
    });
}

// Helper to analyze if index-type operands should be converted to static shapes
mlir::LogicalResult analyzeIndexOperandConversion(mlir::func::CallOp callOp, mlir::func::FuncOp calledFunc,
                                                  OpChainAnalysis& opChainAnalysis,
                                                  IndexOperandConversionInfo& idxOperandInfo, Logger& log) {
    for (auto [operandIdx, operand] : llvm::enumerate(callOp->getOperands())) {
        if (mlir::isa<mlir::RankedTensorType>(operand.getType())) {
            continue;
        }

        if (calledFunc.getNumArguments() <= operandIdx) {
            log.error("Operand index {0} out of bounds for function {1} with {2} arguments", operandIdx,
                      calledFunc.getName(), calledFunc.getNumArguments());
            return mlir::failure();
        }

        auto funcArg = calledFunc.getArgument(operandIdx);
        if (funcArg.use_empty()) {
            continue;
        }

        auto needStaticAnalysis = checkForSliceOpWithDynamicArgOffset(funcArg);
        if (!needStaticAnalysis) {
            return mlir::failure();
        }

        ValueRangeMap emptyValueMap;
        auto staticValuesOpt =
                opChainAnalysis.getOpFoldResultValue(operand, emptyValueMap, OpChainAnalysis::MODE::ALL_VALUES);
        if (!staticValuesOpt.has_value() || staticValuesOpt->empty()) {
            return mlir::failure();
        }

        auto staticValues = staticValuesOpt.value();
        auto uniqueStaticValues =
                llvm::SmallSetVector<int64_t, DEFAULT_ARG_SET_SIZE>(staticValues.begin(), staticValues.end());
        if (uniqueStaticValues.size() > MAX_TILES_PER_DYNAMIC_DIMENSION) {
            return mlir::failure();
        }

        idxOperandInfo.operandStaticValues[checked_cast<int64_t>(operandIdx)] = std::move(staticValues);
    }

    return mlir::success();
}

mlir::LogicalResult AdjustBlockSizeForScfTilingPass::processIndexTypeArgsInCallOp(
        mlir::func::CallOp callOp, const IndexOperandConversionInfo& conversionInfo) {
    llvm::DenseMap<size_t, int64_t> operandToStaticValue;
    auto funcOp = _moduleOp.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
    if (!funcOp) {
        return errorAt(callOp, "Failed to find function '{0}' for call operation", callOp.getCallee());
    }

    for (auto operandIdx : llvm::make_first_range(conversionInfo.operandStaticValues)) {
        if (!isValidOperandIndex(operandIdx, callOp, funcOp)) {
            return mlir::failure();
        }

        auto staticValues = conversionInfo.operandStaticValues.at(operandIdx);
        if (staticValues.empty()) {
            return mlir::failure();
        }

        bool allSame = std::all_of(staticValues.begin(), staticValues.end(), [first = staticValues[0]](int64_t val) {
            return val == first;
        });
        if (!allSame) {
            return errorAt(callOp,
                           "Static values for operand index {0} are not uniform, cannot convert to static shape",
                           operandIdx);
        }
        operandToStaticValue[operandIdx] = staticValues[0];
    }

    auto builder = mlir::OpBuilder(funcOp);
    return updateFunctionAndCallOp(builder, funcOp, callOp, operandToStaticValue);
}

void removeRedundantExtractSliceOp(mlir::func::FuncOp funcOp) {
    SmallVector<mlir::Operation*> opsToErase;

    funcOp.walk([&](mlir::scf::ForOp forOp) {
        for (auto extractSliceOp : forOp.getOps<mlir::tensor::ExtractSliceOp>()) {
            auto allUsersAreDeadCasts = llvm::all_of(extractSliceOp->getUsers(), [](mlir::Operation* user) {
                return mlir::isa<mlir::tensor::CastOp>(user) && user->use_empty();
            });
            if (!allUsersAreDeadCasts) {
                continue;
            }
            for (auto* user : extractSliceOp->getUsers()) {
                opsToErase.push_back(user);
            }
            opsToErase.push_back(extractSliceOp);
        }
    });

    for (auto* op : opsToErase) {
        op->erase();
    }
}

void AdjustBlockSizeForScfTilingPass::safeRunOnModule() {
    _moduleOp = getOperation();
    _mainFuncOp = net::getMainFunc(_moduleOp);
    mlir::OpBuilder builder(_moduleOp.getContext());

    auto checkForPaddedOps = [&](mlir::func::FuncOp funcOp) {
        for (auto& body : funcOp.getBody()) {
            for (auto& op : body.getOperations()) {
                if (VPU::isNceOpWithPadAttr(&op)) {
                    auto padAttr = op.getAttrOfType<VPU::PaddingAttr>("pad");
                    if (padAttr.getTop().getInt() == 0 && padAttr.getBottom().getInt() == 0 &&
                        padAttr.getLeft().getInt() == 0 && padAttr.getRight().getInt() == 0) {
                        continue;
                    }
                    return true;
                }
            }
        }
        return false;
    };

    auto requireBlockAwareFuncOps = [](const llvm::DenseMap<mlir::Operation*, SliceOpData>& sliceOpDataMap) {
        for (auto& [op, data] : sliceOpDataMap) {
            if (data.hasAdjustedDynDims()) {
                return true;
            }
        }
        return false;
    };

    auto forOps = _mainFuncOp.getOps<mlir::scf::ForOp>();
    if (forOps.empty()) {
        _log.trace("No scf.for operations found in the main function. Skipping the pass.");
        return;
    }
    _firstForOp = *forOps.begin();

    _mainFuncOp.walk([&](mlir::scf::ForOp forOp) {
        llvm::DenseMap<mlir::scf::ForOp, int64_t> forOpToDim;
        if (mlir::failed(adjustOutputBlockIdxAndSize(forOp, forOpToDim))) {
            signalPassFailure();
            return;
        }

        llvm::DenseMap<mlir::Operation*, SliceOpData> sliceOpDataMap;
        if (mlir::failed(calculateInputBlockPosAndShapeInformation(forOp, sliceOpDataMap, forOpToDim))) {
            signalPassFailure();
            return;
        }

        if (sliceOpDataMap.empty() || !requireBlockAwareFuncOps(sliceOpDataMap)) {
            _log.trace("No extract slice operations with dynamic shapes found in scf.for loop {0}", forOp);
            return;
        }

        for (auto callOp : make_early_inc_range(forOp.getOps<mlir::func::CallOp>())) {
            _log.trace("Processing call operation {0}", callOp);

            // check if operands of callOp are dynamic tensors
            if (!IE::hasDynamicTensors(callOp)) {
                _log.trace("Call operation {0} do not have dynamic tensors. {0}", callOp);
                return;
            }

            // get the called function
            auto calledFunc = _moduleOp.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
            if (calledFunc == nullptr) {
                _log.trace("Called function {0} not found in module. Pass failed to adjust block sizes.",
                           callOp.getCallee());
                signalPassFailure();
                return;
            }

            IndexOperandConversionInfo conversionInfo;
            auto result = analyzeIndexOperandConversion(callOp, calledFunc, _opChainAnalysis, conversionInfo, _log);
            if (mlir::failed(result)) {
                _log.trace("Failed to analyze index operand conversion for call operation {0}", callOp);
                signalPassFailure();
                return;
            }

            if (checkForPaddedOps(calledFunc)) {
                auto result = generateBlockAwareFuncOps(callOp, sliceOpDataMap, forOpToDim, conversionInfo);
                if (mlir::failed(result)) {
                    _log.trace("Failed to generate block-aware function operations for call operation {0}", callOp);
                    signalPassFailure();
                }
                return;
            }

            if (!conversionInfo.shouldConvert()) {
                return;
            }

            if (mlir::failed(processIndexTypeArgsInCallOp(callOp, conversionInfo))) {
                signalPassFailure();
                return;
            }
        }
        return;
    });

    removeRedundantExtractSliceOp(_mainFuncOp);
}
}  // namespace

//
// createAdjustBlockSizeForScfTilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createAdjustBlockSizeForScfTilingPass(Logger log) {
    return std::make_unique<AdjustBlockSizeForScfTilingPass>(log);
}
