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

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
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

//
// Low resolution range of a dynamic dim, with the notation used in this file:
//   S         - loop step (size of one output tile)
//   D         - runtime output dim; the low resolution range is S < D <= S + halo
//   bt        - offset of the output tile (after backtracking)
//   R         - offset of tile 1, R = D - S; R_max is its largest value over the range
//   O_l, O_r  - number of invalid outputs at the start / end of a tile computed with full left / right pad
//   P         - phase period of the op chain along the dim (LowResInfo::phasePeriod)
// D = S is computed by one FULLBLK tile. The 2-tile part of the range is computed by two FULLBLK tiles with full pads,
// tile 0 at bt = 0 and tile 1 at bt = R; the first O_l outputs of tile 1 are taken from tile 0
//

// Low resolution (2-tile) data of a dim, measured on interior tiles at the bound. Values are in output units; the
// input and output of the op chain have the same scale along the dim
struct LowResInfo {
    // Phase period of the op chain along the dim. Strides and resampling make the chain shift-equivariant only for
    // shifts by a multiple of it: moving the output tile by phasePeriod moves its input slice by phasePeriod, and the
    // halos repeat. The phase of an output offset `o` is o % phasePeriod.
    // Example: a 3x3 stride-2 conv (pad 1) followed by a x2 upsample (DepthToSpace) has phasePeriod 2. Output pixel `o`
    // comes from the half resolution pixel o / 2, which reads the input pixels 2 * (o / 2) - 1 ... 2 * (o / 2) + 1:
    // - even o = 4: half resolution pixel 2, input pixels 3..5, the window starts 1 pixel before o
    // - odd o = 5: half resolution pixel 2, input pixels 3..5, the window starts 2 pixels before o
    // A shift by 1 changes the window, a shift by 2 does not. A tile that starts at an odd offset also samples the
    // stride-2 conv at odd input pixels, while the full image samples it at even ones, so every output of the tile
    // differs. The low resolution tiles must therefore start at a multiple of phasePeriod
    int64_t phasePeriod = 0;
    // Number of invalid outputs at the start / end of a tile computed with full left / right pad
    int64_t invalidOutputLeft = 0;
    int64_t invalidOutputRight = 0;
    // Largest offset R = D - S of tile 1 over the output dims D of the low resolution range
    int64_t maxTileOneOffset = 0;
};

// Input slice [offset, offset + size) of an output tile along a dim
struct SliceRange {
    int64_t offset = 0;
    int64_t size = 0;

    int64_t end() const {
        return offset + size;
    }
};

// Input halos of the output tile at `o` along a dim: its input slice is [o - left, o + S + right)
struct TileHalo {
    int64_t left = 0;
    int64_t right = 0;

    bool operator==(const TileHalo& other) const {
        return left == other.left && right == other.right;
    }
};

// FULLBLK case that the bound alone does not produce (dim with 2 or more tiles by bound). It is used when the whole
// runtime output dim fits in one step-sized tile. Sizes are in units of the slice source (input) dim
struct FullBlkInfo {
    mlir::scf::ForOp forOp;
    mlir::scf::ForOp outermostForOp;
    mlir::Value source;
    int64_t dimIdx = 0;
    // Input slice size of the single tile: how far the input slice moves when the output tile moves by one step
    int64_t inputStep = 0;
    // Largest input dim for which an edge tile is clipped by the tensor edge (low resolution range)
    int64_t lowResLimit = 0;
    // Loop step S, loop upper bound at the bound and bound of the slice source dim
    int64_t step = 0;
    int64_t upperBound = 0;
    int64_t sourceBound = 0;
    // Output dim (loop upper bound) for the input dims inputStep, inputStep + 1, ..., lowResLimit, or std::nullopt if
    // it cannot be evaluated. Output dims equal to the step are the single tile part of the range, the others the
    // 2-tile part
    SmallVector<std::optional<int64_t>> lowResOutputDims;
    // Set if the 2-tile low resolution range of the dim can be computed with the FULLBLK case
    std::optional<LowResInfo> lowRes;
};

struct SliceOpData {
    SmallVector<size_t> dynamicDims;
    std::array<mlir::Value, MAX_NUM_DIMS> blockPositionValues;
    llvm::DenseMap<size_t, vpux::Shape> caseToShapeMap;
    // Dims for which a FULLBLK case was added on top of the positions given by the bound
    llvm::DenseMap<size_t, FullBlkInfo> addedFullBlkDims;

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
    // Values at an added FULLBLK position (single tile, D = step), per operand index
    llvm::DenseMap<int64_t, int64_t> operandFullBlkValues;

    bool shouldConvert() const {
        return !operandStaticValues.empty();
    }
};

// Case values of the scf.index_switch built for a call
struct SwitchCaseValues {
    // Every case of the switch, sorted
    SmallVector<int64_t> caseValues;
    // A case given by the bound: the reference result type of the cases and the body of the default case
    int64_t referenceCaseValue = 0;
    // Set if the added FULLBLK cases are kept
    bool hasAddedFullBlkCases = false;
};

// Builds the body of the switch case `caseVal` into `block`, returns the new call and its callee
using CaseBlockBuilder = llvm::function_ref<std::pair<mlir::func::CallOp, mlir::func::FuncOp>(
        mlir::OpBuilder& builder, mlir::Block& block, int64_t caseVal)>;

//
// AdjustBlockSizeForScfTilingPass
//
class AdjustBlockSizeForScfTilingPass final :
        public VPU::impl::AdjustBlockSizeForScfTilingBase<AdjustBlockSizeForScfTilingPass> {
public:
    AdjustBlockSizeForScfTilingPass(bool enableSubTileFallback, Logger log)
            : _enableSubTileFallback(enableSubTileFallback) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;

    struct AdjustedIndexInfo {
        mlir::OpFoldResult index;
        mlir::Operation* indexDefiningOp = nullptr;
    };

    // Runtime values needed to decide whether a dynamic output dimension holds fewer elements than one tile.
    struct SubTileCheck {
        mlir::tensor::DimOp dimOp;
        mlir::Value blockSize;
    };

    // Untouched copy of a top-level loop nest, taken before block sizes are adjusted. Serves as the sub-tile
    // fallback path: its func.call ops target dedicated dynamic clones of the original kernels.
    struct SubTileSnapshot {
        mlir::scf::ForOp originalForOp;
        mlir::OwningOpRef<mlir::scf::ForOp> fallbackForOp;
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
    SwitchCaseValues filterAddedFullBlkCases(mlir::func::CallOp callOp, const SliceOpData& dispatchSliceOpData,
                                             const llvm::DenseMap<mlir::Operation*, SliceOpData>& mapSliceOpToData,
                                             const llvm::DenseMap<mlir::scf::ForOp, int64_t>& forOpToDimInfo,
                                             ArrayRef<int64_t> validCaseValues,
                                             IndexOperandConversionInfo& indexOperandInfo,
                                             CaseBlockBuilder buildCaseBlock);
    mlir::scf::IfOp getTensorBlockPosition(mlir::scf::ForOp forOp, mlir::OpBuilder builder, mlir::Value offsetVal,
                                           mlir::Operation* insertionPoint, mlir::Value currentBlkSize,
                                           mlir::Value upperBound);
    std::optional<int64_t> evaluateAtSingleTile(mlir::Value val, mlir::scf::ForOp forOp, int64_t inputDim);
    std::optional<FullBlkInfo> getFullBlkInfo(mlir::scf::ForOp forOp, mlir::scf::ForOp outermostForOp,
                                              mlir::Value offset, mlir::Value size, mlir::Value source, size_t dimIdx,
                                              const llvm::DenseMap<int64_t, int64_t>& mapPosToBlockSizes);
    std::optional<LowResInfo> analyzeLowRes(const FullBlkInfo& fullBlkInfo, mlir::Value offset, mlir::Value size);
    std::optional<LowResInfo> checkIfTileRegionsAreValid(int64_t dimIdx, int64_t step,
                                                         ArrayRef<std::optional<LowResInfo>> inputs);
    SmallVector<mlir::Value> emitRuntimeGuardsForLowRes(
            mlir::func::CallOp callOp, const llvm::DenseMap<mlir::Operation*, SliceOpData>& mapSliceOpToData,
            ArrayRef<vpux::Dim> dynDims, ArrayRef<mlir::Value> tileIds);
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
        auto constOp = mlir::arith::ConstantOp::create(builder, takeOpLoc(_firstForOp, "const_val"), constAttr);
        _constOpCache[val] = {constAttr, constOp};
        return _constOpCache[val];
    }

    mlir::tensor::DimOp createDimOp(mlir::OpBuilder builder, mlir::scf::ForOp forOp, mlir::Value tensor, size_t idx) {
        mlir::OpBuilder localBuilder(forOp);
        auto dimOp = forOp.isDefinedOutsideOfLoop(tensor)
                             ? mlir::tensor::DimOp::create(localBuilder, takeOpLoc(forOp, "dim"), tensor, idx)
                             : mlir::tensor::DimOp::create(builder, takeOpLoc(forOp, "dim"), tensor, idx);
        return dimOp;
    }

    mlir::arith::ConstantIndexOp getCstIndexOp(int64_t val) {
        if (_constIndexOpCache.contains(val)) {
            return _constIndexOpCache[val];
        }

        mlir::OpBuilder builder(_firstForOp);
        auto cstOp = mlir::arith::ConstantIndexOp::create(builder,
                                                          takeOpLoc(_firstForOp, "const_" + std::to_string(val)), val);
        _constIndexOpCache[val] = cstOp;
        return cstOp;
    }

    void buildMapOfBlockSizesForSliceOps(
            mlir::scf::ForOp forOp,
            llvm::DenseMap<mlir::Operation*, DimToBlockSizesMap>& mapSliceOpToPerDimensionBlockSizes);

    SubTileSnapshot createSubTileSnapshot(mlir::scf::ForOp forOp, mlir::SymbolTable& symbolTable);
    mlir::func::FuncOp getOrCreateSubTileFunc(mlir::StringAttr calleeAttr, mlir::SymbolTable& symbolTable);
    void emitSubTileChecksAsAsserts(ArrayRef<SubTileCheck> checks);
    mlir::LogicalResult buildSubTileFallback(SubTileSnapshot& snapshot);

    OpChainAnalysis _opChainAnalysis;
    mlir::ModuleOp _moduleOp;
    mlir::scf::ForOp _firstForOp;
    mlir::func::FuncOp _mainFuncOp;
    llvm::DenseMap<int64_t, mlir::arith::ConstantIndexOp> _constIndexOpCache;
    llvm::DenseMap<int64_t, std::pair<mlir::IntegerAttr, mlir::Value>> _constOpCache;
    bool _enableSubTileFallback = false;
    llvm::DenseMap<mlir::scf::ForOp, SmallVector<SubTileCheck>> _forOpToSubTileChecks;
    // Kernel clones created by this pass run, keyed by the original callee. Clones are never resolved by name:
    // a pre-existing symbol with the generated name is unrelated and must not be reused.
    llvm::DenseMap<mlir::StringAttr, mlir::func::FuncOp> _calleeToSubTileFunc;
    // Id of the low resolution data of each scf.for (LOW_RES_ATTR_NAME), shared with its predicate
    llvm::DenseMap<mlir::Operation*, int64_t> _lowResIds;
};

mlir::LogicalResult AdjustBlockSizeForScfTilingPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableSubTileFallback.hasValue()) {
        _enableSubTileFallback = enableSubTileFallback.getValue();
    }
    return mlir::success();
}

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
    auto indexIsZero = mlir::arith::CmpIOp::create(builder, takeOpLoc(insertionPoint, "cmp_index_zero"),
                                                   mlir::arith::CmpIPredicate::eq, offset, zeroConst);

    auto ifIndexZero = mlir::scf::IfOp::create(builder, takeOpLoc(insertionPoint, "if_offset_zero"),
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
        auto checkForFullTile = mlir::arith::CmpIOp::create(thenBuilder, takeOpLoc(ifIndexZero, "full_tile_check"),
                                                            mlir::arith::CmpIPredicate::eq, currentBlkSize, upperBound);
        auto selectIndex = mlir::arith::SelectOp::create(thenBuilder, takeOpLoc(ifIndexZero, "select_full_tile"),
                                                         checkForFullTile, threeConst, twoConst);
        mlir::scf::YieldOp::create(thenBuilder, takeOpLoc(selectIndex, "yield"),
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
        auto sum = mlir::arith::AddIOp::create(elseBuilder, takeOpLoc(ifIndexZero, "else_add"), offset, currentBlkSize);
        auto sumLessThanBound = mlir::arith::CmpIOp::create(elseBuilder, takeOpLoc(ifIndexZero, "pos_check"),
                                                            mlir::arith::CmpIPredicate::slt, sum, upperBound);
        auto selectIndex = mlir::arith::SelectOp::create(elseBuilder, takeOpLoc(ifIndexZero, "select_tile_pos"),
                                                         sumLessThanBound, zeroConst, oneConst);
        mlir::scf::YieldOp::create(elseBuilder, takeOpLoc(ifIndexZero, "yield"), selectIndex->getResult(0));
    }
    return ifIndexZero;
}

// Collects the tensor.dim results that `val` is computed from
llvm::SmallSetVector<mlir::Value, 4> getTensorDimOpsFromOpChain(mlir::Value val, OpChainAnalysis& opChainAnalysis) {
    llvm::SmallSetVector<mlir::Value, 4> leaves;
    if (mlir::isa_and_nonnull<mlir::tensor::DimOp>(val.getDefiningOp())) {
        leaves.insert(val);
        return leaves;
    }
    for (auto* op : opChainAnalysis.collectParentOpsChain(val)) {
        for (auto operand : op->getOperands()) {
            if (mlir::isa_and_nonnull<mlir::tensor::DimOp>(operand.getDefiningOp())) {
                leaves.insert(operand);
            }
        }
    }
    return leaves;
}

// Returns the tensor.dim the loop upper bound is computed from, or nullptr if it depends on zero or several of them
mlir::Value getUpperBoundRootOp(mlir::scf::ForOp forOp, OpChainAnalysis& opChainAnalysis) {
    auto tensorDimOps = getTensorDimOpsFromOpChain(forOp.getUpperBound(), opChainAnalysis);
    return tensorDimOps.size() == 1 ? tensorDimOps.front() : nullptr;
}

// Evaluates `val` with the given seeds; returns std::nullopt unless it evaluates to exactly one value
std::optional<int64_t> evaluateUniqueValue(mlir::Value val, ValueRangeMap seeds, OpChainAnalysis& opChainAnalysis) {
    if (seeds.contains(val)) {
        return seeds[val].size() == 1 ? std::optional<int64_t>(seeds[val].front()) : std::nullopt;
    }
    auto values = opChainAnalysis.getOpFoldResultValue(val, seeds, OpChainAnalysis::MODE::ALL_VALUES);
    if (!values.has_value() || values->empty() || llvm::any_of(values.value(), [&](int64_t v) {
            return v != values->front();
        })) {
        return std::nullopt;
    }
    return values->front();
}

// Returns the bound of `source` along `dimIdx`, or std::nullopt if it is not known
std::optional<int64_t> getDimBound(mlir::Value source, size_t dimIdx) {
    auto sourceType = mlir::cast<NDTypeInterface>(source.getType());
    auto boundedType = mlir::dyn_cast<vpux::Core::BoundedTensorType>(sourceType);
    auto shape = boundedType != nullptr ? to_small_vector(boundedType.getBounds().raw())
                                        : to_small_vector(sourceType.getShape().raw());
    if (dimIdx >= shape.size() || shape[dimIdx] == mlir::ShapedType::kDynamic) {
        return std::nullopt;
    }
    return shape[dimIdx];
}

// Evaluates the input slice of the tile of `forOp` at the output offset `outOffset`
std::optional<SliceRange> evaluateInputSliceAt(mlir::Value offset, mlir::Value size, mlir::scf::ForOp forOp,
                                               int64_t outOffset, OpChainAnalysis& opChainAnalysis) {
    ValueRangeMap seeds;
    seeds[forOp.getInductionVar()] = {outOffset};
    auto sliceOffset = evaluateUniqueValue(offset, seeds, opChainAnalysis);
    auto sliceSize = evaluateUniqueValue(size, seeds, opChainAnalysis);
    if (!sliceOffset.has_value() || !sliceSize.has_value()) {
        return std::nullopt;
    }
    return SliceRange{sliceOffset.value(), sliceSize.value()};
}

// Seeds the root tensor.dim `root` of a loop upper bound with the input dim `inputDim`, for the evaluation of `vals`.
// TODO: temporary workaround. The upper bound may be computed from a tensor.dim other than the slice source dim
// (Model 3: the resize input dim, with the same formula as the resize output dim that is the slice source). Every float
// tensor.extract in the chains of `vals` (interpolate scales) is seeded with 1.0, so that the root equals the input
// dim. This is valid only if the upper bound chain repeats the slice source dim formula, which should be checked
// structurally instead
ValueRangeMap getInputDimSeeds(mlir::Value root, int64_t inputDim, ArrayRef<mlir::Value> vals,
                               OpChainAnalysis& opChainAnalysis) {
    ValueRangeMap seeds;
    seeds[root] = {inputDim};
    for (auto val : vals) {
        for (auto* op : opChainAnalysis.collectParentOpsChain(val)) {
            auto extractOp = mlir::dyn_cast<mlir::tensor::ExtractOp>(op);
            if (extractOp != nullptr && mlir::isa<mlir::FloatType>(extractOp.getType())) {
                seeds[extractOp.getResult()] = {llvm::bit_cast<int64_t>(1.0)};
            }
        }
    }
    return seeds;
}

// Evaluates the loop upper bound of `forOp` (the output dim) when its root tensor.dim `root` is `inputDim`
std::optional<int64_t> evaluateUpperBoundForFullblk(mlir::scf::ForOp forOp, mlir::Value root, int64_t inputDim,
                                                    OpChainAnalysis& opChainAnalysis) {
    auto seeds = getInputDimSeeds(root, inputDim, {forOp.getUpperBound()}, opChainAnalysis);
    return evaluateUniqueValue(forOp.getUpperBound(), seeds, opChainAnalysis);
}

// Returns true if an extract_slice in `funcOp` reads past the bounds of its source. Offsets and sizes are known if they
// are constants or function arguments listed in `argValues`; slices with unknown values are not checked
bool hasOutOfBoundsSlice(mlir::func::FuncOp funcOp, const llvm::DenseMap<int64_t, int64_t>& argValues) {
    auto getKnownValue = [&](mlir::OpFoldResult ofr) -> std::optional<int64_t> {
        if (auto constValue = mlir::getConstantIntValue(ofr)) {
            return constValue;
        }
        auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(mlir::dyn_cast<mlir::Value>(ofr));
        if (blockArg == nullptr || blockArg.getOwner()->getParentOp() != funcOp.getOperation()) {
            return std::nullopt;
        }
        auto it = argValues.find(blockArg.getArgNumber());
        return it != argValues.end() ? std::optional<int64_t>(it->second) : std::nullopt;
    };

    auto result = funcOp.walk([&](mlir::tensor::ExtractSliceOp sliceOp) {
        auto sourceType = mlir::cast<NDTypeInterface>(sliceOp.getSource().getType());
        auto boundedType = mlir::dyn_cast<vpux::Core::BoundedTensorType>(sourceType);
        if (sourceType.getShape().isDynamic() && boundedType == nullptr) {
            return mlir::WalkResult::advance();
        }
        auto sourceShape = boundedType != nullptr ? to_small_vector(boundedType.getBounds().raw())
                                                  : to_small_vector(sourceType.getShape().raw());
        for (auto [offset, size, bound] : llvm::zip(sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(), sourceShape)) {
            auto offsetValue = getKnownValue(offset);
            auto sizeValue = getKnownValue(size);
            if (offsetValue.has_value() && sizeValue.has_value() && offsetValue.value() + sizeValue.value() > bound) {
                return mlir::WalkResult::interrupt();
            }
        }
        return mlir::WalkResult::advance();
    });
    return result.wasInterrupted();
}

/**
 * @brief Evaluates `val` for the single tile of `forOp` (iv = 0) when the slice source dim is `inputDim`.
 *
 * The loop upper bound must be computed from exactly one tensor.dim (the root). The root is seeded with `inputDim` and
 * the upper bound is checked to evaluate to the step. `val` must not depend on any other tensor.dim, because its
 * runtime value for a single tile is not known. Returns std::nullopt if the value cannot be evaluated exactly.
 */
std::optional<int64_t> AdjustBlockSizeForScfTilingPass::evaluateAtSingleTile(mlir::Value val, mlir::scf::ForOp forOp,
                                                                             int64_t inputDim) {
    auto step = _opChainAnalysis.getIntegerFromValue(forOp.getStep());
    auto root = getUpperBoundRootOp(forOp, _opChainAnalysis);
    if (!step.has_value() || root == nullptr) {
        _log.trace("Single tile evaluation: unknown step or upper bound not computed from one tensor.dim");
        return std::nullopt;
    }

    auto seeds = getInputDimSeeds(root, inputDim, {forOp.getUpperBound(), val}, _opChainAnalysis);
    auto upperBound = evaluateUniqueValue(forOp.getUpperBound(), seeds, _opChainAnalysis);
    if (!upperBound.has_value() || upperBound.value() != step.value()) {
        _log.trace("Single tile evaluation: upper bound does not evaluate to the step {0} for input dim {1}",
                   step.value(), inputDim);
        return std::nullopt;
    }

    for (auto leaf : getTensorDimOpsFromOpChain(val, _opChainAnalysis)) {
        if (leaf != root) {
            _log.trace("Single tile evaluation: value depends on a tensor.dim other than the upper bound root");
            return std::nullopt;
        }
    }

    seeds[forOp.getInductionVar()] = {0};
    auto value = evaluateUniqueValue(val, seeds, _opChainAnalysis);
    if (!value.has_value()) {
        _log.trace("Single tile evaluation: value is not unique");
    }
    return value;
}

/**
 * @brief Computes the FULLBLK case of a dim that has 2 or more tiles by bound. Sizes are in input (slice source) units.
 *
 * - Input step: the input slice offset difference between the middle pair of tiles at the bound. These tiles are the
 *   farthest from both edges, so if they are clipped by an edge no tile is interior and FULLBLK is not added. It is the
 *   input slice size of the single tile.
 * - Low resolution limit: the larger of the START and END input sizes. Up to this input dim an edge tile is clipped by
 *   the tensor edge and the START/MIDDLE/END cases do not apply.
 * - Every input dim that gives a single output tile must be served by the static [0, input step) slice: no
 *   smaller input dim gives a single tile, and no larger one needs a different slice. The input dim is linked to
 *   the output dim through the loop upper bound, which must be computed from one tensor.dim (the root). The root is
 *   assumed to be the slice source dim (temporary workaround, see getInputDimSeeds).
 */
std::optional<FullBlkInfo> AdjustBlockSizeForScfTilingPass::getFullBlkInfo(
        mlir::scf::ForOp forOp, mlir::scf::ForOp outermostForOp, mlir::Value offset, mlir::Value size,
        mlir::Value source, size_t dimIdx, const llvm::DenseMap<int64_t, int64_t>& mapPosToBlockSizes) {
    auto notAdded = [&](const char* reason, const auto&... args) -> std::optional<FullBlkInfo> {
        _log.trace("FULLBLK not added for dim {0}: {1}", dimIdx, llvm::formatv(reason, args...).str());
        return std::nullopt;
    };

    if (offset == nullptr) {
        return notAdded("static slice offset");
    }

    auto step = _opChainAnalysis.getIntegerFromValue(forOp.getStep());
    auto upperBound = _opChainAnalysis.getIntegerFromValue(forOp.getUpperBound(), true);
    auto sourceBound = getDimBound(source, dimIdx);
    if (!step.has_value() || !upperBound.has_value() || !sourceBound.has_value()) {
        return notAdded("unknown step, upper bound or source bound");
    }

    // The input step is the offset distance of two adjacent middle tiles. The input offset of a tile depends only on
    // its left side: tile 0 is clipped to offset 0, but the right clip of a slice changes only its size. So the two
    // tiles only need an input offset > 0 and a full output tile (a partial last tile can be backtracked, which moves
    // its offset), and 3 tiles at the bound are enough: tiles 1 and 2.
    // TODO: the input step can be found with fewer tiles by inverting the loop upper bound: inputStep is the smallest
    // input dim d with UB(d) == step, i.e. the unpadded input of a single tile (llvm::partition_point over
    // [1, sourceBound], UB is monotone in d). Seeding the input dim with the step itself is not generic: it is right
    // only if the input and output dims are equal, e.g. UB = (d / 2) * 2. With a x2 upsample input (output step 180,
    // input slice 90) the single tile slice is clipped to 180, and with a stride-2 or valid conv UB(step) != step, so
    // the loop does not even have a single tile. The inversion requires the UB to be computed only from the root
    // tensor.dim and constants; a UB that also uses a runtime value (e.g. a tensor.extract of interpolate scales) has
    // no unique inverse and must be rejected. The UB checks below and the single tile slice check stay as they are
    auto numTiles = (upperBound.value() + step.value() - 1) / step.value();
    if (numTiles < 3) {
        return notAdded("{0} tiles at the bound, at least 3 are needed", numTiles);
    }
    auto middleIdx = (numTiles - 1) / 2;
    SmallVector<int64_t, 2> middleOffsets;
    for (auto tileIdx : {middleIdx, middleIdx + 1}) {
        auto inputSlice = evaluateInputSliceAt(offset, size, forOp, tileIdx * step.value(), _opChainAnalysis);
        if (!inputSlice.has_value()) {
            return notAdded("slice of tile {0} cannot be evaluated", tileIdx);
        }
        _log.trace("Dim {0} tile {1}: input offset {2}, size {3}", dimIdx, tileIdx, inputSlice->offset,
                   inputSlice->size);
        if (inputSlice->offset <= 0 || (tileIdx + 1) * step.value() > upperBound.value()) {
            return notAdded("middle tile {0} is clipped on the left or is a partial tile", tileIdx);
        }
        middleOffsets.push_back(inputSlice->offset);
    }
    auto inputStep = middleOffsets[1] - middleOffsets[0];
    if (inputStep <= 0) {
        return notAdded("input step {0} is not positive", inputStep);
    }

    auto lowResLimit = std::max(mapPosToBlockSizes.lookup(static_cast<int64_t>(TilePosition::START)),
                                mapPosToBlockSizes.lookup(static_cast<int64_t>(TilePosition::END)));

    auto root = getUpperBoundRootOp(forOp, _opChainAnalysis);
    auto rootDimOp = root != nullptr ? root.getDefiningOp<mlir::tensor::DimOp>() : nullptr;
    if (rootDimOp == nullptr) {
        return notAdded("the loop upper bound is not computed from one tensor.dim");
    }

    // Defensive check to make sure that with seed value as inputStep, we still get loop upperBound which
    // is same as loop STEP SIZE
    auto outTensorUpperBound = evaluateUpperBoundForFullblk(forOp, root, inputStep, _opChainAnalysis);
    if (!outTensorUpperBound.has_value() || outTensorUpperBound.value() != step.value()) {
        return notAdded("input dim {0} does not give a single tile", inputStep);
    }

    // Output dim of every input dim in the low resolution range, recorded for analyzeLowRes. The single tile part
    // (output dim equal to the step) comes first and must be served by the static [0, input step) slice. The upper
    // bound is expected to be monotone in the input dim, so no input dim after the 2-tile part starts gives a single
    // tile again
    SmallVector<std::optional<int64_t>> lowResOutputDims{outTensorUpperBound};
    bool isSingleTile = true;
    for (auto inputDim = inputStep + 1; inputDim <= lowResLimit; ++inputDim) {
        auto outTensorUB = evaluateUpperBoundForFullblk(forOp, root, inputDim, _opChainAnalysis);
        lowResOutputDims.push_back(outTensorUB);
        if (isSingleTile && !outTensorUB.has_value()) {
            return notAdded("upper bound cannot be evaluated for input dim {0}", inputDim);
        }

        // Defensive check for monotone UB's
        const auto isDimSingleTile = outTensorUB.has_value() && outTensorUB.value() == step.value();
        if (isDimSingleTile && !isSingleTile) {
            return notAdded("input dim {0} gives a single tile after a multi-tile input dim", inputDim);
        }

        isSingleTile = isDimSingleTile;
        if (!isSingleTile) {
            continue;
        }

        // Defensive check that for the give inputDim and single tile, the upperbound is still step size
        auto singleTileOffset = evaluateAtSingleTile(offset, forOp, inputDim);
        auto singleTileSize = evaluateAtSingleTile(size, forOp, inputDim);
        if (singleTileOffset != std::optional<int64_t>(0) || singleTileSize != std::optional<int64_t>(inputStep)) {
            return notAdded("input dim {0} needs a slice other than [0, {1})", inputDim, inputStep);
        }
    }

    _log.trace("FULLBLK for dim {0}: input step {1}, low resolution limit {2}", dimIdx, inputStep, lowResLimit);
    FullBlkInfo info{forOp,
                     outermostForOp,
                     source,
                     checked_cast<int64_t>(dimIdx),
                     inputStep,
                     lowResLimit,
                     /*step=*/0,
                     /*upperBound=*/0,
                     /*sourceBound=*/0,
                     /*lowResOutputDims=*/{},
                     /*lowRes=*/std::nullopt};
    info.step = step.value();
    info.upperBound = upperBound.value();
    info.sourceBound = sourceBound.value();
    info.lowResOutputDims = std::move(lowResOutputDims);
    return info;
}

/**
 * @brief Measures the 2-tile low resolution data of a dim (S < D <= S + halo) on interior tiles at the bound.
 *
 * Both low resolution tiles run the FULLBLK kernel: input [bt, bt + S) with full pads. This needs the input and output
 * of the op chain to have the same scale along the dim: the input step equals the output step S, and the input offset
 * moves with the output offset. The input slice of the output tile at `o` is [off(o), off(o) + size(o)), with the
 * halos h_l(o) = o - off(o) and h_r(o) = off(o) + size(o) - (o + S).
 * - The halos are sampled on consecutive offsets from an aligned tile t; they repeat with the phase period P of the
 *   chain (LowResInfo::phasePeriod). A repeating h_l also means off(o + P) - off(o) = P, the same scale condition.
 * - The invalid output counts O_l and O_r of a tile at t computed with full pads follow from the halos of one period.
 * - Tile 1 starts at R = D - S, which must be a multiple of P for every runtime dim of the low resolution range.
 * The guard on O_l and O_r is checked in checkIfTileRegionsAreValid, on the data merged over all inputs of the call.
 */
std::optional<LowResInfo> AdjustBlockSizeForScfTilingPass::analyzeLowRes(const FullBlkInfo& fullBlkInfo,
                                                                         mlir::Value offset, mlir::Value size) {
    const auto dimIdx = fullBlkInfo.dimIdx;
    auto notSupported = [&](const char* reason, const auto&... args) -> std::optional<LowResInfo> {
        _log.trace("Low resolution not supported for dim {0}: {1}", dimIdx, llvm::formatv(reason, args...).str());
        return std::nullopt;
    };

    const auto step = fullBlkInfo.step;
    if (fullBlkInfo.inputStep != step) {
        return notSupported("input step {0} differs from the output step {1}", fullBlkInfo.inputStep, step);
    }

    // Halos of the output tiles at the consecutive offsets t, t + 1, ..., where t is the middle tile, the farthest from
    // both edges. Only interior tiles are measured: the backtracking min(o, bound - S) must not move the tile, and its
    // input slice must not touch the input edges
    constexpr int64_t SAMPLE_WINDOW = 16;
    const auto numTiles = (fullBlkInfo.upperBound + step - 1) / step;
    const auto alignedOffset = ((numTiles - 1) / 2) * step;
    SmallVector<TileHalo, SAMPLE_WINDOW> halos;
    for (auto outOffset : llvm::seq(alignedOffset, alignedOffset + SAMPLE_WINDOW)) {
        auto inputSlice = outOffset + step <= fullBlkInfo.upperBound
                                  ? evaluateInputSliceAt(offset, size, fullBlkInfo.forOp, outOffset, _opChainAnalysis)
                                  : std::nullopt;
        if (!inputSlice.has_value() || inputSlice->offset <= 0 || inputSlice->end() >= fullBlkInfo.sourceBound) {
            return notSupported("output offset {0} has no interior input slice", outOffset);
        }
        halos.push_back({outOffset - inputSlice->offset, inputSlice->end() - (outOffset + step)});
    }

    // Phase period P: the smallest shift p for which the halos repeat over the window, halos[k + p] == halos[k] for
    // every k. Compared as the window without its first p halos against the window without its last p halos
    const ArrayRef<TileHalo> window(halos);
    const auto candidates = llvm::seq_inclusive<int64_t>(1, SAMPLE_WINDOW / 2);
    const auto periodIt = llvm::find_if(candidates, [&](int64_t p) {
        return window.drop_front(p) == window.drop_back(p);
    });
    if (periodIt == candidates.end()) {
        return notSupported("the halos do not repeat within the sampling window (no phase period, or different scale)");
    }
    const auto phasePeriod = *periodIt;
    if (step % phasePeriod != 0) {
        return notSupported("step {0} is not a multiple of the phase period {1}", step, phasePeriod);
    }

    // t is a multiple of S and so of P: phases[i] is the halo of every tile at t + i + n * P, also for negative n
    const auto phases = window.take_front(phasePeriod);
    const auto haloAt = [&](int64_t k) {
        return phases[((k % phasePeriod) + phasePeriod) % phasePeriod];
    };
    const auto maxHaloLeft = llvm::max_element(phases, [](const TileHalo& lhs, const TileHalo& rhs) {
                                 return lhs.left < rhs.left;
                             })->left;
    const auto maxHaloRight = llvm::max_element(phases, [](const TileHalo& lhs, const TileHalo& rhs) {
                                  return lhs.right < rhs.right;
                              })->right;

    LowResInfo info;
    info.phasePeriod = phasePeriod;

    // Invalid outputs at the start of the tile at t computed with full left pad. Output t + k (k = 0, 1, ... is the
    // index of the output inside the tile) is the first output of the tile at t + k, so its input window starts at
    // off(t + k) = t + k - h_l(t + k). It is invalid if that is before t, i.e. the pad is read instead of real input:
    // h_l(t + k) > k, which can only hold for k < max h_l.
    // Window starts do not decrease with k, so if output k is invalid, every output before it is invalid too: the
    // invalid outputs are the block k = 0 .. kLast, where kLast is the last invalid index. O_l is a count, not an
    // index, so it is the number of outputs in that block:
    //   O_l = kLast + 1 = 1 + max{k : h_l(t + k) > k}
    // h_l(t + k) depends on the phase of k (k mod P), so each k is compared with the halo of its own phase, not with
    // one halo for all k. Model 3 (P = 2, t even, h_l = 19 at even and 20 at odd k):
    //   k = 18 (even): 19 > 18, invalid
    //   k = 19 (odd):  20 > 19, invalid   <- kLast
    //   k = 20 (even): 19 > 20 is false, valid; every later k is valid as k >= 20 = max h_l
    // so outputs 0..19 are invalid and O_l = 20. With the even halo 19 alone (P = 1) kLast would be 18 and O_l = 19:
    // the odd phase, whose window starts one input earlier, adds one invalid output
    for (int64_t k = 0; k < maxHaloLeft; ++k) {
        if (haloAt(k).left > k) {
            info.invalidOutputLeft = k + 1;
        }
    }

    // Invalid outputs at the end of the tile at t computed with full right pad. The output j positions before the last
    // one, t + S - 1 - j, is the last output of the tile at t - j, so its input window ends at t - j + S + h_r(t - j).
    // It is valid if that is at most t + S, i.e. h_r(t - j) <= j. Window ends do not increase towards the left, so the
    // invalid outputs form a block at the end:
    //   O_r = min{j >= 0 : h_r(t - j) <= j}
    // Here the first valid index is taken, so no + 1 is needed: j = 0 .. O_r - 1 are invalid, which is O_r outputs.
    // j = max(max h_r, 0) is always valid, so the search always succeeds. As for O_l, each j uses the halo of its own
    // phase. Model 3 (P = 2, t even, h_r = 18 at even and 17 at odd j):
    //   j = 16 (even): 18 <= 16 is false, invalid
    //   j = 17 (odd):  17 <= 17, valid    <- first valid
    // so outputs j = 0..16 are invalid and O_r = 17
    info.invalidOutputRight =
            *llvm::find_if(llvm::seq_inclusive<int64_t>(0, std::max<int64_t>(maxHaloRight, 0)), [&](int64_t j) {
                return haloAt(-j).right <= j;
            });

    _log.trace("Low resolution data for dim {0}: phase period {1}, halo left {2}, halo right {3} (per phase: left max "
               "{4}, right max {5}), invalid outputs left {6}, right {7}",
               dimIdx, info.phasePeriod, phases.front().left, phases.front().right, maxHaloLeft, maxHaloRight,
               info.invalidOutputLeft, info.invalidOutputRight);

    // Tile 1 starts at R = D - S. The full pad kernel reproduces the full image only if R is on the phase grid,
    // R % P == 0, which holds for every runtime dim if S (checked above) and every output dim D of the low resolution
    // range are multiples of P. lowResOutputDims[i] is the output dim for the input dim inputStep + i. The largest R
    // is kept for the guard in checkIfTileRegionsAreValid
    for (auto [i, outputDim] : llvm::enumerate(fullBlkInfo.lowResOutputDims)) {
        if (!outputDim.has_value() || outputDim.value() % phasePeriod != 0) {
            return notSupported("output dim for input dim {0} is unknown or not a multiple of the phase period {1}",
                                fullBlkInfo.inputStep + static_cast<int64_t>(i), phasePeriod);
        }
        info.maxTileOneOffset = std::max(info.maxTileOneOffset, outputDim.value() - step);
    }

    _log.trace("Low resolution supported for dim {0}: input dims {1}..{2}, largest tile 1 offset {3}", dimIdx,
               fullBlkInfo.inputStep, fullBlkInfo.lowResLimit, info.maxTileOneOffset);
    return info;
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
            VPUX_THROW_WHEN(forOp == nullptr, "Expected parent scf.for operation for slice size but got none");
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
            if (_enableSubTileFallback) {
                // The decision is deferred to buildSubTileFallback(), which turns the check into scf.if dispatch
                _forOpToSubTileChecks[forOp].push_back({dimOp, blockSize});
                continue;
            }
            addCheckForBlockSize(builder, dimOp, blockSize, _mainFuncOp,
                                 "Not enough elements to backtrack in scf.for loop for Output tensor");
        }

        // Create a new insert_slice operation with adjusted offsets and sizes
        builder.setInsertionPoint(insertSliceOp);
        auto newInsertSliceOp = mlir::tensor::InsertSliceOp::create(
                builder, insertSliceOp.getLoc(), insertSliceOp.getSource(), insertSliceOp.getDest(), newOffsets,
                newSizes, insertSliceOp.getMixedStrides());
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

// Returns (operand index, ExtractSliceOp) pairs for the call operands produced by an ExtractSliceOp,
// looking through an optional tensor.cast
SmallVector<std::pair<int64_t, mlir::Operation*>> getInputSliceOps(mlir::func::CallOp callOp) {
    SmallVector<std::pair<int64_t, mlir::Operation*>> sliceOps;
    for (auto& operand : callOp->getOpOperands()) {
        auto operandIdx = operand.getOperandNumber();
        auto user = operand.get().getDefiningOp();

        if (auto castOp = mlir::dyn_cast_or_null<mlir::tensor::CastOp>(user)) {
            user = castOp.getSource().getDefiningOp();
        }

        if (mlir::isa_and_nonnull<mlir::tensor::ExtractSliceOp>(user)) {
            sliceOps.push_back({operandIdx, user});
        }
    }
    return sliceOps;
}

size_t getNumUniqueBlockShapes(const SliceOpData& sliceOpData) {
    SmallVector<Shape> uniqueShapes;
    for (const auto& [caseVal, shape] : sliceOpData.caseToShapeMap) {
        if (!llvm::is_contained(uniqueShapes, shape)) {
            uniqueShapes.push_back(shape);
        }
    }
    return uniqueShapes.size();
}

/**
 * The slice op whose block shape varies the most across tile positions is selected so that its shape requirements are
 * honored; slice ops with position-invariant block shapes get the same shape for any dispatched case. Ties are resolved
 * by the lowest operand index to keep the selection independent of the DenseMap iteration order.
 */
const SliceOpData& getDispatchSliceOpData(mlir::func::CallOp callOp,
                                          const llvm::DenseMap<mlir::Operation*, SliceOpData>& mapSliceOpToData) {
    auto dispatchSliceOpIt = mapSliceOpToData.begin();
    size_t maxNumUniqueShapes = 0;
    for (auto [operandIdx, sliceOp] : getInputSliceOps(callOp)) {
        auto sliceOpIt = mapSliceOpToData.find(sliceOp);
        if (sliceOpIt == mapSliceOpToData.end()) {
            continue;
        }

        auto numUniqueShapes = getNumUniqueBlockShapes(sliceOpIt->second);
        if (numUniqueShapes > maxNumUniqueShapes) {
            maxNumUniqueShapes = numUniqueShapes;
            dispatchSliceOpIt = sliceOpIt;
        }
    }
    return dispatchSliceOpIt->second;
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

            auto sizeForOp = getEnclosingForOp(val, _opChainAnalysis);
            if (sizeForOp != nullptr) {
                forOpToDim.insert({sizeForOp, static_cast<int64_t>(idx)});
            }

            // The bound gives 2 or more tiles, so FULLBLK is missing. At runtime the whole output dim can still fit in
            // one tile; add the FULLBLK case with the input step as its input size
            if (mapPosToBlockSizes.size() > 1 && sizeForOp != nullptr) {
                auto offsetVal = mlir::dyn_cast_or_null<mlir::Value>(sliceOp.getMixedOffsets()[idx]);
                auto fullBlkInfo = getFullBlkInfo(sizeForOp, outermostForOp, offsetVal, val, sliceOp.getSource(), idx,
                                                  mapPosToBlockSizes);
                if (fullBlkInfo.has_value()) {
                    fullBlkInfo->lowRes = analyzeLowRes(fullBlkInfo.value(), offsetVal, val);
                    mapPosToBlockSizes[static_cast<int64_t>(TilePosition::FULLBLK)] = fullBlkInfo->inputStep;
                    sliceOpData.addedFullBlkDims[idx] = fullBlkInfo.value();
                    _log.trace("Added FULLBLK block size {0} for dim {1}", fullBlkInfo->inputStep, idx);
                }
            }
            dynDimsToTilePositionAndSizeVec.push_back({vpux::Dim(idx), mapPosToBlockSizes});

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
        auto newSliceOp = mlir::tensor::ExtractSliceOp::create(
                builder, appendLoc(sliceOp->getLoc(), "adjusted_input_slice"), sliceOp.getSource(), newOffsets,
                newSizes, sliceOp.getMixedStrides());
        auto newResultType = removeBoundsAttr(newSliceOp.getResultType());
        newSliceOp.getResult().setType(newResultType);

        // ExtractSliceOp -> CastOp -> xxxOp require CastOp to change the input type
        // ExtractSliceOp -> xxxOp require castOp to get the dynamic shape back from static shape
        SmallVector<mlir::OpOperand*> requireCastOp;
        for (auto& use : make_early_inc_range(sliceOp->getUses())) {
            if (mlir::isa<mlir::tensor::CastOp>(use.getOwner())) {
                auto castOp = mlir::cast<mlir::tensor::CastOp>(use.getOwner());
                auto newCastOp = mlir::tensor::CastOp::create(builder, newSliceOp->getLoc(),
                                                              castOp.getResult().getType(), newSliceOp.getResult());
                castOp.getResult().replaceAllUsesWith(newCastOp);
                castOp.erase();
            } else {
                requireCastOp.push_back(&use);
            }
        }
        for (auto* opOperand : requireCastOp) {
            auto newCastOp = mlir::tensor::CastOp::create(builder, newSliceOp->getLoc(),
                                                          sliceOp->getResult(0).getType(), newSliceOp->getResult(0));
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

    auto effectiveDynDimsOrFail = VPU::remapDimsThroughInputPermuteCast(originalFuncOp, dynDims, callOp);
    if (mlir::failed(effectiveDynDimsOrFail)) {
        return mlir::failure();
    }
    auto effectiveDynDims = effectiveDynDimsOrFail.value();
    std::string newFuncName = generateUniqueFunctionSuffix(oldFuncName, caseValue, effectiveDynDims);

    if (moduleOp.lookupSymbol<mlir::func::FuncOp>(newFuncName) != nullptr) {
        // Clone the call operation with new function
        auto newCallOp = mlir::func::CallOp::create(builder, callOp.getLoc(), newFuncName, callOp.getResultTypes(),
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
    auto newCallOp = mlir::func::CallOp::create(builder, callOp.getLoc(), newFuncName, callOp.getResultTypes(),
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
    auto newCallOp = mlir::func::CallOp::create(builder, callOp.getLoc(), callOp.getCallee(), callOp.getResultTypes(),
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
            auto constOp = mlir::arith::ConstantIndexOp::create(builder, funcOp.getLoc(), staticAttr.getInt());
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

        if (tilePosition == TilePosition::FULLBLK && conversionInfo.operandFullBlkValues.contains(operandIdx)) {
            operandToStaticValue[operandIdx] = conversionInfo.operandFullBlkValues.at(operandIdx);
            continue;
        }

        const auto& staticValues = conversionInfo.operandStaticValues.at(operandIdx);
        size_t staticValueIndex = getStaticValueIndexForTilePosition(tilePosition, staticValues.size());

        if (staticValueIndex < staticValues.size()) {
            operandToStaticValue[operandIdx] = staticValues[staticValueIndex];
        }
    }

    return updateFunctionAndCallOp(builder, funcOp, callOp, operandToStaticValue);
}

/**
 * @brief Merges the low resolution data of the inputs of a call along a dim and checks the guard.
 *
 * An output of tile 1 is invalid if the input window of any input is clipped, so the invalid output counts are the
 * largest over the inputs, and so is the largest tile 1 offset R_max (the low resolution ranges of the inputs are
 * combined with OR).
 * The guard is checked here, once for all inputs, as the largest values can come from different inputs:
 * - tile 0 is valid where tile 1 leaves it in place: the invalid head of tile 1, [R, R + O_l), is not written, so
 *   tile 0 must be valid there. Tile 0 is valid on [0, S - O_r), so R + O_l <= S - O_r for every runtime R:
 *   R_max + O_l + O_r <= S. Model 3 W: R_max = 360 - 342 = 18, 18 + 20 + 17 = 55 <= 342
 */
std::optional<LowResInfo> AdjustBlockSizeForScfTilingPass::checkIfTileRegionsAreValid(
        int64_t dimIdx, int64_t step, ArrayRef<std::optional<LowResInfo>> inputs) {
    auto notSupported = [&](const char* reason, const auto&... args) -> std::optional<LowResInfo> {
        _log.trace("Dim {0}: low resolution data not set: {1}", dimIdx, llvm::formatv(reason, args...).str());
        return std::nullopt;
    };

    if (!llvm::all_of(inputs, [](const std::optional<LowResInfo>& input) {
            return input.has_value();
        })) {
        return notSupported("an input does not support low resolution");
    }

    LowResInfo merged;
    for (const auto& input : inputs) {
        merged.invalidOutputLeft = std::max(merged.invalidOutputLeft, input->invalidOutputLeft);
        merged.invalidOutputRight = std::max(merged.invalidOutputRight, input->invalidOutputRight);
        merged.maxTileOneOffset = std::max(merged.maxTileOneOffset, input->maxTileOneOffset);
    }
    _log.trace("Dim {0}: merged low resolution data of {1} inputs: invalid outputs left {2}, right {3}, largest tile 1 "
               "offset {4}",
               dimIdx, inputs.size(), merged.invalidOutputLeft, merged.invalidOutputRight, merged.maxTileOneOffset);

    // tile 0:  [=========valid=========|xx O_r xx]
    //          0                    S-O_r       S
    // tile 1:              [xx O_l xx|===========valid===========]
    //                      R       R+O_l                         D
    //  The xx O_l xx block has to sit entirely under tile 0's valid part.
    if (merged.maxTileOneOffset + merged.invalidOutputLeft + merged.invalidOutputRight > step) {
        return notSupported("largest tile 1 offset {0} + invalid outputs {1} + {2} exceeds the step {3}",
                            merged.maxTileOneOffset, merged.invalidOutputLeft, merged.invalidOutputRight, step);
    }

    return merged;
}

/**
 * @brief Emits the runtime low resolution guards for the call and returns the tile ids with FULLBLK selected
 * when the runtime input dim is in the low resolution range.
 *
 * In this range an edge tile is clipped by the tensor edge, so the START/MIDDLE/END cases do not apply. For every dim
 * with FULLBLK cases the function:
 *   1. Builds the lowRes predicate before the outermost loop of the nest. Inputs that read the same source take the
 *      largest limit, different sources are combined with OR:
 *        lowRes = OR over sources (tensor.dim(source, d) <= limit)
 *   2. Emits a single tile assert next to the predicate. Only the single tile part of the range is supported here, so
 *      the 2-tile part is rejected at runtime:
 *        assert(!lowRes || upperBound == step)
 *   3. If the 2-tile range of the dim is supported for all inputs (checkIfTileRegionsAreValid), gives the predicate and
 * the assert the same id (LOW_RES_PRED_ATTR_NAME, LOW_RES_ASSERT_ATTR_NAME) and puts the merged low resolution data on
 * the scf.for of the dim (LOW_RES_ATTR_NAME). AddKernelsForOverlappingEdgeTiles uses them to add the 2-tile cases and
 * removes the assert once all loops of the id are handled.
 *   4. Overrides the tile id of the dim before the call:
 *        tileId = lowRes ? FULLBLK : tileId
 * The returned tile ids replace `tileIds` in the switch selector.
 */
SmallVector<mlir::Value> AdjustBlockSizeForScfTilingPass::emitRuntimeGuardsForLowRes(
        mlir::func::CallOp callOp, const llvm::DenseMap<mlir::Operation*, SliceOpData>& mapSliceOpToData,
        ArrayRef<vpux::Dim> dynDims, ArrayRef<mlir::Value> tileIds) {
    auto newTileIds = to_small_vector(tileIds);
    for (auto [idx, dim] : llvm::enumerate(dynDims)) {
        llvm::MapVector<mlir::Value, int64_t> sourceToLimit;
        std::optional<FullBlkInfo> dimInfo;
        SmallVector<std::optional<LowResInfo>> inputLowRes;
        for (auto [operandIdx, sliceOp] : getInputSliceOps(callOp)) {
            auto sliceIt = mapSliceOpToData.find(sliceOp);
            if (sliceIt == mapSliceOpToData.end()) {
                continue;
            }
            auto infoIt = sliceIt->second.addedFullBlkDims.find(checked_cast<size_t>(dim.ind()));
            if (infoIt == sliceIt->second.addedFullBlkDims.end()) {
                continue;
            }
            auto& limit = sourceToLimit[infoIt->second.source];
            limit = std::max(limit, infoIt->second.lowResLimit);
            dimInfo = infoIt->second;
            inputLowRes.push_back(infoIt->second.lowRes);
        }
        if (!dimInfo.has_value()) {
            continue;
        }

        // Guards are added here to restrict the FULLBLK case only for dimensions D = S (step size)
        mlir::OpBuilder builder(dimInfo->outermostForOp);
        auto loc = takeOpLoc(dimInfo->forOp, "low_res_check");
        mlir::Value lowRes;
        for (auto [source, limit] : sourceToLimit) {
            auto srcDim = mlir::tensor::DimOp::create(builder, loc, source, dim.ind());
            auto inRange = mlir::arith::CmpIOp::create(builder, loc, mlir::arith::CmpIPredicate::ule, srcDim,
                                                       getCstIndexOp(limit));
            lowRes = lowRes == nullptr ? inRange.getResult()
                                       : mlir::arith::OrIOp::create(builder, loc, lowRes, inRange).getResult();
            _log.trace("Dim {0}: low resolution when the input dim is at most {1}", dim.ind(), limit);
        }
        auto singleTile = mlir::arith::CmpIOp::create(builder, loc, mlir::arith::CmpIPredicate::eq,
                                                      dimInfo->forOp.getUpperBound(), dimInfo->forOp.getStep());
        auto trueVal = mlir::arith::ConstantOp::create(builder, loc, builder.getBoolAttr(true));
        auto isValid = mlir::arith::SelectOp::create(builder, loc, lowRes, singleTile, trueVal);
        auto assertOp = mlir::cf::AssertOp::create(builder, loc, isValid,
                                                   "Low resolution is supported only when the dimension fits in a "
                                                   "single tile");

        if (auto lowResInfo = checkIfTileRegionsAreValid(dim.ind(), dimInfo->inputStep, inputLowRes)) {
            auto forOp = dimInfo->forOp;
            const auto id = _lowResIds.try_emplace(forOp.getOperation(), _lowResIds.size()).first->second;
            lowRes.getDefiningOp()->setAttr(LOW_RES_PRED_ATTR_NAME, builder.getI64IntegerAttr(id));
            assertOp->setAttr(LOW_RES_ASSERT_ATTR_NAME, builder.getI64IntegerAttr(id));
            forOp->setAttr(LOW_RES_ATTR_NAME,
                           VPU::LowResAttr::get(builder.getContext(), id, dim.ind(), dimInfo->inputStep,
                                                lowResInfo->invalidOutputLeft, lowResInfo->invalidOutputRight));
            _log.trace("Dim {0}: low resolution data set with id {1}", dim.ind(), id);
        }

        mlir::OpBuilder positionBuilder(callOp);
        newTileIds[idx] =
                mlir::arith::SelectOp::create(positionBuilder, takeOpLoc(callOp, "select_fullblk"), lowRes,
                                              getCstIndexOp(static_cast<int64_t>(TilePosition::FULLBLK)), tileIds[idx]);
    }
    return newTileIds;
}

/**
 * @brief Selects the cases of the scf.index_switch of `callOp`: the cases given by the bound, and the added FULLBLK
 * cases (any dim at an added FULLBLK position) if they can be kept.
 *
 * The added FULLBLK cases are kept only if every sliced input has a shape for them, every index operand driven by a
 * FULLBLK loop can be evaluated for the single tile, the FULLBLK clone yields the same tile type as the other cases,
 * and no slice of the clone reads past its source. The FULLBLK values of the index operands are recorded in
 * `indexOperandInfo`. If the cases are dropped, the FULLBLK clones created for the check are erased.
 */
SwitchCaseValues AdjustBlockSizeForScfTilingPass::filterAddedFullBlkCases(
        mlir::func::CallOp callOp, const SliceOpData& dispatchSliceOpData,
        const llvm::DenseMap<mlir::Operation*, SliceOpData>& mapSliceOpToData,
        const llvm::DenseMap<mlir::scf::ForOp, int64_t>& forOpToDimInfo, ArrayRef<int64_t> validCaseValues,
        IndexOperandConversionInfo& indexOperandInfo, CaseBlockBuilder buildCaseBlock) {
    auto isAddedFullBlkCase = [&](int64_t caseVal) {
        return llvm::any_of(dispatchSliceOpData.addedFullBlkDims, [&](const auto& entry) {
            return getDimPosition(caseVal, Dim(entry.first)) == TilePosition::FULLBLK;
        });
    };
    SmallVector<int64_t> boundCaseValues;
    SmallVector<int64_t> addedFullBlkCaseValues;
    for (auto caseVal : validCaseValues) {
        (isAddedFullBlkCase(caseVal) ? addedFullBlkCaseValues : boundCaseValues).push_back(caseVal);
    }
    VPUX_THROW_WHEN(boundCaseValues.empty(), "filterAddedFullBlkCases: no case given by the bound");

    SwitchCaseValues boundCases{boundCaseValues, boundCaseValues.front(), false};
    if (addedFullBlkCaseValues.empty()) {
        return boundCases;
    }

    llvm::DenseMap<mlir::Operation*, int64_t> fullBlkInputSteps;
    for (const auto& [dimIdx, fullBlkInfo] : dispatchSliceOpData.addedFullBlkDims) {
        fullBlkInputSteps[mlir::scf::ForOp(fullBlkInfo.forOp).getOperation()] = fullBlkInfo.inputStep;
    }
    for (auto operandIdx : llvm::make_first_range(indexOperandInfo.operandStaticValues)) {
        auto operand = callOp->getOperand(checked_cast<unsigned>(operandIdx));
        auto forOp = getEnclosingForOp(operand, _opChainAnalysis);
        if (forOp == nullptr || !fullBlkInputSteps.contains(forOp.getOperation())) {
            continue;
        }
        auto fullBlkValue = evaluateAtSingleTile(operand, forOp, fullBlkInputSteps.at(forOp.getOperation()));
        if (!fullBlkValue.has_value()) {
            _log.trace("Added FULLBLK cases dropped: index operand {0} cannot be evaluated for a single tile",
                       operandIdx);
            return boundCases;
        }
        _log.trace("Index operand {0}: FULLBLK value {1}", operandIdx, fullBlkValue.value());
        indexOperandInfo.operandFullBlkValues[operandIdx] = fullBlkValue.value();
    }

    for (auto [operandIdx, sliceOp] : getInputSliceOps(callOp)) {
        auto sliceIt = mapSliceOpToData.find(sliceOp);
        if (sliceIt == mapSliceOpToData.end()) {
            continue;
        }
        for (auto caseVal : addedFullBlkCaseValues) {
            if (!sliceIt->second.caseToShapeMap.contains(checked_cast<size_t>(caseVal))) {
                _log.trace("Added FULLBLK cases dropped: operand {0} has no shape for case {1}", operandIdx, caseVal);
                return boundCases;
            }
        }
    }

    // Probe the result types in a detached block that is discarded afterwards
    mlir::Block tmpBlock;
    llvm::SmallSetVector<mlir::func::FuncOp, 4> addedFullBlkFuncs;
    auto getResultTypeFromBoundedCaseValues = [&](int64_t caseVal) -> std::optional<mlir::Type> {
        // The probe block has no parent region, so the context is taken from the call
        mlir::OpBuilder builder(callOp.getContext());
        builder.setInsertionPointToEnd(&tmpBlock);
        auto [callOp, funcOp] = buildCaseBlock(builder, tmpBlock, caseVal);
        if (!callOp) {
            return std::nullopt;
        }
        if (isAddedFullBlkCase(caseVal)) {
            addedFullBlkFuncs.insert(funcOp);
        }
        return callOp.getResult(0).getType();
    };

    auto referenceType = getResultTypeFromBoundedCaseValues(boundCaseValues.front());
    bool canKeep = referenceType.has_value();
    for (auto caseVal : addedFullBlkCaseValues) {
        auto caseType = getResultTypeFromBoundedCaseValues(caseVal);
        if (!caseType.has_value() || caseType != referenceType) {
            _log.trace("Added FULLBLK cases dropped: case {0} yields {1}, expected {2}", caseVal, caseType,
                       referenceType);
            canKeep = false;
            break;
        }

        // The index operands of the FULLBLK dims become constants in the clone, they must not make a slice read past
        // its source
        auto funcOp = addedFullBlkFuncs.back();
        llvm::DenseMap<int64_t, int64_t> argValues;
        if (funcOp.getNumArguments() == callOp.getNumOperands()) {
            for (auto [operandIdx, value] : indexOperandInfo.operandFullBlkValues) {
                auto forOp =
                        getEnclosingForOp(callOp->getOperand(checked_cast<unsigned>(operandIdx)), _opChainAnalysis);
                if (forOpToDimInfo.contains(forOp) &&
                    getDimPosition(caseVal, Dim(forOpToDimInfo.at(forOp))) == TilePosition::FULLBLK) {
                    argValues[operandIdx] = value;
                }
            }
        }
        if (hasOutOfBoundsSlice(funcOp, argValues)) {
            _log.trace("Added FULLBLK cases dropped: case {0} has a slice out of the source bounds", caseVal);
            canKeep = false;
            break;
        }
    }
    tmpBlock.clear();

    if (!canKeep) {
        for (auto funcOp : addedFullBlkFuncs) {
            if (mlir::SymbolTable::symbolKnownUseEmpty(funcOp, _moduleOp)) {
                funcOp.erase();
            }
        }
        return boundCases;
    }
    return SwitchCaseValues{to_small_vector(validCaseValues), boundCaseValues.front(), true};
}

mlir::LogicalResult AdjustBlockSizeForScfTilingPass::generateBlockAwareFuncOps(
        mlir::func::CallOp callOp, llvm::DenseMap<mlir::Operation*, SliceOpData>& mapSliceOpToData,
        const llvm::DenseMap<mlir::scf::ForOp, int64_t>& forOpToDimInfo,
        const IndexOperandConversionInfo& conversionInfo) {
    const auto& dispatchSliceOpData = getDispatchSliceOpData(callOp, mapSliceOpToData);

    mlir::OpBuilder builder(callOp);
    SmallVector<mlir::Value> values;
    SmallVector<vpux::Dim> dynDims;
    for (auto [dim, tensorBlockId] : llvm::enumerate(dispatchSliceOpData.blockPositionValues)) {
        if (tensorBlockId == nullptr) {
            continue;
        }

        values.push_back(tensorBlockId);
        dynDims.push_back(Dim(dim));
    }
    auto shiftAmounts = llvm::to_vector(llvm::map_range(dynDims, [this](vpux::Dim dim) {
        return createConstOpOutsideForOp(getTilePositionShift(dim)).second;
    }));

    // Although the number of cases is 4^N, not all cases are supported
    // E#183027 tracks the support for missing cases
    auto mapOfCaseVals = dispatchSliceOpData.caseToShapeMap;
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

    // Ids of the low resolution dims at which the case is at an added FULLBLK position. Its input slices keep the
    // tiling offset max(0, bt - halo) here, which is right only at D = S (bt = 0); the offset of a low resolution tile
    // is the output tile offset bt. It is set after UnrollSCF, where the loops that can run a low resolution tile are
    // known and the switch is re-encoded, so the slices are tagged with these ids instead of being found by case value
    auto getLowResSliceIds = [&](int64_t caseVal) {
        SmallVector<int64_t> ids;
        for (const auto& [dimIdx, fullBlkInfo] : dispatchSliceOpData.addedFullBlkDims) {
            auto lowResAttr = fullBlkInfo.forOp->getAttrOfType<VPU::LowResAttr>(LOW_RES_ATTR_NAME);
            if (lowResAttr != nullptr && getDimPosition(caseVal, Dim(dimIdx)) == TilePosition::FULLBLK) {
                ids.push_back(lowResAttr.getId());
            }
        }
        llvm::sort(ids);
        return ids;
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
        mlir::scf::YieldOp::create(builder, newCallOp.getLoc(), newCallOp.getResults());
        builder.setInsertionPointToStart(&block);
        auto locString = defaultCase ? "_default_case" : ("_case_" + std::to_string(caseIndex));
        auto lowResSliceIds = defaultCase ? SmallVector<int64_t>{} : getLowResSliceIds(caseIndex);

        for (auto [operandIdx, sliceOp] : extractSliceOps) {
            auto extractSliceOp = mlir::cast<mlir::tensor::ExtractSliceOp>(sliceOp);
            SmallVector<mlir::OpFoldResult> newShape;
            auto shapeMap = getShapeMapForSliceOp(sliceOp);
            getShapeAttrs(shapeMap[caseIndex], newShape);

            if (defaultCase) {
                auto falseAttr = mlir::arith::ConstantOp::create(builder, takeOpLoc(callOp, "bool_" + locString),
                                                                 builder.getBoolAttr(false));
                mlir::cf::AssertOp::create(builder, takeOpLoc(callOp, "assert_valid_shape" + locString), falseAttr,
                                           builder.getStringAttr("Unsupported case"));
            }

            // Create new extract slice op with adjusted shape and remove bounds attribute
            // Add CastOp to convert back to dynamic shape
            auto newSliceOp = mlir::tensor::ExtractSliceOp::create(
                    builder, takeOpLoc(newCallOp, "adjusted_input_slice_" + locString), extractSliceOp.getSource(),
                    extractSliceOp.getMixedOffsets(), newShape, extractSliceOp.getMixedStrides());
            auto newSliceOpResultType = newSliceOp->getResult(0).getType();
            assert(mlir::isa<mlir::RankedTensorType>(newSliceOpResultType) &&
                   "Expected RankedTensorType for extract slice result");
            auto newReturnType = removeBoundsAttr(mlir::cast<mlir::RankedTensorType>(newSliceOpResultType));
            newSliceOp.getResult().setType(newReturnType);
            if (!lowResSliceIds.empty()) {
                newSliceOp->setAttr(LOW_RES_SLICE_ATTR_NAME, builder.getDenseI64ArrayAttr(lowResSliceIds));
            }

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
            auto newCastOp = mlir::tensor::CastOp::create(builder, newCallOp.getLoc(), adjustedCastOpType,
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

            auto updatedCallOp = mlir::func::CallOp::create(builder, newCallOp.getLoc(), funcOp.getName(),
                                                            newCallOp.getResultTypes(), alignedOperands.value());
            newCallOp.replaceAllUsesWith(updatedCallOp.getResults());
            newCallOp.erase();
            newCallOp = updatedCallOp;
        }
        propagateTypeInCallOp(builder, _moduleOp, newCallOp);
        return {newCallOp, funcOp};
    };

    auto buildCaseBlock = [&](mlir::OpBuilder& caseBuilder, mlir::Block& block, int64_t caseVal) {
        auto caseValues = decodePointPosition(caseVal, dynDims);
        return constructBlock(caseBuilder, block, caseVal, dynDims, caseValues, callOp);
    };
    auto indexOperandInfo = conversionInfo;
    auto switchCases = filterAddedFullBlkCases(callOp, dispatchSliceOpData, mapSliceOpToData, forOpToDimInfo,
                                               validCaseValues, indexOperandInfo, buildCaseBlock);
    if (switchCases.hasAddedFullBlkCases) {
        values = emitRuntimeGuardsForLowRes(callOp, mapSliceOpToData, dynDims, values);
    }
    auto encodedIdx = encodePointPosition(builder, callOp.getLoc(), values, shiftAmounts);

    auto dimToEncodingIndex = buildDimToEncodingIndexMap(forOpToDimInfo);
    llvm::DenseSet<llvm::StringRef> processedFunctions;

    auto switchOp = mlir::scf::IndexSwitchOp::create(builder, callOp.getLoc(), callOp.getResultTypes(), encodedIdx,
                                                     switchCases.caseValues, switchCases.caseValues.size());
    for (auto caseIndex : switchOp.getCases()) {
        auto it = llvm::find(switchCases.caseValues, caseIndex);
        auto index = std::distance(switchCases.caseValues.begin(), it);
        auto& region = switchOp.getCaseRegions()[index];
        auto& block = region.empty() ? region.emplaceBlock() : region.front();
        mlir::OpBuilder caseBuilder = mlir::OpBuilder::atBlockBegin(&block);
        auto caseValues = decodePointPosition(caseIndex, dynDims);
        auto [newCallOp, funcOp] = constructBlock(caseBuilder, block, caseIndex, dynDims, caseValues, callOp);
        if (!newCallOp || !funcOp) {
            return mlir::failure();
        }
        if (!indexOperandInfo.shouldConvert() || !funcOp || processedFunctions.contains(funcOp.getName())) {
            continue;
        }
        processedFunctions.insert(funcOp.getName());
        auto result = processIndexTypeArgsInCallOp(caseBuilder, funcOp, newCallOp, indexOperandInfo, forOpToDimInfo,
                                                   dimToEncodingIndex, caseValues);
        if (mlir::failed(result)) {
            return mlir::failure();
        }
    }

    auto& defRegion = switchOp.getDefaultRegion();
    auto& defBlock = defRegion.empty() ? defRegion.emplaceBlock() : defRegion.front();
    mlir::OpBuilder defBuilder = mlir::OpBuilder::atBlockBegin(&defBlock);
    auto defCaseValues = decodePointPosition(switchCases.referenceCaseValue, dynDims);
    auto [defaultCallOp, defaultFuncOp] =
            constructBlock(defBuilder, defBlock, switchCases.referenceCaseValue, dynDims, defCaseValues, callOp, true);
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

constexpr llvm::StringLiteral SUB_TILE_FUNC_SUFFIX = "_sub_tile";

// Returns the dynamic clone of the callee created by this pass run, creating it on first request. The clone is
// inserted through the symbol table so that its name is uniquified when `<callee>_sub_tile` is already taken.
mlir::func::FuncOp AdjustBlockSizeForScfTilingPass::getOrCreateSubTileFunc(mlir::StringAttr calleeAttr,
                                                                           mlir::SymbolTable& symbolTable) {
    auto it = _calleeToSubTileFunc.find(calleeAttr);
    if (it != _calleeToSubTileFunc.end()) {
        return it->second;
    }

    auto originalFuncOp = symbolTable.lookup<mlir::func::FuncOp>(calleeAttr);
    VPUX_THROW_WHEN(originalFuncOp == nullptr, "Callee '{0}' not found in module", calleeAttr.getValue());

    auto subTileFuncOp = originalFuncOp.clone();
    subTileFuncOp.setSymName(calleeAttr.getValue().str() + SUB_TILE_FUNC_SUFFIX.str());
    // Marks the kernel for later passes; the symbol name is not reliable since it may be uniquified
    subTileFuncOp->setAttr(SUB_TILE_FALLBACK_ATTR, mlir::UnitAttr::get(subTileFuncOp->getContext()));
    symbolTable.insert(subTileFuncOp, std::next(mlir::Block::iterator(originalFuncOp.getOperation())));

    _calleeToSubTileFunc.insert({calleeAttr, subTileFuncOp});
    return subTileFuncOp;
}

// Clones the loop nest before any block-size adjustment. Each func.call in the clone is redirected to a dedicated
// dynamic copy of its callee, so that later in-place edits of the original kernels do not affect the fallback path.
AdjustBlockSizeForScfTilingPass::SubTileSnapshot AdjustBlockSizeForScfTilingPass::createSubTileSnapshot(
        mlir::scf::ForOp forOp, mlir::SymbolTable& symbolTable) {
    SubTileSnapshot snapshot;
    snapshot.originalForOp = forOp;
    snapshot.fallbackForOp = mlir::cast<mlir::scf::ForOp>(forOp->clone());

    snapshot.fallbackForOp->walk([&](mlir::Operation* op) {
        if (auto nestedForOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
            nestedForOp->setAttr(SUB_TILE_FALLBACK_ATTR, mlir::UnitAttr::get(op->getContext()));
            return;
        }
        if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
            auto subTileFuncOp = getOrCreateSubTileFunc(callOp.getCalleeAttr().getAttr(), symbolTable);
            callOp.setCallee(subTileFuncOp.getSymName());
        }
    });

    return snapshot;
}

void AdjustBlockSizeForScfTilingPass::emitSubTileChecksAsAsserts(ArrayRef<SubTileCheck> checks) {
    mlir::OpBuilder builder(_mainFuncOp.getContext());
    for (const auto& check : checks) {
        addCheckForBlockSize(builder, check.dimOp, check.blockSize, _mainFuncOp,
                             "Not enough elements to backtrack in scf.for loop for Output tensor");
    }
}

/**
 * @brief Wraps a top-level loop nest into scf.if that dispatches between the adjusted loops and the sub-tile
 * fallback.
 *
 *   %is_sub_tile = or(dim_d < step_d for every dynamic dimension d of the nest)
 *   %result = scf.if %is_sub_tile {
 *     <original loop nest, calling *_sub_tile kernels>     // runs once per dimension with dim < step
 *   } else {
 *     <adjusted loop nest with static tile sizes>
 *   }
 *
 * Falls back to the runtime assert when the check values do not dominate the loop nest or when the nest has no
 * dynamic output dimension.
 */
mlir::LogicalResult AdjustBlockSizeForScfTilingPass::buildSubTileFallback(SubTileSnapshot& snapshot) {
    auto topForOp = snapshot.originalForOp;

    SmallVector<SubTileCheck> checks;
    topForOp->walk([&](mlir::scf::ForOp forOp) {
        if (auto it = _forOpToSubTileChecks.find(forOp); it != _forOpToSubTileChecks.end()) {
            checks.append(it->second);
            _forOpToSubTileChecks.erase(it);
        }
    });
    if (checks.empty()) {
        return mlir::success();
    }

    mlir::DominanceInfo dom(_mainFuncOp);
    const auto dominatesLoopNest = [&](mlir::Value val) {
        return dom.properlyDominates(val, topForOp.getOperation());
    };
    const auto canDispatch = llvm::all_of(checks, [&](SubTileCheck check) {
        return dominatesLoopNest(check.dimOp.getResult()) && dominatesLoopNest(check.blockSize);
    });
    if (!canDispatch) {
        _log.trace("Sub-tile check values do not dominate loop nest at {0}, keeping runtime assert", topForOp.getLoc());
        emitSubTileChecksAsAsserts(checks);
        return mlir::success();
    }

    mlir::OpBuilder builder(topForOp);
    mlir::Value isSubTile;
    for (auto check : checks) {
        auto isDimSubTile = mlir::arith::CmpIOp::create(builder, takeOpLoc(topForOp, "is_sub_tile"),
                                                        mlir::arith::CmpIPredicate::slt, check.dimOp, check.blockSize);
        isSubTile = isSubTile == nullptr ? isDimSubTile.getResult()
                                         : mlir::arith::OrIOp::create(builder, takeOpLoc(topForOp, "any_sub_tile"),
                                                                      isSubTile, isDimSubTile)
                                                   .getResult();
    }

    auto ifOp = mlir::scf::IfOp::create(builder, takeOpLoc(topForOp, "sub_tile_dispatch"), topForOp.getResultTypes(),
                                        isSubTile, /*withElseRegion=*/true);
    topForOp->replaceAllUsesWith(ifOp.getResults());

    auto fallbackForOp = snapshot.fallbackForOp.release();
    auto thenBuilder = ifOp.getThenBodyBuilder();
    thenBuilder.insert(fallbackForOp.getOperation());
    mlir::scf::YieldOp::create(thenBuilder, takeOpLoc(ifOp, "yield_sub_tile"), fallbackForOp.getResults());

    auto elseBuilder = ifOp.getElseBodyBuilder();
    topForOp->moveBefore(ifOp.elseBlock(), ifOp.elseBlock()->end());
    mlir::scf::YieldOp::create(elseBuilder, takeOpLoc(ifOp, "yield_full_tile"), topForOp.getResults());

    return mlir::success();
}

void AdjustBlockSizeForScfTilingPass::safeRunOnModule() {
    _moduleOp = getOperation();
    _mainFuncOp = net::getMainFunc(_moduleOp);
    _lowResIds.clear();
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

    // Snapshots must be taken before the walk below rewrites loop bodies and kernel functions
    SmallVector<SubTileSnapshot> subTileSnapshots;
    _calleeToSubTileFunc.clear();
    _forOpToSubTileChecks.clear();
    if (_enableSubTileFallback) {
        mlir::SymbolTable symbolTable(_moduleOp);
        for (auto forOp : llvm::to_vector(forOps)) {
            subTileSnapshots.push_back(createSubTileSnapshot(forOp, symbolTable));
        }
    }

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

    for (auto& snapshot : subTileSnapshots) {
        if (mlir::failed(buildSubTileFallback(snapshot))) {
            signalPassFailure();
            return;
        }
    }
    // Checks of loops outside every snapshotted nest cannot be dispatched; keep them as runtime asserts
    for (const auto& checks : llvm::make_second_range(_forOpToSubTileChecks)) {
        emitSubTileChecksAsAsserts(checks);
    }
    _forOpToSubTileChecks.clear();
    // Release unused loop-nest clones first so that only materialized fallback paths count as symbol uses
    for (auto& snapshot : subTileSnapshots) {
        if (snapshot.fallbackForOp) {
            snapshot.fallbackForOp.release().erase();
        }
    }
    for (auto subTileFuncOp : llvm::make_second_range(_calleeToSubTileFunc)) {
        if (mlir::SymbolTable::symbolKnownUseEmpty(subTileFuncOp, _moduleOp)) {
            subTileFuncOp.erase();
        }
    }
    _calleeToSubTileFunc.clear();

    removeRedundantExtractSliceOp(_mainFuncOp);
}
}  // namespace

//
// createAdjustBlockSizeForScfTilingPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createAdjustBlockSizeForScfTilingPass(bool enableSubTileFallback, Logger log) {
    return std::make_unique<AdjustBlockSizeForScfTilingPass>(enableSubTileFallback, log);
}
