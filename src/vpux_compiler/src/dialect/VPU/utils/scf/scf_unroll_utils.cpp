//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scf/scf_unroll_utils.hpp"
#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/IRMapping.h>
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"

namespace vpux::VPU {

std::string parseCaseNumber(const std::string& functionName) {
    // Check if string contains "_cases_"
    const std::string casesPattern = "_cases_";
    size_t casesPos = functionName.find(casesPattern);

    if (casesPos == std::string::npos) {
        return "";
    }

    // Find the start position after "_cases_"
    size_t numberStart = casesPos + casesPattern.length();
    if (numberStart >= functionName.length()) {
        return "";
    }

    // Find the end of the numeric part
    size_t numberEnd = numberStart;
    while (numberEnd < functionName.length() && std::isdigit(functionName[numberEnd])) {
        numberEnd++;
    }

    // Extract the numeric string
    if (numberEnd > numberStart) {
        std::string caseNumber = functionName.substr(numberStart, numberEnd - numberStart);
        return caseNumber;
    }

    return "";
}

mlir::Operation* findFirstNonArithmeticOperation(mlir::Block* block) {
    for (auto& op : block->getOperations()) {
        if (mlir::isa<mlir::arith::ArithDialect, mlir::affine::AffineDialect>(op.getDialect())) {
            continue;
        }

        if (mlir::isa<mlir::scf::IfOp>(op)) {
            auto returnTypes = mlir::cast<mlir::scf::IfOp>(op).getResultTypes();
            bool skipOp = true;
            for (auto type : returnTypes) {
                if (!type.isIntOrIndex()) {
                    skipOp = false;
                }
            }

            if (skipOp) {
                continue;
            }
        }

        return &op;
    }
    return block->getTerminator();
}

void cleanUpAfterMerging(mlir::Block* block) {
    SmallVector<mlir::Operation*> toDelete;
    for (auto it = block->rbegin(); it != block->rend(); ++it) {
        mlir::Operation& op = *it;
        if (op.hasTrait<mlir::OpTrait::IsTerminator>()) {
            continue;
        }

        if (mlir::isa<mlir::arith::ArithDialect, mlir::affine::AffineDialect>(op.getDialect())) {
            continue;
        }

        if (mlir::isa<mlir::scf::IfOp>(op)) {
            continue;
        }

        bool safeToDelete = true;
        for (auto user : op.getUsers()) {
            if (!llvm::is_contained(toDelete, user)) {
                safeToDelete = false;
            }
        }

        if (safeToDelete) {
            toDelete.push_back(&op);
        }
    }

    // Delete collected operations
    for (mlir::Operation* op : toDelete) {
        op->erase();
    }
}

template <typename OpType>
SmallVector<OpType> collectOpsFromUnrolledBlocks(llvm::ArrayRef<mlir::Operation*> allUnrolledOps, size_t index,
                                                 size_t totalBlocks, size_t numOriginalOps) {
    SmallVector<OpType> collectedOps;
    collectedOps.reserve(totalBlocks);

    for (size_t i = 0; i < totalBlocks; ++i) {
        size_t opIndex = index + i * numOriginalOps;
        if (opIndex < allUnrolledOps.size()) {
            if (auto op = mlir::dyn_cast<OpType>(allUnrolledOps[opIndex])) {
                collectedOps.push_back(op);
            } else {
                // Handle error: expected operation type not found
                assert(false && "Expected operation type not found at calculated index");
            }
        } else {
            assert(false && "Operation index out of bounds. Failed to collect ops from unrolled blocks.");
        }
    }

    return collectedOps;
}

void setBuilderPositionToEndOfBlock(mlir::OpBuilder& builder, mlir::Block* block, mlir::Location& loc) {
    assert(block != nullptr && "Block pointer is null");

    if (block->empty()) {
        loc = block->getParentOp()->getLoc();
        builder.setInsertionPointToStart(block);
        return;
    }

    // Find last non-terminator operation
    mlir::Operation* nonTerminatorOp = nullptr;
    for (auto it = block->rbegin(); it != block->rend(); ++it) {
        mlir::Operation& op = *it;
        if (!op.hasTrait<mlir::OpTrait::IsTerminator>()) {
            loc = op.getLoc();
            nonTerminatorOp = &op;
            break;
        }
    }

    // set the builder position
    if (block->mightHaveTerminator()) {
        builder.setInsertionPoint(block->getTerminator());
        loc = nonTerminatorOp != nullptr ? nonTerminatorOp->getLoc() : block->getParentOp()->getLoc();
        return;
    }

    if (nonTerminatorOp != nullptr) {
        // Insert after the last non-terminator operation
        builder.setInsertionPointAfter(nonTerminatorOp);
        loc = nonTerminatorOp->getLoc();
    }
}

/**
 * @brief Generates block position combinations recursively for loop unrolling.
 *
 * Block Position Encoding:
 * - 2 = START  (first block in dimension)
 * - 0 = MIDDLE (intermediate block in dimension)
 * - 1 = END    (last block in dimension)
 *
 * Generic NxM Matrix Block Position Layout:
 *
 * For any NxM unrolling pattern (e.g., 3x4), blocks are positioned as:
 *
 * ┌─────────┬─────────┬─────────┬─────────┐
 * │Block 0  │Block 1  │Block 2  │Block 3  │ <- Row 0 (START)
 * │[2,2]    │[2,0]    │[2,0]    │[2,1]    │    H=START(2)
 * ├─────────┼─────────┼─────────┼─────────┤
 * │Block 4  │Block 5  │Block 6  │Block 7  │ <- Row 1 (MIDDLE)
 * │[0,2]    │[0,0]    │[0,0]    │[0,1]    │    H=MIDDLE(0)
 * ├─────────┼─────────┼─────────┼─────────┤
 * │Block 8  │Block 9  │Block 10 │Block 11 │ <- Row 2 (END)
 * │[1,2]    │[1,0]    │[1,0]    │[1,1]    │    H=END(1)
 * └─────────┴─────────┴─────────┴─────────┘
 *     ↑         ↑         ↑         ↑
 *  W=START   W=MIDDLE  W=MIDDLE   W=END
 *   (2)       (0)       (0)       (1)
 *
 * Position Rules:
 * - First row/col: START(2)
 * - Last row/col:  END(1)
 * - Middle rows/cols: MIDDLE(0)
 *
 * Linear Index to 2D Position:
 * - linearIdx → [row, col] = [linearIdx / numCols, linearIdx % numCols]
 * - Block encoding: [H_position, W_position]
 *
 * Valid start-end combinations for dimension with multiple blocks:
 * - {START, MIDDLE} = {2, 0} - spans from first to intermediate
 * - {START, END}    = {2, 1} - spans from first to last
 * - {MIDDLE, MIDDLE}= {0, 0} - spans intermediate blocks
 * - {MIDDLE, END}   = {0, 1} - spans from intermediate to last
 *
 * For single block dimension:
 * - {START, START}  = {2, 2} - single block at start
 * - {MIDDLE, MIDDLE}= {0, 0} - single block in middle
 * - {END, END}      = {1, 1} - single block at end
 */
void generateCombinationsRecursively(int64_t currentDimIndex, ArrayRef<int64_t> blockSizes,
                                     ArrayRef<int64_t> dimensionIdx, int64_t currentChoice,
                                     SmallVector<int64_t>& results) {
    if (currentDimIndex == static_cast<int64_t>(blockSizes.size())) {
        results.push_back(currentChoice);
        return;
    }

    auto blockCount = blockSizes[currentDimIndex];
    SmallVector<std::pair<TilePosition, TilePosition>> validStartEndCombinations;
    if (blockCount > 1) {
        validStartEndCombinations = {{TilePosition::START, TilePosition::MIDDLE},
                                     {TilePosition::START, TilePosition::END},
                                     {TilePosition::MIDDLE, TilePosition::MIDDLE},
                                     {TilePosition::MIDDLE, TilePosition::END}};
    } else {
        validStartEndCombinations = {{TilePosition::START, TilePosition::START},
                                     {TilePosition::MIDDLE, TilePosition::MIDDLE},
                                     {TilePosition::END, TilePosition::END}};
    }

    BlockEncodingUtils encodingUtils;
    for (auto combination : validStartEndCombinations) {
        auto newValue = encodingUtils.encodeValue(vpux::Dim(dimensionIdx[currentDimIndex]), currentChoice,
                                                  static_cast<int64_t>(combination.first),
                                                  static_cast<int64_t>(combination.second));
        generateCombinationsRecursively(currentDimIndex + 1, blockSizes, dimensionIdx, newValue, results);
    }
}

/**
 * @brief Generates all valid block position combinations for loop unrolling optimization.
 *
 * This function creates encoded values representing different ways to partition dimensions
 * during loop unrolling. Each combination specifies start and end block positions for
 * each dimension, following valid forward progression patterns.
 *
 * Block Position Values:
 * - START(2):  First block in a dimension
 * - MIDDLE(0): Intermediate block in a dimension
 * - END(1):    Last block in a dimension
 *
 * Valid Forward Progression Patterns:
 * For dimensions with multiple blocks (blockCount > 1):
 * - START → MIDDLE: Covers blocks from first to intermediate positions
 * - START → END:    Covers blocks from first to last (full span)
 * - MIDDLE → MIDDLE: Covers only intermediate blocks
 * - MIDDLE → END:   Covers blocks from intermediate to last
 *
 * For single block dimensions (blockCount = 1):
 * - START → START:   Single block at start position
 * - MIDDLE → MIDDLE: Single block at middle position
 * - END → END:       Single block at end position
 *
 * Example - 2D unrolling with factors [2, 3] (H=2 blocks, W=3 blocks):
 *
 * H dimension valid combinations: {START→MIDDLE}, {START→END}, {MIDDLE→MIDDLE}, {MIDDLE→END}
 * W dimension valid combinations: {START→MIDDLE}, {START→END}, {MIDDLE→MIDDLE}, {MIDDLE→END}
 *
 * Generated combinations (W varies fastest, then H):
 * - Combination 1: H={START,MIDDLE} + W={START,MIDDLE}
 *   → Covers H[0:0] (first H block) + W[0:1] (first 2 W blocks)
 * - Combination 2: H={START,MIDDLE} + W={START,END}
 *   → Covers H[0:0] (first H block) + W[0:2] (all W blocks)
 * - Combination 3: H={START,MIDDLE} + W={MIDDLE,MIDDLE}
 *   → Covers H[0:0] (first H block) + W[1:1] (middle W block)
 * - Combination 4: H={START,MIDDLE} + W={MIDDLE,END}
 *   → Covers H[0:0] (first H block) + W[1:2] (last 2 W blocks)
 * - Combination 5: H={START,END} + W={START,MIDDLE}
 *   → Covers H[0:1] (all H blocks) + W[0:1] (first 2 W blocks)
 * - Combination 6: H={START,END} + W={START,END}
 *   → Covers H[0:1] (all H blocks) + W[0:2] (all W blocks)
 * - ... (continues with MIDDLE→MIDDLE and MIDDLE→END for H dimension)
 *
 * Encoding Process:
 * Each combination is encoded using BlockEncodingUtils.encodeValue() which packs
 * the start/end positions into a single integer value using bit manipulation.
 * The encoding follows NCHW dimension order with 4 bits per dimension.
 *
 * Usage in Loop Unrolling:
 * These combinations represent different "tiles" or "chunks" of the original
 * tensor that can be processed independently. Each combination becomes a case
 * in an IndexSwitchOp, allowing runtime selection of the appropriate processing
 * path based on the current block position.
 *
 * @param blockSizes Vector of unroll factors for each dimension
 * @param dimensionIdx Vector of dimension indices in NCHW order
 * @return Vector of encoded combination values representing valid block spans
 */
SmallVector<int64_t> generateUnrollCombinations(ArrayRef<int64_t> blockSizes, ArrayRef<int64_t> dimensionIdx) {
    SmallVector<int64_t> results;

    // Validation
    assert(blockSizes.size() == dimensionIdx.size() && "Block sizes and dimension indices must have same length");

    if (blockSizes.empty()) {
        return results;
    }

    // Check if any dimension can be unrolled
    bool canUnroll = false;
    for (int size : blockSizes) {
        if (size > 1) {
            canUnroll = true;
            break;
        }
    }

    if (!canUnroll) {
        return results;
    }

    generateCombinationsRecursively(0, blockSizes, dimensionIdx, 0, results);
    return results;
}

void generateBlockIdsRecursively(int64_t currentDimIndex, ArrayRef<int64_t> blockSizes, ArrayRef<int64_t> dimensionIdx,
                                 int64_t blockValue, int64_t currentValue, SmallVector<int64_t>& results) {
    if (currentDimIndex == static_cast<int64_t>(blockSizes.size())) {
        results.push_back(currentValue);
        return;
    }

    BlockEncodingUtils encodingUtils;
    auto blockCount = blockSizes[currentDimIndex];
    SmallVector<int64_t> currentBlocks = {currentValue};
    auto [startBlk, endBlk] =
            encodingUtils.decodeStartAndEndBlkIds(vpux::Dim(dimensionIdx[currentDimIndex]), blockValue);
    if (blockCount > 1) {
        for (size_t i = 1; i < static_cast<size_t>(blockCount); ++i) {
            currentBlocks.push_back(currentValue);
        }
        encodingUtils.insertBlockId(vpux::Dim(dimensionIdx[currentDimIndex]), currentBlocks[0], startBlk);
        for (size_t i = 1; i < static_cast<size_t>(blockCount - 1); ++i) {
            encodingUtils.insertBlockId(vpux::Dim(dimensionIdx[currentDimIndex]), currentBlocks[i], 0);
        }
        encodingUtils.insertBlockId(vpux::Dim(dimensionIdx[currentDimIndex]), currentBlocks[blockCount - 1], endBlk);
    } else {
        // auto dimStartBlkId = getBlockId(vpux::Dim(dimensionIdx[currentDimIndex]), startBlk);
        encodingUtils.insertBlockId(vpux::Dim(dimensionIdx[currentDimIndex]), currentBlocks[0], startBlk);
    }

    for (auto block : currentBlocks) {
        generateBlockIdsRecursively(currentDimIndex + 1, blockSizes, dimensionIdx, blockValue, block, results);
    }
}

/**
 * @brief Generates individual block position IDs for unrolled blocks based on encoded block value.
 *
 * This function takes an encoded block value (containing start/end block positions) and generates
 * the actual block position IDs for each unrolled block in each dimension. The generation follows
 * a specific pattern where intermediate blocks are assigned MIDDLE positions.
 *
 * Block ID Generation Pattern:
 * For each dimension with unroll factor > 1:
 * - First block gets the START block ID from the encoded value
 * - Last block gets the END block ID from the encoded value
 * - All intermediate blocks get MIDDLE(0) block ID
 *
 * Example 1 - Single dimension H with factor 3 (W constant):
 * Input: blockValue encoded with H_start=2(START), H_end=1(END), W_start=2, W_end=2
 * Generated blocks (3 total):
 * - Block 0: H=2(START), W=2(constant)
 * - Block 1: H=0(MIDDLE), W=2(constant)
 * - Block 2: H=1(END), W=2(constant)
 *
 * This represents unrolling in H dimension while W remains unchanged:
 * ┌─────────┐ ← Block 0: H=START, W=constant
 * │ Block 0 │
 * ├─────────┤ ← Block 1: H=MIDDLE, W=constant
 * │ Block 1 │
 * ├─────────┤ ← Block 2: H=END, W=constant
 * │ Block 2 │
 * └─────────┘
 *
 * Example 2 - Two dimensions H×W with factors [2,3]:
 * Input: blockValue with H_start=2, H_end=1, W_start=2, W_end=1
 * Generated blocks (6 total):
 * - Block 0: H=2(START), W=2(START)
 * - Block 1: H=2(START), W=0(MIDDLE)
 * - Block 2: H=2(START), W=1(END)
 * - Block 3: H=1(END), W=2(START)
 * - Block 4: H=1(END), W=0(MIDDLE)
 * - Block 5: H=1(END), W=1(END)
 *
 * For single block dimensions (factor=1):
 * - Block gets the START block ID (same as start and end)
 *
 * Special handling for single dimension:
 * - Result is right-shifted by 2 bits to match expected format
 *
 * @param config Unroll configuration containing factors and access order
 * @param blockValue Encoded value containing start/end block positions for all dimensions
 * @return Vector of block position IDs, one for each unrolled block
 */
SmallVector<int64_t> generateUnrollBlockIds(const UnrollConfig& config, int64_t blockValue) {
    SmallVector<int64_t> results;
    generateBlockIdsRecursively(0, config.unrollFactors, config.accessOrder, blockValue, 0, results);

    // If there is only one dimension, block id is in first 2 bits
    if (config.accessOrder.size() == 1) {
        for (auto& result : results) {
            switch (config.accessOrder[0]) {
            case 0:
                result = result >> N_SHIFT;
                break;
            case 1:
                result = result >> C_SHIFT;
                break;
            case 2:
                result = result >> H_SHIFT;
                break;
            case 3:
                break;
            default:
                assert(false && "Invalid dimension index in access order");
                break;
            }
            result = result & TWO_BIT_MASK;
        }
    }
    return results;
}

/**
 * @brief Encodes block position IDs into a single value using bit manipulation.
 *
 * This function encodes block position information where each dimension uses 4 bits:
 * - 2 bits for start block ID
 * - 2 bits for end block ID
 * - Dimensions are encoded in NCHW order (N=batch, C=channel, H=height, W=width)
 *
 * Bit Layout (16-bit encoding for 4 dimensions):
 * ┌───────────┬───────────┬───────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
 * │   15:14   │   13:12   │   11:10   │    9:8    │    7:6    │    5:4    │    3:2    │    1:0    │
 * ├───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
 * │   N-end   │   N-start │   C-end   │   C-start │   H-end   │   H-start │   W-end   │   W-start │
 * │  (2-bit)  │  (2-bit)  │  (2-bit)  │  (2-bit)  │  (2-bit)  │  (2-bit)  │  (2-bit)  │  (2-bit)  │
 * └───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
 *
 * Dimension Bit Allocation:
 * - N (Batch):   bits [15:12] = [end-blk(15:14)][start-blk(13:12)]
 * - C (Channel): bits [11:8]  = [end-blk(11:10)][start-blk(9:8)]
 * - H (Height):  bits [7:4]   = [end-blk(7:6)][start-blk(5:4)]
 * - W (Width):   bits [3:0]   = [end-blk(3:2)][start-blk(1:0)]
 *
 * Block ID Values (2-bit encoding):
 * - 00 (0) = MIDDLE block
 * - 01 (1) = END block
 * - 10 (2) = START block
 * - 11 (3) = Reserved/unused
 *
 * Example Encoding:
 * For H×W unrolling where H dimension is unrolled (start=2, end=1) and W is not unrolled:
 * - N: not used → bits[15:12] = 0000 (all zeros)
 * - C: not used → bits[11:8]  = 0000 (all zeros)
 * - H: start=2(10), end=1(01) → bits[7:4] = 0110 (end=01, start=10)
 * - W: start=2(10), end=2(10) → bits[3:0] = 1010 (end=10, start=10, single block)
 * - Final encoded value: 0000000001101010 = 0x006A
 *
 * Breakdown for H dimension unrolling:
 * - H start=2 (START block) means this is the first block in H dimension
 * - H end=1 (END block) means this block spans from start to end of H dimension
 * - W start=2, end=2 (START-START) means single block spanning entire W dimension
 *
 */
mlir::Value encodeBlockPositionId(mlir::OpBuilder& builder, SmallVector<mlir::Value>& inputBlockIdVals,
                                  ArrayRef<int64_t> accessPattern) {
    assert(inputBlockIdVals.size() == 2 && "Need start and end output block position ids");

    SmallVector<int64_t> shiftBits = {6, 4, 2, 0};                // N, C, H, W
    SmallVector<int64_t> shiftBitsForBlockStart = {12, 8, 4, 0};  // N, C, H, W
    SmallVector<int64_t> shiftBitsForBlockEnd = {14, 10, 6, 2};   // N, C, H, W

    mlir::Value returnVal;
    /*auto mask2Bits = builder.create<mlir::arith::ConstantIndexOp>(
        builder.getUnknownLoc(), builder.getIndexType(), builder.getI64IntegerAttr(0x3));*/
    auto mask2Bits = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 3);
    for (auto [idx, dim] : llvm::enumerate(accessPattern)) {
        auto cstStart = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), shiftBits[dim]);
        auto dimVal = builder.create<mlir::arith::ShRUIOp>(builder.getUnknownLoc(), inputBlockIdVals[0], cstStart);
        auto dimStart = builder.create<mlir::arith::AndIOp>(builder.getUnknownLoc(), dimVal, mask2Bits);

        auto dimVal2 = builder.create<mlir::arith::ShRUIOp>(builder.getUnknownLoc(), inputBlockIdVals[1], cstStart);
        auto dimEnd = builder.create<mlir::arith::AndIOp>(builder.getUnknownLoc(), dimVal2, mask2Bits);

        auto cstBlkStart =
                builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), shiftBitsForBlockStart[dim]);
        auto cstBlkEnd =
                builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), shiftBitsForBlockEnd[dim]);

        auto shiftedStart = builder.create<mlir::arith::ShLIOp>(builder.getUnknownLoc(), dimStart, cstBlkStart);
        auto shiftedEnd = builder.create<mlir::arith::ShLIOp>(builder.getUnknownLoc(), dimEnd, cstBlkEnd);

        auto combined = builder.create<mlir::arith::OrIOp>(builder.getUnknownLoc(), shiftedStart, shiftedEnd);
        returnVal = combined->getResult(0);
    }
    return returnVal;
}

void deleteUnrolledBlocks(llvm::SmallSetVector<mlir::Operation*, 8>& opsToDelete) {
    for (auto op : llvm::reverse(opsToDelete)) {
        if (op->use_empty()) {
            op->erase();
        } else {
            assert(false && "Operation to be deleted still has uses");
        }
    }
}

void constructIndexSwitchCaseBlock(mlir::OpBuilder& builder, mlir::Block&, SmallVector<int64_t>& targetCaseValues,
                                   SmallVector<mlir::scf::IndexSwitchOp>& indexSwitchOps, const UnrollConfig&) {
    mlir::IRMapping valueMapper;
    auto cloneBlock = [&](mlir::Block& srcBlock, mlir::scf::IndexSwitchOp srcIndexSwitchOp,
                          SmallVector<mlir::Value>& yieldArgs) {
        for (auto& op : srcBlock.getOperations()) {
            if (!mlir::isa<mlir::scf::YieldOp>(op)) {
                auto* clonedOp = builder.clone(op, valueMapper);

                // Update mapper with new results
                for (auto [originalResult, clonedResult] : llvm::zip(op.getResults(), clonedOp->getResults())) {
                    valueMapper.map(originalResult, clonedResult);
                }
            } else {
                auto yieldOp = mlir::cast<mlir::scf::YieldOp>(op);
                auto yieldOperands = yieldOp.getOperands();
                for (auto [yieldOperand, switchResult] : llvm::zip(yieldOperands, srcIndexSwitchOp.getResults())) {
                    if (valueMapper.contains(yieldOperand)) {
                        valueMapper.map(switchResult, valueMapper.lookup(yieldOperand));
                        yieldArgs.push_back(valueMapper.lookup(yieldOperand));
                    } else {
                        assert(false && "Yield operand not found in value mapper");
                    }
                }
            }
        }
    };

    auto totalBlocks = indexSwitchOps.size();
    SmallVector<mlir::Value> returnValues;
    for (size_t i = 0; i < totalBlocks; ++i) {
        auto currentSwitchOp = indexSwitchOps[i];
        auto currentCaseValue = targetCaseValues[i];
        auto sourceCaseValues = currentSwitchOp.getCases();
        auto sourceCaseIterator = llvm::find(sourceCaseValues, currentCaseValue);
        if (sourceCaseIterator == sourceCaseValues.end()) {
            assert(false && "Case value not found in source switch op");
        }

        size_t caseIdx = std::distance(sourceCaseValues.begin(), sourceCaseIterator);
        auto& region = currentSwitchOp.getCaseRegions()[caseIdx];
        assert(!region.empty() && "Region should not be empty");
        auto& block = region.front();
        cloneBlock(block, currentSwitchOp, returnValues);
    }
}

mlir::LogicalResult updateUnrolledCaseBlock(mlir::Block& block, const UnrollConfig& config) {
    auto startingOp = findFirstNonArithmeticOperation(&block);
    assert(startingOp != nullptr && "No non-arithmetic operation found in the loop body");

    auto firstUnrollSliceBeginIt = mlir::Block::iterator(startingOp);
    auto numOriginalOps = 0;
    while (firstUnrollSliceBeginIt != block.end() && !mlir::isa<mlir::func::CallOp>(&(*firstUnrollSliceBeginIt))) {
        ++firstUnrollSliceBeginIt;
        numOriginalOps++;
    }
    numOriginalOps++;

    if (mlir::failed(mergeUnrollOperationsInBlock(&block, config, numOriginalOps, false))) {
        return mlir::failure();
    }

    auto callOpsIter = block.getOps<mlir::func::CallOp>();
    auto callOps = llvm::to_vector(callOpsIter);
    if (callOpsIter.empty() || callOps.size() <= static_cast<size_t>(config.totalBlocks)) {
        return mlir::failure();
    }

    auto mergedCallOp = callOps.back();
    mlir::OpBuilder builder(mergedCallOp);
    builder.setInsertionPointAfter(mergedCallOp);
    builder.create<mlir::scf::YieldOp>(takeOpLoc(mergedCallOp, "yield"), mergedCallOp->getResult(0));

    return mlir::success();
}

/**
 * @brief Creates a hierarchical structure of ConcatOp operations to concatenate multiple operands
 * according to the specified unroll configuration and access order.
 *
 * This function processes the concatenation in a structured manner, working from the innermost
 * (rightmost) to outermost (leftmost) dimensions as specified in the access order. It groups
 * operands according to the unroll factors and creates concat operations for each group,
 * building up a tree-like structure of concatenations.
 *
 * @example
 * For a 2x3 unroll pattern with accessOrder = [1, 0] and unrollFactors = [2, 3]:
 * Input: 6 operands representing a 2x3 grid: [op0, op1, op2, op3, op4, op5]
 *
 * Step 1 (dim=0, factor=3): Group by rows
 * - Group 0: [op0, op1, op2] -> concat_row0
 * - Group 1: [op3, op4, op5] -> concat_row1
 * Result: [concat_row0, concat_row1]
 *
 * Step 2 (dim=1, factor=2): Group by columns
 * - Group 0: [concat_row0, concat_row1] -> final_concat
 * Result: [final_concat]
 *
 * Final concatenation tree:
 *     final_concat(dim=1)
 *        /        \
 * concat_row0   concat_row1
 *   (dim=0)      (dim=0)
 *   /  |  \      /  |  \
 * op0 op1 op2  op3 op4 op5
 */
SmallVector<vpux::VPU::ConcatOp> insertConcatOp(mlir::OpBuilder& builder, mlir::Location loc,
                                                ArrayRef<mlir::Value> operands, const UnrollConfig& config) {
    SmallVector<vpux::VPU::ConcatOp> concatOps;
    SmallVector<mlir::Value> operandsCopy(operands.begin(), operands.end());

    // Process dimensions from innermost (rightmost) to outermost (leftmost)
    for (int dimIdx = config.accessOrder.size() - 1; dimIdx >= 0; --dimIdx) {
        unsigned dim = config.accessOrder[dimIdx];
        int64_t factor = config.unrollFactors[dimIdx];

        SmallVector<mlir::Value> nextLevel;

        // Group results by this dimension's factor
        for (size_t groupIdx = 0; groupIdx < operandsCopy.size(); groupIdx += factor) {
            SmallVector<mlir::Value> group;

            // Collect up to 'factor' results for this group
            for (auto i = 0; i < factor && (groupIdx + i) < config.totalBlocks; ++i) {
                group.push_back(operandsCopy[groupIdx + i]);
            }

            // Concatenate if more than one element
            if (group.size() > 1) {
                auto lastConcatOp =
                        builder.create<VPU::ConcatOp>(appendLoc(loc, "concat"), mlir::ValueRange(group), dim);
                builder.setInsertionPointAfter(lastConcatOp);
                nextLevel.push_back(lastConcatOp.getResult());
                concatOps.push_back(lastConcatOp);
            } else if (group.size() == 1) {
                nextLevel.push_back(group[0]);
            }
        }

        operandsCopy = std::move(nextLevel);

        if (operandsCopy.size() == 1) {
            break;  // Done - single result remains
        }
    }

    return concatOps;
}

/**
 * @brief Converts a linear index to multi-dimensional indices based on given dimension factors.
 *
 * This function takes a linear (flat) index and converts it to corresponding multi-dimensional
 * indices using the provided factors for each dimension. The conversion follows row-major order
 * where the rightmost dimension varies fastest.
 *
 * @param linearIdx The linear index to convert to multi-dimensional indices
 * @param factors A vector containing the size/factor for each dimension
 * @return A vector of indices corresponding to each dimension, where indices[i] represents
 *         the index in the i-th dimension
 *
 * @example
 * For factors = [2, 3, 4] and linearIdx = 11:
 * - Result would be [1, 1, 3] representing position (1,1,3) in a 2x3x4 tensor
 *
 * @note The function assumes valid input where linearIdx < product of all factors
 */
SmallVector<int64_t> getMultiDimIndices(int64_t linearIdx, ArrayRef<int64_t> factors) {
    SmallVector<int64_t> indices(factors.size());
    for (int i = factors.size() - 1; i >= 0; --i) {
        indices[i] = linearIdx % factors[i];
        linearIdx /= factors[i];
    }
    return indices;
}

// Extract static value from OpFoldResult
int64_t getStaticValue(mlir::OpFoldResult ofr) {
    if (auto attr = mlir::dyn_cast<mlir::Attribute>(ofr)) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
            return intAttr.getInt();
        }
    }
    return mlir::ShapedType::kDynamic;
}

struct SliceSizeInfo {
    llvm::DenseMap<size_t, SmallVector<int64_t>> dimToSizes;  // dim -> [size for each position]
};

/**
 * @brief Collects and analyzes slice sizes from sibling set of operations for loop unrolling optimization.
 *
 * This function processes a collection of tensor operations and organizes their
 * size information according to a specified unroll configuration. It determines the maximum
 * slice size for each dimension at each unrolled position, which is essential for determining
 * optimal memory allocation and scheduling during loop unrolling transformations.
 *
 */
template <typename T>
SliceSizeInfo collectSliceSizes(ArrayRef<T> opsOfType, uint64_t index, const UnrollConfig& config) {
    SliceSizeInfo info;

    // Initialize size vectors for each unrolled dimension
    for (const auto& dim : config.accessOrder) {
        size_t index = &dim - &config.accessOrder[0];
        info.dimToSizes[dim].resize(config.unrollFactors[index], 0);
    }

    // Collect sizes from all extract_slice ops
    for (size_t linearIdx = 0; linearIdx < opsOfType.size(); ++linearIdx) {
        auto currentOp = opsOfType[linearIdx];
        auto rankedType = mlir::cast<mlir::RankedTensorType>(currentOp.getOperation()->getResultTypes()[index]);
        auto outputShape = rankedType.getShape();
        auto sizes = to_small_vector(outputShape);

        auto multiDimIndices = getMultiDimIndices(linearIdx, config.unrollFactors);

        // Store size for each unrolled dimension at this position
        for (size_t i = 0; i < config.accessOrder.size(); ++i) {
            unsigned dim = config.accessOrder[i];
            int64_t blockIndex = multiDimIndices[i];
            int64_t sliceSize = sizes[dim];

            if (sliceSize != mlir::ShapedType::kDynamic) {
                // Take maximum size seen at this position
                info.dimToSizes[dim][blockIndex] = std::max(info.dimToSizes[dim][blockIndex], sliceSize);
            }
        }
    }

    return info;
}

SliceSizeInfo collectSliceSizes(llvm::ArrayRef<mlir::tensor::ExtractSliceOp> extractOps, const UnrollConfig& config) {
    SliceSizeInfo info;

    // Initialize size vectors for each unrolled dimension
    for (const auto& dim : config.accessOrder) {
        size_t index = &dim - &config.accessOrder[0];
        info.dimToSizes[dim].resize(config.unrollFactors[index], 0);
    }

    // Collect sizes from all extract_slice ops
    for (size_t linearIdx = 0; linearIdx < extractOps.size(); ++linearIdx) {
        auto extractOp = extractOps[linearIdx];
        auto sizes = extractOp.getMixedSizes();
        auto multiDimIndices = getMultiDimIndices(linearIdx, config.unrollFactors);

        // Store size for each unrolled dimension at this position
        for (size_t i = 0; i < config.accessOrder.size(); ++i) {
            unsigned dim = config.accessOrder[i];
            int64_t blockIndex = multiDimIndices[i];
            int64_t sliceSize = getStaticValue(sizes[dim]);

            if (sliceSize != mlir::ShapedType::kDynamic) {
                // Take maximum size seen at this position
                info.dimToSizes[dim][blockIndex] = std::max(info.dimToSizes[dim][blockIndex], sliceSize);
            }
        }
    }

    return info;
}

SmallVector<int64_t> computeMergedShape(llvm::ArrayRef<mlir::tensor::ExtractSliceOp> extractOps,
                                        const UnrollConfig& config) {
    auto firstOp = extractOps[0];
    auto rank = firstOp.getType().getRank();
    SmallVector<int64_t> mergedShape(rank);

    // Get sizes from first extract_slice for non-unrolled dimensions
    auto firstSliceSize = firstOp.getMixedSizes();
    for (size_t dim = 0; dim < static_cast<size_t>(rank); ++dim) {
        mergedShape[dim] = getStaticValue(firstSliceSize[dim]);
    }

    // Collect actual sizes from all slices
    auto sizeInfo = collectSliceSizes(extractOps, config);

    // Accumulate sizes for each unrolled dimension
    for (auto dim : config.accessOrder) {
        int64_t totalSize = 0;
        for (auto size : sizeInfo.dimToSizes[dim]) {
            totalSize += size;
        }
        if (totalSize > 0) {
            mergedShape[dim] = totalSize;
        }
    }

    return mergedShape;
}

template <typename T>
SmallVector<int64_t> calculateVPUSliceOffset(int64_t linearIdx, llvm::ArrayRef<T> allExtractOps, uint64_t index,
                                             const UnrollConfig& config) {
    auto firstElement = allExtractOps[0];
    auto firstOp = firstElement.getOperation();
    auto rankedType = mlir::cast<mlir::RankedTensorType>(firstOp->getResultTypes()[index]);
    SmallVector<int64_t> offset(rankedType.getRank(), 0);

    // Get multi-dimensional indices for this operation
    auto multiDimIndices = getMultiDimIndices(linearIdx, config.unrollFactors);

    // Collect size information
    auto sizeInfo = collectSliceSizes(allExtractOps, index, config);

    // Calculate offset for each unrolled dimension
    for (size_t i = 0; i < config.accessOrder.size(); ++i) {
        unsigned dim = config.accessOrder[i];
        int64_t blockIndex = multiDimIndices[i];

        // Accumulate sizes of all blocks before this one
        int64_t accumulatedOffset = 0;
        for (int64_t prevIdx = 0; prevIdx < blockIndex; ++prevIdx) {
            accumulatedOffset += sizeInfo.dimToSizes[dim][prevIdx];
        }

        offset[dim] = accumulatedOffset;
    }

    return offset;
}

SmallVector<int64_t> calculateVPUSliceOffset(int64_t linearIdx,
                                             llvm::ArrayRef<mlir::tensor::ExtractSliceOp> allExtractOps,
                                             const UnrollConfig& config) {
    auto firstSliceOp = allExtractOps[0];
    auto rank = firstSliceOp.getType().getRank();
    SmallVector<int64_t> offset(rank, 0);

    // Get multi-dimensional indices for this operation
    auto multiDimIndices = getMultiDimIndices(linearIdx, config.unrollFactors);

    // Collect size information
    auto sizeInfo = collectSliceSizes(allExtractOps, config);

    // Calculate offset for each unrolled dimension
    for (size_t i = 0; i < config.accessOrder.size(); ++i) {
        unsigned dim = config.accessOrder[i];
        int64_t blockIndex = multiDimIndices[i];

        // Accumulate sizes of all blocks before this one
        int64_t accumulatedOffset = 0;
        for (int64_t prevIdx = 0; prevIdx < blockIndex; ++prevIdx) {
            accumulatedOffset += sizeInfo.dimToSizes[dim][prevIdx];
        }

        offset[dim] = accumulatedOffset;
    }

    return offset;
}

mlir::tensor::ExtractSliceOp createNewExtractSliceOp(mlir::OpBuilder& builder,
                                                     llvm::ArrayRef<mlir::tensor::ExtractSliceOp> extractSliceOps,
                                                     const UnrollConfig& config) {
    auto firstExtractSliceOp = extractSliceOps[0];
    auto mergedSliceOpShape = computeMergedShape(extractSliceOps, config);
    SmallVector<mlir::OpFoldResult> mergedSliceOpSize;
    for (auto dim : mergedSliceOpShape) {
        mergedSliceOpSize.push_back(builder.getIndexAttr(dim));
    }
    auto newOutputRetType =
            mlir::RankedTensorType::get(mergedSliceOpShape, firstExtractSliceOp.getType().getElementType(),
                                        firstExtractSliceOp.getType().getEncoding());
    return builder.create<mlir::tensor::ExtractSliceOp>(
            vpux::takeOpLoc(firstExtractSliceOp, "_new_block_shape"), newOutputRetType, firstExtractSliceOp.getSource(),
            firstExtractSliceOp.getMixedOffsets(), mergedSliceOpSize, firstExtractSliceOp.getMixedStrides());
}

SmallVector<int64_t> calculateCombinedShape(llvm::ArrayRef<mlir::RankedTensorType> tensorTypes,
                                            const UnrollConfig& config) {
    if (tensorTypes.empty()) {
        return {};
    }

    // Get rank from first tensor
    size_t rank = tensorTypes[0].getRank();
    SmallVector<int64_t> combinedShape(rank);

    // Initialize with first tensor's shape
    auto firstShape = tensorTypes[0].getShape();
    for (size_t dim = 0; dim < rank; ++dim) {
        combinedShape[dim] = firstShape[dim];
    }

    // Create a map: unroll dimension -> sizes at each position
    llvm::DenseMap<size_t, SmallVector<int64_t>> dimToSizes;

    // Initialize size vectors for unrolled dimensions
    for (size_t i = 0; i < config.accessOrder.size(); ++i) {
        size_t dim = config.accessOrder[i];
        int64_t factor = config.unrollFactors[i];
        dimToSizes[dim].resize(factor, 0);
    }

    // Helper: convert linear index to multi-dimensional grid position
    auto getMultiDimIndices = [&](int64_t linearIdx) -> SmallVector<int64_t> {
        SmallVector<int64_t> indices(config.unrollFactors.size());
        for (int i = config.unrollFactors.size() - 1; i >= 0; --i) {
            indices[i] = linearIdx % config.unrollFactors[i];
            linearIdx /= config.unrollFactors[i];
        }
        return indices;
    };

    // Collect sizes from all tensors
    for (size_t tensorIdx = 0; tensorIdx < tensorTypes.size(); ++tensorIdx) {
        auto shape = tensorTypes[tensorIdx].getShape();
        auto gridPos = getMultiDimIndices(tensorIdx);

        // For each unrolled dimension, track the size at this grid position
        for (size_t i = 0; i < config.accessOrder.size(); ++i) {
            size_t dim = config.accessOrder[i];
            int64_t posInGrid = gridPos[i];
            int64_t size = shape[dim];

            // Take maximum size at this position (in case of varying sizes)
            if (size != mlir::ShapedType::kDynamic) {
                dimToSizes[dim][posInGrid] = std::max(dimToSizes[dim][posInGrid], size);
            }
        }
    }

    // Accumulate sizes for each unrolled dimension
    for (auto dim : config.accessOrder) {
        int64_t totalSize = 0;

        // Sum all sizes across positions in this dimension
        for (int64_t size : dimToSizes[dim]) {
            totalSize += size;
        }

        if (totalSize > 0) {
            combinedShape[dim] = totalSize;
        }
    }

    return combinedShape;
}

mlir::func::FuncOp mergeFuncOps(mlir::FunctionType newFuncType, mlir::ModuleOp module, mlir::IRMapping& mapper,
                                SmallVector<mlir::Value>& inputValues, llvm::SetVector<mlir::Operation*>& opsToMerge,
                                SmallVector<mlir::Value>& newReturnValues, const std::string& funcNameSuffix) {
    static int functionCounter = 0;
    mlir::OpBuilder moduleBuilder(module->getContext());
    auto funcOps = module.getOps<mlir::func::FuncOp>();
    moduleBuilder.setInsertionPoint(*funcOps.begin());
    auto suffix = funcNameSuffix.empty() ? ("_" + std::to_string(functionCounter++)) : funcNameSuffix;
    auto newFunc = moduleBuilder.create<mlir::func::FuncOp>(module.getLoc(), "merged_vpu_func" + suffix, newFuncType);

    // Clone operations into new function
    auto* entryBlock = newFunc.addEntryBlock();
    for (size_t i = 0; i < inputValues.size(); ++i) {
        mapper.map(inputValues[i], entryBlock->getArgument(i));
    }

    moduleBuilder.setInsertionPointToStart(entryBlock);
    for (auto* op : opsToMerge) {
        if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
            // Inline function call body
            auto oldFuncOp = module.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
            assert(oldFuncOp && "Function not found");

            mlir::IRMapping localMapper = mapper;
            for (auto [callOperand, funcArg] : llvm::zip(callOp.getOperands(), oldFuncOp.getArguments())) {
                localMapper.map(funcArg, mapper.lookupOrDefault(callOperand));
            }

            for (auto& bodyOp : oldFuncOp.getBody().front().without_terminator()) {
                auto* clonedOp = moduleBuilder.clone(bodyOp, localMapper);
                for (auto [orig, clone] : llvm::zip(bodyOp.getResults(), clonedOp->getResults())) {
                    localMapper.map(orig, clone);
                }
            }

            auto returnOp = mlir::cast<mlir::func::ReturnOp>(oldFuncOp.getBody().front().getTerminator());
            for (auto [callResult, returnOperand] : llvm::zip(callOp.getResults(), returnOp.getOperands())) {
                mapper.map(callResult, localMapper.lookup(returnOperand));
            }
        } else {
            auto* clonedOp = moduleBuilder.clone(*op, mapper);
            for (auto [orig, clone] : llvm::zip(op->getResults(), clonedOp->getResults())) {
                mapper.map(orig, clone);
            }
        }
    }

    SmallVector<mlir::Value> funcReturnValues;
    for (auto resultValue : newReturnValues) {
        funcReturnValues.push_back(mapper.lookup(resultValue));
    }
    moduleBuilder.create<mlir::func::ReturnOp>(module.getLoc(), funcReturnValues);
    return newFunc;
}

template <typename T>
void handleOpResults(mlir::OpBuilder& builder, mlir::IRMapping& valueMapper, ArrayRef<T> opsOfType,
                     mlir::Operation* inputOp, const UnrollConfig& config) {
    auto firstOp = opsOfType.front();
    for (auto [resultIdx, result] : llvm::enumerate(firstOp.getResults())) {
        auto hasNonTerminalOpAsUser = false;
        mlir::Operation* firstNonTerminalOp = nullptr;
        for (auto user : result.getUsers()) {
            if (mlir::isa<mlir::tensor::InsertSliceOp>(user) || mlir::isa<mlir::scf::YieldOp>(user) ||
                mlir::isa<mlir::tensor::CastOp>(user) || mlir::isa<vpux::VPU::ConcatOp>(user)) {
                continue;
            }

            hasNonTerminalOpAsUser = true;
            firstNonTerminalOp = user;
            break;
        }

        if (hasNonTerminalOpAsUser) {
            llvm::outs() << "hasNonTerminalOpAsUser true for resultIdx: " << resultIdx << "\n";
            for (auto [callOpIdx, currentCallOp] : llvm::enumerate(opsOfType)) {
                auto vpuSliceOffset =
                        calculateVPUSliceOffset<mlir::func::CallOp>(callOpIdx, opsOfType, resultIdx, config);
                auto callResultType = mlir::cast<mlir::RankedTensorType>(currentCallOp->getResult(resultIdx).getType());
                auto callResultShape = callResultType.getShape();
                SmallVector<int64_t> vpuSliceSize = to_small_vector(callResultShape);

                builder.setInsertionPoint(firstNonTerminalOp);
                auto newSliceOp = builder.create<vpux::VPU::SliceOp>(
                        takeOpLoc(firstOp, "_from_merged"), inputOp->getResult(resultIdx),
                        builder.getI64ArrayAttr(vpuSliceOffset), builder.getI64ArrayAttr(vpuSliceSize));

                currentCallOp->getResult(resultIdx).replaceAllUsesWith(newSliceOp.getResult());
            }
        }

        for (auto [callOpIdx, currentCallOp] : llvm::enumerate(opsOfType)) {
            valueMapper.map(currentCallOp->getResult(resultIdx), inputOp->getResult(resultIdx));
        }
    }
}

/**
 * @brief Merges multiple VPU and function dialect operations into a single reusable function
 *
 * This function analyzes unrolled call operations and their associated VPU operations,
 * then creates a new function in the module containing these operations. The original
 * operations are replaced with a call to the newly created function, reducing code
 * duplication and improving memory efficiency.
 *
 * The function performs the following steps:
 * 1. Collects VPU.Slice operations that feed into the call operations
 * 2. Creates VPU.Concat operations to merge the results from multiple calls
 * 3. Inlines the called function bodies into a new merged function
 * 4. Replaces the original operations with a single call to the merged function
 *
 * @example
 * Before merge (unrolled):
 *   %slice0 = VPU.Slice %input[0, 0] [1, 4] -> tensor<1x4xf32>
 *   %slice1 = VPU.Slice %input[0, 4] [1, 4] -> tensor<1x4xf32>
 *   %result0 = func.call @compute_func(%slice0) : (tensor<1x4xf32>) -> tensor<1x4xf32>
 *   %result1 = func.call @compute_func(%slice1) : (tensor<1x4xf32>) -> tensor<1x4xf32>
 *   %concat = VPU.Concat(%result0, %result1) {axis = 1} -> tensor<1x8xf32>
 *
 * After merge:
 *   // New merged function created in module:
 *   func.func @merged_vpu_func_0(%arg0: tensor<1x8xf32>) -> tensor<1x8xf32> {
 *     %slice0 = VPU.Slice %arg0[0, 0] [1, 4] -> tensor<1x4xf32>
 *     %slice1 = VPU.Slice %arg0[0, 4] [1, 4] -> tensor<1x4xf32>
 *     // inlined compute_func body for both slices
 *     %result = VPU.Concat(%processed0, %processed1) {axis = 1} -> tensor<1x8xf32>
 *     return %result : tensor<1x8xf32>
 *   }
 *
 *   // Original code replaced with:
 *   %merged_result = func.call @merged_vpu_func_0(%input) : (tensor<1x8xf32>) -> tensor<1x8xf32>
 */
mlir::LogicalResult handleCallOps(mlir::OpBuilder& builder, mlir::IRMapping& valueMapper,
                                  ArrayRef<mlir::func::CallOp> callOps, const UnrollConfig& config) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::IRMapping mapper;

    // Collect VPU and function dialect operations to merge
    llvm::SetVector<mlir::Operation*> opsToMerge;
    auto module = callOps.front()->getParentOfType<mlir::ModuleOp>();

    std::string suffix = "";
    for (auto callOp : callOps) {
        for (auto operand : callOp.getOperands()) {
            if (mlir::isa<VPU::SliceOp>(operand.getDefiningOp())) {
                opsToMerge.insert(operand.getDefiningOp());
            }
        }

        opsToMerge.insert(callOp.getOperation());
        auto functionName = callOp.getCallee().str();
        auto caseStr = parseCaseNumber(functionName);
        if (caseStr != "") {
            suffix += "_" + caseStr;
        }
    }

    auto block = callOps.front()->getBlock();
    mlir::Location insertionLoc = builder.getUnknownLoc();
    setBuilderPositionToEndOfBlock(builder, block, insertionLoc);

    // insert concat ops for the soon to be merged callOps
    auto firstCallOp = callOps.front();
    SmallVector<mlir::Value> newReturnValues;
    for (auto [idx, resultType] : llvm::enumerate(firstCallOp.getResultTypes())) {
        if (mlir::isa<mlir::RankedTensorType>(resultType)) {
            SmallVector<mlir::Value> callOpResults;
            for (auto callOp : callOps) {
                callOpResults.push_back(callOp.getResult(idx));
            }

            auto concatResultOp = insertConcatOp(builder, insertionLoc, callOpResults, config);
            newReturnValues.push_back(concatResultOp.back().getResult());
            opsToMerge.insert(concatResultOp.begin(), concatResultOp.end());
        } else {
            return mlir::failure();
        }
    }

    SmallVector<mlir::Type> resultTypes;
    for (auto val : newReturnValues) {
        auto rankedType = mlir::cast<mlir::RankedTensorType>(val.getType());
        resultTypes.push_back(rankedType);
    }

    // Determine function signature based on external inputs/outputs
    SmallVector<mlir::Type> inputTypes;
    SmallVector<mlir::Value> inputValues;

    // Collect unique external inputs
    llvm::SetVector<mlir::Value> uniqueInputs;
    for (auto* op : opsToMerge) {
        for (auto operand : op->getOperands()) {
            auto isDefiningOpInMergeList = llvm::is_contained(opsToMerge, operand.getDefiningOp());
            if (valueMapper.contains(operand)) {
                auto newDefiningOp = valueMapper.lookup(operand).getDefiningOp();
                isDefiningOpInMergeList |= llvm::is_contained(opsToMerge, newDefiningOp);
            }
            if (!isDefiningOpInMergeList) {
                if (operand.getDefiningOp() == nullptr) {
                    uniqueInputs.insert(operand);
                } else {
                    if (valueMapper.contains(operand)) {
                        uniqueInputs.insert(valueMapper.lookup(operand));
                    } else {
                        uniqueInputs.insert(operand);
                    }
                }
            }
        }
    }
    inputValues = uniqueInputs.takeVector();
    for (auto val : inputValues) {
        inputTypes.push_back(val.getType());
    }

    // Create new function
    auto funcType = builder.getFunctionType(inputTypes, resultTypes);
    auto newFuncOp = mergeFuncOps(funcType, module, mapper, inputValues, opsToMerge, newReturnValues, suffix);
    if (newFuncOp == nullptr) {
        return mlir::failure();
    }

    auto callOpToMergedFunc =
            builder.create<mlir::func::CallOp>(module.getLoc(), newFuncOp.getName(), resultTypes, inputValues);

    handleOpResults(builder, valueMapper, callOps, callOpToMergedFunc.getOperation(), config);
    return mlir::success();
}

/* @brief Handles extract slice operations by creating a merged extract slice and VPU slice operations
 *
 * This function takes unrolled extract slice operations and replaces them with:
 * 1. A single merged extract slice operation that extracts a larger block
 * 2. Multiple VPU.Slice operations that slice the merged block into individual pieces
 *
 * Example transformation:
 * Before (unrolled):
 *   %0 = tensor.extract_slice %input[0, 0] [1, 4] [1, 1] -> tensor<1x4xf32>
 *   %1 = tensor.extract_slice %input[0, 4] [1, 4] [1, 1] -> tensor<1x4xf32>
 *   %2 = tensor.extract_slice %input[0, 8] [1, 4] [1, 1] -> tensor<1x4xf32>
 *
 * After (merged):
 *   %merged = tensor.extract_slice %input[0, 0] [1, 12] [1, 1] -> tensor<1x12xf32>
 *   %0 = VPU.Slice %merged[0, 0] [1, 4] -> tensor<1x4xf32>
 *   %1 = VPU.Slice %merged[0, 4] [1, 4] -> tensor<1x4xf32>
 *   %2 = VPU.Slice %merged[0, 8] [1, 4] -> tensor<1x4xf32>
 */
mlir::LogicalResult handleExtractSliceOps(mlir::OpBuilder& builder, mlir::IRMapping& valueMapper, int64_t idx,
                                          int64_t originalOpCount, size_t totalBlocks,
                                          SmallVector<mlir::Operation*>& allUnrolledOps, const UnrollConfig& config) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    SmallVector<mlir::tensor::ExtractSliceOp> extractSliceOps;
    for (size_t i = 0; i < totalBlocks; ++i) {
        auto currentExtractSliceOp =
                mlir::cast<mlir::tensor::ExtractSliceOp>(allUnrolledOps[idx + i * originalOpCount]);
        extractSliceOps.push_back(currentExtractSliceOp);
    }

    builder.setInsertionPoint(extractSliceOps.front());
    auto mergedExtractSliceOp = createNewExtractSliceOp(builder, extractSliceOps, config);

    // Create slices from merged extract with calculated offsets
    for (size_t i = 0; i < extractSliceOps.size(); ++i) {
        auto extractOp = extractSliceOps[i];

        // Calculate offset using actual accumulated sizes
        auto vpuSliceOffset = calculateVPUSliceOffset(i, extractSliceOps, config);

        // Get sizes from current extract_slice
        SmallVector<int64_t> vpuSliceSize;
        for (auto size : extractOp.getMixedSizes()) {
            vpuSliceSize.push_back(getStaticValue(size));
        }

        auto newSliceOp = builder.create<vpux::VPU::SliceOp>(
                takeOpLoc(extractOp, "_from_merged"), mergedExtractSliceOp.getResult(),
                builder.getI64ArrayAttr(vpuSliceOffset), builder.getI64ArrayAttr(vpuSliceSize));
        builder.setInsertionPointAfter(newSliceOp);
        extractOp.replaceAllUsesWith(newSliceOp.getResult());

        // Except for the ExtractSliceOp, all the merged ops are created anew. valueMapper needs to be updated
        // for the newly created ops.
        valueMapper.map(newSliceOp.getResult(), newSliceOp.getResult());
    }

    for (auto op : llvm::reverse(extractSliceOps)) {
        assert(op.use_empty() && "ExtractSliceOp to be deleted still has uses");
        op.erase();
    }

    return mlir::success();
}

/**
 * @brief Merges multiple InsertSliceOp operations from unrolled blocks into a single operation.
 *
 * This function handles the consolidation of tensor::InsertSliceOp operations that were created
 * during loop unrolling. It collects all InsertSliceOps from the unrolled blocks, validates
 * their count, and creates a single merged InsertSliceOp with appropriate size calculations.
 *
 * @example
 * Before merge (unrolled):
 *   %result0 = tensor.insert_slice %slice0 into %dest[0, 0] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<2x8xf32>
 *   %result1 = tensor.insert_slice %slice1 into %dest[0, 4] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<2x8xf32>
 *   %result2 = tensor.insert_slice %slice2 into %dest[1, 0] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<2x8xf32>
 *   %result3 = tensor.insert_slice %slice3 into %dest[1, 4] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<2x8xf32>
 *
 * After merge:
 *   %merged_result = tensor.insert_slice %concatenated_source into %dest[0, 0] [2, 8] [1, 1] : tensor<2x8xf32> into
 * tensor<2x8xf32>
 *
 * Where %concatenated_source is the result of previous VPU.Concat operations
 */
mlir::LogicalResult handleInsertSliceOps(mlir::OpBuilder& builder, mlir::Block* block, mlir::IRMapping& valueMapper,
                                         int64_t index, int64_t originalOpCount, size_t totalBlocks,
                                         SmallVector<mlir::Operation*>& allUnrolledOps) {
    auto sliceOps = collectOpsFromUnrolledBlocks<mlir::tensor::InsertSliceOp>(
            allUnrolledOps, index, static_cast<int64_t>(totalBlocks), originalOpCount);
    if (sliceOps.size() != totalBlocks) {
        llvm::errs() << "Mismatch in number of InsertSliceOps collected. Expected: " << totalBlocks
                     << ", Found: " << sliceOps.size() << "\n";
        return mlir::failure();
    }

    auto firstInsertSliceOp = sliceOps.front();
    auto dstOperand = firstInsertSliceOp.getDest();
    auto origOffsets = firstInsertSliceOp.getMixedOffsets();
    auto origStrides = firstInsertSliceOp.getMixedStrides();

    if (!valueMapper.contains(firstInsertSliceOp.getSource())) {
        llvm::errs() << "Source of InsertSliceOp not found in value mapper. Cannot proceed with merge.\n";
        return mlir::failure();
    }

    auto srcOperand = valueMapper.lookup(firstInsertSliceOp.getSource());
    auto srcOperation = srcOperand.getDefiningOp();
    SmallVector<mlir::OpFoldResult> mergedSize;

    // If the source operand is a cast op, we need to get the shape from its input
    // with the dynamic dimensions resolved to mlir::ShapedType::kDynamic to avoid compilation issues
    if (auto castOp = mlir::dyn_cast<mlir::tensor::CastOp>(srcOperation)) {
        auto castOpInputShape = mlir::cast<mlir::RankedTensorType>(castOp.getSource().getType()).getShape();
        auto castOpOutShape = mlir::cast<mlir::RankedTensorType>(srcOperation->getResult(0).getType()).getShape();
        for (auto [idx, shape] : llvm::enumerate(castOpOutShape)) {
            if (shape == mlir::ShapedType::kDynamic) {
                auto cstVal = builder.create<mlir::arith::ConstantIndexOp>(takeOpLoc(srcOperation, "_dynamic_dim_size"),
                                                                           castOpInputShape[idx]);
                mergedSize.push_back(cstVal.getResult());
            } else {
                mergedSize.push_back(builder.getI64IntegerAttr(shape));
            }
        }
    } else {
        auto inputShape = mlir::cast<mlir::RankedTensorType>(srcOperand.getType()).getShape();
        for (int64_t size : inputShape) {
            mergedSize.push_back(builder.getI64IntegerAttr(size));
        }
    }

    mlir::Location insertionLoc = builder.getUnknownLoc();
    setBuilderPositionToEndOfBlock(builder, block, insertionLoc);
    auto newInsertSliceOp = builder.create<mlir::tensor::InsertSliceOp>(
            appendLoc(insertionLoc, "_merged_insert_op"), srcOperand, dstOperand, origOffsets, mergedSize, origStrides);

    // InsertSliceOps are used by terminator ops. Need to find the insertSliceOp used by the terminator
    // and replace its uses with the newInsertSliceOp result.
    SmallVector<mlir::tensor::InsertSliceOp> sliceOpsUsedByTerminator;
    for (auto sliceOp : sliceOps) {
        for (auto user : sliceOp->getUsers()) {
            if (user->mightHaveTrait<mlir::OpTrait::IsTerminator>()) {
                sliceOpsUsedByTerminator.push_back(sliceOp);
            }
        }
    }

    if (!sliceOpsUsedByTerminator.size()) {
        return mlir::failure();
    }

    sliceOpsUsedByTerminator.front().getResult().replaceAllUsesWith(newInsertSliceOp.getResult());
    if (sliceOpsUsedByTerminator.front() != firstInsertSliceOp) {
        valueMapper.map(firstInsertSliceOp.getResult(), newInsertSliceOp.getResult());
    }

    valueMapper.map(sliceOpsUsedByTerminator.front(), newInsertSliceOp.getResult());
    return mlir::success();
}

mlir::LogicalResult handleTensorCastOps(mlir::OpBuilder& builder, mlir::Block* block, mlir::IRMapping& valueMapper,
                                        mlir::tensor::CastOp firstCastOp) {
    if (!valueMapper.contains(firstCastOp.getSource())) {
        llvm::errs() << "Source of CastOp not found in value mapper. Cannot proceed with merge.\n";
        return mlir::failure();
    }

    auto srcOperand = valueMapper.lookup(firstCastOp.getSource());
    auto srcShape = mlir::cast<mlir::RankedTensorType>(srcOperand.getType()).getShape();
    auto prevDstType = mlir::cast<mlir::RankedTensorType>(firstCastOp.getDest().getType());
    SmallVector<int64_t> boundsVec(srcShape.begin(), srcShape.end());
    BoundsRef boundsRef(boundsVec);
    auto boundedDstType = mlir::dyn_cast<Core::BoundedTensorType>(prevDstType);
    const auto updatedDynamicInputType = boundedDstType.changeBounds(boundsRef);

    mlir::Location insertionLoc = builder.getUnknownLoc();
    setBuilderPositionToEndOfBlock(builder, block, insertionLoc);

    auto updatedCastOp = builder.create<mlir::tensor::CastOp>(appendLoc(insertionLoc, "_updated_cast_op"),
                                                              updatedDynamicInputType, srcOperand);
    valueMapper.map(firstCastOp.getResult(), updatedCastOp.getResult());
    return mlir::success();
}

/**
 * @brief Handles merging multiple IndexSwitchOp operations during loop unrolling.
 *
 * This function takes multiple IndexSwitchOp operations that were created during loop unrolling
 * and merges them into a single IndexSwitchOp with combined cases. It calculates the combined
 * output shape, generates block position combinations based on unroll factors, and constructs
 * the merged switch operation with appropriate case blocks.
 *
 * Example:
 * Input: Multiple IndexSwitchOps from unrolled loop iterations:
 *   switch %pos0 { case 0: %result0, case 1: %result1 }
 *   switch %pos1 { case 0: %result2, case 1: %result3 }
 *
 * Output: Single merged IndexSwitchOp:
 *   switch %encoded_pos {
 *     case 0: %merged_result_00,
 *     case 1: %merged_result_01,
 *     case 2: %merged_result_10,
 *     case 3: %merged_result_11
 *   }
 *
 * @param builder MLIR OpBuilder for creating new operations
 * @param valueMapper IRMapping for tracking value mappings during transformation
 * @param idx Starting index in the allUnrolledOps array for IndexSwitchOps
 * @param originalOpCount Number of operations in the original (pre-unroll) loop body
 * @param totalBlocks Total number of unrolled blocks to process
 * @param allUnrolledOps Vector containing all operations from unrolled loop iterations
 * @param config Unroll configuration containing factors and access order
 * @return LogicalResult indicating success or failure of the merge operation
 */
mlir::LogicalResult handleIndexSwitchOps(mlir::OpBuilder& builder, mlir::IRMapping& valueMapper, int64_t idx,
                                         int64_t originalOpCount, size_t totalBlocks,
                                         SmallVector<mlir::Operation*>& allUnrolledOps, const UnrollConfig& config) {
    mlir::OpBuilder::InsertionGuard guard(builder);

    SmallVector<mlir::scf::IndexSwitchOp> indexSwitchOps;
    SmallVector<mlir::RankedTensorType> outputTypes;
    for (size_t i = 0; i < totalBlocks; ++i) {
        auto currentIndexSwitchOp = mlir::cast<mlir::scf::IndexSwitchOp>(allUnrolledOps[idx + i * originalOpCount]);
        indexSwitchOps.push_back(currentIndexSwitchOp);
        outputTypes.push_back(mlir::cast<mlir::RankedTensorType>(currentIndexSwitchOp.getResult(0).getType()));
    }

    if (indexSwitchOps.size() <= 1) {
        return mlir::success();
    }

    // Calculate combined output shape
    auto mergedShape = calculateCombinedShape(outputTypes, config);
    auto returnType = mlir::RankedTensorType::get(mergedShape, outputTypes.front().getElementType(),
                                                  outputTypes.front().getEncoding());

    // Index switch case operand represents the output block id currently being processed
    SmallVector<mlir::Value> boundaryBlocksPosIds = {indexSwitchOps.front().getOperand(),
                                                     indexSwitchOps.back().getOperand()};

    // Before merging the switch ops, all possible block positions need to be generated
    auto blockPosCombinations = generateUnrollCombinations(config.unrollFactors, config.accessOrder);
    auto encodedBlockPosition = encodeBlockPositionId(builder, boundaryBlocksPosIds, config.accessOrder);
    auto encodedBlockPositionOp = encodedBlockPosition.getDefiningOp();
    builder.setInsertionPointAfter(encodedBlockPositionOp);
    auto switchOp = builder.create<mlir::scf::IndexSwitchOp>(builder.getUnknownLoc(), returnType, encodedBlockPosition,
                                                             blockPosCombinations, blockPosCombinations.size());

    for (auto caseIndex : switchOp.getCases()) {
        auto blockPosCombinationIter = llvm::find(blockPosCombinations, caseIndex);
        auto index = std::distance(blockPosCombinations.begin(), blockPosCombinationIter);
        auto& region = switchOp.getCaseRegions()[index];
        auto& block = region.empty() ? region.emplaceBlock() : region.front();

        auto blockPosIds = generateUnrollBlockIds(config, caseIndex);
        mlir::OpBuilder caseBuilder = mlir::OpBuilder::atBlockBegin(&block);
        constructIndexSwitchCaseBlock(caseBuilder, block, blockPosIds, indexSwitchOps, config);

        if (mlir::failed(updateUnrolledCaseBlock(block, config))) {
            return mlir::failure();
        }
        cleanUpAfterMerging(&block);
    }

    // Default case
    {
        auto& defRegion = switchOp.getDefaultRegion();
        auto& defaultBlock = defRegion.empty() ? defRegion.emplaceBlock() : defRegion.front();
        mlir::OpBuilder defBuilder = mlir::OpBuilder::atBlockBegin(&defaultBlock);
        auto falseAttr = defBuilder.create<mlir::arith::ConstantOp>(vpux::takeOpLoc(switchOp, "_assert_false_"),
                                                                    builder.getBoolAttr(false));
        defBuilder.create<mlir::cf::AssertOp>(vpux::takeOpLoc(switchOp, "_default"), falseAttr,
                                              "Invalid block position");

        auto blockPosCombinationIter = llvm::find(blockPosCombinations, switchOp.getCases().front());
        auto index = std::distance(blockPosCombinations.begin(), blockPosCombinationIter);
        auto& region = switchOp.getCaseRegions()[index];
        assert(!region.empty() && "Region should not be empty");
        auto& block = region.front();
        assert(!block.empty() && "Block should not be empty");
        mlir::IRMapping valueMapper;
        for (auto& op : block) {
            defBuilder.clone(op, valueMapper);
        }
    }

    valueMapper.map(indexSwitchOps.front().getResult(0), switchOp.getResult(0));
    return mlir::success();
}

mlir::LogicalResult mergeUnrollOperationsInBlock(mlir::Block* block, const UnrollConfig& config, int64_t numOriginalOps,
                                                 bool cleanupAfterMerge) {
    auto totalBlocks = config.totalBlocks;
    auto startingOp = findFirstNonArithmeticOperation(block);
    auto firstUnrollSliceBeginIt = mlir::Block::iterator(startingOp);
    auto firstUnrollSliceEndIt = std::next(firstUnrollSliceBeginIt, numOriginalOps);
    auto unrollSliceEndIt = std::next(firstUnrollSliceBeginIt, totalBlocks * numOriginalOps);
    bool nonTensorDialectOps = false;
    for (auto it = firstUnrollSliceBeginIt; it != firstUnrollSliceEndIt; ++it) {
        auto op = &*it;
        if (!llvm::isa<mlir::tensor::TensorDialect>(op->getDialect())) {
            nonTensorDialectOps = true;
        }
    }

    if (!nonTensorDialectOps) {
        return mlir::success();
    }

    SmallVector<mlir::Operation*> allUnrolledOps;
    for (auto it = firstUnrollSliceBeginIt; it != unrollSliceEndIt; ++it) {
        allUnrolledOps.push_back(&*it);
    }

    // Iterate through the block's operations and merge similar ops together.
    // use IRMapping to keep track of value replacements.
    mlir::OpBuilder mergeBuilder(startingOp);
    mlir::IRMapping valueMapper;
    if (block->mightHaveTerminator()) {
        mergeBuilder.setInsertionPoint(block->getTerminator());
    } else {
        mergeBuilder.setInsertionPoint(&(*unrollSliceEndIt));
    }

    for (auto index = 0; index < numOriginalOps; ++index) {
        auto currentOp = allUnrolledOps[index];

        auto result = llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(currentOp)
                              .Case<mlir::tensor::ExtractSliceOp>([&](auto) {
                                  return handleExtractSliceOps(mergeBuilder, valueMapper, index, numOriginalOps,
                                                               totalBlocks, allUnrolledOps, config);
                              })
                              .Case<mlir::tensor::InsertSliceOp>([&](auto) {
                                  return handleInsertSliceOps(mergeBuilder, block, valueMapper, index, numOriginalOps,
                                                              totalBlocks, allUnrolledOps);
                              })
                              .Case<mlir::func::CallOp>([&](auto) {
                                  auto callOps = collectOpsFromUnrolledBlocks<mlir::func::CallOp>(
                                          allUnrolledOps, index, static_cast<int64_t>(totalBlocks), numOriginalOps);
                                  return handleCallOps(mergeBuilder, valueMapper, callOps, config);
                              })
                              .Case<mlir::tensor::CastOp>([&](auto castOp) {
                                  return handleTensorCastOps(mergeBuilder, block, valueMapper, castOp);
                              })
                              .Case<mlir::scf::IndexSwitchOp>([&](auto) {
                                  return handleIndexSwitchOps(mergeBuilder, valueMapper, index, numOriginalOps,
                                                              totalBlocks, allUnrolledOps, config);
                              })
                              .Default([&](mlir::Operation* op) -> mlir::LogicalResult {
                                  llvm::errs() << "Unhandled operation during unroll merging: " << *op << "\n";
                                  return mlir::failure();
                              });

        if (mlir::failed(result)) {
            return mlir::failure();
        }
    }

    if (cleanupAfterMerge) {
        cleanUpAfterMerging(block);
    }
    return mlir::success();
}

/* @brief Merges unrolled operations from an SCF for loop into optimized patterns
 *
 * This function analyzes unrolled operations within an SCF for loop and applies optimizations:
 * 1. Converts multiple extract_slice ops into a single merged extract + VPU.Slice ops
 * 2. Converts multiple insert_slice ops into VPU.Concat + single insert_slice
 * 3. Merges VPU and function dialect operations into reusable functions
 *
 */
mlir::LogicalResult mergeUnrolledOperations(mlir::scf::ForOp forOp, SmallVector<TileDimensionInfo>& tileDimInfoVec) {
    mlir::Operation* terminatorOp = forOp.getBody()->getTerminator();
    mlir::OpBuilder builder(forOp);
    builder.setInsertionPoint(terminatorOp);

    UnrollConfig config;
    for (auto tileDimInfo : tileDimInfoVec) {
        config.unrollFactors.push_back(tileDimInfo.numBlocks);
        config.accessOrder.push_back(tileDimInfo.dimension.ind());
    }

    config.totalBlocks = 1;
    for (auto factor : config.unrollFactors) {
        config.totalBlocks *= factor;
    }

    if (config.totalBlocks <= 1) {
        return mlir::success();
    }

    auto currentBlock = forOp.getBody();
    if (currentBlock->empty()) {
        return mlir::success();
    }

    auto startingOp = findFirstNonArithmeticOperation(currentBlock);
    if (startingOp == &(currentBlock->back())) {
        return mlir::success();
    }

    auto firstUnrollSliceBeginIt = mlir::Block::iterator(startingOp);
    auto numOriginalOps = 0;
    while (firstUnrollSliceBeginIt != currentBlock->end() &&
           !mlir::isa<mlir::tensor::InsertSliceOp>(&(*firstUnrollSliceBeginIt))) {
        ++firstUnrollSliceBeginIt;
        numOriginalOps++;
    }

    if (firstUnrollSliceBeginIt != currentBlock->end()) {
        numOriginalOps++;
    }

    return mergeUnrollOperationsInBlock(currentBlock, config, numOriginalOps);
}

/**
 * @brief Fuses two sibling SCF for loops into a single loop by merging their bodies and iteration arguments.
 *
 * This function combines two SCF for loops that operate on the same iteration space into a single
 * fused loop. The target loop's body is executed first, followed by the source loop's body within
 * each iteration of the fused loop.
 *
 * @param target The target SCF for loop that will be fused (executed first in the fused loop)
 * @param source The source SCF for loop that will be fused (executed second in the fused loop)
 * @param rewriter The MLIR rewriter used to create and modify operations
 * @param residualLoops If true, the fused loop uses target's bounds; if false, uses source's upper bound
 *
 * @return The newly created fused SCF for loop that replaces both input loops
 *
 */
mlir::scf::ForOp fuseSiblingForLoops(mlir::scf::ForOp target, mlir::scf::ForOp source, mlir::RewriterBase& rewriter,
                                     bool residualLoops) {
    // Create fused init_args, with target's init_args before source's init_args.
    llvm::SmallVector<mlir::Value> fusedInitArgs;
    llvm::append_range(fusedInitArgs, target.getInitArgs());

    // Create a new scf.for op after the source loop (with scf.yield terminator
    // (without arguments) only in case its init_args is empty).
    rewriter.setInsertionPoint(target);
    mlir::scf::ForOp fusedLoop;
    if (residualLoops) {
        fusedLoop = rewriter.create<mlir::scf::ForOp>(target.getLoc(), target.getLowerBound(), target.getUpperBound(),
                                                      target.getStep(), fusedInitArgs);
    } else {
        fusedLoop = rewriter.create<mlir::scf::ForOp>(target.getLoc(), target.getLowerBound(), source.getUpperBound(),
                                                      target.getStep(), fusedInitArgs);
    }

    // Map original induction variables and operands to those of the fused loop.
    mlir::IRMapping mapping;
    mapping.map(target.getInductionVar(), fusedLoop.getInductionVar());
    mapping.map(target.getRegionIterArgs(), fusedLoop.getRegionIterArgs());
    mapping.map(source.getInductionVar(), fusedLoop.getInductionVar());
    mapping.map(source.getRegionIterArgs(), fusedLoop.getRegionIterArgs());

    // Merge target's body into the new (fused) for loop and then source's body.
    rewriter.setInsertionPointToStart(fusedLoop.getBody());
    for (mlir::Operation& op : target.getBody()->without_terminator()) {
        rewriter.clone(op, mapping);
    }
    for (mlir::Operation& op : source.getBody()->without_terminator()) {
        rewriter.clone(op, mapping);
    }

    // Build fused yield results by appropriately mapping original yield operands.
    llvm::SmallVector<mlir::Value> yieldResults;
    for (mlir::Value operand : target.getBody()->getTerminator()->getOperands()) {
        yieldResults.push_back(mapping.lookupOrDefault(operand));
    }

    if (!yieldResults.empty()) {
        rewriter.create<mlir::scf::YieldOp>(source.getLoc(), yieldResults);
    }

    // Replace old loops by substituting their uses by results of the fused loop.
    rewriter.replaceOp(target, fusedLoop.getResults());
    rewriter.replaceOp(source, fusedLoop.getResults());

    // copy all attributes of source op into new op
    for (auto attr : source->getAttrs()) {
        fusedLoop->setAttr(attr.getName(), attr.getValue());
    }

    return fusedLoop;
}

}  // namespace vpux::VPU
