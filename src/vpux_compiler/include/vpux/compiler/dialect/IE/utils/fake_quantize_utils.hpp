//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Value.h>
#include "mlir/Support/LogicalResult.h"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {
namespace IE {

struct FqData {
    // Note: using Const::Content (instead of raw vectors) because of potential
    // need to broadcast these values.
    Const::Content low;
    Const::Content high;
};

mlir::FailureOr<FqData> applyScaleShift(mlir::MLIRContext* ctx, const Const::ContentAttr& scale,
                                        const Const::ContentAttr& shift, float low, float high,
                                        vpux::NDTypeInterface storageType, const Logger& log);

mlir::FailureOr<FqData> revertScaleShift(mlir::MLIRContext* ctx, const Const::ContentAttr& scale,
                                         const Const::ContentAttr& shift, float low, float high,
                                         vpux::NDTypeInterface storageType, const Logger& log);

/// Returns quantization levels for a given type.
int64_t getQuantizationLevels(mlir::Type type);

// Returns the real element type of weights that they have during import.
mlir::Type getTrueElemTypeOfWeights(Const::DeclareOp op);

// Returns the real element type of weights that they have during import.
// Historically, assume that convert op's input is weights and their type is the
// real type. This assumption holds when WeightsDequantizeStructureInfo is
// constructed successfully.
mlir::Type getTrueElemTypeOfWeights(IE::ConvertOp op);

class WeightsDequantizeStructureInfo final {
    //                     --- Constant Input Case ---
    //
    //   +----------------------------------------------------------------+
    //   | Weights Const - i8 with transformations                        |
    //   |  [#const.CastElemType<i4>] || [#const.CastElemType<u4>]  |
    //   | [#const.CastElemType<f16>] || [#const.CastElemType<f32>] |
    //   | Weights Const - u8 with transformations                        |
    //   |  [#const.CastElemType<i4>] || [#const.CastElemType<u4>]  |
    //   | [#const.CastElemType<f16>] || [#const.CastElemType<f32>] |
    //   | Weights Const - f16 with transformations                       |
    //   |  [#const.CastElemType<i4>] || [#const.CastElemType<u4>]  |
    //   +----------------------------------------------------------------+
    //             |
    //             |      +-------------+
    //             |      | Shift Const |
    //             |      +-------------+
    //             |           |
    //          +-------------------+
    //          | Optional Subtract |
    //          +-------------------+
    //                    |
    //                    |   +-------------+
    //                    |   | Scale Const |
    //                    |   +-------------+
    //                    |          |
    //                +-------------------+
    //                | Optional Multiply |
    //                +-------------------+
    //                          |

    //        --- Block Argument Input Case ---
    //
    //      [Block Argument]    (si8/ui8/si4/ui4)
    //             |
    //   +--------------------+
    //   | Convert to f16/f32 |
    //   +--------------------+
    //             |
    //             |      +-------------+
    //             |      | Shift Const |
    //             |      +-------------+
    //             |           |
    //          +-------------------+
    //          | Optional Subtract |
    //          +-------------------+
    //                    |
    //                    |   +-------------+
    //                    |   | Scale Const |
    //                    |   +-------------+
    //                    |          |
    //                +-------------------+
    //                | Optional Multiply |
    //                +-------------------+
    //                          |

    //               --- Result ---
    //
    //                      |
    //            +--------------------+
    //            | Convert to f16/f32 |
    //            | (kept if present)  |
    //            +--------------------+
    //                      |
    //   +--------------------------------------+
    //   |             FakeQuantize             |
    //   |  inLow   = type_min                  |
    //   |  inHigh  = type_max                  |
    //   |  outLow  = (inLow - shift) * scale   |
    //   |  outHigh = (inHigh - shift) * scale  |
    //   |  levels  = 256 (i8), 16 (i4)         |
    //   +--------------------------------------+
    //                      |

    //   Subtract and Multiply operation are optional in the dequantization pattern, because they can be folded

private:
    Const::ContentAttr shift = {};  // From subtract op (if present)
    Const::ContentAttr scale = {};  // From multiply op (if present)
    mlir::Value dynamicShift = {};  // From subtract op (if present), coming from blockArgument
    mlir::Value dynamicScale = {};  // From multiply op (if present), coming from blockArgument

    mlir::Value inputValue = nullptr;            // Input of the WD structure (sometimes with Convert Op)
    SmallVector<mlir::Operation*> opChain = {};  // The operations that are part of WD structure
    bool inputIsConst = true;

    [[nodiscard]] mlir::LogicalResult initializeStructure(IE::MultiplyOp& multiplyOp);
    [[nodiscard]] mlir::LogicalResult initializeStructure(IE::SubtractOp& subtractOp);
    [[nodiscard]] mlir::LogicalResult initializeStructure(IE::ConvertOp& convertOp);
    [[nodiscard]] mlir::LogicalResult initializeStructure(IE::AffineReshapeOp& affineReshapeOp);
    [[nodiscard]] mlir::LogicalResult initializeStructure(IE::TransposeOp& transposeOp);
    [[nodiscard]] mlir::LogicalResult initializeStructure(Const::DeclareOp& declareOp);

    vpux::NDTypeInterface getInputType() const;

    WeightsDequantizeStructureInfo(bool constInput, const Logger& log);

public:
    const Logger log;

    static mlir::FailureOr<WeightsDequantizeStructureInfo> create(Const::DeclareOp origOp, const Logger& log);
    static mlir::FailureOr<WeightsDequantizeStructureInfo> create(IE::ConvertOp origOp, const Logger& log);

    // Rewriting-related APIs:
    mlir::Operation* getLastOp() const;

    mlir::Value getInput() const;

    // Manually cleans up the currently found WD structure to ensure consecutive
    // searches on the same root operation would discover *new* WD structures
    // when they exist.
    void cleanUpCurrentWdChain(mlir::PatternRewriter& rewriter) const;

    // Quantization-related APIs:
    [[nodiscard]] std::pair<mlir::Value, mlir::Value> getInputQuantizationInterval(mlir::OpBuilder& builder,
                                                                                   mlir::Location loc, float low,
                                                                                   float high) const;
    [[nodiscard]] std::pair<mlir::Value, mlir::Value> getOutputQuantizationInterval(mlir::OpBuilder& builder,
                                                                                    mlir::Location loc, float low,
                                                                                    float high) const;

    mlir::Value getDynamicScale() const;
    mlir::Value getDynamicShift() const;
    Const::ContentAttr getScale() const;
    Const::ContentAttr getShift() const;
};

std::set<int64_t> findAxes(IE::FakeQuantizeOp origOp);
std::set<int64_t> findAxes(IE::DynamicDequantizeOp origOp);

}  // namespace IE
}  // namespace vpux
