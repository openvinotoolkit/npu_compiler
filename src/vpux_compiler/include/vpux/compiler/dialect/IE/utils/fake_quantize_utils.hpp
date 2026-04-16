//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Value.h>
#include "mlir/Support/LogicalResult.h"

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

// Returns the real element type of weights/zp that they have during import.
mlir::Type getTrueElemType(mlir::Operation* op);

// Returns the real input of weights/zp that they have during import.
mlir::Value getTrueInputValue(mlir::Operation* op, mlir::PatternRewriter& rewriter);

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
    mlir::Value shift = nullptr;  // From subtract op (if present)
    mlir::Value scale = nullptr;  // From multiply op (if present)

    mlir::Value inputValue = nullptr;            // Input of the WD structure (sometimes with Convert Op)
    mlir::Type lowPrecisionType = nullptr;       // Low precision type
    SmallVector<mlir::Operation*> opChain = {};  // The operations that are part of WD structure

    [[nodiscard]] mlir::LogicalResult initializeStructure(IE::MultiplyOp& multiplyOp);
    [[nodiscard]] mlir::LogicalResult initializeStructure(IE::SubtractOp& subtractOp);
    [[nodiscard]] mlir::LogicalResult initializeStructure(IE::ConvertOp& convertOp);
    [[nodiscard]] mlir::LogicalResult initializeStructure(Const::DeclareOp& declareOp);

    mlir::LogicalResult checkAndSet(mlir::Value& out, mlir::Value value, bool allowConstant) const;

    vpux::NDTypeInterface getInputType() const;

    WeightsDequantizeStructureInfo(const Logger& log);

public:
    const Logger log;

    static mlir::FailureOr<WeightsDequantizeStructureInfo> create(Const::DeclareOp origOp, const Logger& log);
    static mlir::FailureOr<WeightsDequantizeStructureInfo> create(IE::ConvertOp origOp, const Logger& log);

    // Rewriting-related APIs:
    mlir::Operation* getLastOp() const;

    SmallVector<mlir::Operation*> getOpChain() const;

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

    bool hasConstWeights() const;
    bool hasScale() const;
    bool hasShift() const;
    bool isKVcachedPattern() const;
    // Returns true when the WD-chain's last op has exactly one use and that use is a GatherOp with i4/ui4 weights.
    // The Gather accesses only a small subset of rows at inference time, so dequantizing the full table offline is
    // wasteful. The WD chain is instead routed to DynamicDequantize so that dequantization happens after Gather.
    bool isI4ConsumedByGather() const;

    mlir::Type getInputElemType() const;
    mlir::Type getLowPrecisionElemType() const;

    Const::ContentAttr getStaticScaleAttr() const;
    Const::ContentAttr getStaticShiftAttr() const;

    mlir::Value getStaticScale() const;
    mlir::Value getStaticShift() const;
    mlir::Value getDynamicScale() const;
    mlir::Value getDynamicShift() const;

    vpux::NDTypeInterface getScaleType() const;
    vpux::NDTypeInterface getShiftType() const;

    int64_t getQuantizedAxisCount() const;
};

std::set<int64_t> findAxes(IE::FakeQuantizeOp origOp);
std::set<int64_t> findAxes(IE::DynamicDequantizeOp origOp);

template <typename AnyOp>
type::QuantileFloatType tryParsingNF4(AnyOp) {
    return nullptr;
}

template <>
type::QuantileFloatType tryParsingNF4(Const::DeclareOp constOp);

}  // namespace IE
}  // namespace vpux
