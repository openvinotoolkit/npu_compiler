//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/cost_model_shave_utils.hpp"

#include <map>
#include <string>

namespace vpux {
namespace VPU {
namespace arch40xx {
/** @brief Provides a static mapping between operation names and their corresponding SHAVE implementation names.
 *
 * This struct contains a static constant map (`shaveMap`) that associates operation names (as strings)
 * with their respective SHAVE implementation names. The mapping is used to translate operation names
 * into their SHAVE-specific counterparts for the VPUNN architecture.
 */
struct Shave40NamingMap {
    inline static const MapShaveNamesToVPUNN shaveMap{{"Abs", "abs"},
                                                      {"Acos", "acos"},
                                                      {"Acosh", "acosh"},
                                                      {"Add", "add"},
                                                      {"AffineReshape", "AffineReshape"},
                                                      {"And", "logicaland"},
                                                      {"Asin", "asin"},
                                                      {"Asinh", "asinh"},
                                                      {"Atan", "atan"},
                                                      {"Atanh", "atanh"},
                                                      {"Broadcast", "Broadcast"},
                                                      {"Ceiling", "ceiling"},
                                                      {"Clamp", "clamp"},
                                                      {"Concat", "Concat"},
                                                      {"Cos", "cos"},
                                                      {"Cosh", "cosh"},
                                                      {"CumSum", "cumsum"},
                                                      {"DepthToSpace", "DepthToSpace"},
                                                      {"Divide", "div"},
                                                      {"Elu", "elu"},
                                                      {"Equal", "equal"},
                                                      {"Erf", "erf"},
                                                      {"Exp", "exp"},
                                                      {"FakeQuantize", "fakequantize"},
                                                      {"Floor", "floor"},
                                                      {"FloorMod", "floormod"},
                                                      {"Gather", "Gather"},
                                                      {"Gelu", "gelu"},
                                                      {"Greater", "greater"},
                                                      {"GreaterEqual", "greatereq"},
                                                      {"HardSigmoid", "hardsigmoid"},
                                                      {"HardSwish", "hardswish"},
                                                      {"HSigmoid", "hsigmoid"},
                                                      {"HSwish", "hswish"},
                                                      {"Less", "less"},
                                                      {"LessEqual", "lesseq"},
                                                      {"Log", "log"},
                                                      {"LogicalNot", "logicalnot"},
                                                      {"LogicalOr", "logicalor"},
                                                      {"LogicalXor", "logicalxor"},
                                                      {"MVN", "MVN"},
                                                      {"MVN6", "MVN6"},
                                                      {"Maximum", "max"},
                                                      {"MemPermute", "MemPermute"},
                                                      {"Minimum", "min"},
                                                      {"Mish", "mish"},
                                                      {"Multiply", "mul"},
                                                      {"Negative", "negative"},
                                                      {"NormalizeL2", "normalizel2onlyc"},
                                                      {"NotEqual", "notequal"},
                                                      {"PRelu", "prelu"},
                                                      {"PermuteCast", "PermuteCast"},
                                                      {"PermuteQuantize", "PermuteQuantize"},
                                                      {"Power", "power"},
                                                      {"Quantize", "QuantizeCast"},
                                                      {"Relu", "relu"},
                                                      {"Reshape", "Reshape"},
                                                      {"Roll", "Roll"},
                                                      {"Round", "round"},
                                                      {"ScaleShift", "ScaleShift"},
                                                      {"ScatterNDUpdate", "ScatterNDUpdate"},
                                                      {"ScatterUpdate", "ScatterUpdate"},
                                                      {"Select", "select"},
                                                      {"Selu", "selu"},
                                                      {"ShuffleChannels", "ShuffleChannels"},
                                                      {"Sigmoid", "sigmoid"},
                                                      {"Sign", "sign"},
                                                      {"Sin", "sin"},
                                                      {"Sinh", "sinh"},
                                                      {"SoftPlus", "softplus"},
                                                      {"SoftMax", "softmax"},
                                                      {"SpaceToDepth", "SpaceToDepthOp"},
                                                      {"Sqrt", "sqrt"},
                                                      {"SquaredDifference", "squaredifference"},
                                                      {"Squeeze", "Squeeze"},
                                                      {"Subtract", "sub"},
                                                      {"Swish", "swish"},
                                                      {"Tan", "tan"},
                                                      {"Tanh", "tanh"},
                                                      {"Transpose", "Transpose"},
                                                      {"Unsqueeze", "Unsqueeze"},
                                                      {"YuvToRgb", "YuvToRgb"}};
};

// generates based on template the specific class with the needed map
using CostModelShaveUtil = SHAVECMUtilsBase<Shave40NamingMap>;
}  // namespace arch40xx
}  // namespace VPU
}  // namespace vpux
