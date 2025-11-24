//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>

#include "vpux/compiler/dialect/VPU/interfaces/cost_model_shave_utils_interface.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace VPU {
/** @brief template class that will be managing the specific map for each architecture
 *  @tparam ShaveMap Struct that contains the static map with the needed mapping
 *  @note it implements the IShaveCostModelUtils interface to expose only the required methods
 *       to the CostModelConfig class
 */
template <typename ShaveMap>
class SHAVECMUtilsBase : public IShaveCostModelUtils {
public:
    // Retrieve the mapping of the transformation from VPUx to VPUNN of SW kernel names
    const MapShaveNamesToVPUNN& getSwKernelContainer() const override {
        return _isShave2ApiUsedInVPUNN ? _swKernelNameShave2API : _swKernelNameShave1API;
    }

    // retrieve if a kernel is supported in the current mapping
    bool isSwKernelOpSupported(const std::string& swKernelName) const override {
        auto swKernelMapGeneric = getSwKernelContainer();
        auto it = swKernelMapGeneric.find(swKernelName);
        return it != swKernelMapGeneric.end();
    }

    SHAVECMUtilsBase(bool isShave2ApiUsed): _isShave2ApiUsedInVPUNN(isShave2ApiUsed) {
    }

private:
    bool _isShave2ApiUsedInVPUNN;
    inline static const MapShaveNamesToVPUNN _swKernelNameShave2API{ShaveMap::shaveMap};
    inline static const MapShaveNamesToVPUNN _swKernelNameShave1API{{"Abs", "Abs"},
                                                                    {"Acos", "Acos"},
                                                                    {"Acosh", "Acosh"},
                                                                    {"Add", "Add"},
                                                                    {"AffineReshape", "AffineReshape"},
                                                                    {"And", "And"},
                                                                    {"Asin", "Asin"},
                                                                    {"Asinh", "Asinh"},
                                                                    {"Atan", "Atan"},
                                                                    {"Atanh", "Atanh"},
                                                                    {"Broadcast", "Broadcast"},
                                                                    {"Ceiling", "Ceiling"},
                                                                    {"Clamp", "Clamp"},
                                                                    {"Concat", "Concat"},
                                                                    {"Cos", "Cos"},
                                                                    {"Cosh", "Cosh"},
                                                                    {"DepthToSpace", "DepthToSpace"},
                                                                    {"Divide", "Divide"},
                                                                    {"Elu", "ELU"},
                                                                    {"Equal", "Equal"},
                                                                    {"Erf", "Erf"},
                                                                    {"Exp", "Exp"},
                                                                    {"FakeQuantize", "FakeQuantize"},
                                                                    {"Floor", "Floor"},
                                                                    {"FloorMod", "FloorMod"},
                                                                    {"Gather", "Gather"},
                                                                    {"Gelu", "Gelu"},
                                                                    {"Greater", "Greater"},
                                                                    {"GreaterEqual", "GreaterEqual"},
                                                                    {"HardSigmoid", "HardSigmoid"},
                                                                    {"HardSwish", "HardSwish"},
                                                                    {"Less", "Less"},
                                                                    {"LessEqual", "LessEqual"},
                                                                    {"Log", "Log"},
                                                                    {"LogicalNot", "LogicalNot"},
                                                                    {"LogicalOr", "LogicalOr"},
                                                                    {"LogicalXor", "LogicalXor"},
                                                                    {"MVN", "MVN"},
                                                                    {"Maximum", "Maximum"},
                                                                    {"MemPermute", "MemPermute"},
                                                                    {"Minimum", "Minimum"},
                                                                    {"Mish", "Mish"},
                                                                    {"Multiply", "Multiply"},
                                                                    {"Negative", "Negative"},
                                                                    {"NotEqual", "NotEqual"},
                                                                    {"PermuteCast", "PermuteCast"},
                                                                    {"PermuteQuantize", "PermuteQuantize"},
                                                                    {"Power", "Power"},
                                                                    {"Quantize", "QuantizeCast"},
                                                                    {"Reshape", "Reshape"},
                                                                    {"Roll", "Roll"},
                                                                    {"Round", "Round"},
                                                                    {"ScaleShift", "ScaleShift"},
                                                                    {"ScatterNDUpdate", "ScatterNDUpdate"},
                                                                    {"ScatterUpdate", "ScatterUpdate"},
                                                                    {"Selu", "Selu"},
                                                                    {"ShuffleChannels", "ShuffleChannels"},
                                                                    {"Sigmoid", "Sigmoid"},
                                                                    {"Sign", "Sign"},
                                                                    {"Sin", "Sin"},
                                                                    {"Sinh", "Sinh"},
                                                                    {"SoftPlus", "SoftPlus"},
                                                                    {"SoftMax", "Softmax"},
                                                                    {"SpaceToDepth", "SpaceToDepthOp"},
                                                                    {"Sqrt", "Sqrt"},
                                                                    {"SquaredDifference", "SquaredDiff"},
                                                                    {"Squeeze", "Squeeze"},
                                                                    {"Subtract", "Subtract"},
                                                                    {"Swish", "Swish"},
                                                                    {"Tanh", "Tanh"},
                                                                    {"Transpose", "Transpose"},
                                                                    {"Unsqueeze", "Unsqueeze"},
                                                                    {"YuvToRgb", "YuvToRgb"}};
};
}  // namespace VPU
}  // namespace vpux
