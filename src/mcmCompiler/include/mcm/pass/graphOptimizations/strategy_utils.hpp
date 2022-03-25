//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#ifndef STRATEGY_UTILS_HPP
#define STRATEGY_UTILS_HPP

#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"

namespace mv
{
    std::size_t realTensorSize(const mv::Data::TensorIterator tensorToSize, const mv::Shape& streamingPool, bool isCMConv);
    std::size_t activationTensorSize(mv::Op& op, const mv::Data::TensorIterator tensorToSize, std::string clustering, const mv::Shape& streamingPool, 
                                            bool isCMConv, int totalClusters, bool isInput, mv::Shape& streamedShape, bool dilation = false);
    std::size_t alignedWeightsSize(const mv::Data::TensorIterator tensorToSize, const Shape& streamConfig, std::string clustering, int totalClusters);
    std::tuple<std::size_t,std::size_t,std::size_t> memorySize(mv::Op& op, int totalClusters, std::string clustering, 
                                            bool inputActivationSparsity, bool outputActivationSparsity, bool weightsSparsity, const Shape& streamConfig,
                                            bool fakeSparsity, bool spilling = false, bool parentSpilling = true);
}

#endif