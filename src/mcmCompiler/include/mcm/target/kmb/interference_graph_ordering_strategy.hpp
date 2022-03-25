//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#ifndef MV_INTERFERENCE_GRAPH_ORDERING_STRATEGY_HPP_
#define MV_INTERFERENCE_GRAPH_ORDERING_STRATEGY_HPP_

namespace mv
{
    enum class OrderingStrategy
    {
        IG_RANDOM_ORDER,
        IG_LARGEST_FIRST_ORDER,
        IG_SMALLEST_FIRST_ORDER,
        IG_LARGEST_NEIGHBORS_FIRST,
        IG_SMALLEST_NEIGHBORS_FIRST
    };
}

#endif //MV_INTERFERENCE_GRAPH_ORDERING_STRATEGY_HPP_
