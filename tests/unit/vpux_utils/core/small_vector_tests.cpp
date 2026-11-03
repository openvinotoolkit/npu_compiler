//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/utils/core/algo.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <unordered_map>

using namespace vpux;

TEST(MLIR_SmallVectorTest, SimpleUsage) {
    std::vector<int> vec1;
    SmallVector<int, 5> vec2;

    for (int i = 0; i < 5; ++i) {
        vec1.push_back(i);
        vec2.push_back(i);
    }

    for (size_t i = 0; i < vec1.size(); ++i) {
        EXPECT_EQ(vec1[i], vec2[i]) << i;
    }

    vec1.clear();
    vec2.clear();

    for (int i = 0; i < 5; ++i) {
        vec1.push_back(i);
    }
    vec2.insert(vec2.end(), vec1.begin(), vec1.end());

    const auto it1 = std::find(vec1.begin(), vec1.end(), 2);
    const auto it2 = std::find(vec2.begin(), vec2.end(), 2);

    EXPECT_NE(it1, vec1.end());
    EXPECT_NE(it2, vec2.end());

    vec1.erase(it1);
    vec2.erase(it2);

    for (size_t i = 0; i < vec1.size(); ++i) {
        EXPECT_EQ(vec1[i], vec2[i]) << i;
    }

    vec1.push_back(15);
    vec1.push_back(16);

    vec2.push_back(15);
    vec2.push_back(16);

    for (size_t i = 0; i < vec1.size(); ++i) {
        EXPECT_EQ(vec1[i], vec2[i]) << i;
    }
}

TEST(MLIR_SmallVectorTest, Equal) {
    SmallVector<int, 2> vec1;
    SmallVector<int, 5> vec2;
    SmallVector<int, 15> vec3;

    for (int i = 0; i < 5; ++i) {
        vec1.push_back(i);
        vec2.push_back(i);
        vec3.push_back(i + 1);
    }

    EXPECT_EQ(vec1, vec2);
    EXPECT_NE(vec1, vec3);
}

TEST(MLIR_SmallVectorTest, Swap) {
    SmallVector<int, 5> vec1;
    SmallVector<int, 5> vec2;

    for (int i = 0; i < 5; ++i) {
        vec1.push_back(i);
        vec2.push_back(5 - i);
    }

    vec1.swap(vec2);

    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(vec1[i], 5 - i) << i;
        EXPECT_EQ(vec2[i], i) << i;
    }
}

TEST(MLIR_SmallVectorTest, BroadcastRepeatsValues) {
    SmallVector<int> values = {1, 2};

    broadcast(values, 6);

    EXPECT_EQ(values, (SmallVector<int>{1, 2, 1, 2, 1, 2}));
}

TEST(MLIR_SmallVectorTest, BroadcastThrowsForNonDivisibleSize) {
    SmallVector<int> values = {1, 2};

    EXPECT_THROW(broadcast(values, 5), vpux::Exception);
}
