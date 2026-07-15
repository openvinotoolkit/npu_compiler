//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUASM/dma_transaction.hpp"

#include <gtest/gtest.h>

namespace {

using namespace vpux;

struct DMAConfigTestParams {
    DMATransaction inTransaction = {};
    VPUASM::DMATransactionConfig expectedConfig = {};
};

class DMAConfigTest : public testing::TestWithParam<DMAConfigTestParams> {};

std::vector<DMAConfigTestParams> DMAConfigTestValues = {
        /**/
        {/* 1D */
         /* Input transaction */
         DMATransaction(
                 /* Input pattern*/
                 {DMAPattern({48}, {48})},
                 /* Output pattern*/
                 {DMAPattern({48}, {48})}),
         /* Expected config */
         {/**/
          {48},
          {0},
          {48},
          {0},
          0}},

        {/* 2D */
         /* Input transaction */
         DMATransaction(
                 /* Input pattern*/
                 {DMAPattern({30, 8}, {480, 16})},
                 /* Output pattern*/
                 {DMAPattern({30, 8}, {480, 16})}),
         /* Expected config */
         {/**/
          {8, 29},
          {0, 16},
          {8, 29},
          {0, 16},
          1}},

        {/* 6D */
         /* Input transaction */
         DMATransaction(
                 /* Input pattern*/
                 {DMAPattern({2, 2, 2, 2, 2, 2}, {4096, 1024, 256, 64, 16, 4})},
                 /* Output pattern*/
                 {DMAPattern({2, 2, 2, 2, 2, 2}, {4096, 1024, 256, 64, 16, 4})}),
         /* Expected config */
         {/**/
          {2, 1, 1, 1, 1, 1},
          {0, 4, 16, 64, 256, 1024},
          {2, 1, 1, 1, 1, 1},
          {0, 4, 16, 64, 256, 1024},
          5}},

        {/* 6D */
         /* Input transaction */
         DMATransaction(
                 /* Input pattern*/
                 {DMAPattern({2, 3, 4, 5, 6, 8}, {368640, 92160, 15630, 1920, 192, 16})},
                 /* Output pattern*/
                 {DMAPattern({2, 3, 4, 5, 6, 8}, {368640, 92160, 15630, 1920, 192, 16})}),
         /* Expected config */
         {/**/
          {8, 5, 4, 3, 2, 1},
          {0, 16, 192, 1920, 15630, 92160},
          {8, 5, 4, 3, 2, 1},
          {0, 16, 192, 1920, 15630, 92160},
          5}},

        {/* 6D to 1D */
         /* Input transaction */
         DMATransaction(
                 /* Input pattern*/
                 {DMAPattern({2, 3, 4, 5, 6, 8}, {368640, 92160, 15630, 1920, 192, 16})},
                 /* Output pattern*/
                 {DMAPattern({5760}, {5760})}),
         /* Expected config */
         {/**/
          {8, 5, 4, 3, 2, 1},
          {0, 16, 192, 1920, 15630, 92160},
          {5760},
          {0},
          5}},

        /**/};

TEST_P(DMAConfigTest, GetParams) {
    const auto params = GetParam();
    auto inTransaction = params.inTransaction;
    auto expectedConfig = params.expectedConfig;

    auto generatedConfig = VPUASM::getDMATransactionConfigFromTransaction(inTransaction);

    EXPECT_EQ(generatedConfig.srcDimSizes, expectedConfig.srcDimSizes);
    EXPECT_EQ(generatedConfig.srcStrides, expectedConfig.srcStrides);
    EXPECT_EQ(generatedConfig.dstDimSizes, expectedConfig.dstDimSizes);
    EXPECT_EQ(generatedConfig.dstStrides, expectedConfig.dstStrides);
    EXPECT_EQ(generatedConfig.numDims, expectedConfig.numDims);
}

INSTANTIATE_TEST_SUITE_P(NPUReg40XX, DMAConfigTest, testing::ValuesIn(DMAConfigTestValues));

class DMAConfigExpectThrowTest : public testing::TestWithParam<DMAConfigTestParams> {};

std::vector<DMAConfigTestParams> DMAConfigExpectThrowTestValues = {
        /**/
        {/* 7 dims */
         /* Input transaction */
         DMATransaction(
                 /* Input pattern*/
                 {DMAPattern({2, 2, 2, 2, 2, 2, 2}, {16384, 4096, 1024, 256, 64, 16, 4})},
                 /* Output pattern*/
                 {DMAPattern({128}, {128})}),
         /* Expected config */
         {
                 /**/
         }},

        {/* empty input patterns */
         /* Input transaction */
         DMATransaction(
                 /* Input pattern*/
                 {},
                 /* Output pattern*/
                 {DMAPattern({30, 8}, {480, 16})}),
         /* Expected config */
         {
                 /**/
         }},

        {/* empty output patterns */
         /* Input transaction */
         DMATransaction(
                 /* Input pattern*/
                 {DMAPattern({30, 8}, {480, 16})},
                 /* Output pattern*/
                 {}),
         /* Expected config */
         {
                 /**/
         }},

        {/* 2 input patterns */
         /* Input transaction */
         DMATransaction(
                 /* Input pattern*/
                 {DMAPattern({30, 8}, {480, 16}), DMAPattern({30, 8}, {480, 16})},
                 /* Output pattern*/
                 {DMAPattern({30, 8}, {480, 16})}),
         /* Expected config */
         {
                 /**/
         }},

        {/* 2 output patterns */
         /* Input transaction */
         DMATransaction(
                 /* Input pattern*/
                 {DMAPattern({30, 8}, {480, 16})},
                 /* Output pattern*/
                 {DMAPattern({30, 8}, {480, 16}), DMAPattern({30, 8}, {480, 16})}),
         /* Expected config */
         {
                 /**/
         }},

        {/* input dim count vs stride count mismatch */
         /* Input transaction */
         DMATransaction(
                 /* Input pattern*/
                 {DMAPattern({30, 8}, {480})},
                 /* Output pattern*/
                 {DMAPattern({30, 8}, {480, 16})}),
         /* Expected config */
         {
                 /**/
         }},

        {/* output dim count vs stride count mismatch */
         /* Input transaction */
         DMATransaction(
                 /* Input pattern*/
                 {DMAPattern({30, 8}, {480, 16})},
                 /* Output pattern*/
                 {DMAPattern({30, 8}, {480})}),
         /* Expected config */
         {
                 /**/
         }}

        /**/};

TEST_P(DMAConfigExpectThrowTest, GetParams) {
    const auto params = GetParam();
    auto inTransaction = params.inTransaction;

    EXPECT_ANY_THROW(VPUASM::getDMATransactionConfigFromTransaction(inTransaction));
}

INSTANTIATE_TEST_SUITE_P(NPUReg40XX, DMAConfigExpectThrowTest, testing::ValuesIn(DMAConfigExpectThrowTestValues));

}  // namespace
