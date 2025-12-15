//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/dialect/IE/utils/function_outlining_splitter.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;
using namespace vpux::IE;

using MLIR_FunctionOutliningSplitterOptions = MLIR_UnitBase;

TEST_F(MLIR_FunctionOutliningSplitterOptions, ParamsWellFormed) {
    std::string param = "'repeating-blocks=max-num-iterations=11 min-ops-in-block=22 weights-as-inputs=true, "
                        "naive=num-parts=33, batching='";
    auto options = OutlinerPassOptions::createFromString(param);

    ASSERT_EQ(options.getIf<NaiveOptions>(0), nullptr);
    ASSERT_NE(options.getIf<NaiveOptions>(1), nullptr);
    ASSERT_EQ(options.getIf<RepeatingBlocksOptions>(1), nullptr);
    ASSERT_NE(options.getIf<RepeatingBlocksOptions>(0), nullptr);
    ASSERT_NE(options.getIf<BatchingOptions>(2), nullptr);

    ASSERT_EQ(options.getIf<RepeatingBlocksOptions>(0)->minOpsInBlock, 22);
    ASSERT_EQ(options.getIf<RepeatingBlocksOptions>(0)->maxNumIterations, 11);
    ASSERT_EQ(options.getIf<RepeatingBlocksOptions>(0)->weightsAsInputs, true);
    ASSERT_EQ(options.getIf<NaiveOptions>(1)->numParts, 33);
}

TEST_F(MLIR_FunctionOutliningSplitterOptions, ParamsWellFormedInsaneSpacing) {
    std::string param =
            " '  repeating-blocks= min-ops-in-block=22     max-num-iterations=11  weights-as-inputs=false  ,   naive=  "
            "num-parts=33,   batching=   '    ";
    auto options = OutlinerPassOptions::createFromString(param);

    ASSERT_EQ(options.getIf<NaiveOptions>(0), nullptr);
    ASSERT_NE(options.getIf<NaiveOptions>(1), nullptr);
    ASSERT_EQ(options.getIf<RepeatingBlocksOptions>(1), nullptr);
    ASSERT_NE(options.getIf<RepeatingBlocksOptions>(0), nullptr);
    ASSERT_NE(options.getIf<BatchingOptions>(2), nullptr);

    ASSERT_EQ(options.getIf<RepeatingBlocksOptions>(0)->minOpsInBlock, 22);
    ASSERT_EQ(options.getIf<RepeatingBlocksOptions>(0)->maxNumIterations, 11);
    ASSERT_EQ(options.getIf<RepeatingBlocksOptions>(0)->weightsAsInputs, false);
    ASSERT_EQ(options.getIf<NaiveOptions>(1)->numParts, 33);
}

TEST_F(MLIR_FunctionOutliningSplitterOptions, ParamsWellFormedDefaultValues) {
    std::string param = "repeating-blocks, naive=";
    auto options = OutlinerPassOptions::createFromString(param);

    ASSERT_EQ(options.getIf<NaiveOptions>(0), nullptr);
    ASSERT_NE(options.getIf<NaiveOptions>(1), nullptr);
    ASSERT_EQ(options.getIf<RepeatingBlocksOptions>(1), nullptr);
    ASSERT_NE(options.getIf<RepeatingBlocksOptions>(0), nullptr);

    ASSERT_EQ(options.getIf<RepeatingBlocksOptions>(0)->minOpsInBlock,
              RepeatingBlocksOptions::MIN_OPS_IN_BLOCK_DEFAULT);
    ASSERT_EQ(options.getIf<RepeatingBlocksOptions>(0)->maxNumIterations,
              RepeatingBlocksOptions::MAX_NUM_ITERATIONS_DEFAULT);
    ASSERT_EQ(options.getIf<RepeatingBlocksOptions>(0)->weightsAsInputs,
              RepeatingBlocksOptions::WEIGHTS_AS_INPUTS_DEFAULT);
    ASSERT_EQ(options.getIf<NaiveOptions>(1)->numParts, NaiveOptions::NUM_PARTS_DEFAULT);
}

TEST_F(MLIR_FunctionOutliningSplitterOptions, ParamsIllFormedNoInteger) {
    std::string param = " '   repeating-blocks= min-ops-in-block=def     max-num-iterations=11    ,   naive=   "
                        "num-parts=33'   ";
    ASSERT_THROW(OutlinerPassOptions::createFromString(param), vpux::Exception);
}

TEST_F(MLIR_FunctionOutliningSplitterOptions, ParamsIllFormedBadIntegerValue) {
    std::string param = "   ' repeating-blocks= min-ops-in-block=-1     max-num-iterations=11    ,   naive=   "
                        "num-parts=33'    ";
    ASSERT_THROW(OutlinerPassOptions::createFromString(param), vpux::Exception);
}

TEST_F(MLIR_FunctionOutliningSplitterOptions, ParamsIllFormedMissingDelimiter) {
    std::string param = "  '  repeating-blocks= min-ops-in-block=22      naive=num-parts=33'    ";
    ASSERT_THROW(OutlinerPassOptions::createFromString(param), vpux::Exception);
}
