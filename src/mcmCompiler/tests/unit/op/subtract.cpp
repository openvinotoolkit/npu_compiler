//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "gtest/gtest.h"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(ops, subtract)
{

    mv::OpModel om("testModel");
    auto input0 = om.input("", {256, 512}, mv::DType("Float16"), mv::Order("HW"));
    std::vector<double> input1Data = mv::utils::generateSequence<double>(256u * 512u);
    std::vector<double> input2Data = mv::utils::generateSequence<double>(256u * 512u);
    auto input1 = om.constant("", input1Data, {256, 512}, mv::DType("Float16"), mv::Order("HW"));
    auto input2 = om.constant("", input2Data, {256, 512}, mv::DType("Float16"), mv::Order("HW"));

    auto subtract = om.eltwise("", {input1, input2}, "Subtract");
    auto subtractOp = om.getSourceOp(subtract);
    auto output = om.output("", subtract);

    ASSERT_EQ(subtract->getShape(), mv::Shape({256, 512}));
    ASSERT_EQ(subtractOp->getOpType(), "Eltwise");
    ASSERT_EQ(subtract->attrsCount(), 6);
    ASSERT_EQ(subtractOp->attrsCount(), 2);
    ASSERT_EQ(subtractOp->inputSlots(), 2);
    ASSERT_EQ(subtractOp->outputSlots(), 1);
    ASSERT_TRUE(subtractOp->hasTypeTrait("executable"));

}
