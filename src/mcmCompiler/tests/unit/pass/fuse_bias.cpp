//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"

TEST(fuse_bias, case_conv)
{

    mv::OpModel om("testModel");
    auto input = om.input("", {64, 64, 16, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3 * 3 * 16 * 32);
    auto weights = om.constant("weights", weightsData, {3, 3, 16, 32}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(4)));
    auto conv = om.conv("", input, weights, {1, 1}, {1, 1, 1, 1}, 1);
    auto convOp = om.getSourceOp(conv);
    std::vector<double> biasesData = mv::utils::generateSequence<double>(32);
    auto biases = om.constant("biases", biasesData, {32}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(1)));
    auto bias = om.bias("", conv, biases);
    auto biasOp = om.getSourceOp(bias);
    om.output("", bias);
    
    auto outputOp = biasOp.leftmostChild();

    mv::Element dummyPassDesc("");
    mv::TargetDescriptor dummyTargDesc;
    mv::Element compOutput("CompilationOutput");

    mv::pass::PassRegistry::instance().find("FuseBias")->run(om, dummyTargDesc, dummyPassDesc, compOutput);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 4);
    ASSERT_EQ(dm.tensorsCount(), 4);

    // Check predecessing operation
    ASSERT_EQ(convOp.childrenSize(), 1);

    for (unsigned i = 0; i < dm.getTensor(convOp->get<std::string>("bias"))->getDoubleData().size(); ++i)
        ASSERT_FLOAT_EQ(dm.getTensor(convOp->get<std::string>("bias"))->getDoubleData()[i], biasesData[i]);

}
