#include "gtest/gtest.h"
#include "include/fathom/computation/model/op_model.hpp"
#include "include/fathom/computation/model/data_model.hpp"

TEST(computation_model, data_model_construction)
{

    mv::OpModel om;

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto inIt = om.input(mv::Shape(1, 32, 32, 1), mv::DType::Float, mv::Order::NWHC);
    mv::vector<mv::float_type> weightsData =
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
     15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f};
    mv::ConstantTensor weights(mv::Shape(1, 3, 3, 3), mv::DType::Float, mv::Order::NWHC, weightsData);
    auto convIt = om.conv2D(inIt, weights, 4, 4, 1, 1);
    auto outIt = om.output(convIt);

    mv::DataModel dm(om);

    // Check if model is valid
    ASSERT_TRUE(dm.isValid());

    // Check input and output tensor shape
    ASSERT_EQ((*dm.getInput()).getTensor().getAttr<

    std::cout << "Input op: " << (*om.getInput()).getName() << std::endl;
    std::cout << "Input tensor (output tensor of the input op): " << (*dm.getInput()).getTensor().getName() << std::endl;
    std::cout << "Output op: " << (*om.getOutput()).getName() << std::endl;
    std::cout << "Output tensor (input tensor of the output op): " << (*dm.getOutput()).getTensor().getName() << std::endl;

}