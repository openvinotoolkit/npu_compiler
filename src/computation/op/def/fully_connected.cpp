#include "include/mcm/computation/op/def/fully_connected.hpp"

mv::op::FullyConnected::FullyConnected(const std::string &name) :
ComputationOp(OpType::FullyConnected, name),
SinkOp(OpType::FullyConnected, 2, name),
SourceOp(OpType::FullyConnected, 1, name)
{
    set<bool>("executable", true);
}


mv::Tensor mv::op::FullyConnected::getOutputDef(std::size_t idx)
{
    
    // Will throw on error
    validOutputDef_(idx);

    auto input0 = getInputTensor(0);
    auto input0Shape = input0->getShape(); 
    auto input1Shape = getInputTensor(1)->getShape();

    if (input1Shape.ndims() != 2)
        throw(OpError(*this, "Invalid shape of the weights tensor (input 1) - must have a dimensionality of 2, "
            " has " + std::to_string(input1Shape.ndims())));

    if (input0Shape.totalSize() != input1Shape[0])
        throw(OpError(*this, "Inconsistent total size of input tensor (input 0) " + std::to_string(input0Shape.totalSize()) + 
            " and 1st dimension of weights tensor (input 1) " + std::to_string(input1Shape[0])));

    Order outputOrder(Order::getColMajorID(2));
    if(input0->getOrder().isRowMajor())
        outputOrder = Order(Order::getRowMajorID(2));
    else if(input0->getOrder().isRowMajorPlanar())
        outputOrder = Order(Order::getRowMajorPlanarID(2));
    return Tensor(name_ + ":0", {1, input1Shape[1]}, input0->getDType(), outputOrder);
    
}

bool mv::op::FullyConnected::isHardwarizeable(json::Object&)
{
    return false;
}
