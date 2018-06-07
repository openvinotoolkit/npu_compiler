#ifndef CONV_HPP_
#define CONV_HPP_

#include "include/fathom/computation/op/kernel_op.hpp"
#include "include/fathom/computation/op/multisink_op.hpp"

namespace mv
{
    /// \todo Add assertions (dimensions)   
    class Conv : public KernelOp, public MultiSinkOp
    {

    public:

        Conv(const Logger &logger, byte_type strideX, byte_type strideY, byte_type padX, byte_type padY, const string &name) :
        ComputationOp(logger, "conv", name),
        KernelOp(logger, "conv", strideX, strideY, padX, padY, name),
        MultiSinkOp(logger, "conv", 2, name)
        {

        }

        UnpopulatedTensor getOutputDef()
        {

            if (!validOutputDef_())
                return UnpopulatedTensor(logger_);

            auto input = getInput(0);
            auto inputShape = input->getShape();
            auto weights = getInput(1);
            auto weightsShape = weights->getShape();

            if (inputShape.ndims() != 4)
            {
                logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because of incorrect shape " + inputShape.toString() + " of input");
                return UnpopulatedTensor(logger_);
            }
            
            if (weightsShape.ndims() != 4)
            {
                logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because of incorrect shape " + inputShape.toString() + " of weights");
                return UnpopulatedTensor(logger_);
            }

            if (inputShape[3] != weightsShape[2])
            {
                logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because of mismatch in channels dimensions between input (" + Printable::toString(inputShape[3])
                         + ") and weights (" + Printable::toString(weightsShape[2]) + ")");
                return UnpopulatedTensor(logger_);
            }

            auto padX = getAttr("padX").getContent<byte_type>();
            auto padY = getAttr("padY").getContent<byte_type>();
            auto strideX = getAttr("strideX").getContent<byte_type>();
            auto strideY = getAttr("strideY").getContent<byte_type>();

            Shape outputShape(inputShape[0], getOutputDim_(inputShape[1], weightsShape[0], padX, strideX), getOutputDim_(inputShape[2], weightsShape[1], padY, strideY), weightsShape[3]);

            return UnpopulatedTensor(logger_, getOutputName(), outputShape, input->getDType(), input->getOrder());

        }

    };

}

#endif // CONV_HPP_