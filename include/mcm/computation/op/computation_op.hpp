#ifndef COMPUTATION_OP_HPP_
#define COMPUTATION_OP_HPP_

#include <string>
#include "include/mcm/base/element.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/computation/op/op_type.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"

namespace mv
{

    class ComputationOp : public Element
    {

        OpType opType_;

    protected:

        void validOutputDef_(std::size_t idx);

    public:

        ComputationOp(OpType opType, const std::string& name);
        ComputationOp(json::Value& value);
        virtual ~ComputationOp() = 0;

        OpType getOpType() const;
        std::string toString() const override;

        virtual bool setInputTensor(Data::TensorIterator& tensor, std::size_t idx);
        virtual bool setOutputTensor(Data::TensorIterator& tensor, std::size_t idx);
        virtual Data::TensorIterator getInputTensor(std::size_t idx);
        virtual Data::TensorIterator getOutputTensor(std::size_t idx);
        virtual bool hasInputDef();
        virtual bool hasInputDef(std::size_t idx);
        virtual Tensor getOutputDef(std::size_t idx) = 0;
        virtual std::size_t inputSlots();
        virtual std::size_t outputSlots();
        bool isExecutable() const;
        virtual bool isHardwarizeable(mv::json::Object& TargetDescriptor) = 0;
        bool operator==(const ComputationOp &other) const;
        virtual std::string getLogID() const override;

    };

}

#endif // COMPUTATION_OP_HPP_
