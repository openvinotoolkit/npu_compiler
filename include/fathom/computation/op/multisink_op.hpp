#ifndef MULTISINK_OP_HPP_
#define MULTISINK_OP_HPP_

#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class MultiSinkOp : public virtual ComputationOp
    {
        
        vector<TensorContext::TensorIterator> inputs_;

    public:

        MultiSinkOp(const string &opType, byte_type inputsCount, const string &name);
        virtual ~MultiSinkOp() = 0;
        virtual bool setInput(TensorContext::TensorIterator &tensor, byte_type idx);
        virtual TensorContext::TensorIterator getInput(byte_type idx);
        bool hasInputDef();
        bool hasInputDef(byte_type idx);
        byte_type inputSlots();

    };

}

#endif // MULTISINK_OP_HPP_