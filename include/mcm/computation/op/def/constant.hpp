#ifndef CONSTANT_HPP_
#define CONSTANT_HPP_

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/computation/tensor/tensor.hpp"

namespace mv
{

    namespace op
    {

        class Constant : public SourceOp
        {

            dynamic_vector<float_type> data_;

        public:

            Constant(const dynamic_vector<float_type> &data, const Shape &shape, DType dType, Order order, const string &name);
            Constant(mv::json::Value &obj);
            Tensor getOutputDef(byte_type idx);
            mv::json::Value toJsonValue() const;
            bool isHardwarizeable(mv::json::Object& TargetDescriptor);

        };

    }

}

#endif // CONSTANT_HPP_
