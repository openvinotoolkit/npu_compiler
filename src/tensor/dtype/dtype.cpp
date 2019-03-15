#include "include/mcm/tensor/dtype/dtype.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "include/mcm/tensor/binary_data.hpp"
#include "include/mcm/base/exception/dtype_error.hpp"
#include "include/mcm/tensor/dtype/dtype_registry.hpp"

mv::DType::DType():
DType("Float16")
{

}

mv::DType::DType(const DType& other) :
dType_(other.dType_)
{

}

mv::DType::DType(const std::string& value)
{
    if(!mv::DTypeRegistry::checkDType(value))
        throw DTypeError(*this, "Invalid string passed for DType construction " + value);
    dType_ = value;
}

std::string mv::DType::toString() const
{
    return dType_;
}

mv::BinaryData mv::DType::toBinary(const std::vector<DataElement>& data) const
{
    const std::function<mv::BinaryData(const std::vector<DataElement>&)>& func = mv::DTypeRegistry::getToBinaryFunc(dType_);
    return func(data);
}

unsigned mv::DType::getSizeInBits() const
{
    return mv::DTypeRegistry::getSizeInBits(dType_);
}

bool mv::DType::isDoubleType() const
{
    return mv::DTypeRegistry::isDoubleType(dType_);
}

mv::DType& mv::DType::operator=(const DType& other)
{
    dType_ = other.dType_;
    return *this;
}

bool mv::DType::operator==(const DType &other) const
{
    return dType_ == other.dType_;
}

bool mv::DType::operator!=(const DType &other) const
{
    return !operator==(other);
}

std::string mv::DType::getLogID() const
{
    return "DType:" + toString();
}
