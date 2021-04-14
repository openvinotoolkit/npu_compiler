#include "include/mcm/tensor/dtype/dtype.hpp"
#include "include/mcm/base/exception/dtype_error.hpp"
#include "include/mcm/tensor/dtype/dtype_registry.hpp"

mv::DType::DType():
DType("Default")
{

}

mv::DType::DType(const DType& other) :
dType_(other.dType_)
{

}

mv::DType::DType(const std::string& value)
{
    dType_ = value;
}

std::string mv::DType::toString() const
{
    return dType_;
}

unsigned mv::DType::getSizeInBits() const
{
    return mv::DTypeRegistry::getSizeInBits(dType_);
}

unsigned mv::DType::getSizeInBytes() const
{
    return getSizeInBits() / 8;
}

bool mv::DType::isDoubleType() const
{
    return mv::DTypeRegistry::isDoubleType(dType_);
}

bool mv::DType::isSigned() const
{
    return mv::DTypeRegistry::isSigned(dType_);
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

bool mv::DType::operator<(const DType &other) const
{
    return dType_ < other.dType_;
}

std::string mv::DType::getLogID() const
{
    return "DType:" + toString();
}
