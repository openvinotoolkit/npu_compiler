#include "include/mcm/base/attribute_registry.hpp"

namespace mv
{

    MV_DEFINE_REGISTRY(attr::AttributeRegistry, std::type_index, mv::attr::AttributeEntry)

}

const std::set<std::string> mv::attr::AttributeRegistry::typeTraits_ = 
{
    // Large content - this attribute type can potentailly store a large content
    "large"
};

const std::set<std::string> mv::attr::AttributeRegistry::instanceTraits_ = 
{
    // Read-only (constant) attribute
    "const"
};

mv::attr::AttributeRegistry& mv::attr::AttributeRegistry::instance()
{
    
    return Registry<AttributeRegistry, std::type_index, AttributeEntry>::instance();

}

bool mv::attr::AttributeRegistry::checkType(const std::type_index& typeID)
{
    return instance().find(typeID) != nullptr;
}

bool mv::attr::AttributeRegistry::checkType(const std::string& typeName)
{
    return checkType(getTypeID(typeName));
}

std::string mv::attr::AttributeRegistry::getTypeName(std::type_index typeID)
{
    if (!checkType(typeID))
    {
        throw AttributeError("AttributeRegistry", "Attempt of obtaining the name of an unregistered attribute type "
            + std::string(typeID.name()));
    }

    AttributeEntry* const attrPtr = instance().find(typeID);

    if (attrPtr)
    {
        return attrPtr->getTypeName();
    }

    throw MasterError("AttributeRegistry", "Registered attribute type " + std::string(typeID.name()) +
        " not found in the attribute registry");

}

std::type_index mv::attr::AttributeRegistry::getTypeID(const std::string& typeName)
{
    for (auto it = instance().reg_.begin(); it != instance().reg_.end(); ++it)
        if (it->second->getTypeName() == typeName)
            return it->first;
    throw AttributeError("AttributeRegistry", "Type ID undefined for an unregistered attribute type "
            + typeName);
}

bool mv::attr::AttributeRegistry::checkValue(std::type_index typeID, const Attribute& val, std::string& msg)
{

    if (!checkType(typeID))
    {
        throw AttributeError("AttributeRegistry", "Attempt of checking the value for an unregistered attribute type "
            + std::string(typeID.name()));
    }

    AttributeEntry* const attrPtr = instance().find(typeID);

    if (attrPtr)
    {
        if (!attrPtr->hasCheckFunc())
            return true;
        auto fcnPtr = attrPtr->getCheckFunc();
        return fcnPtr(val, msg);
    }

    throw MasterError("AttributeRegistry", "Registered attribute type " + std::string(typeID.name()) +
        " not found in the attribute registry");

}

const std::function<mv::json::Value(const mv::Attribute&)>& mv::attr::AttributeRegistry::getToJSONFunc(std::type_index typeID)
{

    if (!checkType(typeID))
    {
        throw AttributeError("AttributeRegistry", "Attempt of obtaining to-JSON conversion function for an unregistered attribute type "
            + std::string(typeID.name()));
    }

    AttributeEntry* const attrPtr = instance().find(typeID);

    if (attrPtr)
    {
        return attrPtr->getToJSONFunc();
    }

    throw MasterError("AttributeRegistry", "Registered attribute type " + std::string(typeID.name()) +
        " not found in the attribute registry");
}

const std::function<mv::Attribute(const mv::json::Value&)>& mv::attr::AttributeRegistry::getFromJSONFunc(std::type_index typeID)
{
    if (!checkType(typeID))
    {
        throw AttributeError("AttributeRegistry", "Attempt of obtaining from-JSON conversion function for an unregistered attribute type "
            + std::string(typeID.name()));
    }

    AttributeEntry* const attrPtr = instance().find(typeID);

    if (attrPtr)
    {
        return attrPtr->getFromJSONFunc();
    }

    throw MasterError("AttributeRegistry", "Registered attribute type " + std::string(typeID.name()) +
        " not found in the attribute registry");
}

const std::function<mv::Attribute(const mv::json::Value&)>& mv::attr::AttributeRegistry::getFromSimplifiedJSONFunc(std::type_index typeID)
{
    if (!checkType(typeID))
    {
        throw AttributeError("AttributeRegistry", "Attempt of obtaining from-Simplified-JSON conversion function for an unregistered attribute type "
            + std::string(typeID.name()));
    }

    AttributeEntry* const attrPtr = instance().find(typeID);

    if (attrPtr)
    {
        return attrPtr->getFromSimplifiedJSONFunc();
    }

    throw MasterError("AttributeRegistry", "Registered attribute type " + std::string(typeID.name()) +
        " not found in the attribute registry");
}

const std::function<mv::Attribute(const mv::json::Value&)>& mv::attr::AttributeRegistry::getFromJSONFunc(std::string typeName)
{
    return getFromJSONFunc(getTypeID(typeName));
}

const std::function<std::string(const mv::Attribute&)>& mv::attr::AttributeRegistry::getToStringFunc(std::type_index typeID, bool forceLong)
{

    if (!checkType(typeID))
    {
        throw AttributeError("AttributeRegistry", "Attempt of obtaining to-string conversion function for an unregistered attribute type "
            + std::string(typeID.name()));
    }

    AttributeEntry* const attrPtr = instance().find(typeID);

    if (attrPtr)
    {
        if (hasTypeTrait(typeID, "large") && forceLong)
            return attrPtr->getToLongStringFunc();
        return attrPtr->getToStringFunc();
    }

    throw MasterError("AttributeRegistry", "Registered attribute type " + std::string(typeID.name()) +
        " not found in the attribute registry");

}

const std::function<std::vector<uint8_t>(const mv::Attribute&)>& mv::attr::AttributeRegistry::getToBinaryFunc(std::type_index typeID)
{
    if (!checkType(typeID))
    {
        throw AttributeError("AttributeRegistry", "Attempt of obtaining from-binary conversion function for an unregistered attribute type "
            + std::string(typeID.name()));
    }

    AttributeEntry* const attrPtr = instance().find(typeID);

    if (attrPtr)
    {
        return attrPtr->getToBinaryFunc();
    }

    throw MasterError("AttributeRegistry", "Registered attribute type " + std::string(typeID.name()) +
        " not found in the attribute registry");
}


bool mv::attr::AttributeRegistry::checkTypeTrait(const std::string& typeTrait)
{
    if (typeTraits_.find(typeTrait) != typeTraits_.end())
        return true;
    return false;
}

bool mv::attr::AttributeRegistry::hasTypeTrait(std::type_index typeID, const std::string& trait)
{

    if (!checkTypeTrait(trait))
    {
        throw AttributeError("AttributeRegistry", "Attempt of checking an illegal type trait " + trait);
    }

    if (!checkType(typeID))
    {
        throw AttributeError("AttributeRegistry", "Attempt of checking type trait for an unregistered attribute type "
            + std::string(typeID.name()));
    }

    AttributeEntry* const attrPtr = instance().find(typeID);

    if (attrPtr)
    {
        return attrPtr->hasTypeTrait(trait);
    }

    throw MasterError("AttributeRegistry", "Registered attribute type " + std::string(typeID.name()) +
        " not found in the attribute registry");
}

bool mv::attr::AttributeRegistry::checkInstanceTrait(const std::string& trait)
{
    if (instanceTraits_.find(trait) != instanceTraits_.end())
        return true;
    return false;
}
