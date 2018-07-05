#ifndef MV_JSON_TEXT_PARSER_HPP_
#define MV_JSON_TEXT_PARSER_HPP_

#include <fstream>
#include <string>
#include <algorithm>
#include <queue>
#include "include/mcm/base/json/json.hpp"
#include "include/mcm/utils/parser/exception/argument_error.hpp"
#include "include/mcm/utils/parser/exception/parsing_error.hpp"
#include <stack>
#include <map>
#include <regex>

namespace mv
{

    class JSONTextParser
    {

        enum class JSONSymbol
        {
            LBrace,
            RBrace,
            LBracket,
            RBracket,
            Colon,
            Comma,
            String,
            Number,
            True,
            False,
            Null,
            EOFSymbol,
            Invalid
        };

        enum class ParserState
        {
            Start,
            ObjectInit,
            MemberKey,
            KeyValueDelim,
            MemeberValue,
            MemberDelim,
            ObjectFinish,
            ArrayInit,
            Element,
            ElementDelim,
            ArrayFinish
        };

        static const std::map<ParserState, std::map<JSONSymbol, ParserState>> pushdownAutomata_;

        char *buffer_;
        std::string bufferStr_;
        unsigned bufferLength_;
        std::ifstream inputStream_;

        unsigned readStream_();
        std::pair<JSONSymbol, std::string> lexer_();

    public:

        JSONTextParser(unsigned bufferLength = 64);
        bool parseFile(const std::string& fileName, json::Value& outputObject);

    };

}

#endif // MV_PARSER_JSON_TEXT_HPP_