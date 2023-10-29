from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from io import TextIOWrapper
import argparse
import typing
import enum
import sys
import re
import os

VERSION = "0.0.1"
RT = typing.TypeVar('RT')  # return type


class TokenType(enum.Enum):
    COMMENT = re.compile(r"#.*(?=\n)?")
    IDENTIFIER = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
    EQUAL = re.compile(r"=")
    PIPE = re.compile(r"\|")
    AMPERSAND = re.compile(r"&")
    EXCLAMATION_MARK = re.compile(r"!")
    STAR = re.compile(r"\*")
    PLUS = re.compile(r"\+")
    QUESTION_MARK = re.compile(r"\?")
    LPAR = re.compile(r"\(")
    RPAR = re.compile(r"\)")
    LCBRACKET = re.compile(r"{")
    RCBRACKET = re.compile(r"}")
    PERCENT = re.compile(r"%")
    COLON = re.compile(r":")
    STRING = re.compile(r"(?<!\\)\".+?(?<!\\)\"")
    CHARACTER_CLASS = re.compile(r"(?<!\\)\[.*?(?<!\\)\]")
    # special tokens
    CODE_SECTION = enum.auto()
    ACTION = enum.auto()

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f'{cls_name}.{self.name}'


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int


class Tokenizer:
    def __init__(self, file: typing.TextIO):
        self.filename = os.path.basename(file.name)
        self.src = file.read()
        self.tokens: list[Token] = []
        self.pos = 0

    def tokenize(self) -> list[Token]:
        if self.tokens:
            return self.tokens

        while self.pos < len(self.src):
            if self.src[self.pos].isspace():
                self.pos += 1
                continue

            # special token processing
            elif len(self.tokens) >= 2 and \
                    self.tokens[-2].type == TokenType.PERCENT and self.tokens[-1].type == TokenType.IDENTIFIER and \
                    ((name := self.tokens[-1].value) == "cpp" or name == "hpp"):
                start_pos = self.pos

                while self.src[self.pos].isspace():
                    self.pos += 1

                open_brackets = 0
                if self.peek() == "{":
                    self.pos += 1
                    code_start = self.pos
                    open_brackets += 1
                    while self.pos < len(self.src):
                        if self.peek() == "{":
                            open_brackets += 1
                        elif self.peek() == "}":
                            open_brackets -= 1
                        self.pos += 1
                        if open_brackets == 0:
                            action_end = self.pos - 1
                            value = self.src[code_start:action_end]
                            line_number = self.calc_line(start_pos)
                            col = self.calc_column(start_pos, line_number)
                            self.tokens.append(Token(
                                TokenType.CODE_SECTION,
                                value,
                                line_number,
                                col))
                            break
                    else:
                        self.error("'}' is expected")
                else:
                    self.error("'{' is expected")
                continue
            elif self.src[self.pos] == "{":
                open_brackets = 0
                self.pos += 1
                action_start = self.pos
                open_brackets += 1
                while self.pos < len(self.src):
                    if self.peek() == "{":
                        open_brackets += 1
                    elif self.peek() == "}":
                        open_brackets -= 1
                    self.pos += 1
                    if open_brackets == 0:
                        action_end = self.pos - 1
                        value = self.src[action_start:action_end]
                        line_number = self.calc_line(action_start)
                        col = self.calc_column(action_start, line_number)
                        self.tokens.append(Token(
                            TokenType.ACTION,
                            value,
                            line_number,
                            col))
                        break
                else:
                    self.error("'}' is expected")
                continue

            for token_type in TokenType:
                # skip special tokens
                if token_type in (TokenType.CODE_SECTION, TokenType.ACTION):
                    continue
                if result := token_type.value.match(self.src, self.pos):
                    self.pos = result.end()
                    self.add_token(token_type, result.group())
                    break
            else:
                self.error(f"unknown character '{self.src[self.pos]}'")

        return self.tokens

    def peek(self):
        if self.pos >= len(self.src):
            return None
        return self.src[self.pos]

    def calc_line(self, position: int) -> int:
        return self.src.count("\n", 0, position) + 1

    def calc_column(self, position: int, line: int) -> int:
        lines = self.src.splitlines(keepends=True)
        col = position if line == 1 else position - sum(map(len, lines[:line - 1]))
        return col + 1

    def add_token(self, token_type: TokenType, value: str = ""):
        if token_type == TokenType.COMMENT:
            return
        position = self.pos - len(value)
        line = self.calc_line(position)
        col = self.calc_column(position, line)
        self.tokens.append(Token(token_type, value, line, col))

    def error(self, message):
        line_number = self.src.count("\n", 0, self.pos)
        lines = self.src.splitlines(keepends=True)
        col = self.pos + 1 if line_number == 0 else self.pos - sum(map(len, lines[:line_number])) + 1
        print(f"{self.filename}:{line_number + 1}:{col}: {message}", file=sys.stderr)
        sys.exit(1)


class Node():
    pass


@dataclass
class BlockStatementNode(Node):
    statements: list[Node]


@dataclass
class NameNode(Node):
    name: str


@dataclass
class HeaderBlockNode(Node):
    header: str


@dataclass
class CodeBlockNode(Node):
    code: str


@dataclass
class RuleTypeNode(Node):
    type_name: str


@dataclass
class RootRuleNode(Node):
    name: str


@dataclass
class ParsingExpressionContext:
    name: str | None = None
    lookahead: bool = False
    lookahead_positive: bool | None = None
    loop: bool = False
    loop_nonempty: bool | None = None
    optional: bool = False


@dataclass(kw_only=True)
class ParsingExpressionNode:
    ctx: ParsingExpressionContext = field(default_factory=ParsingExpressionContext)


@dataclass
class ParsingExpressionRuleNameNode(ParsingExpressionNode):
    name: str


@dataclass
class ParsingExpressionStringNode(ParsingExpressionNode):
    value: str


@dataclass
class ParsingExpressionGroupNode(ParsingExpressionNode):
    parsing_expression: list[list[ParsingExpressionNode]]


@dataclass
class ParsingExpressionCharacterClassNode(ParsingExpressionNode):
    characters: str


@dataclass
class ParsingExpressionsNode(Node):
    items: list[ParsingExpressionNode]
    action: str | None


@dataclass
class RuleNode(Node):
    name: str
    parsing_expression: list[ParsingExpressionsNode]


class ParsingFail(Exception):
    pass


class ParserManager(AbstractContextManager):
    pos: int = 0

    def __init__(self, parser: "Parser"):
        self.parser = parser

    def __enter__(self):
        self.pos = self.parser.mark()

    def __exit__(self, type, value, traceback) -> bool:
        if type is not None:
            if type == ParsingFail:
                self.parser.reset(self.pos)
            else:
                raise
        return True


class Parser:
    def __init__(self, tokenizer: Tokenizer):
        self.filename = tokenizer.filename
        self.tokens = tokenizer.tokenize()
        self.pos = 0

    def parse(self) -> BlockStatementNode:
        return self.root_block()

    def root_block(self):
        statements = []
        while True:
            try:
                statements.append(self.statement())
            except ParsingFail:
                break
        if self.pos != len(self.tokens):
            self.error("parsing fail")
        return BlockStatementNode(statements)

    def statement(self):
        with self.manager:
            return self.name_statement()
        with self.manager:
            return self.header_statement()
        with self.manager:
            return self.code_statement()
        with self.manager:
            return self.rule_type_statement()
        with self.manager:
            return self.root_rule_statement()
        with self.manager:
            return self.rule_statement()
        raise ParsingFail

    def name_statement(self):
        with self.manager:
            self.match(TokenType.PERCENT)
            if self.match(TokenType.IDENTIFIER) == "name":
                return NameNode(self.match(TokenType.IDENTIFIER))
        raise ParsingFail

    def header_statement(self):
        with self.manager:
            self.match(TokenType.PERCENT)
            if self.match(TokenType.IDENTIFIER) == "hpp":
                return HeaderBlockNode(self.match(TokenType.CODE_SECTION))
        raise ParsingFail

    def code_statement(self):
        with self.manager:
            self.match(TokenType.PERCENT)
            if self.match(TokenType.IDENTIFIER) == "cpp":
                return CodeBlockNode(self.match(TokenType.CODE_SECTION))
        raise ParsingFail

    def rule_type_statement(self):
        with self.manager:
            self.match(TokenType.PERCENT)
            if self.match(TokenType.IDENTIFIER) == "type":
                return RuleTypeNode(self.match(TokenType.STRING)[1:-1])
        raise ParsingFail

    def root_rule_statement(self):
        with self.manager:
            self.match(TokenType.PERCENT)
            if self.match(TokenType.IDENTIFIER) == "root":
                return RootRuleNode(self.match(TokenType.IDENTIFIER))
        raise ParsingFail

    def parsing_expression_atom(self):
        with self.manager:
            id = self.match(TokenType.IDENTIFIER)
            self.lookahead(False, TokenType.EQUAL)
            return ParsingExpressionRuleNameNode(id)
        with self.manager:
            return ParsingExpressionStringNode(self.match(TokenType.STRING)[1:-1])
        with self.manager:
            self.match(TokenType.LPAR)
            parsing_expressions = self.loop(True, self.parsing_expression_)
            group = ParsingExpressionGroupNode(parsing_expressions)
            self.match(TokenType.RPAR)
            return group
        with self.manager:
            return ParsingExpressionCharacterClassNode(self.match(TokenType.CHARACTER_CLASS)[1:-1])
        raise ParsingFail

    def parsing_expression_item(self):
        with self.manager:
            atom = self.parsing_expression_atom()
            self.match(TokenType.PLUS)
            atom.ctx.loop = True
            atom.ctx.loop_nonempty = True
            return atom
        with self.manager:
            atom = self.parsing_expression_atom()
            self.match(TokenType.STAR)
            atom.ctx.loop = True
            atom.ctx.loop_nonempty = False
            return atom
        with self.manager:
            atom = self.parsing_expression_atom()
            self.match(TokenType.QUESTION_MARK)
            atom.ctx.optional = True
            return atom
        with self.manager:
            return self.parsing_expression_atom()
        with self.manager:
            self.match(TokenType.AMPERSAND)
            atom = self.parsing_expression_atom()
            atom.ctx.lookahead = True
            atom.ctx.lookahead_positive = True
            return atom
        with self.manager:
            self.match(TokenType.EXCLAMATION_MARK)
            atom = self.parsing_expression_atom()
            atom.ctx.lookahead = True
            atom.ctx.lookahead_positive = False
            return atom
        raise ParsingFail

    def parsing_expression_named_item_or_item(self) -> ParsingExpressionNode:
        with self.manager:
            id = self.match(TokenType.IDENTIFIER)
            self.match(TokenType.COLON)
            item = self.parsing_expression_item()
            item.ctx.name = id
            return item
        with self.manager:
            return self.parsing_expression_item()
        raise ParsingFail

    def parsing_expression_(self):
        with self.manager:
            return self.loop(True, self.parsing_expression_named_item_or_item)
        with self.manager:
            self.match(TokenType.PIPE)
            return self.loop(True, self.parsing_expression_named_item_or_item)
        raise ParsingFail

    def parsing_expression(self):
        with self.manager:
            parsing_expression = self.parsing_expression_()
            action = self.optional(TokenType.ACTION)
            return ParsingExpressionsNode(parsing_expression, action)
        raise ParsingFail

    def rule_statement(self):
        with self.manager:
            rule_name = self.match(TokenType.IDENTIFIER)
            self.match(TokenType.EQUAL)
            parsing_expressions = self.loop(True, self.parsing_expression)
            return RuleNode(rule_name, parsing_expressions)
        raise ParsingFail

    @property
    def manager(self):
        return ParserManager(self)

    def mark(self):
        return self.pos

    def reset(self, pos):
        self.pos = pos

    def get_token(self) -> Token | None:
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            return token
        return None

    def match(self, token_type: TokenType) -> str:
        token = self.get_token()
        if token and token.type == token_type:
            self.pos += 1
            return token.value
        raise ParsingFail

    def optional(self, token_type: TokenType) -> str | None:
        token = self.get_token()
        if token and token.type == token_type:
            self.pos += 1
            return token.value
        return None

    def loop(self, nonempty, func: typing.Callable[..., RT], *args) -> list[RT]:
        nodes = []
        try:
            while True:
                node = func(*args)
                nodes.append(node)
        except ParsingFail:
            pass
        if len(nodes) >= nonempty:
            return nodes
        raise ParsingFail

    def lookahead(self, positive: bool, token_type: TokenType):
        if self.pos < len(self.tokens):
            foo = self.tokens[self.pos].type == token_type
            if foo == positive:
                return
        elif not positive:
            return
        raise ParsingFail

    def error(self, message):
        token = self.tokens[self.pos]
        print(f"{self.filename}:{token.line}:{token.col}: {message}, token: \"{token.value}\"", file=sys.stderr)
        sys.exit(1)


def write_lines(file: TextIOWrapper, *lines: str):
    for line in lines:
        file.write(line)
        file.write("\n")


def add_indent(string: str, indent: int) -> str:
    lines = string.split("\n")
    new_lines = []

    for line in lines:
        new_lines.append((' ' * indent + line) if line.strip() else '')

    return '\n'.join(new_lines)


class CodeGenerator:
    cpp_file: TextIOWrapper
    hpp_file: TextIOWrapper

    def __init__(self, root_node: BlockStatementNode, filename: str):
        self.root_node = root_node
        self.parser_name = filename
        self.header_from_directive = ""
        self.code_from_directive = ""
        self.rule_type = "size_t"
        self.root_rule = ""

        if name_node := self.check_count_and_get_node(NameNode):
            self.parser_name = name_node.name

        if header_node := self.check_count_and_get_node(HeaderBlockNode):
            self.header_from_directive = header_node.header

        if code_node := self.check_count_and_get_node(CodeBlockNode):
            self.code_from_directive = code_node.code

        if rule_type_node := self.check_count_and_get_node(RuleTypeNode):
            self.rule_type = rule_type_node.type_name

        if root_rule_node := self.check_count_and_get_node(RootRuleNode):
            self.root_rule = root_rule_node.name

    def check_count_and_get_node(self, node_type: typing.Type[RT]) -> RT | None:
        node = [node for node in self.root_node.statements if isinstance(node, node_type)]
        if len(node) > 1:
            name = ""
            if node_type == NameNode:
                name = "name"
            elif node_type == CodeBlockNode:
                name = "cpp"
            elif node_type == RuleTypeNode:
                name = "type"
            elif node_type == RootRuleNode:
                name = "root"
            print(f"there can be only one %{name} directive", file=sys.stderr)
            exit(1)
        if len(node):
            return node[0]
        return None

    def start(self):
        self.cpp_file = open(f"{self.parser_name}.cpp", "w", encoding="utf-8")
        self.hpp_file = open(f"{self.parser_name}.hpp", "w", encoding="utf-8")

        write_lines(
            self.cpp_file,
            f"#include \"{self.parser_name}.hpp\"",
            "",
            "#include <algorithm>",
            "",

        )

        if self.code_from_directive:
            write_lines(
                self.cpp_file,
                "// code from %code",
                self.code_from_directive,
                "// end %code",
                "",
            )

        write_lines(
            self.cpp_file,
            "namespace PParser",
            "{",
            "",
            "    ////////// BEGINNING OF RULES //////////",
            "",
        )

        write_lines(
            self.hpp_file,
            "#ifndef PPARSER_HPP_",
            "#define PPARSER_HPP_",
            "",
            "#include <string_view>",
            "",
        )

        if self.header_from_directive:
            write_lines(
                self.hpp_file,
                "",
                "// code from %hpp",
                self.header_from_directive,
                "// end %hpp",
            )

        write_lines(
            self.hpp_file,
            "",
            "namespace PParser",
            "{",
            "",
            f"    using ExprResult = {self.rule_type};",
            "",
            "    struct Token",
            "    {",
            "        std::string value;",
            "        size_t firstLine;",
            "        size_t firstColumn;",
            "        size_t lastLine;",
            "        size_t lastColumn;",
            "    };",
            "",
            "    class Parser",
            "    {",
            "    private:",
            "        const std::string_view src;",
            "        size_t position = 0;",
            "",
            "        ////////// BEGINNING OF RULES //////////",
        )

        for statement in self.root_node.statements:
            self.generate(statement)

        write_lines(
            self.cpp_file,
            "    ////////// END OF RULES //////////",
            "",
            "    Token Parser::newToken(std::string_view value)",
            "    {",
            "        Token token;",
            "        token.value = std::move(value);",
            "        token.firstLine = calcLine(position);",
            "        token.firstColumn = calcColumn(position, token.firstLine);",
            "        size_t endPosition = position + value.size();",
            "        token.lastLine = calcLine(endPosition);",
            "        token.lastColumn = calcColumn(endPosition, token.lastLine);",
            "        return token;",
            "    }",
            "",
            "    Token Parser::newToken(size_t startOfToken)",
            "    {",
            "        return newToken(src.substr(startOfToken, position));",
            "    }",
            "",
            "    size_t Parser::calcLine(size_t position)",
            "    {",
            "        return std::count(src.begin(), src.begin() + position, '\\n') + 1;",
            "    }",
            "",
            "    size_t Parser::calcColumn(size_t position, size_t line)",
            "    {",
            "        if (line == 1)",
            "        {",
            "            return position + 1;",
            "        }",
            "       auto it = std::find(src.rbegin() + (src.size() - position), src.rend(), '\\n');",
            "       size_t startLinePosition = std::distance(it, src.rend()) - 1;",
            "",
            "        return position - startLinePosition + 1;",
            "    }",
            "",
            "    ExprResult Parser::parse()",
            "    {",
            "        this->position = 0;",
            f"        return {self.root_rule}();",
            "    }",
            "",
            "    Parser::Parser(std::string_view src) : src(src) {}",
            "",
            "}",
        )

        write_lines(
            self.hpp_file,
            "        ////////// END OF RULES //////////",
            "",
            "        Token newToken(std::string_view value);",
            "        Token newToken(size_t startOfToken);",
            "        size_t calcLine(size_t position);",
            "        size_t calcColumn(size_t position, size_t line);",
            "",
            "    public:",
            "        explicit Parser(std::string_view src);",
            "",
            "        ExprResult parse();",
            "    };",
            "}",
            "",
            "#endif // PPARSER_HPP_",
        )

    def generate(self, node):
        match node:
            case NameNode() | HeaderBlockNode() | CodeBlockNode() | RuleTypeNode() | RootRuleNode():
                pass
            case RuleNode():
                if not self.root_rule:
                    self.root_rule = node.name
                self.gen_Rule(node)
            case _:
                self.gen_type_error(node)

    def gen_type_error(self, node: Node):
        print(f"generator for node with type <{type(node).__name__}> not implemented", file=sys.stderr)
        exit(1)

    def gen_Rule(self, node: RuleNode):
        write_lines(self.hpp_file, f"        bool {node.name}();")
        write_lines(
            self.cpp_file,
            f"    bool Parser::{node.name}()",
            "    {",
        )
        code = "auto __mark = this->position;\n"
        for i, parsing_expression in enumerate(node.parsing_expression):
            if i > 0:
                code += f"NEXT_{i}:\n"
                code += "this->position = __mark;\n"
            next = f"NEXT_{i + 1}" if i + 1 < len(node.parsing_expression) else "FAIL"
            code += self.gen_ParsingExpression(parsing_expression, next)
            code += "\n"
        self.cpp_file.write(add_indent(code, 8))
        write_lines(
            self.cpp_file,
            "    FAIL:",
            "        this->position = __mark;",
            "        return false;",
            "    SUCCESS:",
            "        return true;",
        )
        write_lines(self.cpp_file, "    }", "")

    def gen_ParsingExpression(self, node: ParsingExpressionsNode, next):
        code = "{\n"
        for i in node.items:
            match i:
                case ParsingExpressionRuleNameNode():
                    code += add_indent(self.gen_ParsingExpressionRuleName(i, next), 4)
                case ParsingExpressionStringNode():
                    code += add_indent(self.gen_ParsingExpressionStringNode(i, next), 4)
                case ParsingExpressionCharacterClassNode():
                    code += add_indent(self.gen_ParsingExpressionCharacterClassNode(i, next), 4)
                case _:
                    print(node)
                    self.gen_type_error(node)
            code += "\n"
        code += "    goto SUCCESS;\n"
        code += "}\n"
        return code

    def gen_ParsingExpressionRuleName(self, node: ParsingExpressionRuleNameNode, next: str):
        code = ""

        if node.ctx.lookahead:
            code += "{\n"
            code += "   size_t tempMark = position;\n"
            code += f"   if({'!' if node.ctx.lookahead_positive else ''}("
            if node.ctx.name:
                code += f"auto {node.ctx.name} = "
            code += f"{node.name}())) goto {next};\n"
            code += "   position = tempMark;\n"
            code += "}\n"
        elif node.ctx.optional:
            if node.ctx.name:
                code += f"auto {node.ctx.name} = "
            code += f"({node.name}());\n"
        elif node.ctx.loop:
            # TODO: implement storage of the result in a variable(node.ctx.name)
            assert node.ctx.name is None, "Not implemented yet"

            code += "{\n"
            if node.ctx.loop_nonempty:
                code += "    size_t i = 0;\n"
            code += "    for (;;)\n"
            code += "    {\n"
            code += f"        if (!({node.name}())) break;\n"
            if node.ctx.loop_nonempty:
                code += "        i++;\n"
            code += "    }\n"
            if node.ctx.loop_nonempty:
                code += f"\n    if (!i) goto {next};\n"
            code += "}\n"
        else:
            code += "if (!("
            if node.ctx.name:
                code += f"auto {node.ctx.name} = "
            code += f"{node.name}())) goto {next};\n"
        return code

    def gen_ParsingExpressionStringNode(self, node: ParsingExpressionStringNode, next: str):
        # TODO: implement storage of the result in a variable(node.ctx.name)
        assert node.ctx.name is None, "Not implemented yet"

        code = ""
        str_len = len(node.value)
        str_condition = ""
        for i, ch in enumerate(node.value):
            str_condition += f"   && this->src[this->position + {i}] == '{ch}'\n"

        if node.ctx.lookahead:
            if node.ctx.lookahead_positive:
                code += f"if (this->position + {str_len - 1} > this->src.size()) goto {next};\n"
                code += "if (!(true\n"
                code += str_condition
                code += f")) goto {next};\n"
            else:
                code += f"if (this->position + {str_len - 1} < this->src.size())\n"
                code += "{\n"
                code += "    if(true\n"
                code += add_indent(str_condition, 4)
                code += f"    ) goto {next};\n"
                code += "}\n"
        elif node.ctx.optional:
            code += f"if (this->position + {str_len - 1} <= this->src.size())\n"
            code += "{\n"
            code += "    if ((true\n"
            code += add_indent(str_condition, 4)
            code += f"    )) this->position += {str_len};\n"
            code += "}\n"
        elif node.ctx.loop:
            code += "{\n"
            if node.ctx.loop_nonempty:
                code += "    size_t i = 0;\n"
            code += "    for (;;)\n"
            code += "    {\n"
            code += f"        if (this->position + {str_len - 1} > this->src.size()) break;\n"
            code += "        if (!(true\n"
            code += add_indent(str_condition, 8)
            code += "        )) break;\n"
            code += f"        this->position += {str_len};\n"
            if node.ctx.loop_nonempty:
                code += "        i++;\n"
            code += "    }\n"
            if node.ctx.loop_nonempty:
                code += f"\n    if (!i) goto {next};\n"
            code += "}\n"
        else:
            code += f"if (this->position + {str_len - 1} > this->src.size()) goto {next};\n"
            code += "if (!(true\n"
            code += str_condition
            code += f")) goto {next};\n"
            code += f"this->position += {str_len};\n"
        return code

    def gen_ParsingExpressionCharacterClassNode(self, node: ParsingExpressionCharacterClassNode, next: str):
        # TODO: implement storage of the result in a variable(node.ctx.name)
        assert node.ctx.name is None, "Not implemented yet"

        code = ""
        condition = ""
        for ch in node.characters:
            condition += f"    || this->src[this->position] == '{ch}'\n"

        if node.ctx.lookahead:
            if node.ctx.lookahead_positive:
                code += f"if (this->position > this->size()) goto {next};\n"
                code += "if (!(false\n"
                code += condition
                code += f")) goto {next};\n"
            else:
                code += "if (this->position < this->src.size())\n"
                code += "{\n"
                code += "    if(!(false\n"
                code += add_indent(condition, 4)
                code += f"    )) goto {next};\n"
                code += "}\n"
                code += f"else goto {next};\n"
        elif node.ctx.optional:
            code += "if (this->position < this->src.size())\n"
            code += "{\n"
            code += "    if ((false\n"
            code += add_indent(condition, 4)
            code += "    )) this->position++;\n"
            code += "}\n"
        elif node.ctx.loop:
            code += "{\n"
            if node.ctx.loop_nonempty:
                code += "    size_t i = 0;"
            code += "    for(;;)\n"
            code += "    {\n"
            code += "        if (this->position > this->src.size()) break;\n"
            code += "        if (!(false\n"
            code += add_indent(condition, 8)
            code += "        )) break;\n"
            code += "        this->position++;"
            if node.ctx.loop_nonempty:
                code += "        i++;\n"
            code += "    }\n"
            if node.ctx.loop_nonempty:
                code += f"\nif (!i) goto {next};\n"
            code += "}\n"
        else:
            code += f"if (this->position > this->src.size()) goto {next};\n"
            code += "if (!(false\n"
            code += condition
            code += f")) goto {next};\n"
            code += "this->position++;\n"
        return code


def generate_parser(file):
    filename = os.path.basename(file.name)[0]
    tokenizer = Tokenizer(file)
    parser = Parser(tokenizer)
    root_node = parser.parse()
    code_gen = CodeGenerator(root_node, filename)
    code_gen.start()


argument_parser = argparse.ArgumentParser(description="Peg parser generator")
argument_parser.add_argument('--version', action='version', version=f"%(prog)s {VERSION}")
argument_parser.add_argument("path", type=argparse.FileType(encoding="utf-8"))
arguments = argument_parser.parse_args()
generate_parser(arguments.path)
