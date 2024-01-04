# MIT License
#
# Copyright (c) 2023 Roman Feduniak
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from io import TextIOWrapper
import argparse
import typing
import enum
import sys
import re
import os

VERSION = "0.1.2"
RT = typing.TypeVar('RT')  # return type


class TokenType(enum.Enum):
    COMMENT = re.compile(r"#.*(?=\n)?")
    IDENTIFIER = re.compile(r"[^\d\W]\w*")
    EQUAL = re.compile(r"=")
    SLASH = re.compile(r"/")
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
    STRING = re.compile(r"\"[^\"\\]*(\\.[^\"\\]*)*\"")
    CHARACTER_CLASS = re.compile(r"\[[^\]\\]*(\\.[^\]\\]*)*\]")
    DOT = re.compile(r"\.")
    TILDE = re.compile(r"~")
    MINUS = re.compile(r"-")
    # special tokens
    CODE_SECTION = enum.auto()
    ACTION = enum.auto()
    RULE_TYPE = enum.auto()

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
            # special token processing
            elif len(self.tokens) >= 2 and \
                    self.tokens[-2].type == TokenType.PERCENT and self.tokens[-1].type == TokenType.IDENTIFIER and \
                    ((name := self.tokens[-1].value) == "cpp" or name == "hpp"):
                while self.src[self.pos].isspace():
                    self.pos += 1

                if self.peek() == "{":
                    action_start, action_end = self.match_paired_characters("{", "}")
                    self.pos = action_end + 1
                    value = self.src[action_start + 1:action_end]
                    line_number = self.calc_line(action_start)
                    col = self.calc_column(action_start, line_number)
                    self.tokens.append(Token(TokenType.CODE_SECTION, value, line_number, col))
                else:
                    self.error("'{' is expected")
            elif self.peek() == "{":
                action_start, action_end = self.match_paired_characters("{", "}")
                self.pos = action_end + 1
                value = self.src[action_start:action_end + 1]
                line_number = self.calc_line(action_start)
                col = self.calc_column(action_start, line_number)
                self.tokens.append(Token(TokenType.ACTION, value, line_number, col))
            elif self.peek() == "<":
                type_start, type_end = self.match_paired_characters("<", ">")
                self.pos = type_end + 1
                value = self.src[type_start:type_end + 1]
                line_number = self.calc_line(type_start)
                col = self.calc_column(type_start, line_number)
                self.tokens.append(Token(TokenType.RULE_TYPE, value, line_number, col))
            else:
                for token_type in TokenType:
                    # skip special tokens
                    if token_type in (TokenType.CODE_SECTION, TokenType.ACTION, TokenType.RULE_TYPE):
                        continue
                    if result := token_type.value.match(self.src, self.pos):
                        self.pos = result.end()
                        self.add_token(token_type, result.group())
                        break
                else:
                    self.error(f"unknown character '{self.src[self.pos]}'")

        return self.tokens

    def match_paired_characters(self, open_char: str, close_char: str) -> tuple[int, int]:
        mark = self.pos
        open = 1
        self.pos += 1
        while self.pos < len(self.src):
            if self.peek() == open_char: open += 1
            elif self.peek() == close_char: open -= 1
            if open == 0:
                end_pos = self.pos
                self.pos = mark
                return mark, end_pos
            self.pos += 1
        self.error(f"'{close_char}' is expected")

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

    def error(self, message) -> typing.NoReturn:
        line_number = self.src.count("\n", 0, self.pos)
        lines = self.src.splitlines(keepends=True)
        col = self.pos + 1 if line_number == 0 else self.pos - sum(map(len, lines[:line_number])) + 1
        print(f"{self.filename}:{line_number + 1}:{col}: {message}", file=sys.stderr)
        sys.exit(1)


TNode = typing.TypeVar("TNode", bound="Node")

@dataclass(kw_only=True)
class Node():
    line: int = 0
    col: int = 0

    def set_pos(self: TNode, start_token: Token) -> TNode:
        self.line = start_token.line
        self.col = start_token.col
        return self

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
class ParsingExpressionNode(Node):
    ctx: ParsingExpressionContext = field(default_factory=ParsingExpressionContext)


@dataclass
class ParsingExpressionRuleNameNode(ParsingExpressionNode):
    name: str


@dataclass
class ParsingExpressionStringNode(ParsingExpressionNode):
    value: str


@dataclass
class ParsingExpressionGroupNode(ParsingExpressionNode):
    parsing_expression: list["ParsingExpressionSequence"]


@dataclass
class ParsingExpressionCharacterClassNode(ParsingExpressionNode):
    characters: str


@dataclass
class ParsingExpressionDotNode(ParsingExpressionNode):
    pass


@dataclass
class ParsingExpressionSequence(Node):
    items: list[ParsingExpressionNode]
    action: str | None = None
    error_action: str | None = None
    position_vars: set[str] = field(default_factory=set)


@dataclass
class RuleNode(Node):
    name: str
    expression_sequences: list[ParsingExpressionSequence]
    return_type: str | None = None
    is_left_recursive: bool = False
    memo: bool = True
    inline: bool = False


STRING_UNESCAPE_TABLE = {
    "\\": "\\",
    "\"": "\"",
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
}

CHARACTER_CLASS_UNESCAPE_TABLE = STRING_UNESCAPE_TABLE.copy()
CHARACTER_CLASS_UNESCAPE_TABLE.update({
    "]": "]",
    "[": "[",
})


def unescape_string(string: str, table: dict[str, str]) -> str:
    new_string = ""
    i = 0
    while i < len(string) - 1:
        ch = string[i]
        next_ch = string[i + 1]
        if ch == "\\" and next_ch in table:
            new_string += table[next_ch]
            i += 2
            if i > len(string):
                new_string += string[-1:]
        else:
            new_string += ch
            i += 1
    if i < len(string):
        new_string += string[-1:]

    return new_string


def escape_string(string: str) -> str:
    escape_table = {
        "\a": "a",
        "\b": "b",
        "\f": "f",
        "\n": "n",
        "\r": "r",
        "\t": "t",
        "\v": "v",
        "\\": "\\",
    }
    new_string = ""
    for ch in string:
        if ch in escape_table:
            new_string += "\\"
            new_string += escape_table[ch]
        else:
            new_string += ch
    return new_string


def string_to_bytes(string: str) -> bytes:
    return re.sub(
            rb"\\x[0-9a-fA-F]{2}",
            lambda x: int(x.group(0)[-2:], 16).to_bytes(1, "big"),
            string.encode()
        )


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
            if self.match(TokenType.IDENTIFIER).value == "name":
                id = self.match(TokenType.IDENTIFIER)
                return NameNode(id.value).set_pos(id)
        raise ParsingFail

    def header_statement(self):
        with self.manager:
            percent = self.match(TokenType.PERCENT)
            if self.match(TokenType.IDENTIFIER).value == "hpp":
                return HeaderBlockNode(self.match(TokenType.CODE_SECTION).value).set_pos(percent)
        raise ParsingFail

    def code_statement(self):
        with self.manager:
            self.match(TokenType.PERCENT)
            if self.match(TokenType.IDENTIFIER).value == "cpp":
                code = self.match(TokenType.CODE_SECTION)
                return CodeBlockNode(code.value).set_pos(code)
        raise ParsingFail

    def rule_type_statement(self):
        with self.manager:
            self.match(TokenType.PERCENT)
            if self.match(TokenType.IDENTIFIER).value == "type":
                string = self.match(TokenType.STRING)
                return RuleTypeNode(string.value[1:-1]).set_pos(string)
        raise ParsingFail

    def root_rule_statement(self):
        with self.manager:
            self.match(TokenType.PERCENT)
            if self.match(TokenType.IDENTIFIER).value == "root":
                id = self.match(TokenType.IDENTIFIER)
                return RootRuleNode(id.value).set_pos(id)
        raise ParsingFail

    def parsing_expression_atom(self):
        with self.manager:
            id = self.match(TokenType.IDENTIFIER)
            self.lookahead(False, TokenType.EQUAL)
            self.lookahead(False, TokenType.RULE_TYPE)
            self.lookahead(False, TokenType.MINUS)
            return ParsingExpressionRuleNameNode(id.value).set_pos(id)
        with self.manager:
            str_token = self.match(TokenType.STRING)
            string = unescape_string(str_token.value[1:-1], STRING_UNESCAPE_TABLE)
            return ParsingExpressionStringNode(string).set_pos(str_token)
        with self.manager:
            lpar = self.match(TokenType.LPAR)
            parsing_expressions = self.loop(True, self.parsing_expression_)
            group = ParsingExpressionGroupNode([ParsingExpressionSequence(i).set_pos(lpar) for i in parsing_expressions]).set_pos(lpar)
            self.match(TokenType.RPAR)
            return group
        with self.manager:
            char_class = self.match(TokenType.CHARACTER_CLASS)
            string = unescape_string(char_class.value[1:-1], CHARACTER_CLASS_UNESCAPE_TABLE)
            return ParsingExpressionCharacterClassNode(string).set_pos(char_class)
        with self.manager:
            dot = self.match(TokenType.DOT)
            return ParsingExpressionDotNode().set_pos(dot)
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
            item.ctx.name = id.value
            return item
        with self.manager:
            return self.parsing_expression_item()
        raise ParsingFail

    def parsing_expression_(self):
        with self.manager:
            return self.loop(True, self.parsing_expression_named_item_or_item)
        with self.manager:
            self.match(TokenType.SLASH)
            return self.loop(True, self.parsing_expression_named_item_or_item)
        raise ParsingFail

    def error_action(self):
        with self.manager:
            self.match(TokenType.TILDE)
            return self.match(TokenType.ACTION)
        return None

    def parsing_expression(self):
        with self.manager:
            parsing_expression = self.parsing_expression_()
            action = self.optional(TokenType.ACTION)
            error_action = self.error_action()
            node = ParsingExpressionSequence(
                parsing_expression,
                action.value if action else None,
                error_action.value if error_action else None,
            )
            if action: node.position_vars.update(re.findall(r"\$([1-9][0-9]*)", action.value))
            node.line = parsing_expression[0].line
            node.col = parsing_expression[0].col
            return node
        raise ParsingFail

    def rule_attribute(self):
        attr: Token
        with self.manager:
            self.match(TokenType.MINUS)
            attr = self.match(TokenType.IDENTIFIER)
            if attr.value in ("nomemo", "inline"):
                return attr
            self.error(f"Unknown rule attribute \"{attr.value}\"")
        return None

    def rule_statement(self):
        with self.manager:
            rule_name = self.match(TokenType.IDENTIFIER)
            rule_type = self.optional(TokenType.RULE_TYPE)
            attr = self.rule_attribute()
            self.match(TokenType.EQUAL)
            parsing_expressions = self.loop(True, self.parsing_expression)
            rule_node = RuleNode(rule_name.value, parsing_expressions).set_pos(rule_name)
            if rule_type:
                rule_node.return_type = rule_type.value[1:-1].strip()
            if attr:
                match attr.value:
                    case "nomemo":
                        rule_node.memo = False
                    case "inline":
                        rule_node.inline = True
            return rule_node
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

    def match(self, token_type: TokenType) -> Token:
        token = self.get_token()
        if token and token.type == token_type:
            self.pos += 1
            return token
        raise ParsingFail

    def optional(self, token_type: TokenType) -> Token | None:
        token = self.get_token()
        if token and token.type == token_type:
            self.pos += 1
            return token
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
        print(f"{self.filename}:{token.line}:{token.col}: {message}", file=sys.stderr)
        sys.exit(1)


def write_lines(file: TextIOWrapper, *lines: str):
    for line in lines:
        file.write(line)
        file.write("\n")


def add_indent(string: str, indent: int) -> str:
    lines = string.split("\n")
    new_lines = []

    for line in lines:
        new_lines.append((' ' * indent + line.rstrip()) if line.strip() else '')

    return '\n'.join(new_lines)


def get_indent(string: str) -> int:
    return len(string) - len(string.lstrip())


def remove_indent(string: str) -> str:
    lines = string.split("\n")
    new_lines = []
    indent = min((get_indent(line) for line in lines if line.strip()))

    for line in lines:
        new_indent = get_indent(line) - indent
        new_lines.append((' ' * new_indent + line.lstrip()) if line.strip() else '')

    return '\n'.join(new_lines)


def set_indent(string: str, indent: int) -> str:
    return add_indent(remove_indent(string), indent)


def get_return_type_of_parsing_expression_sequence(parsing_expression: ParsingExpressionSequence) -> "CppType":
    if parsing_expression.action is None or "$$" not in parsing_expression.action:
        return CppType("bool")
    else:
        return CppType("ExprResult", is_optional=True)


def count_by_predicate(container: typing.Iterable, predicate: typing.Callable[[typing.Any], bool]) -> int:
    c = 0
    for item in container:
        if predicate(item):
            c += 1
    return c


def find_by_predicate(container: typing.Iterable, predicate: typing.Callable[[typing.Any], bool]):
    for item in container:
        if predicate(item):
            return item
    return None


@dataclass
class GeneratedExpression:
    code: str
    user_defined_var: str | None = None


@dataclass
class GeneratedGroupExpression:
    code: str
    user_defined_vars: list[str] = field(default_factory=list)


@dataclass
class CppType:
    raw_type: str
    is_optional: bool = False

    def __str__(self) -> str:
        if self.is_optional:
            return f"std::optional<{self.raw_type}>"
        return self.raw_type

    @property
    def null(self) -> str:
        if self.is_optional:
            return "std::nullopt"
        if self.raw_type == "bool":
            return "false"
        assert False, f"not implemented for type '{self.raw_type}'"

    @property
    def getter(self) -> str:
        if self.is_optional:
            return ".value()"
        return ""


class AstContext:
    def __init__(self, root_node: BlockStatementNode, filename: str):
        self.root_node = root_node
        self.filename = filename
        if r := self.get_node_or_none(RootRuleNode):
            self.root_rule: RuleNode = self.get_rule_by_name(r.name)  # type: ignore
        else:
            self.root_rule: RuleNode = self.get_node_or_none(RuleNode)  # type: ignore

    def get_rule_by_name(self, name: str) -> RuleNode | None:
        return find_by_predicate(
            self.root_node.statements, lambda node: isinstance(node, RuleNode) and node.name == name)

    def get_node_or_none(self, node_type: typing.Type[RT]) -> RT | None:
        for item in self.root_node.statements:
            if isinstance(item, node_type):
                return item

    def get_rules(self) -> list[RuleNode]:
        rules = []
        for statement in self.root_node.statements:
            if isinstance(rule := statement, RuleNode):
                rules.append(rule)
        return rules

    def get_rule_names(self) -> list[str]:
        return [rule.name for rule in self.get_rules()]

    def get_node_count(self, node_type: typing.Type[Node]) -> int:
        return count_by_predicate(self.root_node.statements, lambda node: isinstance(node, node_type))


class LeftRecursiveAnalyzer:
    def __init__(self, ctx: AstContext):
        self.ctx = ctx

    def analyze(self):
        for rule in self.ctx.get_rules():
            if not rule.inline:
                rule.is_left_recursive = self.is_direct_left_recursive(rule)

    def is_direct_left_recursive(self, rule: RuleNode) -> bool:
        for sequence in rule.expression_sequences:
            if first_rule := self.get_first_rule_or_none(sequence):
                if first_rule.name == rule.name:
                    return True
        return False

    def get_first_rule_or_none(self, sequence: ParsingExpressionSequence) -> ParsingExpressionRuleNameNode | None:
        for item in sequence.items:
            if isinstance(item, ParsingExpressionRuleNameNode):
                return item
            elif isinstance(item, ParsingExpressionGroupNode):
                return self.get_first_rule_or_none(item.parsing_expression[0])
            elif not self.is_parsing_expr_consume_zero(item):
                break
        return None

    def is_parsing_expr_consume_zero(self, expr: ParsingExpressionNode) -> bool:
        if expr.ctx.optional \
            or expr.ctx.lookahead \
            or expr.ctx.loop and not expr.ctx.loop_nonempty \
            or isinstance(expr, ParsingExpressionRuleNameNode) and self.is_rule_consume_zero(expr.name):
            return True
        return False

    def is_rule_consume_zero(self, rule_name: str) -> bool:
        rule: RuleNode = self.ctx.get_rule_by_name(rule_name)  # type: ignore
        for sequence in rule.expression_sequences:
            zero_consume = True
            for item in sequence.items:
                if not self.is_parsing_expr_consume_zero(item):
                    zero_consume = False
                    break
            if zero_consume: return True
        return False


class StaticAnalyzer:
    def __init__(self, ctx: AstContext):
        self.ctx = ctx

    def analyze(self):
        self.rules_presence()
        self.same_rule_names()
        self.check_directives()
        self.check_rule_name_in_root_directive()
        self.rule_not_exist_but_used()
        self.unused_rules()
        LeftRecursiveAnalyzer(self.ctx).analyze()  # only after check unused rules
        self.wrong_left_recursive_rules()
        self.check_action_presence()  # check for the presence of an action when variables are present
        self.same_var_names_in_parsing_expr_sequence()
        self.group_with_repetition_has_variables_inside()
        self.lookahead_false_assigned_to_var()
        self.string_assigned_to_var()
        # The return types in all parsing expression sequences must match within the rule
        self.check_return_types_in_parsing_expression_sequences()
        self.check_characters_inside_character_class()
        self.check_position_vars_in_action()
        self.check_nomemo_attr()
        self.check_inline_attr()

    def rules_presence(self):
        if self.ctx.get_node_or_none(RuleNode) is None:
            self.error("No rule is defined")

    def same_rule_names(self):
        rule_names = self.ctx.get_rule_names()
        for i, rule_name in enumerate(rule_names[:-1], 1):
            for rule_name_ in rule_names[i:]:
                if rule_name == rule_name_:
                    self.error(f"Rule '{rule_name}' has more than one definition")

    def check_directives(self):
        error_message = "The '%{}' directive has more than one definition"
        if self.ctx.get_node_count(NameNode) > 1:
            self.error(error_message.format("name"))
        if self.ctx.get_node_count(HeaderBlockNode) > 1:
            self.error(error_message.format("hpp"))
        if self.ctx.get_node_count(CodeBlockNode) > 1:
            self.error(error_message.format("cpp"))
        if self.ctx.get_node_count(RuleTypeNode) > 1:
            self.error(error_message.format("type"))
        if self.ctx.get_node_count(RootRuleNode) > 1:
            self.error(error_message.format("root"))

    def check_rule_name_in_root_directive(self):
        if root_rule_node := self.ctx.get_node_or_none(RootRuleNode):
            if root_rule_node.name not in self.ctx.get_rule_names():
                self.error(f"The directive '%root' contains a non-existing rule: '{root_rule_node.name}'", root_rule_node)

    def rule_not_exist_but_used(self):
        rule_names = self.ctx.get_rule_names()
        for rule in self.ctx.get_rules():
            for sequence in rule.expression_sequences:
                for item in sequence.items:
                    if isinstance(item, ParsingExpressionRuleNameNode):
                        if item.name not in rule_names:
                            self.error(f"The '{rule.name}' rule invokes a nonexistent rule '{item.name}'", item)

    def unused_rules(self):
        checked_rules = []
        def group_traversal(group: ParsingExpressionGroupNode) -> set[str]:
            rules = set()
            for sequence in group.parsing_expression:
                for item in sequence.items:
                    if isinstance(r := item, ParsingExpressionRuleNameNode):
                        if r.name not in checked_rules:
                            rules.add(r.name)
                            rules.update(rule_traversal(self.ctx.get_rule_by_name(r.name)))  # type: ignore
                    elif isinstance(g := item, ParsingExpressionGroupNode):
                        rules.update(group_traversal(g))
            return rules

        def rule_traversal(rule: RuleNode) -> set[str]:
            checked_rules.append(rule.name)
            rules = set()
            for sequence in rule.expression_sequences:
                for item in sequence.items:
                    if isinstance(r := item, ParsingExpressionRuleNameNode):
                        if r.name not in checked_rules:
                            rules.add(r.name)
                            rules.update(rule_traversal(self.ctx.get_rule_by_name(r.name)))  # type: ignore
                    elif isinstance(g := item, ParsingExpressionGroupNode):
                        rules.update(group_traversal(g))
            return rules

        used_rules = rule_traversal(self.ctx.root_rule)
        used_rules.add(self.ctx.root_rule.name)
        all_rules = set(self.ctx.get_rule_names())
        if len(unused_rules := all_rules - used_rules):
            error_messages = []
            for unused_rule_name in unused_rules:
                unused_rule_node: RuleNode = self.ctx.get_rule_by_name(unused_rule_name)  # type: ignore
                error_messages.append(f"{self.ctx.filename}:{unused_rule_node.line}:{unused_rule_node.col}:"
                                      f" Rule '{unused_rule_name}' defined but not used")
            self.error("\n".join(error_messages))

    def wrong_left_recursive_rules(self):
        for rule in self.ctx.get_rules():
            if rule.is_left_recursive:
                if len(rule.expression_sequences) == 1:
                    self.error(f"In the '{rule.name}' name, a left-recursive rule must be at least 2 sequences of expressions", rule)

    def check_action_presence(self):
        for rule in self.ctx.get_rules():
            is_rule_type_specified = bool(rule.return_type)
            for parsing_expression_sequence in rule.expression_sequences:
                is_var_presence = False
                for item in parsing_expression_sequence.items:
                    if isinstance(group := item, ParsingExpressionGroupNode):
                        if len(self.get_vars_from_group(group)):
                            is_var_presence = True
                            break
                    if item.ctx.name:
                        is_var_presence = True
                        break
                if is_var_presence and parsing_expression_sequence.action is None:
                    self.error(f"In the '{rule.name}' rule, variables are declared, but there is no action", parsing_expression_sequence)
                if is_rule_type_specified:
                    if parsing_expression_sequence.action is None:
                        self.error(f"In the '{rule.name}' rule, the return type is defined, but the action not specified",
                                    parsing_expression_sequence)
                    elif "$$" not in parsing_expression_sequence.action:
                        self.error(f"In the '{rule.name}' rule, the return type is defined, but '$$' variable in the action is not",
                                    parsing_expression_sequence)

    def same_var_names_in_parsing_expr_sequence(self):
        error_message = "In the '{}' rule, variable '{}' is declared multiple times"
        for rule in self.ctx.get_rules():
            for parsing_expression_sequence in rule.expression_sequences:
                var_names = []
                for item in parsing_expression_sequence.items:
                    if isinstance(group := item, ParsingExpressionGroupNode):
                        for var in self.get_vars_from_group(group):
                            if var in var_names:
                                self.error(error_message.format(rule.name, var), parsing_expression_sequence)
                            var_names.append(var)
                    if (var := item.ctx.name):
                        if var in var_names:
                            self.error(error_message.format(rule.name, var), parsing_expression_sequence)
                        var_names.append(var)

    def group_with_repetition_has_variables_inside(self):
        for rule in self.ctx.get_rules():
            for parsing_expression_sequence in rule.expression_sequences:
                for item in parsing_expression_sequence.items:
                    if isinstance(group := item, ParsingExpressionGroupNode):
                        if group.ctx.loop and len(self.get_vars_from_group(group)):
                            self.error(f"In the '{rule.name}' rule, the group uses variables inside itself"
                                        " and repetitions operators simultaneously", group)

    def lookahead_false_assigned_to_var(self):
        for rule in self.ctx.get_rules():
            for parsing_expression_sequence in rule.expression_sequences:
                for item in parsing_expression_sequence.items:
                    if item.ctx.lookahead and not item.ctx.lookahead_positive and item.ctx.name:
                        self.error(f"In the '{rule.name}' rule, a parsing expression with the '!' operator"
                                    " cannot be assigned to a variable", item)

    def string_assigned_to_var(self):
        for rule in self.ctx.get_rules():
            for parsing_expression_sequence in rule.expression_sequences:
                for item in parsing_expression_sequence.items:
                    if isinstance(string := item, ParsingExpressionStringNode):
                        if string.ctx.name:
                            if string.ctx.lookahead:
                                self.error(f"In the '{rule.name}' rule, a string with the '&' operator"
                                            " cannot be assigned to a variable", string)
                            if not string.ctx.loop and not string.ctx.optional:
                                self.error(f"In the '{rule.name}' rule, simple string cannot be assigned to a variable", string)

    def check_return_types_in_parsing_expression_sequences(self):
        for rule in self.ctx.get_rules():
            if len(rule.expression_sequences) > 1:
                return_type = get_return_type_of_parsing_expression_sequence(rule.expression_sequences[0])
                for parsing_expression_sequence in rule.expression_sequences[1:]:
                    if return_type != get_return_type_of_parsing_expression_sequence(parsing_expression_sequence):
                        self.error(f"In the '{rule.name}' rule, parsing expression sequences return different types", rule)

    def check_characters_inside_character_class(self):
        for rule in self.ctx.get_rules():
            for parsing_expression_sequence in rule.expression_sequences:
                for item in parsing_expression_sequence.items:
                    if isinstance(character_class := item, ParsingExpressionCharacterClassNode):
                        characters = []
                        ranges: list[tuple[str, str]] = []
                        i = 0
                        while i < len(character_class.characters):
                            ch = character_class.characters[i]
                            if i + 2 < len(character_class.characters) and character_class.characters[i + 1] == "-":
                                from_ = ch
                                to = character_class.characters[i + 2]
                                error_message = (
                                        f"In the '{rule.name}' rule, inside the character class"
                                        f" '[{escape_string(character_class.characters)}]',"
                                        " {}"
                                        f" '{escape_string(from_)}-{escape_string(to)}'"
                                )
                                if from_ == to:
                                    self.error(error_message.format("the first and second characters in the range are the same"),
                                                character_class)
                                elif ord(from_) > ord(to):
                                    self.error(error_message.format("the first character is 'greater' than the second in a range"),
                                                character_class)
                                i += 2
                                ranges.append((from_, to))
                            else:
                                if ch in characters:
                                    self.error(f"In the '{rule.name}' rule, the character class has the same characters: {escape_string(ch)}",
                                                character_class)
                                characters.append(ch)
                            i += 1

                        for from_, to in ranges:
                            for ch in characters:
                                if ord(ch) >= ord(from_) and ord(ch) <= ord(to):
                                    self.error(
                                        f"In the '{rule.name}' rule, inside the character class"
                                        f" '[{escape_string(character_class.characters)}]',"
                                        f" the character '{escape_string(ch)}' intersects with the range"
                                        f" '{escape_string(from_)}-{escape_string(to)}'",
                                        character_class
                                    )

    def check_position_vars_in_action(self):
        for rule in self.ctx.get_rules():
            for sequence in rule.expression_sequences:
                if vars := sequence.position_vars:
                    for var in vars:
                        if int(var) > len(sequence.items):
                            self.error(f"'${var}', the index exceeds the number of expressions", sequence)

    def check_nomemo_attr(self):
        for rule in self.ctx.get_rules():
            if rule.is_left_recursive and rule.memo is False:
                self.error(f"Rule '{rule.name}': the -nomemo attribute has no effect on left-recursive rules", rule)

    def check_inline_attr(self):
        for rule in self.ctx.get_rules():
            if rule.inline:
                if rule.return_type:
                    self.error(f"Rule '{rule.name}': if the -inline attribute is defined, the rule cannot have a return type", rule)
                for sequence in rule.expression_sequences:
                    if sequence.action or sequence.error_action:
                        self.error(f"Rule '{rule.name}': if the -inline attribute is defined, the rule cannot have actions", rule)

    def get_vars_from_group(self, group: ParsingExpressionGroupNode) -> list[str]:
        vars = []
        for parsing_expression_sequences in group.parsing_expression:
            for item in parsing_expression_sequences.items:
                if isinstance(group_ := item, ParsingExpressionGroupNode):
                    vars.extend(self.get_vars_from_group(group_))
                elif (var := item.ctx.name):
                    vars.append(var)
        return vars

    def error(self, message: str, node: Node | None = None) -> typing.NoReturn:
        if node:
            print(f"{self.ctx.filename}:{node.line}:{node.col}: ", end='', file=sys.stderr)
        print(message, file=sys.stderr)
        sys.exit(1)


class CodeGenerator:
    cpp_file: TextIOWrapper
    hpp_file: TextIOWrapper

    def __init__(self, ctx: AstContext):
        self.ctx = ctx
        self.parser_name = ctx.filename.split(".")[0]
        self.header_from_directive = ""
        self.code_from_directive = ""
        self.rule_type = "size_t"
        self.rules_return_type: dict[str, CppType] = dict()

        if name_node := self.get_node_or_none(NameNode):
            self.parser_name = name_node.name

        if header_node := self.get_node_or_none(HeaderBlockNode):
            self.header_from_directive = header_node.header

        if code_node := self.get_node_or_none(CodeBlockNode):
            self.code_from_directive = code_node.code

        if rule_type_node := self.get_node_or_none(RuleTypeNode):
            self.rule_type = rule_type_node.type_name

        self.type_analysis()  # it should come after processing the %type directive

    def get_node_or_none(self, node_type: typing.Type[RT]) -> RT | None:
        return find_by_predicate(self.ctx.root_node.statements, lambda item: isinstance(item, node_type))

    def type_analysis(self):
        for node in self.ctx.root_node.statements:
            if not isinstance(rule := node, RuleNode):
                continue
            if rule.return_type:
                self.rules_return_type[rule.name] = CppType(rule.return_type, True)
            else:
                self.rules_return_type[rule.name] = get_return_type_of_parsing_expression_sequence(rule.expression_sequences[0])

    def start(self):
        self.cpp_file = open(f"{self.parser_name}.cpp", "w", encoding="utf-8")
        self.hpp_file = open(f"{self.parser_name}.hpp", "w", encoding="utf-8")

        copyright_comment = f"// Generated by pparser {VERSION} (https://github.com/romanfedyniak/pparser) from {self.ctx.filename}"
        memo_size = count_by_predicate(
            self.ctx.root_node.statements, lambda item: isinstance(item, RuleNode) and item.memo and not item.inline)

        write_lines(
            self.cpp_file,
            copyright_comment,
            f"#include \"{self.parser_name}.hpp\"",
            "",
            "#include <algorithm>",
            "",
        )

        if self.code_from_directive:
            write_lines(
                self.cpp_file,
                "// code from %cpp",
                remove_indent(self.code_from_directive),
                "// end %cpp",
                "",
            )

        write_lines(
            self.cpp_file,
            "namespace PParser",
            "{",
            "",
            "    struct ParsingFail",
            "    {",
            "        std::string message;",
            "        size_t position;",
            "    };",
            "",
            "    ////////// BEGINNING OF RULES //////////",
            "",
        )

        write_lines(
            self.hpp_file,
            copyright_comment,
            "#ifndef PPARSER_HPP_",
            "#define PPARSER_HPP_",
            "",
            "#include <string>",
            "#include <string_view>",
            "#include <optional>",
            "#include <functional>",
            "#include <array>",
            "#include <unordered_map>",
            "#include <any>",
            "#include <tuple>",
            "#include <iostream>",
            "#include <vector>",
            "",
        )

        if self.header_from_directive:
            write_lines(
                self.hpp_file,
                "// code from %hpp",
                remove_indent(self.header_from_directive),
                "// end %hpp",
                "",
            )

        write_lines(
            self.hpp_file,
            "namespace PParser",
            "{",
            "",
            f"    using ExprResult = {self.rule_type};",
            "",
            "    struct TokenPos",
            "    {",
            "        size_t startCol;",
            "        size_t startLine;",
            "        size_t endCol;",
            "        size_t endLine;",
            "    };",
            "",
            "    class Parser",
            "    {",
            "    private:",
            "        using errorHandler_t = std::function<void(std::string message, size_t position)>;",
            "        errorHandler_t errorHandler;",
            "        const std::string_view src;",
            "        size_t position = 0;",
            f"        std::array<std::unordered_map<size_t, std::tuple<std::any, size_t>>, {memo_size}> memos;",
            "        std::vector<size_t> lineNumbers;",
            "",
            "        ////////// BEGINNING OF RULES //////////",
        )

        self.generate()

        write_lines(
            self.cpp_file,
            "    ////////// END OF RULES //////////",
            "",
            "    size_t Parser::getUtf8Size() const",
            "    {",
            "        if (position >= src.size()) return 0;",
            "        auto uc = (unsigned char)src[position];",
            "        if (uc < 128) return 1;",
            "        else if ((uc & 0xE0) == 0xC0) return 2;",
            "        else if ((uc & 0xF0) == 0xE0) return 3;",
            "        else if ((uc & 0xF8) == 0xF0) return 4;",
            "        else return 0;",
            "    }",
            "",
            "    size_t Parser::getUtf32Char(char32_t& c32) const",
            "    {",
            "        size_t n = getUtf8Size();",
            "        if (n == 0) return 0;",
            "        if (position + n > src.size()) return 0;",
            "",
            "        switch(n) {",
            "        case 1:",
            "            c32 = src[position];",
            "            break;",
            "        case 2:",
            "            if ((src[position + 1] & 0xC0) != 0x80) return 0;",
            "            c32 = ((src[position + 0] & 0x1F) << 6) |",
            "                  ((src[position + 1] & 0x3F));",
            "            break;",
            "        case 3:",
            "            if ((src[position + 1] & 0xC0) != 0x80) return 0;",
            "            if ((src[position + 2] & 0xC0) != 0x80) return 0;",
            "            c32 = ((src[position + 0] & 0xF) << 12) |",
            "                  ((src[position + 1] & 0x3F) << 6) |",
            "                  ((src[position + 2] & 0x3F));",
            "            break;",
            "        case 4:",
            "            if ((src[position + 1] & 0xC0) != 0x80) return 0;",
            "            if ((src[position + 2] & 0xC0) != 0x80) return 0;",
            "            if ((src[position + 3] & 0xC0) != 0x80) return 0;",
            "            c32 = ((src[position + 0] & 0x7) << 18)  |",
            "                  ((src[position + 1] & 0x3F) << 12) |",
            "                  ((src[position + 2] & 0x3F) << 6)  |",
            "                  ((src[position + 3] & 0x3F));",
            "            break;",
            "        }",
            "",
            "        return n;",
            "    }",
            "",
            "    std::optional<std::tuple<std::any, size_t>> Parser::memoGet(size_t ruleId) const",
            "    {",
            "        auto memo = memos[ruleId];",
            "        if (const auto& search = memo.find(position); search != memo.cend())",
            "        {",
            "            return memo[this->position];",
            "        }",
            "        return std::nullopt;",
            "    }",
            "",
            "    void Parser::memoSet(size_t ruleId, std::any value, size_t start_position)",
            "    {",
            "        memos[ruleId][start_position] = { value, this->position };",
            "    }",
            "",
            "    void Parser::parseError(const std::string& msg) const",
            "    {",
            "        throw ParsingFail{msg, this->position};",
            "    }",
            "",
            "    size_t Parser::getLineFromPosition(size_t pos) const",
            "    {",
            "        auto it = std::lower_bound(this->lineNumbers.cbegin(), this->lineNumbers.cend(), pos + 1);",
            "        if (it == this->lineNumbers.cend()) return this->lineNumbers.size() + 1;",
            "        return it - this->lineNumbers.cbegin() + 1;",
            "    }",
            "",
            "    size_t Parser::getColFromPosition(size_t pos, size_t line) const",
            "    {",
            "        if (line == 1) return pos + 1;",
            "        if (line >= this->lineNumbers.size()) return pos - this->lineNumbers.back() + 1;",
            "        size_t start_line = this->lineNumbers[line - 1];",
            "        return start_line - pos + 1;",
            "    }",
            "",
            "    void Parser::calculateLineNumbers()",
            "    {",
            "        if (this->lineNumbers.size() != 0) return;",
            "        for (size_t i = 0; i < this->src.size(); ++i)",
            "            if (this->src[i] == '\\n') this->lineNumbers.push_back(i + 1);",
            "    }",
            "",
            "    void Parser::setErrorHandler(errorHandler_t handler)",
            "    {",
            "        errorHandler = handler;",
            "    }",
            "",
            f"    {self.rules_return_type[self.ctx.root_rule.name]} Parser::parse() noexcept",
            "    {",
            "        this->calculateLineNumbers();",
            "        this->position = 0;",
            "        try {",
            f"            return rule__{self.ctx.root_rule.name}();",
            "        } catch (const ParsingFail& error) {",
            "            if (errorHandler) errorHandler(error.message, error.position);",
            "            else { std::cerr << \"Error at position \" << error.position << \": \" << error.message << std::endl; }",
            f"            return {self.rules_return_type[self.ctx.root_rule.name].null};",
            "        }",
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
            "        size_t getUtf8Size() const;",
            "        size_t getUtf32Char(char32_t& c32) const;",
            "        std::optional<std::tuple<std::any, size_t>> memoGet(size_t ruleId) const;",
            "        void memoSet(size_t ruleId, std::any value, size_t start_position);",
            "        void parseError(const std::string& msg) const;",
            "        size_t getLineFromPosition(size_t pos) const;",
            "        size_t getColFromPosition(size_t pos, size_t line) const;",
            "        void calculateLineNumbers();",
            "",
            "    public:",
            "        void setErrorHandler(errorHandler_t handler);",
            f"        {self.rules_return_type[self.ctx.root_rule.name]} parse() noexcept;",
            "",
            "        explicit Parser(std::string_view src);",
            "    };",
            "}",
            "",
            "#endif // PPARSER_HPP_",
        )

        self.cpp_file.close()
        self.hpp_file.close()

    def generate(self):
        rule_id = 0
        for node in self.ctx.root_node.statements:
            match node:
                case NameNode() | HeaderBlockNode() | CodeBlockNode() | RuleTypeNode() | RootRuleNode():
                    pass
                case RuleNode():
                    if not node.inline:
                        self.gen_rule(node, rule_id)
                        if node.memo:
                            rule_id += 1
                case _:
                    self.gen_type_error(node)

    def gen_type_error(self, node: Node) -> typing.NoReturn:
        print(f"generator for node with type <{type(node).__name__}> not implemented", file=sys.stderr)
        exit(1)

    def gen_rule(self, node: RuleNode, rule_id: int):
        return_type = self.rules_return_type[node.name]
        write_lines(self.hpp_file, f"        {return_type} rule__{node.name}();")
        if not node.is_left_recursive:
            write_lines(
                self.cpp_file,
                f"    {return_type} Parser::rule__{node.name}()",
                "    {",
            )
        else:
            write_lines(self.hpp_file, f"        {return_type} rule__{node.name}__body();")
            write_lines(
                self.cpp_file,
                f"    {return_type} Parser::rule__{node.name}()",
                "    {",
                "        auto mark = this->position;",
                f"        auto memoized = this->memoGet({rule_id});",
                "        if (memoized.has_value())",
                "        {",
                "            auto& [memoized_value, memoized_position] = memoized.value();",
                "            this->position = memoized_position;",
                f"            if (!memoized_value.has_value()) return {return_type.null};",
                f"            return std::any_cast<{return_type.raw_type}>(memoized_value);",
                "        }",
                "        else",
                "        {",
                "            auto last_position = mark;",
                f"            this->memoSet({rule_id}, {{}}, mark);",
                f"            {return_type.raw_type} last_result;",
                "",
                "            for(;;)",
                "            {",
                "                this->position = mark;",
                f"                auto result = rule__{node.name}__body();",
                "                auto end_position = this->position;",
                "                if (end_position <= last_position) break;",
                f"                this->memoSet({rule_id}, result{return_type.getter}, mark);",
                f"                last_result = result{return_type.getter};",
                f"                last_position = end_position;",
                "            }",
                "",
                f"            if (last_position == mark) return {return_type.null};",
                "            this->position = last_position;",
                f"            return std::any_cast<{return_type.raw_type}>(last_result);",
                "        }",
                "    }",
                "",
            )

            write_lines(
                self.cpp_file,
                f"    {return_type} Parser::rule__{node.name}__body()",
                "    {",
            )
        code = ""
        if not node.is_left_recursive and node.memo:
            code += f"auto __memoized = this->memoGet({rule_id});\n"
            code += f"if (__memoized.has_value())\n"
            code += "{\n"
            code += "    auto& [__memoized_value, __memoized_position] = __memoized.value();\n"
            code += "    this->position = __memoized_position;\n"
            code += f"    if (!__memoized_value.has_value()) return {return_type.null};\n"
            code += f"    return std::any_cast<{return_type.raw_type}>(__memoized_value);\n"
            code += "}\n\n"
        code += "auto __mark = this->position;\n"

        for i, parsing_expression in enumerate(node.expression_sequences):
            if i > 0:
                code += f"NEXT_{i}:\n"
                code += "this->position = __mark;\n"
            next = f"NEXT_{i + 1}" if i + 1 < len(node.expression_sequences) else "FAIL"
            code += self.gen_parsing_expr(parsing_expression, next, return_type, i + 1, rule_id, node)
            code += "\n"
        self.cpp_file.write(add_indent(code, 8))

        write_lines(self.cpp_file,
            "    FAIL:",
            "        this->position = __mark;",
        )

        if not node.is_left_recursive and node.memo:
            write_lines(self.cpp_file, f"        this->memoSet({rule_id}, {{}}, __mark);")
        write_lines(self.cpp_file, f"        return {return_type.null};")
        if not return_type.is_optional:
            write_lines(self.cpp_file, "    SUCCESS:")
            if not node.is_left_recursive and node.memo:
                write_lines(self.cpp_file, f"        this->memoSet({rule_id}, true, __mark);")
            write_lines(self.cpp_file, "        return true;")
        write_lines(self.cpp_file, "    }", "")

    def gen_parsing_expr(
            self, node: ParsingExpressionSequence, next: str, return_type: CppType, expr_index: int, rule_id: int, rule: RuleNode):
        group_index = 1
        generated_exprs: list[GeneratedExpression | GeneratedGroupExpression] = []
        if node.error_action:
            real_next = next
            next = f"ERROR_ACTION_{expr_index}"
        for i in node.items:
            match i:
                case ParsingExpressionRuleNameNode():
                    if (rule_ := self.ctx.get_rule_by_name(i.name)) and rule_.inline:
                        group = ParsingExpressionGroupNode(rule_.expression_sequences)
                        group.ctx = i.ctx
                        generated_exprs.append(self.gen_parsing_expr_group(group, next, f"group_{expr_index}_{group_index}"))
                        group_index += 1
                    else:
                        generated_exprs.append(self.gen_parsing_expr_rule_name(i, next))
                case ParsingExpressionStringNode():
                    generated_exprs.append(self.gen_parsing_expr_string(i, next))
                case ParsingExpressionCharacterClassNode():
                    generated_exprs.append(self.gen_parsing_expr_character_class(i, next))
                case ParsingExpressionGroupNode():
                    generated_exprs.append(self.gen_parsing_expr_group(i, next, f"group_{expr_index}_{group_index}"))
                    group_index += 1
                case ParsingExpressionDotNode():
                    generated_exprs.append(self.gen_parsing_expr_dot(i, next))
                case _:
                    self.gen_type_error(node)

        vars_declaration = ""
        for g in generated_exprs:
            match g:
                case GeneratedExpression():
                    if var := g.user_defined_var:
                        vars_declaration += var
                        vars_declaration += "\n"
                case GeneratedGroupExpression():
                    if len(vars := g.user_defined_vars):
                        vars_declaration += "\n".join(vars)
                        vars_declaration += "\n"

        code = "{\n"
        if len(vars_declaration):
            code += "    // User defined variables\n"
            code += add_indent(vars_declaration, 4)
            code += "    // end variables\n\n"
        for i, g in enumerate(generated_exprs, 1):
            save_pos = str(i) in node.position_vars
            if save_pos:
                code += f"    TokenPos __token_pos_{i};\n"
                code += f"    __token_pos_{i}.startLine = this->getLineFromPosition(this->position);\n"
                code += f"    __token_pos_{i}.startCol = this->getColFromPosition(this->position, __token_pos_{i}.startLine);\n\n"
            code += add_indent(g.code, 4)
            code += "\n"
            if save_pos:
                code += f"    __token_pos_{i}.endLine = this->getLineFromPosition(this->position);\n"
                code += f"    __token_pos_{i}.endCol = this->getColFromPosition(this->position, __token_pos_{i}.endLine);\n\n"
        if action_code := node.action:
            code += "    { // action\n"
            for var in node.position_vars:
                if f"${var}" in action_code:
                    action_code = action_code.replace(f"${var}", f"__token_pos_{var}")
            if "$$" in node.action:
                code += f"        {return_type.raw_type} __rule_result;\n"
                action_code = action_code.replace("$$", "__rule_result")
                code += set_indent(action_code, 8)
                code += "\n"
                if not rule.is_left_recursive and rule.memo:
                    code += f"        this->memoSet({rule_id}, __rule_result, __mark);\n"
                code += "        return __rule_result;\n"
            else:
                code += set_indent(action_code, 8)
                code += "\n"
            code += "    } // end of action\n"
        if node.action is None or "$$" not in node.action:
            code += "    goto SUCCESS;\n"
        if node.error_action:
            code +="\n"
            code += f"{next}:\n"
            code += "    { // error action\n"
            code += set_indent(node.error_action, 8)
            code += "\n"
            code += "    } // end of error action\n"
            code += f"    goto {real_next};\n"  # type: ignore
        code += "}\n"
        return code

    def gen_parsing_expr_rule_name(self, node: ParsingExpressionRuleNameNode, next: str) -> GeneratedExpression:
        code = ""
        var = None
        return_type = self.rules_return_type[node.name]

        if node.ctx.lookahead:
            code += "{\n"
            code += "   size_t __tempMark = position;\n"
            if node.ctx.name:
                code += f"   {self.rules_return_type[node.name]} __result;\n"
                var = f"{return_type.raw_type} {node.ctx.name};"
            code += f"   if({'!' if node.ctx.lookahead_positive else ''}("
            if node.ctx.name:
                code += "__result = "
            code += f"rule__{node.name}())) goto {next};\n"
            if node.ctx.name:
                code += f"else {node.ctx.name} = __result{return_type.getter};\n"
            code += "   position = __tempMark;\n"
            code += "}\n"
        elif node.ctx.optional:
            code += "{\n"
            if node.ctx.name:
                var = f"std::optional<{return_type.raw_type}> {node.ctx.name};"
                code += "    auto __result = "
            code += f"    (rule__{node.name}());\n"
            if node.ctx.name:
                code += f"    if (__result) {node.ctx.name} = __result{return_type.getter};\n"
            code += "}\n"
        elif node.ctx.loop:
            if node.ctx.name:
                var = f"std::vector<{return_type.raw_type}> {node.ctx.name};"
            code += "{\n"
            if node.ctx.loop_nonempty:
                code += "    size_t __i = 0;\n"
            code += "    for (;;)\n"
            code += "    {\n"
            if node.ctx.name:
                code += f"    {self.rules_return_type[node.name]} __result;\n"
            code += f"        if (!({'__result = ' if node.ctx.name else ''} rule__{node.name}())) break;\n"
            if node.ctx.name:
                code += f"        {node.ctx.name}.push_back(__result{return_type.getter});\n"
            if node.ctx.loop_nonempty:
                code += "        __i++;\n"
            code += "    }\n"
            if node.ctx.loop_nonempty:
                code += f"\n    if (!__i) goto {next};\n"
            code += "}\n"
        else:
            code += "{\n"
            if node.ctx.name:
                var = f"{return_type.raw_type} {node.ctx.name};"
                code += f"    {return_type} __result;\n"
            code += "    if (!("
            if node.ctx.name:
                code += "__result = "
            code += f"rule__{node.name}())) goto {next};\n"
            if node.ctx.name:
                code += f"    {node.ctx.name} = __result{return_type.getter};\n"
            code += "}\n"
        return GeneratedExpression(code, var)

    def gen_parsing_expr_string(self, node: ParsingExpressionStringNode, next: str) -> GeneratedExpression:
        var = None
        code = ""
        str_bytes = string_to_bytes(node.value)
        str_len = len(str_bytes)
        str_condition = ""

        for i, b in enumerate(str_bytes):
            if b < 128:
                str_condition += f"   && this->src[this->position + {i}] == '{escape_string(chr(b))}'\n"
            else:
                str_condition += f"   && this->src[this->position + {i}] == '\\x{b:x}'\n"

        if node.ctx.lookahead:
            if node.ctx.lookahead_positive:
                code += f"if (this->position + {str_len - 1} >= this->src.size()) goto {next};\n"
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
            code += f"if (this->position + {str_len - 1} < this->src.size())\n"
            code += "{\n"
            code += "    if ((true\n"
            code += add_indent(str_condition, 4)
            code += "    ))\n"
            code += "    {\n"
            if node.ctx.name:
                var = f"bool {node.ctx.name} = false;"
                code += f"        {node.ctx.name} = true;\n"
            code += f"        this->position += {str_len};\n"
            code += "    }\n"
            code += "}\n"
        elif node.ctx.loop:
            code += "{\n"
            if node.ctx.loop_nonempty:
                code += "    size_t __i = 0;\n"
            code += "    for (;;)\n"
            code += "    {\n"
            code += f"        if (this->position + {str_len - 1} >= this->src.size()) break;\n"
            code += "        if (!(true\n"
            code += add_indent(str_condition, 8)
            code += "        )) break;\n"
            if node.ctx.name:
                var = f"size_t {node.ctx.name} = 0;"
                code += f"        {node.ctx.name}++;\n"
            code += f"        this->position += {str_len};\n"
            if node.ctx.loop_nonempty:
                code += "        __i++;\n"
            code += "    }\n"
            if node.ctx.loop_nonempty:
                code += f"\n    if (!__i) goto {next};\n"
            code += "}\n"
        else:
            code += f"if (this->position + {str_len - 1} >= this->src.size()) goto {next};\n"
            code += "if (!(true\n"
            code += str_condition
            code += f")) goto {next};\n"
            code += f"this->position += {str_len};\n"
        return GeneratedExpression(code, var)

    def generate_character_class_condition(self, characters: str) -> str:
        condition = ""
        i = 0
        while i < len(characters):
            ch = characters[i]
            escaped_ch = escape_string(ch) if ch != "\\" else "'\\'"
            if i + 2 < len(characters) and characters[i + 1] == "-":
                condition += f"    || __ch >= 0x{ord(ch):06x}"
                escaped_ch2 = escape_string(characters[i + 2]) if characters[i + 2] != "\\" else "'\\'"
                condition += f" && __ch <= 0x{ord(characters[i + 2]):06x} // {escaped_ch}, {escaped_ch2}\n"
                i += 2
            else:
                condition += f"    || __ch == 0x{ord(ch):06x} // {escaped_ch}\n"
            i += 1
        return condition

    def gen_parsing_expr_character_class(self, node: ParsingExpressionCharacterClassNode, next: str) -> GeneratedExpression:
        var = ""
        code = ""
        condition = self.generate_character_class_condition(node.characters)

        if node.ctx.lookahead:
            if node.ctx.lookahead_positive:
                code += "{\n"
                code += "    size_t __n;\n"
                code += "    char32_t __ch;\n"
                code += f"    if (!(__n = getUtf32Char(__ch))) goto {next};\n"
                code += "    if (!(false\n"
                code += add_indent(condition, 4)
                code += f"    )) goto {next};\n"
                if node.ctx.name:
                    var = f"std::string {node.ctx.name};"
                    code += f"    {node.ctx.name} = this->src.substr(this->position, __n);\n"
                code += "}\n"
            else:
                code += "if (char32_t __ch; getUtf32Char(__ch))\n"
                code += "{\n"
                code += "    if((false\n"
                code += add_indent(condition, 4)
                code += f"    )) goto {next};\n"
                code += "}\n"
                code += f"else goto {next};\n"
        elif node.ctx.optional:
            code += "{\n"
            code += "    char32_t __ch;\n"
            code += "    if (size_t __n = getUtf32Char(__ch))\n"
            code += "    {\n"
            code += "        if ((false\n"
            code += add_indent(condition, 8)
            code += "        ))\n"
            code += "        {\n"
            if node.ctx.name:
                var = f"std::optional<std::string> {node.ctx.name};"
                code += f"            {node.ctx.name} = this->src.substr(this->position, __n);\n"
            code += "            this->position += __n;\n"
            code += "        }\n"
            code += "    }\n"
            code += "}\n"
        elif node.ctx.loop:
            code += "{\n"
            if node.ctx.loop_nonempty:
                code += "    size_t __i = 0;\n"
            code += "    size_t __n;\n"
            code += "    char32_t __ch;\n"
            code += "    for(;;)\n"
            code += "    {\n"
            code += "        if (!(__n = getUtf32Char(__ch))) break;\n"
            code += "        if (!(false\n"
            code += add_indent(condition, 8)
            code += "        )) break;\n"
            if node.ctx.name:
                var = f"std::string {node.ctx.name};"
                code += f"        {node.ctx.name} += this->src.substr(this->position, __n);\n"
            code += "        this->position += __n;\n"
            if node.ctx.loop_nonempty:
                code += "        __i++;\n"
            code += "    }\n"
            if node.ctx.loop_nonempty:
                code += f"\n    if (!__i) goto {next};\n"
            code += "}\n"
        else:
            code += "{\n"
            code += "    size_t __n;\n"
            code += "    char32_t __ch;\n"
            code += f"    if (!(__n = getUtf32Char(__ch))) goto {next};\n"
            code += "    if (!(false\n"
            code += add_indent(condition, 4)
            code += f"    )) goto {next};\n"
            if node.ctx.name:
                var = f"std::string {node.ctx.name};"
                code += f"    {node.ctx.name} = this->src.substr(this->position, __n);\n"
            code += "    this->position += __n;\n"
            code += "}\n"
        return GeneratedExpression(code, var)

    def gen_parsing_expr_inside_group(
            self, node: ParsingExpressionSequence, next: str, expr_index: int, prefix: str,) -> tuple[str, list[str]]:
        group_index = 1
        generated_exprs: list[GeneratedExpression | GeneratedGroupExpression] = []
        for i in node.items:
            match i:
                case ParsingExpressionRuleNameNode():
                    if (rule := self.ctx.get_rule_by_name(i.name)) and rule.inline:
                        group = ParsingExpressionGroupNode(rule.expression_sequences)
                        group.ctx = i.ctx
                        generated_exprs.append(self.gen_parsing_expr_group(group, next, f"{prefix}_{expr_index}_{group_index}"))
                        group_index += 1
                    else:
                        generated_exprs.append(self.gen_parsing_expr_rule_name(i, next))
                case ParsingExpressionStringNode():
                    generated_exprs.append(self.gen_parsing_expr_string(i, next))
                case ParsingExpressionCharacterClassNode():
                    generated_exprs.append(self.gen_parsing_expr_character_class(i, next))
                case ParsingExpressionGroupNode():
                    generated_exprs.append(self.gen_parsing_expr_group(i, next, f"{prefix}_{expr_index}_{group_index}"))
                    group_index += 1
                case ParsingExpressionDotNode():
                    generated_exprs.append(self.gen_parsing_expr_dot(i, next))
                case _:
                    self.gen_type_error(node)

        vars = []
        for g in generated_exprs:
            match g:
                case GeneratedExpression():
                    if var := g.user_defined_var:
                        vars.append(var)
                case GeneratedGroupExpression():
                    vars.extend(g.user_defined_vars)

        code = "{\n"
        for g in generated_exprs:
            code += add_indent(g.code, 4)
            code += "\n"
        code += f"    goto {prefix}_SUCCESS;\n"
        code += "}\n"
        return code, vars

    def gen_parsing_expr_group(self, node: ParsingExpressionGroupNode, next: str, prefix: str) -> GeneratedGroupExpression:
        code = ""
        vars = []
        body = ""

        body += "auto __mark = this->position;\n"
        for i, parsing_expression in enumerate(node.parsing_expression):
            if i > 0:
                body += f"{prefix}_NEXT_{i}:\n"
                body += "this->position = __mark;\n"
            group_next = f"{prefix}_NEXT_{i + 1}" if i + 1 < len(node.parsing_expression) else f"{prefix}_FAIL"
            code_, vars_ = self.gen_parsing_expr_inside_group(parsing_expression, group_next, i + 1, prefix)
            body += code_
            vars.extend(vars_)
            body += "\n"

        if node.ctx.name:
            code += f"auto {prefix}_start_position = this->position;\n"

        if node.ctx.lookahead:
            code += "{\n"
            code += add_indent(body, 4)
            code += f"{prefix}_FAIL:\n"
            code += "    this->position = __mark;\n"
            if node.ctx.lookahead_positive:
                code += f"    goto {next};\n"
            else:
                code += f"    goto {prefix}_END;\n"
            code += f"{prefix}_SUCCESS:\n"
            code += "    this->position = __mark;\n"
            if not node.ctx.lookahead_positive:
                code += f"    goto {next};\n"
                code += f"{prefix}_END:;"
            code += "}\n"
        elif node.ctx.optional:
            code += "{\n"
            code += add_indent(body, 4)
            code += f"{prefix}_FAIL:\n"
            code += "    this->position = __mark;\n"
            code += "    // fallthrough\n"
            code += f"{prefix}_SUCCESS:;\n"
            code += "}\n"
        elif node.ctx.loop:
            code += "{\n"
            if node.ctx.loop_nonempty:
                code += "    size_t __i = 0;\n"
            code += "    for (;;)\n"
            code += "    {\n"
            code += add_indent(body, 8)
            code += f"    {prefix}_FAIL:\n"
            code += "        this->position = __mark;\n"
            code += "        break;\n"
            code += f"    {prefix}_SUCCESS:;\n"
            if node.ctx.loop_nonempty:
                code += "        __i++;\n"
            code += "    }\n"
            if node.ctx.loop_nonempty:
                code += f"    if (!__i) goto {next};\n"
            code += "}\n"
        else:
            code += "{\n"
            code += add_indent(body, 4)
            code += f"{prefix}_FAIL:\n"
            code += "    this->position = __mark;\n"
            code += f"    goto {next};\n"
            code += f"{prefix}_SUCCESS:;\n"
            code += "}\n"

        if node.ctx.name:
            vars.append(f"{'std::optional<std::string>' if node.ctx.optional else'std::string'} {node.ctx.name};")
            code += f"if (this->position != {prefix}_start_position)\n"
            code += "{\n"
            code += f"    {node.ctx.name} = std::string{{this->src.substr({prefix}_start_position,"
            code += f"    this->position - {prefix}_start_position)}};\n"
            code += "}\n"
        return GeneratedGroupExpression(code, vars)

    def gen_parsing_expr_dot(self, node: ParsingExpressionDotNode, next: str) -> GeneratedExpression:
        code = ""
        var = ""

        if node.ctx.lookahead:
            if not node.ctx.lookahead_positive:
                code += f"if (this->position < this->src.size()) goto {next};\n"
            else:
                code += f"if (this->position >= this->src.size()) goto {next};\n"
            if node.ctx.name:
                var = f"std::string {node.ctx.name};"
                code += f"{node.ctx.name} = this->src[this->position];\n"
            code += "this->position++;\n"
        elif node.ctx.optional:
            code += "if (size_t __n = getUtf8Size())\n"
            code += "{\n"
            if node.ctx.name:
                var = f"std::optional<std::string> {node.ctx.name};"
                code += f"    {node.ctx.name} = this->src.substr(this->position, __n);\n"
            code += "    this->position += __n;\n"
            code += "}\n"
        elif node.ctx.loop:
            code += "{\n"
            if node.ctx.loop_nonempty:
                code += "    size_t __i = 0;\n"
            code += "    for (;;)\n"
            code += "    {\n"
            code += "        size_t __n = getUtf8Size();\n"
            code += "        if (!__n) break;\n"
            if node.ctx.name:
                var = f"std::string {node.ctx.name};"
                code += f"        {node.ctx.name} += this->src.substr(this->position, __n);\n"
            code += "        this->position += __n;\n"
            if node.ctx.loop_nonempty:
                code += "        __i++;\n"
            code += "    }\n"
            if node.ctx.loop_nonempty:
                code += f"    if (!__i) goto {next};\n"
            code += "}\n"
        else:
            code += "{\n"
            code += "    size_t __n = getUtf8Size();\n"
            code += f"    if (!__n) goto {next};\n"
            if node.ctx.name:
                var = f"std::string {node.ctx.name};"
                code += f"    {node.ctx.name} = this->src.substr(this->position, __n);\n"
            code += "    this->position += __n;\n"
            code += "}\n"
        return GeneratedExpression(code, var)


def generate_parser(file):
    filename = os.path.basename(file.name)
    tokenizer = Tokenizer(file)
    parser = Parser(tokenizer)
    root_node = parser.parse()
    ctx = AstContext(root_node, filename)
    StaticAnalyzer(ctx).analyze()
    code_gen = CodeGenerator(ctx)
    code_gen.start()


argument_parser = argparse.ArgumentParser(description="Peg parser generator")
argument_parser.add_argument('--version', action='version', version=f"%(prog)s {VERSION}")
argument_parser.add_argument("path", type=argparse.FileType(encoding="utf-8"))
arguments = argument_parser.parse_args()
generate_parser(arguments.path)
