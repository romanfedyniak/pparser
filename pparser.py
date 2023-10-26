from contextlib import AbstractContextManager
from dataclasses import dataclass, field
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
    CODE_SECTION = enum.auto()
    COLON = re.compile(r":")
    STRING = re.compile(r"(?<!\\)\".*?(?<!\\)\"")
    CHARACTER_CLASS = re.compile(r"(?<!\\)\[.*?(?<!\\)\]")
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
            elif self.pos + 4 < len(self.src) and self.src[self.pos:self.pos+4] == "%cpp":
                start_pos = self.pos
                self.pos += 5
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

    def code_statement(self):
        with self.manager:
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


def generate_parser(file):
    # parser_name = os.path.basename(file.name)[0]
    tokenizer = Tokenizer(file)
    for token in tokenizer.tokenize():
        print(token)
    print("="*20)
    parser = Parser(tokenizer)
    for node in parser.parse().statements:
        print(node)


argument_parser = argparse.ArgumentParser(description="Peg parser generator`")
argument_parser.add_argument('--version', action='version', version=f"%(prog)s {VERSION}")
argument_parser.add_argument("path", type=argparse.FileType(encoding="utf-8"))
arguments = argument_parser.parse_args()
generate_parser(arguments.path)
