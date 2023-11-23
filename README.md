# pparser

pparser is [peg](https://en.wikipedia.org/wiki/Parsing_expression_grammar) parser generator for C++ 17.

## Features
- [x] generation of a parser in C++ code
- [x] support for unicode
- [x] support for various return types for rules
- [x] support for direct left-recursive rules
- [x] support for the Packrat parsing algorithm

## Usage
calc.peg:
```
%name calc

%type "int"
%root Sum

Value = a:[0-9]+ { $$ = std::stoi(a); }
Sum = a:Value _ "+" _ b:Value { $$ = a + b; }
_ = [ \t]*
```

generate parser:
```console
$ python pparser.py calc.peg
```
this will generate a .cpp and .hpp file with the parser implementation. The parser will be located within the namespace `PParser`. Now you can include it into your project and use it:

```cpp
#include "parser.hpp"

int main() {
    char str[] = "1 + 2";
    PParser::Parser parser(str); // create parser
    auto result = parser.parse(); // use it
    if (result.has_value())
        std::cout << result.value() << std::endl;
    return 0;
}
```

## Docs
### Comments
Comments start with the symbol #. Anything following # until the end of the line is considered a comment.
```
# this is a comment
```

### Directives
Directives begin with the symbol `%`. The directive can be undeclared, but if declared, it can only be declared once.

- `%name` - Specifies the names for the output .cpp and .hpp files. If not set, the name will be based on the input file's name.
```
%name calculator
```

- `%type` - The type that the parser will return, which can also be used for certain rules if a type is not specified but there is an action.
```
%type "std::string"
```

- `%hpp` - Inserts code at the beginning of the `.hpp` file.
```
%hpp {
    #include <iostream>

    struct Node
    {
        std::string value;
    }
}
```

- `%cpp` - Inserts code into the `.cpp` file.
```
%cpp {
    int main()
    {
        char str[] = "1 + 2";
        Pparser::Parser parser(str);
        auto result = parser.parse();
        std::cout << result << std::endl;
        return 0;
    }
}
```

- `%root` - Specifies the rule from which parsing will begin. If this directive is not specified, parsing will start from the first declared rule.
```
%root Statement

Value = [0-9]+
Statement = Value # starts from here
```

### Rules
Syntax:
```
<RuleName> <return type> <attribute> = <parsing expression sequence> { <C++ code> }  ~{ <C++ code> }
```
- RuleName - this is the name of a rule, and it can consist of letters, digits, and underscores, but it cannot start with a digit. Names are case-sensitive.

- parsing expression sequence - this is the expression being parsed. Separated by spaces, checked from left to right.

```
Rule = "1" "2" "3"
```

#### Parsing Expression
Consists of the following element:

- String - іs enclosed in double quotes, cannot be empty.
```
Rule = "string"
```
Escape sequences are also supported:

| Escape sequence | Hex value in ASCII | Description |
| :-------------: | :----------------: | ----------- |
|     `\a`        |        07          |             |
|     `\b`        |        08          |             |
|     `\f`        |        0C          |             |
|     `\n`        |        0A          |             |
|     `\r`        |        0D          |             |
|     `\t`        |        09          |             |
|     `\v`        |        0B          |             |
|     `\\`        |        5C          | Backslash   |
|     `\"`        |        22          | Double quotation mark |
|     `\x`hh      |        any         | hh - hexadecimal number |

- Character class - this is a set of characters. It is enclosed within square brackets, matches only one character from the sequence, and cannot be empty. You can use a hyphen inside a character class to specify a range of characters. [0-9] matches a single digit between 0 and 9.
```
Rule = [abcd]
Hexadecimal = [0-9a-fA-F] # matches a single hexadecimal digit
```

- Dot - matches any single character. Fails only when the input has ended.
```
AnyCharacter = .
```

- Another rule - invokes another rule.
```
Rule = SecondRule
SecondRule = "hello"
```

#### Operators
Various operators can also be applied to the elements mentioned above:

- `&` - and-predicate. Look ahead into the input string without actually consuming it. True if it exists.
```
Rule1 = "hello" &" world"
```

- `!` - not-predicate. The same as an and-predicate, but true only if the element does not exist.
```
Rule = "hello" !" world"
EOF = !. # if use a not-predicate with dot, you can determine if the input has ended
```

- `*` - zero-or-more. Consumes zero or more occurrences of the element. Always true.
```
Numbers = [0123456789]*
```

- `+` - one-or-more. Consumes one or more occurrences of the element. True when consuming more than one element.
```
Numbers = [0123456789]+
```

- `?` - optional. Looks ahead for the element. If found, consumes it. Always true/
```
Rule = "hello" "world"?
```

- `()` - group. In essence, this is an anonymous rule. It executes expressions within itself. Becomes true if its contents are true.
```
Rule = "1" ("2" "3")
```

- `|` - the choice operator. Splits parsing expression sequence into several different sequences. If the sequence on the left succeeds, returns it; if not, moves to the sequence on the right and executes it. If both sequences fail, returns false.
```
Rule = "foo" | "bar" | "baz"
Rule = [123456789] " " ("+" | "-" | "*" | "/") [123456789]

# You can also write an extra operator at the beginning for aesthetic purposes
Rule =
    | "foo"
    | "bar"
    | "baz"
```

#### Action
The action is arbitrary C++ code to be executed at the end of a matching.
```
Rule = [0-9]+ { std::cout << "number\n"; }
```

Inside actions, you can use special variables.
+ `$$` - represents the output variable. The type of this variable is determined by the `%type` directive, or if it is not defined, it defaults to `size_t`.
```
Rule = "42" { $$ = 42; }
```

+ `$`n - position of the expression. Where n is a number from 1 to the number of expressions.
```cpp
namespace PParser
{
    // The type of the variable is a structure:
    struct TokenPos
    {
        size_t startCol;
        size_t startLine;
        size_t endCol;
        size_t endLine;
    };
}
```
```
Number = [1-9] [0-9]+
Rule = Number "+" Number {
    std::cout << $1.startCol << std::endl;  # position of the first 'Number'
    std::cout << $3.startCol << std::endl;  # position of the second 'Number'
}
```

#### Error Action
The action is arbitrary C++ code to be executed at the end of a matching only if matching fails.
```
Rule = [0-9]+ ~{ parseError("expected number"); }

# Actions and error actions can be used together
Rule = [0-9]+ { std::cout << "number\n"; } ~{ parseError("expected number"); }
```

#### Variable
You can assign a name to any parsing expression in a rule; this will create a variable that can be used in an action. An exception is the '!' operator; if this operator is used, it is not possible to assign the expression to a variable.
```
Number = number:[0-9]+ { $$ = std::stoi(number); }
```
Each parsing expression has its own C++ return type, which also depends on the operators used.
+ String
    + "string" - it is not possible to assign to a variable
    + "string"+ or "string"* - `size_t` (the number of repetitions of the string)
    + "string"? - `bool`
    + &"string" - it is not possible to assign to a variable
+ Character class
    + [a-z] - `std::string`
    + [a-z]+ or [a-z]* - `std::string`
    + [a-z]? - `std::optional<std::string>`
    + &[a-z] - `std::string`
+ Dot
    + . - `std::string`
    + .+ or .* - `std::string`
    + .? - `std::optional<std::string>`
    + &. - `std::string`
+ Group - returns a string consisting of matches within a group
    + ("1" | "2") - `std::string`
    + ("1" | "2")+ or ("1" | "2")* - `std::string`
    + ("1" | "2")? - `std::optional<std::string>`
    + &("1" | "2") - `std::string`
+ Another rule
    + Rule - `the type returned by the invoked rule`
    + Rule+ or Rule* - `std::vector<the type returned by the invoked rule>`
    + Rule? - `std::optional<the type returned by the invoked rule>`
    + &Rule - `the type returned by the invoked rule`

#### Rule return type
The return type of a rule is the return type of the parsing expression sequences. There can only be one return type in a rule, so all sequences of parsing expressions must return the same type.
```
# correct, all sequences return bool
Rule =
    | "1"
    | "2"
    | "3"

# incorrect, the return types of the sequences are different
Rule =
    | "1"
    | "2"
    | n:"3" { $$=std::stoi(n) }
```
If no action is defined with the variable '`$$`', the return type is `bool`. If variable '`$$`' is not defined, the return type is determined by the `%type` directive, or if the directive is not defined, it defaults to `size_t`. Also, the return type can be set manually.
```
Rule<Node*> [0-9]+ { $$ = new Node{}; }
```

#### Attributes
Attributes are specified after the rule name and return type. Attributes start with the - sign.
+ `nomemo` - does not cache the results of the rule.
+ `inline` - inserts expressions directly into the rule. Behaves like a group. The rule cannot have a return type and actions.

```
EOF -nomemo = !.
number<int> -nomemo = num:([1-9][0-9]*) { $$ = std::stoi(num); }
```

## C++ API
The parser is defined in the `PParser` namespace and is named `Parser`.
```cpp
Parser::Parser(std::string_view src);
```
Constructor. Accepts a sequence of characters to be parsed.

```cpp
"return type" Parser::parse();
```
Initiates parsing. The return type of the method is the same as in the root rule, wrapped in `std::optional`, in case the return type is not `bool`.

```cpp
void Parser::parseError(const std::string& msg);
```
This is private method; it can be called withins actions if you want to terminate parsing, calls an error handler if one was set.

```cpp
using errorHandler_t = std::function<void(std::string message, size_t position)>;
void Parser::setErrorHandler(errorHandler_t handler);
```
Sets a handler for errors triggered through parseError.
## License
[MIT](LICENSE)
