%name calc

%hpp {
    #include <node.h>
}

%cpp {

    #include "test.h"

    struct Node {

    }

}

# comment
%type "Node"
%root Expr

Value = a:[0-9.]+ { $$ = a } | "(" Expr ")"
Product = Expr (("*" | "/") Expr)*
Sum = Expr (("+" | "-") Expr)*
Expr = Product | Sum | Value

_ = [ \t]+
