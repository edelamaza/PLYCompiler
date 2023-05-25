import ply.yacc as yacc
import ply.lex as lex
from lexer import tokens

# Grammar Rules
# PIla operandos, pila opradores, pila tipos, pila saltos
operators = []
operands = []
jumps = []
types = []
cuadruples = [operators, operands, types, jumps]

# Variable Table
varTable = {
    # 'a' : ['int', value]
}

varTempArr = []


def p_program(p):
    'program : PROGRAM IDENTIFIER LCURLYBRACE vars block RCURLYBRACE'
    p[0] = p[1]


def p_vars(p):
    '''vars : VAR varsp COLON type seentype SEMICOLON
            | VAR varsp COLON type seentype SEMICOLON vars
            | empty '''


def p_varsp(p):
    '''varsp : IDENTIFIER seenid
            | IDENTIFIER seenid COMMA varsp'''


def p_seenid(p):
    '''seenid : '''
    # add into variable table
    global varTempArr
    varTempArr.append(p[-1])


def p_seentype(p):
    'seentype : '
    # agregar a tabla variables
    # agrego a pila tipos
    # type = p[-1]
    global varTempArr
    global varTable
    for id in varTempArr:
        varTable[id] = [p[-1], None]

    # TODO
    # Error handling for duplicate variables
    # id = p[-3] si ya esta es error (duplicate)

    # Reset var arraytemp
    varTempArr = []


def p_type(p):
    '''type : INT
            | REAL
            | STRING
            | BOOLEAN'''
    p[0] = p[1]


def p_block(p):
    'block : BEGIN SEMICOLON statement END SEMICOLON'


# def p_statement(p):
#     '''statement : empty
#                 | assign
#                 | condition
#                 | for
#                 | while
#                 | write
#                 | assign statement
#                 | condition statement
#                 | for statement
#                 | while statement
#                 | write statement'''

def p_statement(p):
    '''statement : empty
                | assign
                | assign statement'''

# def p_condition(p):
#     '''condition : IF LPAREN expression RPAREN THEN LCURLYBRACE statement RCURLYBRACE
#                  | IF LPAREN expression RPAREN THEN LCURLYBRACE statement RCURLYBRACE ELSE LCURLYBRACE statement RCURLYBRACE'''
#     # queda pendiente el else if


def p_assign(p):
    'assign : IDENTIFIER ASSIGNOP expression SEMICOLON'


def p_expression(p):
    '''expression : simpleexpression
                    | simpleexpression LESS_THAN simpleexpression
                    | simpleexpression LESS_THAN_EQUALS simpleexpression
                    | simpleexpression GREATER_THAN simpleexpression
                    | simpleexpression GREATER_THAN_EQUALS simpleexpression
                    | simpleexpression NOT_EQUALS simpleexpression
                    | simpleexpression EQUALS simpleexpression
                    '''
    p[0] = p[1]
    # Assing value into Variable Table
    # TODO
    # Handle errors where variables arent defined
    varTable[p[-2]][1] = p[0]
    print(varTable)


def p_simpleexpression(p):
    '''simpleexpression : term
                        | term PLUS simpleexpression
                        | term MINUS simpleexpression
                        | term OR simpleexpression
                        '''
    p[0] = p[1]


def p_term(p):
    '''term : factor
            | factor DIV term
            | factor MULTIPLY term
            | factor DIVIDE term
            | factor MOD term
            | factor AND term
            | factor PLUSPLUS
            | factor MINUSMINUS
    '''
    p[0] = p[1]


def p_factor(p):
    '''factor : const
                | LPAREN expression RPAREN'''
    p[0] = p[1]


def p_const(p):
    '''const : PLUS IDENTIFIER
            | MINUS IDENTIFIER
            | IDENTIFIER
            | PLUS NUMBER_CONST
            | MINUS NUMBER_CONST
            | NUMBER_CONST
    '''
    p[0] = p[1]


def p_empty(p):
    'empty : '

# Error rule for syntax errors


def p_error(p):
    print("Syntax error in input! Near '%s' line: %s" % (p.value, p.lineno))


input_file_path = "lexerOut.txt"
input_file = open(input_file_path, "r")
raw_input = input_file.read().split()
print(raw_input)


class PA2Lexer(object):
    def token(self):
        global raw_input

        if len(raw_input) == 0:
            return None

        token_lexeme = raw_input.pop(0)
        line_number = raw_input.pop(0)
        token_type = raw_input.pop(0)
        return_token = lex.LexToken()
        return_token.lineno = int(line_number)
        return_token.value = token_lexeme
        return_token.type = token_type
        return_token.lexpos = 0
        print(return_token)
        return return_token


lexy = PA2Lexer()

# Build the parser
parser = yacc.yacc()


result = parser.parse(lexer=lexy)
print(result)
