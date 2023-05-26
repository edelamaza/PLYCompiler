import ply.yacc as yacc
import ply.lex as lex
from lexer import tokens

# PIla operandos, pila opradores, pila tipos, pila saltos

currToken = ''
prevToken = ''


class Quadruple:
    def __init__(self, operator, operand1, operand2, result):
        self.operator = operator
        self.operand1 = operand1
        self.operand2 = operand2
        self.result = result

    def print(self):
        print(self.operator, self.operand1, self.operand2, self.result)


class QuadrupleManager:
    def __init__(self):
        self.operators = []
        self.operands = []
        self.types = []
        self.jumps = []
        self.quadruples = []
        self.temp_count = 0

    def generate_temp(self):
        temp = f"t{self.temp_count}"
        self.temp_count += 1
        return temp

    def add_quadruple(self, operator, operand1, operand2, result):
        quadruple = Quadruple(operator, operand1, operand2, result)
        self.quadruples.append(quadruple)

    def push_operator(self, operator):
        self.operators.append(operator)

    def pop_operator(self):
        if self.operators:
            return self.operators.pop()
        return None

    def push_type(self, data_type):
        self.types.append(data_type)

    def pop_type(self):
        if self.types:
            return self.types.pop()
        return None

    def push_jump(self, jump):
        self.jumps.append(jump)

    def pop_jump(self):
        if self.jumps:
            return self.jumps.pop()
        return None

    def push_operand(self, operation):
        self.operands.append(operation)

    def pop_operand(self):
        if self.operands:
            return self.operands.pop()
        return None

    def print_stacks(self):
        print("Operators:", self.operators)
        print("Types:", self.types)
        print("Jumps:", self.jumps)
        print("Operands:", self.operands)
        for quad in self.quadruples:
            quad.print()

    def generate_arithmetic(self):
        # Check Types
        type1 = self.pop_type()
        type2 = self.pop_type()

        if (type1 == type2 and len(self.operators) != 0):
            operand2 = self.pop_operand()
            operand1 = self.pop_operand()
            operator = self.pop_operator()
            print(operator)
            result = self.generate_temp()
            self.add_quadruple(operator, operand1, operand2, result)
            self.push_operand(result)
            self.push_type(type1)

    def generate_assignment(self, identifier):
        operator = self.pop_operator()
        operand1 = self.pop_operand()
        result = identifier
        self.add_quadruple(operator, operand1, None, result)
        self.pop_type()

    def generate_repeat_op(self):
        operator = self.pop_operator()
        operand = self.pop_operand()
        result = self.generate_temp()
        self.push_operand(result)
        self.add_quadruple(operator, operand, 1, result)

    def generate_write(self):
        operand = self.pop_operand()
        self.pop_type()
        self.add_quadruple('write', operand, None, None)


quadrupleMan = QuadrupleManager()

# Variable Table
varTable = {
    # 'a' : ['int', value]
}

varTempArr = []

# Grammar Rules


def p_program(p):
    'program : PROGRAM IDENTIFIER LCURLYBRACE vars block RCURLYBRACE'
    p[0] = p[1]


def p_vars(_):
    '''vars : VAR varsp COLON type seentype SEMICOLON
            | VAR varsp COLON type seentype SEMICOLON vars
            | empty '''


def p_varsp(_):
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
        if p[-1] in ['int', 'real']:
            varTable[id] = ['NUMBER_CONST', None]

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


def p_block(_):
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

def p_statement(_):
    '''statement : empty
                | assign
                | assign statement
                | writefunction
                | writefunction statement'''

# def p_condition(p):
#     '''condition : IF LPAREN expression RPAREN THEN LCURLYBRACE statement RCURLYBRACE
#                  | IF LPAREN expression RPAREN THEN LCURLYBRACE statement RCURLYBRACE ELSE LCURLYBRACE statement RCURLYBRACE'''
#     # queda pendiente el else if


def p_assign(_):
    'assign : IDENTIFIER ASSIGNOP expression assignnow SEMICOLON'


def p_assignnow(p):
    'assignnow : '
    # TODO
    # Handle errors where variables arent defined
    # Assing value into Variable Table
    # varTable[p[-3]][1] = p[-1]
    # Push assign into operator stack
    identifier = p[-3]
    quadrupleMan.push_operator(':=')
    quadrupleMan.generate_assignment(identifier)
    print(varTable)


def p_expression(p):
    '''expression : simpleexpression
                | simpleexpression LESS_THAN seenoperator simpleexpression genquad
                | simpleexpression LESS_THAN_EQUALS seenoperator simpleexpression genquad
                | simpleexpression GREATER_THAN seenoperator simpleexpression genquad
                | simpleexpression GREATER_THAN_EQUALS seenoperator simpleexpression genquad
                | simpleexpression NOT_EQUALS seenoperator simpleexpression genquad
                | simpleexpression EQUALS seenoperator simpleexpression genquad
                    '''
    # if len(p) == 2:
    p[0] = p[1]
    # quadrupleMan.generate_arithmetic()
    # elif p[2] == '>':
    #     p[0] = float(p[1]) > float(p[3])


def p_simpleexpression(p):
    '''simpleexpression : term seenterm simpleexpressionp'''
    if len(p) == 2:
        p[0] = p[1]
    elif p[2] == '+':
        quadrupleMan.push_operator('+')
        quadrupleMan.generate_arithmetic()
        # p[0] = float(p[1]) + float(p[3])
    elif p[2] == '-':
        quadrupleMan.push_operator('-')
        quadrupleMan.generate_arithmetic()

        # p[0] = float(p[1]) - float(p[3])
    # elif p[2] == 'or':
    #     p[0] = float(p[1]) / float(p[3])


def p_seenterm(p):
    '''seenterm :  '''
    # check operator
    if quadrupleMan.operators and quadrupleMan.operators[-1] in ['+', '-', 'OR']:
        # generate arithmetic
        quadrupleMan.generate_arithmetic()
    if quadrupleMan.operators and quadrupleMan.operators[-1] in ['++', '--']:
        # generate arithmetic
        quadrupleMan.generate_repeat_op()


def p_simpleexpressionp(p):
    '''simpleexpressionp : empty
                        | PLUS seenoperator simpleexpression
                        | MINUS seenoperator simpleexpression
                        | OR seenoperator simpleexpression'''


def p_term(p):
    '''term : factor seenfactor termp'''

    # p[0] = float(p[1]) % float(p[3])
    # elif p[2] == '++':
    #     quadrupleMan.push_operator('+')
    #     p[0] = float(p[1]) + 1
    # elif p[2] == '--':
    #     p[0] = float(p[1]) - 1
    # elif p[2] == 'AND':
    #     p[0] = float(p[1]) % float(p[3])


def p_termp(p):
    '''termp : empty
            | MULTIPLY seenoperator term
            | DIV seenoperator term
            | DIVIDE seenoperator term
            | MOD seenoperator term
            | AND seenoperator term 
            | PLUSPLUS seenoperator
            | MINUSMINUS seenoperator'''


def p_seenoperator(p):
    '''seenoperator : '''
    # push p[-1]
    quadrupleMan.push_operator(p[-1])


def p_seenfactor(p):
    '''seenfactor :  '''
    # check operator
    if quadrupleMan.operators and quadrupleMan.operators[-1] in ['*', '/', 'DIV', 'MOD', 'AND']:
        # generate arithmetic
        quadrupleMan.generate_arithmetic()


def p_genquad(_):
    'genquad : '
    # print('entro genquad')
    quadrupleMan.generate_arithmetic()


def p_factor(p):
    '''factor : const
                | LPAREN expression RPAREN'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[2]


def p_const(p):
    '''const : PLUS IDENTIFIER
            | MINUS IDENTIFIER
            | IDENTIFIER
            | PLUS NUMBER_CONST
            | MINUS NUMBER_CONST
            | NUMBER_CONST
            | STRING_CONST
    '''
    print('entro const')
    print(prevToken.type)
    if len(p) == 2:
        if (prevToken.type == 'IDENTIFIER'):
            # Check VarTable to get Type
            quadrupleMan.push_type(varTable[prevToken.value][0])
            quadrupleMan.push_operand(prevToken.value)
        elif prevToken.type == "NUMBER_CONST":
            quadrupleMan.push_type(prevToken.type)
            quadrupleMan.push_operand(float(p[1]))
        else:
            quadrupleMan.push_type(prevToken.type)
            quadrupleMan.push_operand(str(p[1]))
        # p[0] = p[1]
    elif p[1] == '-':
        if (prevToken.type == 'IDENTIFIER'):
            # Check VarTable to get Type
            quadrupleMan.push_type(varTable[prevToken.value][0])
            quadrupleMan.push_operand(-prevToken.value)
            # p[0] = -float(p[2])
        elif prevToken.type == "NUMBER_CONST":
            quadrupleMan.push_type(prevToken.type)
            quadrupleMan.push_operand(-float(p[2]))
            # p[0] = -float(p[2])
        else:
            # TODO
            # Strings with negative sign
            print('error string cannot be negative')
    else:
        if (prevToken.type == 'IDENTIFIER'):
            # Check VarTable to get Type
            quadrupleMan.push_type(varTable[prevToken.value][0])
            quadrupleMan.push_operand(prevToken.value)
        elif prevToken.type == "NUMBER_CONST":
            quadrupleMan.push_type(prevToken.type)
            quadrupleMan.push_operand(-float(p[2]))
        else:
            # TODO
            # Strings with positive sign
            print('error string cannot be positive')


def p_writefunction(p):
    '''writefunction : PRINT LPAREN expression RPAREN SEMICOLON
                    | WRITE LPAREN expression RPAREN SEMICOLON'''
    # print(p[3])
    quadrupleMan.generate_write()


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
        global currToken
        global prevToken

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
        prevToken = currToken
        currToken = return_token
        return return_token


lexy = PA2Lexer()

# Build the parser
parser = yacc.yacc()


result = parser.parse(lexer=lexy)
# print(result)
quadrupleMan.print_stacks()
