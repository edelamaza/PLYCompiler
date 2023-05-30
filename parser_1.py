import ply.yacc as yacc
import ply.lex as lex
from lexer import tokens

# Global Variables
currToken = ''
prevToken = ''

# Operations Type Table
opTypeTable = {
    'NUMBER_CONST': ['NUMBER_CONST', 'int', 'real'],
    'STRING_CONST': ['STRING_CONST'],
    'BOOLEAN': ['BOOLEAN'],
}


class Quadruple:
    def __init__(self, index, operator, operand1, operand2, result):
        self.index = index
        self.operator = operator
        self.operand1 = operand1
        self.operand2 = operand2
        self.result = result

    def print(self):
        print(self.index, self.operator, self.operand1,
              self.operand2, self.result)


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
        index = len(self.quadruples)
        quadruple = Quadruple(index, operator, operand1, operand2, result)
        self.quadruples.append(quadruple)
        return index

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

    def write_quads(self):
        with open('quadruples.txt', 'w') as f:
            for quad in self.quadruples:
                f.write(
                    f"{quad.index}~{quad.operator}~{quad.operand1}~{quad.operand2}~{quad.result}\n")

    def generate_arithmetic(self):
        # Check Types
        type1 = self.pop_type()
        type2 = self.pop_type()

        if (type2 in opTypeTable[type1] and len(self.operators) != 0):
            operand2 = self.pop_operand()
            operand1 = self.pop_operand()
            operator = self.pop_operator()
            result = self.generate_temp()
            # Add it into the variable table
            if type1 == 'NUMBER_CONST':
                varTable[result] = [type1, 0]
            else:
                varTable[result] = [type1, None]

            self.add_quadruple(operator, operand1, operand2, result)
            self.push_operand(result)
            if operator in ['<', '>', '<=', '>=', '<>', '=']:
                self.push_type('BOOLEAN')
            else:
                self.push_type(type1)
        else:
            print('Error, type mismatch')
            return False

    def generate_assignment(self, identifier):
        # Check to see if identifier is in the variable table
        if identifier not in varTable:
            print('Error: Variable {} not defined'.format(identifier))
            p_error('p')

        # Check if identifier has the same type as the expression
        type1 = self.pop_type()
        type2 = varTable[identifier][0]
        if type1 in opTypeTable[type2]:
            operator = self.pop_operator()
            operand1 = self.pop_operand()
            result = identifier
            self.add_quadruple(operator, operand1, None, result)
        else:
            print('Error, type mismatch')
            p_error('p')

    def generate_write(self):
        operand = self.pop_operand()
        self.pop_type()
        self.add_quadruple('write', operand, None, None)

    def check_bool(self):
        type = self.pop_type()
        if type == 'BOOLEAN':
            return True
        print('Error, boolean expression expected')
        return False

    def get_index(self):
        return len(self.quadruples)


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
    global varTempArr
    global varTable

    for id in varTempArr:
        # Check to see if the variable is already in the table
        if id in varTable:
            print('Error, duplicate variable')
            p_error(p)
        if p[-1] in ['int', 'real']:
            varTable[id] = ['NUMBER_CONST', 0]
        elif p[-1] in ['boolean']:
            varTable[id] = ['BOOLEAN', None]
        elif p[-1] in ['string']:
            varTable[id] = ['STRING_CONST', None]

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


def p_statement(_):
    '''statement : empty
                | assign
                | assign statement
                | writefunction
                | writefunction statement
                | condition
                | condition statement
                | while
                | while statement
                | for
                | for statement
                | IDENTIFIER PLUSPLUS seenunary SEMICOLON
                | IDENTIFIER MINUSMINUS seenunary SEMICOLON
                | IDENTIFIER PLUSPLUS seenunary SEMICOLON statement
                | IDENTIFIER MINUSMINUS seenunary SEMICOLON statement'''


def p_condition(p):
    '''condition : IF LPAREN expression RPAREN checkbool seenif THEN LCURLYBRACE statement RCURLYBRACE seencurlyif seencurlyelse
                | IF LPAREN expression RPAREN checkbool seenif THEN LCURLYBRACE statement RCURLYBRACE seencurlyif ELSE condition seencurlyelse
                | IF LPAREN expression RPAREN checkbool seenif THEN LCURLYBRACE statement RCURLYBRACE seencurlyif ELSE LCURLYBRACE statement RCURLYBRACE seencurlyelse'''


def p_checkbool(p):
    'checkbool : '
    if not quadrupleMan.check_bool():
        p_error(p)


def p_seenif(_):
    'seenif : '
    result = quadrupleMan.pop_operand()
    index = quadrupleMan.add_quadruple('gotof', result, None, None)
    quadrupleMan.push_jump(index)


def p_seencurlyif(_):
    'seencurlyif : '
    index_goto = quadrupleMan.add_quadruple('goto', None, None, None)
    index_gotof = quadrupleMan.pop_jump()
    quadrupleMan.push_jump(index_goto)
    quadrupleMan.quadruples[index_gotof].result = index_goto + 1


def p_seencurlyelse(_):
    'seencurlyelse : '
    index_goto = quadrupleMan.pop_jump()
    quadrupleMan.quadruples[index_goto].result = quadrupleMan.get_index()


def p_while(_):
    'while : WHILE LPAREN expression RPAREN checkbool seenwhile DO LCURLYBRACE statement RCURLYBRACE seencurlywhile '


def p_seenwhile(_):
    'seenwhile : '
    index = quadrupleMan.get_index() - 1
    quadrupleMan.push_jump(index)
    result = quadrupleMan.pop_operand()
    index_gotof = quadrupleMan.add_quadruple('gotof', result, None, None)
    quadrupleMan.push_jump(index_gotof)


def p_seencurlywhile(_):
    'seencurlywhile : '
    index_gotof = quadrupleMan.pop_jump()
    index_goto = quadrupleMan.pop_jump()
    quadrupleMan.add_quadruple('goto', None, None, index_goto)
    quadrupleMan.quadruples[index_gotof].result = quadrupleMan.get_index()


def p_for(_):
    '''for : FOR LPAREN assign expression checkbool seenboolfor SEMICOLON expression seenchangefor RPAREN LCURLYBRACE statement RCURLYBRACE seencurlyfor
            | FOR LPAREN assign expression checkbool seenboolfor SEMICOLON assignfor seenchangefor RPAREN LCURLYBRACE statement RCURLYBRACE seencurlyfor'''


def p_seenboolfor(_):
    'seenboolfor : '
    result = quadrupleMan.pop_operand()
    index_gotov = quadrupleMan.add_quadruple('gotov', result, None, None)
    quadrupleMan.push_jump(index_gotov)
    index_gotof = quadrupleMan.add_quadruple('gotof', result, None, None)
    quadrupleMan.push_jump(index_gotof)


def p_seenchangefor(_):
    'seenchangefor : '
    index_gotof = quadrupleMan.pop_jump()
    index_gotov = quadrupleMan.pop_jump()
    index_condition = index_gotov - 1
    quadrupleMan.add_quadruple('goto', None, None, index_condition)
    index = quadrupleMan.get_index()
    quadrupleMan.quadruples[index_gotov].result = index
    quadrupleMan.push_jump(index_gotof)


def p_seencurlyfor(_):
    'seencurlyfor : '
    index_gotof = quadrupleMan.pop_jump()
    index_change = index_gotof + 1
    quadrupleMan.add_quadruple('goto', None, None, index_change)
    index = quadrupleMan.get_index()
    quadrupleMan.quadruples[index_gotof].result = index


def p_assign(_):
    'assign : IDENTIFIER ASSIGNOP expression assignnow SEMICOLON'


def p_assignfor(_):
    'assignfor : IDENTIFIER ASSIGNOP expression assignnow'


def p_assignnow(p):
    'assignnow : '
    identifier = p[-3]
    quadrupleMan.push_operator(':=')
    quadrupleMan.generate_assignment(identifier)


def p_expression(p):
    '''expression : simpleexpression
                | simpleexpression LESS_THAN seenoperator simpleexpression genquad
                | simpleexpression LESS_THAN_EQUALS seenoperator simpleexpression genquad
                | simpleexpression GREATER_THAN seenoperator simpleexpression genquad
                | simpleexpression GREATER_THAN_EQUALS seenoperator simpleexpression genquad
                | simpleexpression NOT_EQUALS seenoperator simpleexpression genquad
                | simpleexpression EQUALS seenoperator simpleexpression genquad
                | IDENTIFIER PLUSPLUS seenunary
                | IDENTIFIER MINUSMINUS seenunary
                    '''
    p[0] = p[1]


def p_seenunary(p):
    'seenunary : '
    quadrupleMan.add_quadruple(p[-1], p[-2], 1, p[-2])


def p_simpleexpression(p):
    '''simpleexpression : term seenterm simpleexpressionp'''


def p_seenterm(_):
    'seenterm :  '
    # check operator
    if quadrupleMan.operators and quadrupleMan.operators[-1] in ['+', '-', 'OR']:
        # generate arithmetic
        quadrupleMan.generate_arithmetic()


def p_simpleexpressionp(p):
    '''simpleexpressionp : empty
                        | PLUS seenoperator simpleexpression
                        | MINUS seenoperator simpleexpression
                        | OR seenoperator simpleexpression'''


def p_term(p):
    '''term : factor seenfactor termp'''


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
    quadrupleMan.push_operator(p[-1])


def p_seenfactor(p):
    '''seenfactor :  '''
    # check operator
    if quadrupleMan.operators and quadrupleMan.operators[-1] in ['*', '/', 'DIV', 'MOD', 'AND']:
        # generate arithmetic
        quadrupleMan.generate_arithmetic()


def p_genquad(_):
    'genquad : '
    quadrupleMan.generate_arithmetic()


def p_factor(p):
    '''factor : const
                | LPAREN seenoperator expression RPAREN exitparen'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[2]


def p_exitparen(_):
    'exitparen : '
    quadrupleMan.pop_operator()


def p_const(p):
    '''const : PLUS IDENTIFIER
            | MINUS IDENTIFIER
            | IDENTIFIER
            | PLUS NUMBER_CONST
            | MINUS NUMBER_CONST
            | NUMBER_CONST
            | STRING_CONST
    '''
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
            # Strings with negative sign
            print('Error string cannot be negative')
            p_error(p)
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
            print('Error string cannot be positive')
            p_error(p)


def p_writefunction(p):
    '''writefunction : PRINT LPAREN expression RPAREN SEMICOLON
                    | WRITE LPAREN expression RPAREN SEMICOLON'''
    quadrupleMan.generate_write()


def p_empty(p):
    'empty : '


# Error rule for syntax errors
def p_error(p):
    print("Syntax error in input! Near '%s' line: %s" % (p.value, p.lineno))


input_file_path = "lexerOut.txt"
input_file = open(input_file_path, "r")
raw_input = input_file.read().split('~')
input_file.close()


class MyLexer(object):
    def token(self):
        global raw_input
        global currToken
        global prevToken

        if len(raw_input) == 1:
            return None

        token_lexeme = raw_input.pop(0)
        line_number = raw_input.pop(0)
        token_type = raw_input.pop(0)
        return_token = lex.LexToken()
        return_token.lineno = int(line_number)
        return_token.value = token_lexeme
        return_token.type = token_type
        return_token.lexpos = 0
        prevToken = currToken
        currToken = return_token
        return return_token


lexy = MyLexer()

# Build the parser
parser = yacc.yacc()


result = parser.parse(lexer=lexy)
quadrupleMan.write_quads()
quadrupleMan.print_stacks()


# # Virtual Machine Execution

op = {'+': lambda x, y: x + y,
      '-': lambda x, y: x - y,
      '/': lambda x, y: x / y,
      '*': lambda x, y: x * y,
      'DIV': lambda x, y: x // y,
      'MOD': lambda x, y: x % y,
      'AND': lambda x, y: x and y,
      'OR': lambda x, y: x or y,
      '<': lambda x, y: x < y,
      '>': lambda x, y: x > y,
      '<=': lambda x, y: x <= y,
      '>=': lambda x, y: x >= y,
      '=': lambda x, y: x == y,
      '<>': lambda x, y: x != y}

opUnary = {'++': lambda x: x + 1,
           '--': lambda x: x - 1}

i = 0
while i < len(quadrupleMan.quadruples):
    quadruple = quadrupleMan.quadruples[i]
    # Switch Case
    if quadruple.operator in ['write', 'print']:
        # Check if operand is vartable
        if quadruple.operand1 in varTable:
            print(varTable[quadruple.operand1][1])
        else:
            print(quadruple.operand1)
    elif quadruple.operator in ['+', '-', '*', '/', 'DIV', 'MOD', 'AND', 'OR', '<', '>', '<=', '>=', '=', '<>']:
        # Check if oprand1 is variable
        if quadruple.operand1 in varTable:
            val1 = varTable[quadruple.operand1][1]
        else:
            val1 = quadruple.operand1
        # Check if operand2 is variable
        if quadruple.operand2 in varTable:
            val2 = varTable[quadruple.operand2][1]
        else:
            val2 = quadruple.operand2

        varTable[quadruple.result][1] = op[quadruple.operator](val1, val2)
    elif quadruple.operator in ['++', '--']:
        varTable[quadruple.result][1] = opUnary[quadruple.operator](
            varTable[quadruple.operand1][1])

    elif quadruple.operator == ':=':
        # Check if operand1 is variable
        if quadruple.operand1 in varTable:
            varTable[quadruple.result][1] = varTable[quadruple.operand1][1]
        else:
            varTable[quadruple.result][1] = quadruple.operand1
    elif quadruple.operator == 'goto':
        i = quadruple.result
        continue
    elif quadruple.operator == 'gotof':
        if not varTable[quadruple.operand1][1]:
            i = quadruple.result
            continue
    elif quadruple.operator == 'gotov':
        if varTable[quadruple.operand1][1]:
            i = quadruple.result
            continue

    i += 1

print(varTable)
