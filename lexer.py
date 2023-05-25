import ply.lex as lex

# List of token names
tokens = (
    # Keywords
    'PROGRAM',
    'VAR',
    'BEGIN',
    'END',
    'IDENTIFIER',

    # Types
    'STRING_CONST',
    'BOOLEAN',
    'INT',
    'REAL',
    'CHAR',
    'NUMBER_CONST',

    # Arithmetic Operators
    'PLUS',
    'MINUS',
    'MULTIPLY',
    'DIVIDE',
    'MOD',
    'PLUSPLUS',
    'MINUSMINUS',
    'DIV',


    # Logical Operators
    'AND',
    'OR',
    'EQUALS',
    'NOT_EQUALS',
    'LESS_THAN',
    'LESS_THAN_EQUALS',
    'GREATER_THAN',
    'GREATER_THAN_EQUALS',
    'ASSIGNOP',

    # Syntax Elements
    'LPAREN',
    'RPAREN',
    'LBRACKET',
    'RBRACKET',
    'LCURLYBRACE',
    'RCURLYBRACE',
    'COMMA',
    'SEMICOLON',
    'COLON',
    'PERIOD',

    # Repeat Statements
    'FOR',
    'WHILE',
    'DO',

    # Conditionals
    'IF',
    'ELSE',
    'THEN',
    'STRING',
    'TRUE',
    'FALSE',
    'NOT',

    # Misc
    'WRITE',
    'PRINT',

)


# Regular expression rules for simple tokens
t_PLUS = r'\+'
t_MINUS = r'-'
t_MULTIPLY = r'\*'
t_DIVIDE = r'/'
t_ASSIGNOP = r':='
t_EQUALS = r'='
t_NOT_EQUALS = r'<>'
t_LESS_THAN = r'<'
t_LESS_THAN_EQUALS = r'<='
t_GREATER_THAN = r'>'
t_GREATER_THAN_EQUALS = r'>='
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_COMMA = r','
t_SEMICOLON = r';'
t_COLON = r':'
t_PERIOD = r'\.'
t_LCURLYBRACE = r'\{'
t_RCURLYBRACE = r'\}'
t_PLUSPLUS = r'\+\+'
t_MINUSMINUS = r'--'


# Identifiers can start with a letter or underscore, followed by any number of letters, digits, or underscores.
# They cannot start with a digit.
# Identifier token
def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = t.value.upper() if t.value.upper() in tokens else 'IDENTIFIER'
    return t


def t_NUMBER_CONST(t):
    r'\d+(\.\d*)?([eE][+-]?\d+)?'
    t.value = float(t.value)
    return t


# String constants
def t_STRING_CONST(t):
    r'\"([^\\\n]|(\\.))*?\"'
    t.value = t.value[1:-1]  # Remove the quotes from the string
    return t


# Error handling rule
def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)


# Define a rule so we can track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)


# Ignore whitespace characters
t_ignore = ' \t\r\n'

# Build the lexer
lexer = lex.lex()

input_file_path = "inputFile2.txt"
input_file = open(input_file_path, "r")
file_contents = input_file.read()
lexer.input(file_contents)
# Define output file
output_file_path = "lexerOut.txt"
output_file = open(output_file_path, "w")

# Tokenize
while True:
    tok = lexer.token()
    if not tok:
        break  # No more input
    output_file.write(str(tok.value) + ' ')
    output_file.write(str(tok.lineno) + ' ')
    output_file.write(str(tok.type) + ' ')

output_file.close()
