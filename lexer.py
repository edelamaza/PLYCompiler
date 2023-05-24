import ply.lex as lex

# List of token names
tokens = (
    'IDENTIFIER',
    'STRING_CONST',
    'PLUS',
    'MINUS',
    'MULTIPLY',
    'DIVIDE',
    'MOD',
    'AND',
    'OR',
    'ASSIGN',
    'EQUALS',
    'NOT_EQUALS',
    'LESS_THAN',
    'LESS_THAN_EQUALS',
    'GREATER_THAN',
    'GREATER_THAN_EQUALS',
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
    'PROGRAM',
    'VAR',
    'BEGIN',
    'END',
    'IF',
    'THEN',
    'ELSE',
    'WHILE',
    'DO',
    'FOR',
    'BOOLEAN',
    'CHAR',
    'STRING',
    'TRUE',
    'FALSE',
    'NOT',
    'INT',
    'REAL',
    'WRITE',
    'PRINT',
    'NUMBER_CONST',
    'DIV'
)


# Regular expression rules for simple tokens
t_PLUS = r'\+'
t_MINUS = r'-'
t_MULTIPLY = r'\*'
t_DIVIDE = r'/'
t_MOD = r'MOD'
t_AND = r'AND'
t_OR = r'OR'
t_ASSIGN = r':='
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

# Test it out
data = '''
program main{
var i,n,f,x : int;
begin;
write("texto dump");
f := 1;
x := 5;
if( x > f) then 
{
	n := x + f;
}else{
	n := x - 4;
}
write(n);
	end;
}

'''

# Give the lexer some input
lexer.input(data)

# Tokenize
while True:
    tok = lexer.token()
    if not tok:
        break  # No more input
    print(tok)
