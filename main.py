from enum import Enum
from Tree import Node
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:\\Users\\User\\PycharmProjects\\CustomProgrammingLanguage\\Graphviz-12.0.0-win64\\Bin'
#import networkx as nx
#import matplotlib.pyplot as plt

# Programming lang action plan
# Define Tokens
# Create a Lexer 
# Use lexer to convert statements to tokens
# organise into a tree (abstract syntax trees) :)
# parse tokens into machine code using custom compiler


class Ttypes(Enum):
    OP = 1
    INT = 2
    STRING = 3
    BOOL = 4
    ID = 5
    NULL = 6
    PUNCTUATOR = 7
    NEWLINE = 8
    IDENTIFIER = 9
    FUNC = 0


class Token:
    def __init__(self, _type=None, value=""):
        self.value = value
        self.type = _type
    def __str__(self):
        return f"[{self.type}: {self.value}]"


class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = -1
        self.current = None
        self.operators = "+-*/^%&|!="
        self.punctuation = "():"
        self.newline = ";"
        self.tokens = []
        self.inString = False
        self.advance()

    def advance(self):
        self.pos += 1
        i = self.pos
        self.current = self.text[i]
        c = self.current

        # add a space at the beginning to make sure the prev character check is valid
        text = " " + self.text

        # Check if current char can be tokenised
        if self.inString:
            self.tokens[-1].value += c
            if c == "'":
                self.inString = False

        elif c == "'":
            self.tokens.append(Token(Ttypes.STRING, c))
            self.inString = True

        elif c in self.operators:
            self.tokens.append(Token(Ttypes.OP, c))

        # Convert to integer
        elif c.isdigit():
            if text[i].isdigit():
                self.tokens[-1].value += c
            else:
                self.tokens.append(Token(Ttypes.INT, c))

        elif c.isalpha():
            self.tokens.append(Token(_type=Ttypes.IDENTIFIER))
            name = self.generate_word()
            if name == "Nocap" or name == "cap":
                self.tokens[-1].type = Ttypes.BOOL
            elif name == "empty":
                self.tokens[-1].type = Ttypes.NULL


            # Write elif/if statments for selection and iteration statements



        elif c in self.punctuation:
            if c == "(":
                if self.tokens[-1].type == Ttypes.IDENTIFIER:
                    self.tokens[-1].type = Ttypes.FUNC
            self.tokens.append(Token(Ttypes.PUNCTUATOR, c))

        elif c in self.newline:
            self.tokens.append(Token(Ttypes.NEWLINE, c))

        # Figure out a way to change things in the language

    def generate_word(self):
        c = self.current
        while c not in self.punctuation and c != " " and c != ';':
            self.tokens[-1].value += c
            self.pos += 1
            c = self.text[self.pos]
        self.pos -= 1
        return self.tokens[-1].value



class Interpreter:
    def __init__(self, ast):
        self.ast = ast

    # Evaluate expression method with interpreter
    def evaluate(self):
        pass

    # def generate_string(self):
    #     self.pos +=1
    #     c = self.text[self.pos]
    #     while c != "'" and c != "\"":
    #         self.tokens[-1].value += c
    #         self.pos += 1
    #         c = self.text[self.pos]
    #     self.tokens[-1].value += c

# Converts a list of Tokens to an AST
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.expression = Node("Expression")


    def parse(self):
        i = 0
        while i < len(self.tokens):
            temp = []
            if self.tokens[i].type == Ttypes.OP:
                opRoot = Node(self.tokens[i])
                opRoot.children.append(Node(self.tokens[i-1]))
                opRoot.children.append(Node(self.tokens[i + 1]))
                funcRoot.children.append(opRoot)
            if self.tokens[i].value == ")":
                funcRoot.children.append(Node(self.tokens[i]))
                self.expression.children.append(funcRoot)
            if self.tokens[i].type == Ttypes.FUNC:
                funcRoot = Node(self.tokens[i])
                funcRoot.children.append(Node(self.tokens[i+1]))
            if self.tokens[i].type == Ttypes.NEWLINE:
                self.expression.children.append(Node(self.tokens[i]))
            i += 1

        return self.expression.to_string(1)

class Interpreter:
    def interpret(self, expr):
        for branch1 in expr.children:
            if branch1.data.type == Ttypes.FUNC:
                for branch2 in branch1.children:
                    if branch2.data.type == Ttypes.OP:
                        print(int(branch2.children[0].data.value) + int(branch2.children[1].data.value))
text = "PRINT(1513+129);"
lex = Lexer(text)
parser = Parser(lex.tokens)

while lex.pos < len(text)-1:
    lex.advance()

for i in lex.tokens:
    print(i)

print(parser.parse())
interpretor = Interpreter()
interpretor.interpret(parser.expression)
# g = graphviz.Graph('G', filename='process.gv')
#
# g.edge('(', 'PRINT')
# g.edge('expr', 'PRINT')
# g.edge(')', 'PRINT')
# g.edge('1513', 'expr')
# g.edge('+', 'expr')
# g.edge('122', 'expr')
#
# g.view()
