"""AI-assisted bounded physics/chemistry grammar, preserving slash and [] meaning.

No formula lookup or algebraic correction. Charges are parsed explicitly.
"""
import re
from .config import CLASSES, FUNCTIONS

OPS = {'=': ('eq', 0), '\\approx': ('approx', 0), '+': ('add', 1),
       '-': ('minus', 1), '\\pm': ('pm', 1), '\\times': ('cross', 2),
       '\\cdot': ('dot', 2), '/': ('div', 2)}
REVERSE = {value[0]: (key, value[1]) for key, value in OPS.items()}


def tokenize(text):
    return re.findall(r'\\[a-zA-Z]+|\\.|[^\s]', text)


class Parser:
    def __init__(self, tokens):
        self.tokens = [t for t in tokens if t not in ('\\left', '\\right', '\\,', '\\!', '\\;')]
        self.index = 0

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self, expected=None):
        value = self.peek()
        if value is None or expected is not None and value != expected:
            raise ValueError('Expected ' + str(expected) + ', got ' + str(value))
        self.index += 1
        return value

    def group(self):
        self.take('{'); node = self.expression(); self.take('}')
        return node

    def script(self):
        # H^{+}, SO_4^{2-}: a terminal sign is a charge, not a missing operand.
        if self.peek() == '{':
            end = self.index + 1
            while end < len(self.tokens) and self.tokens[end] != '}':
                end += 1
            raw = self.tokens[self.index + 1:end]
            if raw and raw[-1] in ('+', '-') and all(t.isdigit() for t in raw[:-1]):
                self.index = end
                self.take('}')
                return ('charge', ''.join(raw))
            return self.group()
        return self.atom(scripts=False)

    def atom(self, scripts=True):
        token = self.take()
        if token in ('(', '[', '{'):
            node = self.expression()
            self.take({'(': ')', '[': ']', '{': '}'}[token])
            if token == '[':
                node = ('square', node)  # Concentration brackets are meaningful.
        elif token == '\\frac':
            node = ('div', self.group(), self.group())
        elif token == '\\sqrt':
            node = ('sqrt', self.group())
        elif token == '\\mathrm':
            node = self.group()
        elif token in FUNCTIONS:
            index = (self.take(), self.script()) if self.peek() in ('^', '_') else None
            child = self.group() if self.peek() == '{' else self.atom()
            node = ('indexed_function', token, index[0], index[1], child) if index else ('function', token, child)
        elif token in ('+', '-'):
            node = ('unary', token, self.atom())
        elif token.isdigit() or token == '.':
            value = token
            while self.peek() is not None and (self.peek().isdigit() or self.peek() == '.'):
                value += self.take()
            if not re.fullmatch(r'\d+(?:\.\d+)?', value):
                raise ValueError('Invalid decimal.')
            node = ('symbol', value)
        elif token in CLASSES and token not in OPS and token not in (')', ']', '\\sqrt'):
            node = ('symbol', token)
        else:
            raise ValueError('Unsupported atom: ' + token)
        while scripts and self.peek() in ('^', '_'):
            operator = self.take()
            node = ('script', operator, node, self.script())
        return node

    def expression(self, minimum=0):
        node = self.atom()
        while self.peek() is not None and self.peek() not in (')', ']', '}'):
            token = self.peek()
            if token in OPS:
                name, priority = OPS[token]
                if priority < minimum:
                    break
                self.take(); right = self.expression(priority + 1)
            else:
                if minimum > 2:
                    break
                name, right = 'multiply', self.expression(3)
            node = (name, node, right)
        return node


def parse(tokens):
    parser = Parser(tokens)
    tree = parser.expression()
    if parser.peek() is not None:
        raise ValueError('Trailing tokens.')
    return tree


def serialize(tree, fractions=False):
    k = tree[0]
    if k in ('symbol', 'charge'):
        return tree[1] + (' ' if tree[1].startswith('\\') else '')
    if k == 'square':
        return '[' + serialize(tree[1], fractions) + ']'
    if k == 'sqrt':
        return '\\sqrt{' + serialize(tree[1], fractions) + '}'
    if k == 'function':
        return tree[1] + '{' + serialize(tree[2], fractions) + '}'
    if k == 'indexed_function':
        return tree[1] + tree[2] + '{' + serialize(tree[3], fractions) + '}{' + serialize(tree[4], fractions) + '}'
    if k == 'unary':
        return tree[1] + '(' + serialize(tree[2], fractions) + ')'
    if k == 'script':
        base = serialize(tree[2], fractions)
        if tree[2][0] in REVERSE or tree[2][0] in ('multiply', 'unary'):
            base = '(' + base + ')'
        return base + tree[1] + '{' + serialize(tree[3], fractions) + '}'
    a, b = serialize(tree[1], fractions), serialize(tree[2], fractions)
    if k == 'div':
        return '\\frac{' + a + '}{' + b + '}' if fractions else '(' + a + ')/(' + b + ')'
    priority = 2 if k == 'multiply' else REVERSE[k][1]
    def child(text, node, right):
        # A sign after an implicit product is otherwise read as binary +/-:
        # a*(-b) must serialize as a (-(b)), never a -(b).
        if k == 'multiply' and right and node[0] == 'unary':
            return '(' + text + ')'
        child_priority = 2 if node[0] == 'multiply' else REVERSE.get(node[0], ('', 10))[1]
        return '(' + text + ')' if child_priority < priority or right and child_priority == priority else text
    a, b = child(a, tree[1], False), child(b, tree[2], True)
    if k == 'multiply' and a[-1:].isdigit() and b[:1].isdigit():
        b = '(' + b + ')'
    return a + (' ' if k == 'multiply' else ' ' + REVERSE[k][0] + ' ') + b


def normalize(tokens, notation='slash'):
    tree = parse(tokens)
    if notation not in ('slash', 'stacked'):
        raise ValueError('Unknown fraction notation.')
    result = tokenize(serialize(tree, fractions=notation == 'stacked'))
    if parse(result) != tree:
        raise ValueError('Normalization changed the syntax tree.')
    return result, repr(tree)


def display_latex(tokens, fractions=False):
    return serialize(parse(tokens), fractions)


def content_key(tokens):
    return normalize(tokens)[1]


def kind(tokens):
    return 'fraction' if '/' in tokens or '\\frac' in tokens else 'sub' if '_' in tokens else 'sup' if '^' in tokens else 'basic'
