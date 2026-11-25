"""AI-assisted report templates, real-ink sources, and immutable shared inputs.

Synthetic test scores are explicitly NOT handwriting/photo benchmark scores.
Fractions in actual source images are excluded: relabelling a stacked fraction
as slash does not produce a slash image. Generated images really contain '/'.
"""
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw
from .config import CLASSES, FUNCTIONS, PREVIOUS, PROJECT, ROOT, TOKENS
from .notation import content_key, kind, normalize, parse, tokenize
from .progress import Progress, save_json


# Split entire formula families before drawing constants or choosing glyphs.
# Validation/test use different formula templates, not random image augmentations.
FAMILIES = {
 'train': [r'F=m a', r'P V=n R T', r'q=m c \Delta T', r'E_k=(m v^{2})/2',
    r'v=v_0+a t', r'x=x_0+v_0 t+(a t^{2})/2', r'\rho=m/V', r'P=F/A',
    r'W=F d \cos{\theta}', r'P=W/t', r'V=I R', r'Q=I t', r'P=I V',
    r'v=\lambda \nu', r'E=h \nu', r'\lambda=h/p', r'T=2 \pi \sqrt{l/g}',
    r'F=(G m_1 m_2)/(r^{2})', r'c=n/V', r'n=m/M', r'\Delta G=\Delta H-T \Delta S',
    r'\Delta G=-(R T \ln{K})', r'p H=-\log_{10}{[H^{+}]}',
    r'K_a=([H^{+}][A^{-}])/([H A])', r'A=\epsilon l c',
    r'\ln{k}=\ln{A}-E_a/(R T)', r'C=Q/V', r'F=q v B \sin{\theta}',
    r'k=A \exp{-(E_a/(R T))}', r'[H^{+}][O H^{-}]=K_w'],
 'val': [r'p=m v', r'\omega=2 \pi \nu', r'\Delta U=q+W',
    r'E=(k Q)/r^{2}', r'\Delta G=\Delta G_0+R T \ln{Q}', r'p K_a=-\log_{10}{K_a}'],
 'test': [r'F=(m v^{2})/r', r'Q=m L', r'R=(\rho l)/A',
    r'p H+p O H=14', r'K_p=K_c(R T)^{\Delta n}', r'\Delta S=q/T']}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pixel_hash(image):
    image = image.convert('L')
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


FONT_FAMILIES = ('stix', 'cm', 'dejavusans')


@lru_cache(maxsize=512)
def _cached_font_symbol(token, font):
    """Return a standalone glyph; sqrt never relies on a MathText alias.

    The input convention is sqrt(argument), so this draws just the radical
    symbol. No dummy argument or extra recognized glyph is introduced.
    """
    if token == '\\sqrt':
        factor = 4
        canvas = Image.new('L', (54*factor, 56*factor), 255)
        draw = ImageDraw.Draw(canvas)
        points = [(4, 32), (12, 28), (21, 47), (33, 7), (49, 7)]
        draw.line([(x*factor, y*factor) for x, y in points],
                  fill=0, width=3*factor, joint='curve')
        return canvas.resize((54, 56), Image.Resampling.LANCZOS)
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser
    parser = MathTextParser('agg')
    text = r'{}^{\circ}' if token == '\\circ' else token
    result = parser.parse('$' + text + '$', dpi=120,
        prop=FontProperties(size=30, math_fontfamily=font))
    return Image.fromarray(255 - np.asarray(result.image))


def render_font_symbol(parser, token, font):
    # The returned image is private: caller transforms cannot mutate the cache.
    return _cached_font_symbol(token, font).copy()


def validate_rendering(output, progress):
    """User-run preflight: all fallback symbols/fonts before source preparation."""
    from matplotlib.mathtext import MathTextParser
    parser = MathTextParser('agg')
    total = len(FONT_FAMILIES)*len(CLASSES)
    checked = 0
    for font in FONT_FAMILIES:
        for token in CLASSES:
            try:
                image = render_font_symbol(parser, token, font)
                if not image.width or not image.height or not np.any(np.asarray(image) < 200):
                    raise ValueError('Blank fallback glyph.')
            except (ValueError, RuntimeError) as exc:
                save_json(output/'render_preflight_failure.json',
                    {'token': token, 'font': font, 'error': str(exc)})
                progress.message(f'대체 기호 렌더링 검사 실패: {token!r}, font={font}')
                raise ValueError(f'Render preflight failed: token={token!r}, '
                                 f'font={font}; {exc}') from exc
            checked += 1
            if checked % 30 == 0 or checked == total:
                progress.update('대체 기호 렌더링 검사', checked, total)
    progress.message('모든 대체 기호/폰트의 렌더링 검사 완료.')


class Renderer:
    def __init__(self, glyphs, settings, rng, split):
        from matplotlib.mathtext import MathTextParser
        self.parser = MathTextParser('agg')
        self.glyphs, self.settings, self.rng, self.split = glyphs, settings, rng, split
        self.handwritten = self.font = self.procedural = 0

    def symbol(self, token, height=42):
        pool = self.glyphs.pools[self.split].get(token, [])
        use_hand = bool(pool) and self.rng.random() < self.settings.handwritten_probability
        if use_hand:
            image = Image.fromarray(self.glyphs.bank[self.rng.choice(pool)])
            self.handwritten += 1
        else:
            font = self.rng.choice(['stix', 'cm']) if self.split == 'train' else 'dejavusans'
            image = render_font_symbol(self.parser, token, font)
            if token == '\\sqrt':
                self.procedural += 1
            else:
                self.font += 1
        box = Image.fromarray((np.asarray(image) < 200).astype('uint8') * 255).getbbox()
        if box:
            image = image.crop(box)
        desired = max(2, round(height * (.13 if token in ('.', '\\cdot') else .25 if token == '\\circ' else 1.)))
        scale = min(desired / max(1, image.height), height * 1.5 / max(1, image.width))
        return image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))), Image.Resampling.LANCZOS)

    def text(self, text, height=42):
        tokens = tokenize(text)
        # Function names use separate letter images, not a fictitious log class.
        images = [self.symbol(t, height) for t in tokens]
        width = sum(i.width for i in images) + max(0, len(images)-1)*5
        out = Image.new('L', (max(1, width), height+8), 255)
        at = 0
        for image in images:
            out.paste(image, (at, height+3-image.height)); at += image.width+5
        return out

    def join(self, images, height=42):
        baseline = max(image[1] for image in images)
        bottom = max(im.height-base for im, base in images)
        gap = self.rng.randint(4, 9)
        out = Image.new('L', (sum(im.width for im, _ in images)+gap*(len(images)-1), baseline+bottom), 255)
        at = 0
        for image, base in images:
            out.paste(image, (at, baseline-base)); at += image.width+gap
        return out, baseline

    def draw(self, tree, height=42):
        k = tree[0]
        if k in ('symbol', 'charge'):
            image = self.text(tree[1], height); return image, height
        if k == 'script':
            base, baseline = self.draw(tree[2], height)
            small, _ = self.draw(tree[3], max(12, round(height*.55)))
            top = max(0, small.height-baseline//2) if tree[1] == '^' else 0
            y = top if tree[1] == '^' else top+baseline-3
            out = Image.new('L', (base.width+small.width+3, max(base.height+top, y+small.height)), 255)
            out.paste(base, (0, top)); out.paste(small, (base.width+3, y))
            return out, baseline+top
        if k == 'square':
            inside = self.draw(tree[1], height)
            return self.join([(self.symbol('[', height), height), inside, (self.symbol(']', height), height)])
        if k == 'indexed_function':
            label = self.draw(('script', tree[2], ('symbol', tree[1][1:]), tree[3]), height)
            return self.join([label, (self.symbol('(', height), height), self.draw(tree[4], height),
                              (self.symbol(')', height), height)])
        if k in ('sqrt', 'function', 'unary'):
            name = '\\sqrt' if k == 'sqrt' else tree[1][1:] if k == 'function' else tree[1]
            child = tree[1] if k == 'sqrt' else tree[2]
            label = self.text(name, height) if k != 'sqrt' else self.symbol(name, height)
            return self.join([(label, height), (self.symbol('(', height), height), self.draw(child, height),
                              (self.symbol(')', height), height)])
        a, b = self.draw(tree[1], height), self.draw(tree[2], height)
        operator = {'div': '/', 'eq': '=', 'approx': '\\approx', 'add': '+', 'minus': '-',
                    'pm': '\\pm', 'cross': '\\times', 'dot': '\\cdot'}.get(k)
        if k == 'div':
            # Actual slash raster, including explicit numerator/denominator grouping.
            return self.join([(self.symbol('(', height), height), a, (self.symbol(')', height), height),
                (self.symbol('/', height), height), (self.symbol('(', height), height), b,
                (self.symbol(')', height), height)])
        pieces = []
        for child_index, (child, rendered) in enumerate([(tree[1], a), (tree[2], b)]):
            if pieces and operator:
                pieces.append((self.symbol(operator, height), height))
            # Explicit grouping preserves the generated tree even when nested.
            adjacent_numbers = (k == 'multiply' and child_index == 1 and tree[1][0] == child[0] == 'symbol'
                                and tree[1][1][-1:].isdigit() and child[1][:1].isdigit())
            if child[0] in ('eq', 'add', 'minus', 'pm', 'div', 'multiply', 'cross', 'dot', 'approx') or adjacent_numbers:
                pieces.extend([(self.symbol('(', height), height), rendered, (self.symbol(')', height), height)])
            else:
                pieces.append(rendered)
        return self.join(pieces)


def sample_formula(template, rng):
    tokens = tokenize(template)
    # Consistent variable renaming within each sample; no correction by formula lookup.
    reserved = set('pHOHART')
    variables = sorted({t for t in tokens if len(t) == 1 and t.isalpha() and t not in reserved})
    choices = list('bcdfgjkmnqrsuvxyz')
    rng.shuffle(choices)
    names = dict(zip(variables, choices))
    tokens = [names.get(t, t) for t in tokens]
    # Overall scalar coefficient varies content without rewriting the relationship.
    if rng.random() < .7:
        coefficient = str(rng.randint(2, 9999))
        equals = tokens.index('=') if '=' in tokens else None
        if equals is not None:
            tokens = tokens[:equals+1] + list(coefficient) + ['('] + tokens[equals+1:] + [')']
    return normalize(tokens)[0]


def validate_templates(settings):
    """Run by the user's entry point before costly glyph/data preparation.

    Include scaled variants: testing an unscaled template alone would miss
    implicit multiplication followed by a negative logarithm.
    """
    for split, templates in FAMILIES.items():
        for index, template in enumerate(templates):
            original = tokenize(template)
            variants = [original]
            if '=' in original:
                at = original.index('=')
                variants.append(original[:at+1] + list('1234') + ['('] + original[at+1:] + [')'])
            for tokens in variants:
                try:
                    canonical, _ = normalize(tokens)
                    if set(canonical) - set(TOKENS[4:]):
                        raise ValueError('Template contains unsupported target tokens.')
                    if len(canonical) >= settings.max_tokens:
                        raise ValueError('Template exceeds max_tokens.')
                except (ValueError, RecursionError) as exc:
                    raise ValueError(f'Template preflight failed: split={split}, family={index}, '
                                     f'template={template!r}, error={exc}') from exc
