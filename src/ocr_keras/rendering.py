"""AI-assisted AST composition with pixel ownership, relations and token anchors.

Annotations are TRAINING targets only. No label/anchor is passed to recognition.
All fallback font families occur in every split; original ink split is preserved.
"""
from dataclasses import dataclass, field
import random
import numpy as np
from PIL import Image
from .config import CLASSES
from .glyph_rendering import Renderer as GlyphRenderer, FAMILIES
from .notation import tokenize,parse,normalize

HARD_FORMULAS=(
    r'Q=(\rho a+\rho b)/t',
    r'c=p O + p 0',
    r'A_0=B_1 + C',
    r'v_a=t_b^{2}+x',
    r'q=(A_i+B_j)/(C_k+D_l)',
    r'M_i=(x_j)^{\Delta m}+A',
)
FRESH_FORMULAS=(
    r'F=\rho (u-v)^{2}',
    r'n_a=c_a V_a+c_b V_b',
    r'D=(u_2-u_1)/(t_2-t_1)',
    r'Q=n C_{p}\Delta T',
    r'c_b=(c_a V_a)/(V_a+V_b)',
    r'K=\exp{-(\Delta G_0/(RT))}',
)

@dataclass
class Layout:
    image:Image.Image
    owner:Image.Image
    glyphs:list
    tokens:list
    anchors:list
    edges:list=field(default_factory=list)
    baseline:float=None
    def __post_init__(self):
        if self.baseline is None:self.baseline=self.image.height

def compose(parts,positions,size,link='Right'):
    image=Image.new('L',size,255);owner=Image.new('I',size,0)
    glyphs=[];tokens=[];anchors=[];edges=[];previous=None
    for part,(x,y) in zip(parts,positions):
        offset=len(glyphs);image.paste(part.image,(x,y))
        a=np.asarray(part.owner,dtype=np.int32);a=np.where(a>0,a+offset,0)
        owner.paste(Image.fromarray(a),(x,y))
        glyphs += [{**g,'box':[g['box'][0]+x,g['box'][1]+y,g['box'][2]+x,g['box'][3]+y]} for g in part.glyphs]
        tokens+=part.tokens;anchors += [[i+offset for i in v] for v in part.anchors]
        edges += [[a+offset,b+offset,r] for a,b,r in part.edges]
        if previous is not None and part.glyphs and link:edges.append([previous,offset,link])
        if part.glyphs:previous=offset+len(part.glyphs)-1
    return Layout(image,owner,glyphs,tokens,anchors,edges)

class Renderer(GlyphRenderer):
    def __init__(self,glyphs,settings,rng,split,notation,compact_scripts=False):
        super().__init__(glyphs,settings,rng,split);self.notation=notation
        self.glyph_seed=str(rng.getstate());self.occurrences={}
        self.compact_scripts=compact_scripts
    def symbol(self,token,height=42):
        from .io import identity
        count=self.occurrences.get(token,0);self.occurrences[token]=count+1
        previous=self.rng;self.rng=random.Random(identity([self.glyph_seed,token,count]))
        try:
            # Reuse audited glyph drawing; all three fonts can appear at train/val/test.
            pool=self.glyphs.pools[self.split].get(token,[])
            if pool and self.rng.random()<self.settings.handwritten_probability:
                im=self.glyphs.image(self.rng.choice(pool))
            else:
                from .glyph_rendering import render_font_symbol
                im=render_font_symbol(self.parser,token,self.rng.choice(('stix','cm','dejavusans')))
            box=Image.fromarray((np.asarray(im)<200).astype('uint8')*255).getbbox()
            if box:im=im.crop(box)
            desired=height*(.14 if token in ('.',r'\cdot') else .25 if token==r'\circ' else 1)
            scale=min(desired/max(1,im.height),height*1.5/max(1,im.width))
            im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS)
            # Per-glyph scale/rotation/ink-width variation preserves exact mask ownership.
            angle=self.rng.uniform(-3,3);im=im.rotate(angle,expand=True,fillcolor=255)
            own=Image.fromarray((np.asarray(im)<200).astype(np.int32))
            return Layout(im,own,[{'label':token,'box':[0,0,im.width,im.height]}],[token],[[0]])
        finally:self.rng=previous
    def line(self,parts,h):
        gap=self.rng.randint(3,10);baseline=max(p.baseline for p in parts)
        height=round(baseline+max(p.image.height-p.baseline for p in parts))
        xy=[];x=0
        for p in parts:xy.append((x,round(baseline-p.baseline)));x+=p.image.width+gap
        out=compose(parts,xy,(max(1,x-gap),height));out.baseline=baseline;return out
    def grouped(self,node,h):return self.line([self.symbol('(',h),node,self.symbol(')',h)],h)
    def draw(self,tree,height=42):
        k=tree[0];h=height
        if k in ('symbol','charge'):
            return self.line([self.symbol(t,h) for t in tokenize(tree[1])],h)
        if k=='square':return self.line([self.symbol('[',h),self.draw(tree[1],h),self.symbol(']',h)],h)
        if k=='script':
            base=self.draw(tree[2],h)
            if tree[2][0] not in ('symbol','charge','script','square'):base=self.grouped(base,h)
            ratio=self.rng.uniform(.38,.66) if self.compact_scripts else .55
            small=self.draw(tree[3],max(9 if self.compact_scripts else 12,round(h*ratio)))
            top=max(0,small.image.height-round(h*.35)) if tree[1]=='^' else 0
            sy=0 if tree[1]=='^' else top+max(0,base.image.height-round(h*.2))
            out=compose([base,small],[(0,top),(base.image.width+3,sy)],(base.image.width+3+small.image.width,max(top+base.image.height,sy+small.image.height)),None)
            n=len(base.glyphs);a=[[i+n for i in v] for v in small.anchors]
            out.tokens=base.tokens+[tree[1],'{']+small.tokens+['}']
            out.anchors=base.anchors+[[n],[]]+a+[[]]
            near_baseline=[i for i,g in enumerate(base.glyphs)
                           if abs(g['box'][3]-base.baseline)<=max(3,.22*h)]
            parent=max(near_baseline or list(range(n)),key=lambda i:base.glyphs[i]['box'][2])
            out.edges.append([parent,n,'Sup' if tree[1]=='^' else 'Sub'])
            out.baseline=base.baseline+top
            return out
        if k in ('sqrt','function','indexed_function','unary'):
            name=r'\sqrt' if k=='sqrt' else tree[1]
            child=tree[1] if k=='sqrt' else tree[4] if k=='indexed_function' else tree[2]
            label=self.symbol(name,h) if k=='sqrt' else self.line([self.symbol(t,h) for t in tokenize(name[1:] if name.startswith('\\') else name)],h)
            if k=='indexed_function':
                label=self.draw(('script',tree[2],('symbol',name[1:]),tree[3]),h)
            inside=self.draw(child,h);out=self.line([label,self.grouped(inside,h)],h)
            if k in ('sqrt','function','indexed_function'):
                offset=len(label.glyphs)+1
                aa=[[i+offset for i in a] for a in inside.anchors]
                if k=='indexed_function':
                    # The function-name token is anchored to all printed name letters.
                    idx=self.draw_index_tokens(tree[3])
                    name_count=len(tokenize(name[1:]));out.tokens=[name,tree[2],'{']+idx+['}','{']+inside.tokens+['}']
                    # Index tokens need not be guided; lexical child/function anchors suffice.
                    out.anchors=[list(range(name_count)),[],[]]+[[] for _ in idx]+[[],[]]+aa+[[]]
                else:out.tokens=[name,'{']+inside.tokens+['}'];out.anchors=[list(range(len(label.glyphs))),[]]+aa+[[]]
                out.edges.append([0,offset,'Inside'])
            return out
        a=self.draw(tree[1],h);b=self.draw(tree[2],h)
        if k=='div' and self.notation=='stacked':
            w=max(a.image.width,b.image.width)+12;gap=8
            # A fraction bar is procedural and must not consume a shared '-' glyph.
            bar=Layout(Image.new('L',(w-4,3),0),Image.new('I',(w-4,3),1),[{'label':'-','box':[0,0,w-4,3]}],['-'],[[0]])
            out=compose([a,bar,b],[((w-a.image.width)//2,0),(2,a.image.height+gap),((w-b.image.width)//2,a.image.height+2*gap+3)],(w,a.image.height+b.image.height+2*gap+3),None)
            ai=len(a.glyphs);bi=ai+1
            out.tokens=[r'\frac','{']+a.tokens+['}','{']+b.tokens+['}']
            out.anchors=[[ai],[]]+a.anchors+[[],[]]+[[i+bi for i in v] for v in b.anchors]+[[]]
            out.edges += [[ai,0,'Above'],[ai,bi,'Below']];out.baseline=a.image.height+gap+2;return out
        if k=='div':return self.line([self.grouped(a,h),self.symbol('/',h),self.grouped(b,h)],h)
        op={'eq':'=','approx':r'\approx','add':'+','minus':'-','pm':r'\pm','cross':r'\times','dot':r'\cdot'}.get(k)
        # Explicit raster parentheses make the target unambiguous for every AST.
        if tree[1][0] in ('eq','add','minus','pm','multiply','div','cross','dot','approx'):a=self.grouped(a,h)
        if tree[2][0] in ('eq','add','minus','pm','multiply','div','cross','dot','approx'):b=self.grouped(b,h)
        # A*(-b) must remain a product, not turn into A-b. The serializer also
        # needs this grouping; scaled negative logarithms are a common case.
        if k=='multiply' and tree[2][0]=='unary':b=self.grouped(b,h)
        if k=='multiply' and a.tokens[-1].isdigit() and b.tokens[0].isdigit():b=self.grouped(b,h)
        return self.line([a,b] if op is None else [a,self.symbol(op,h),b],h)
    @staticmethod
    def draw_index_tokens(tree):
        from .notation import serialize
        return tokenize(serialize(tree))

def sample_formula(rng,split,index,fresh=False):
    """Exact content dedup is applied later across ALL splits and real sources."""
    if fresh:
        at=index%len(FRESH_FORMULAS)
        text=FRESH_FORMULAS[at]
        family=f'fresh_test_{at}'
    elif split=='train' and index%4==0:
        at=(index//4)%len(HARD_FORMULAS)
        text=HARD_FORMULAS[at]
        family=f'hard_train_{at}'
    elif split=='test' or split=='val' and index%2:
        at=(index//2 if split=='val' else index)%len(FAMILIES[split])
        text=FAMILIES[split][at]
        family=f'heldout_{split}_{at}'
    elif rng.random()<.35:
        text=rng.choice(FAMILIES['train']);family='report_template'
    else:
        variables=list('FmavtqPnRVIEQkcxyz');rng.shuffle(variables)
        def expr(depth):
            if depth<=0 or rng.random()<.35:return rng.choice(variables) if rng.random()<.8 else str(rng.randint(1,20))
            op=rng.choice(('product','sum','minus','div','sup','sub','function'))
            a=expr(depth-1)
            if op=='sup':return '('+a+')^{'+str(rng.randint(2,4))+'}'
            if op=='sub':return '('+a+')_{'+rng.choice(('0','1','2','a','b'))+'}'
            if op=='function':return rng.choice((r'\sin',r'\cos',r'\ln'))+'{'+a+'}'
            b=expr(depth-1);operator={'product':' ','sum':'+','minus':'-','div':'/'}[op]
            return '('+a+')'+operator+'('+b+')'
        text=variables[0]+'='+expr(rng.randint(1,3));family='grammar'
    tokens=tokenize(text)
    # Retain authentic coefficients most often; long numbers remain a minority.
    chance=rng.random()
    if chance<.25 and '=' in tokens:
        coefficient=str(rng.randint(2,20) if chance<.20 else rng.randint(21,9999));at=tokens.index('=')
        tokens=tokens[:at+1]+list(coefficient)+['(']+tokens[at+1:]+[')']
    # Consistent renaming expands content while preserving fixed report notations.
    names=sorted({t for t in tokens if len(t)==1 and t.isalpha() and t not in set('pHOHART')})
    choices=list('bcdfgjkmnqrsuvxyz');rng.shuffle(choices);mapping=dict(zip(names,choices))
    return normalize([mapping.get(t,t) for t in tokens])[0],family

def validate_renderer(settings):
    """Fail before merge training if any font/template/scaled template is invalid."""
    from .glyph_rendering import render_font_symbol
    class FontGlyphs:
        bank=np.empty((0,48,48),np.uint8)
        pools={s:{c:[] for c in CLASSES} for s in ('train','val','test')}
    for token in CLASSES:
        for font in ('stix','cm','dejavusans'):
            im=render_font_symbol(None,token,font)
            if not np.any(np.asarray(im)<200):raise ValueError(f'Blank fallback: {token}/{font}')
    count=0
    templates_by_split={**FAMILIES,'train':[*FAMILIES['train'],*HARD_FORMULAS],'test':[*FAMILIES['test'],*FRESH_FORMULAS]}
    for split,templates in templates_by_split.items():
        for text in templates:
            at=text.index('=')+1
            for source in (text,text[:at]+'1234('+text[at:]+')'):
                tree=parse(tokenize(source))
                for notation in ('stacked','slash'):
                    layout=Renderer(FontGlyphs(),settings,random.Random(42),split,notation).draw(tree)
                    if parse(layout.tokens)!=tree:raise ValueError('Render preflight changed AST: '+source)
                    if len(layout.tokens)!=len(layout.anchors):raise ValueError('Token/anchor count mismatch: '+source)
                    count+=1
    print(f'[preflight] 폰트 {len(CLASSES)*3}개 / 기본·계수 변형·양쪽 표기 {count}개 검사 완료',flush=True)
    return count
