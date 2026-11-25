"""AI-assisted immutable split preparation and per-formula NumPy caches."""
from pathlib import Path
from collections import Counter
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import time
import json,random,os
import hashlib
import numpy as np
from PIL import Image
from .config import CLASSES,TOKENS,VERSION,PROJECT
from .io import read,save,identity,digest,code_hashes,resolve
from .components import preprocess,masked_patch
from .raw import raw_components
from .rendering import Renderer,sample_formula
from .notation import normalize,parse,kind,content_key
from .supervision import match,targets
from .progress import Progress

class Glyphs:
    def __init__(self,sources):
        self.bank=np.load(sources/'glyphs/glyph_bank.npy',mmap_mode='r')
        self.pools={s:{c:[] for c in CLASSES} for s in ('train','val','test')}
        for i,r in enumerate(read(sources/'glyphs/glyph_metadata.json')):
            self.pools[r['split']][r['label']].append(i)
        bank=PROJECT/'resources/hard_glyphs.npz'
        with np.load(bank,allow_pickle=False) as extra:
            self.hard_images=extra['images']
            for i,label in enumerate(extra['labels']):
                self.pools['train'][str(label)].append(len(self.bank)+i)

    def image(self,index):
        if index<len(self.bank):
            return Image.fromarray(self.bank[index])
        return Image.fromarray(self.hard_images[index-len(self.bank)])

def pixel_key(image):
    im=image.convert('L')
    return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()

def crop(im):
    im=im.convert('L');box=Image.fromarray((np.asarray(im)<180).astype('uint8')*255).getbbox()
    if box:im=im.crop(box)
    im.thumbnail((42,42),Image.Resampling.LANCZOS);out=Image.new('L',(48,48),255)
    out.paste(im,((48-im.width)//2,(48-im.height)//2))
    return ((255-np.asarray(out,dtype='float32'))/255)[...,None]

def geometry(boxes,size):
    b=np.asarray(boxes,np.float32);c=(b[:,:2]+b[:,2:])/2;h=max(1.,float(np.median(b[:,3]-b[:,1])))
    delta=np.vstack((np.zeros((1,2),np.float32),np.diff(c,axis=0)))/h
    g=np.column_stack((c[:,0]/size[0],c[:,1]/size[1],(b[:,2]-b[:,0])/size[0],(b[:,3]-b[:,1])/size[1],delta,(b[:,2]-b[:,0])/h,(b[:,3]-b[:,1])/h)).astype('float32')
    g[:,4:]=np.clip(g[:,4:],-4,4);return g

def segment(image,merger,s):
    im,mask=preprocess(image);labels,groups=raw_components(mask);groups=merger.groups(labels,groups)
    if not groups or len(groups)>s.max_symbols:raise ValueError('empty_or_excess_segments')
    boxes=[list(g.box) for g in groups]
    return np.stack([crop(masked_patch(im,labels,g)) for g in groups]),geometry(boxes,im.size),boxes,im,labels,groups

def _load_job(r,sources,out):
    with Image.open(resolve(sources if r['root']=='sources' else out,r['image'])) as image:
        original=image.size;im,mask=preprocess(image)
    labels,groups=raw_components(mask)
    return r,original,im,labels,groups

def _save_job(job,groups,out,s,compress,owner_root=None):
    r,original,im,labels,_=job;path=out/'items'/(r['id']+'.npz')
    try:
        if not groups or len(groups)>s.max_symbols:raise ValueError('empty_or_excess_segments')
        boxes=[list(g.box) for g in groups]
        x=np.stack([crop(masked_patch(im,labels,g)) for g in groups]);g=geometry(boxes,im.size)
        truth=r.get('glyphs') or [];owner=None;sx,sy=im.width/original[0],im.height/original[1]
        truth=[{**v,'box':[v['box'][0]*sx,v['box'][1]*sy,v['box'][2]*sx,v['box'][3]*sy]} for v in truth]
        if r.get('owner'):
            with Image.open((owner_root or out)/r['owner']) as own:owner=np.asarray(own.resize(im.size,Image.Resampling.NEAREST),np.int32)
        mapping=match(boxes,truth,owner,labels,groups)
        classes,relations,attention=targets(boxes,truth,r.get('edges') or [],r['tokens'],r['anchors'],mapping)
        y=np.asarray([TOKENS.index(t) for t in r['tokens']]+[2],np.int32)
        writer=np.savez_compressed if compress else np.savez
        temp=path.with_name(path.stem+'.tmp.npz')
        writer(temp,x=x.astype('float16'),g=g,classes=classes,relations=relations,attention=attention,y=y,boxes=np.asarray(boxes,np.float32))
        temp.replace(path)
        return {'id':r['id'],'failed':False,'matched':int(np.sum(classes>=0)),'sha256':digest(path)}
    except ValueError as exc:return {'id':r['id'],'failed':True,'matched':0,'error':str(exc)}

def prepare_items(records,sources,out,s,merger,signature,progress):
    """CPU workers -> one GPU batch -> CPU workers. No per-worker TF/GPU state."""
    workers=int(os.environ.get('OCR_PREP_WORKERS',str(min(4,s.threads))))
    chunk=int(os.environ.get('OCR_PREP_BATCH','32'));compress=os.environ.get('OCR_CACHE_COMPRESS','0')=='1'
    if not 1<=workers<=32 or not 1<=chunk<=128:raise ValueError('Invalid preprocessing worker/batch setting.')
    journal=out/'items_progress.jsonl';done={}
    if journal.exists():
        with journal.open(encoding='utf-8') as f:
            if json.loads(f.readline())!={'signature':signature}:raise ValueError('Item journal identity changed.')
            for line in f:
                try:v=json.loads(line)
                except json.JSONDecodeError:break  # A partial final line is not committed.
                done[v['id']]=v
        # Rewrite only committed records once, removing a torn final line before append.
        temp=journal.with_suffix('.tmp')
        with temp.open('w',encoding='utf-8') as f:
            f.write(json.dumps({'signature':signature})+'\n')
            for v in done.values():f.write(json.dumps(v)+'\n')
        temp.replace(journal)
    else:journal.write_text(json.dumps({'signature':signature})+'\n',encoding='utf-8')
    pending=[];failures=[];matched=Counter();completed=0;timing=Counter()
    def apply(r,v):
        r['failed']=v['failed'];matched[r['source']+'/'+r['split']]+=v['matched']
        if v['failed']:failures.append({'id':r['id'],'error':v['error']})
    for r in records:
        r['cache']='items/'+r['id']+'.npz';v=done.get(r['id']);path=out/r['cache']
        if v and (v['failed'] or path.exists() and digest(path)==v['sha256']):apply(r,v);completed+=1
        else:pending.append(r)
    progress.message(f'전처리: CPU {workers} workers, GPU image batch {chunk}, 재사용 {completed}/{len(records)}, 압축={compress}')
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool,journal.open('a',encoding='utf-8') as f:
        for offset in range(0,len(pending),chunk):
            group=pending[offset:offset+chunk];t=time.monotonic()
            jobs=list(pool.map(lambda r:_load_job(r,sources,out),group));timing['load_preprocess']+=time.monotonic()-t;t=time.monotonic()
            merged=merger.groups_many([(j[3],j[4]) for j in jobs]);timing['merge']+=time.monotonic()-t;t=time.monotonic()
            values=list(pool.map(lambda pair:_save_job(pair[0],pair[1],out,s,compress),zip(jobs,merged)));timing['supervision_save']+=time.monotonic()-t
            for r,v in zip(group,values):apply(r,v);f.write(json.dumps(v)+'\n')
            f.flush();completed+=len(group)
            progress.update('자동 분할/기호 정답 매칭',completed,len(records),stage_elapsed=time.monotonic()-start,**{k:round(v,2) for k,v in timing.items()})
    return matched,failures,dict(timing)

def real_rows(sources):
    rows=[]
    for r in read(sources/'mathwriting/manifest.json'):
        rows.append({**r,'id':'mw_'+r['sample_id'],'source':'MathWriting','image':'mathwriting/'+r['image'],'glyphs':None})
    for split in ('train','val','test'):
        for line in (sources/'crohme'/(split+'.jsonl')).read_text(encoding='utf-8').splitlines():
            r=json.loads(line);rows.append({**r,'id':'crohme_'+r['id'],'split':split,'source':'CROHME','image':'crohme/'+r['image']})
    return rows

@lru_cache(maxsize=192)
def load_item(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}

def batch(rows,shared):
    from .packing import packed_item
    items=[packed_item(shared,r['packed']) if 'packed' in r else load_item(str(shared/r['cache'])) for r in rows];b=len(items);n=max(len(i['x']) for i in items);t=max(len(i['y']) for i in items)
    images=np.zeros((b,n,48,48,1),np.float32);g=np.zeros((b,n,8),np.float32);mask=np.zeros((b,n),bool)
    y=np.zeros((b,t),np.int32);classes=np.full((b,n),-1,np.int32);rel=np.full((b,n,n),-1,np.int32);a=np.zeros((b,t,n),np.float32)
    for j,v in enumerate(items):
        nn=len(v['x']);tt=len(v['y']);images[j,:nn]=v['x'];g[j,:nn]=v['g'];mask[j,:nn]=True
        y[j,:tt]=v['y'];classes[j,:nn]=v['classes'];rel[j,:nn,:nn]=v['relations'];a[j,:tt,:nn]=v['attention']
    previous=np.concatenate([np.ones((b,1),np.int32),y[:,:-1]],axis=1)
    return {'images':images,'geometry':g,'mask':mask,'previous':previous},{'tokens':y,'classes':classes,'relations':rel,'attention':a}
