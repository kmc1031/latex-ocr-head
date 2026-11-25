"""AI-assisted Keras component merge model. No junk class."""
from pathlib import Path
import random, time, zipfile
import numpy as np
from PIL import Image
from scipy import ndimage
import tensorflow as tf
import keras
from keras import layers as L
from .components import Group,combine,preprocess,masked_patch
from .annotations import stroke_masks,owners
from .io import read,save,digest,resolve
from .progress import Progress

def raw_components(mask):
    labels,_ = ndimage.label(mask,np.ones((3,3)))
    groups=[]
    for index, bounds in enumerate(ndimage.find_objects(labels),1):
        if bounds and np.count_nonzero(labels[bounds]==index)>=3:
            y,x=bounds; groups.append(Group((index,),(x.start,y.start,x.stop,y.stop)))
    return labels,groups


def pair_input(labels,a,b):
    union=combine(a,b); x0,y0,x1,y1=union.box
    channels=[]
    for group in (a,b):
        mask=np.isin(labels[y0:y1,x0:x1],group.ids).astype('uint8')*255
        im=Image.fromarray(mask); im.thumbnail((42,42),Image.Resampling.LANCZOS)
        canvas=Image.new('L',(48,48),0)
        canvas.paste(im,((48-im.width)//2,(48-im.height)//2))
        channels.append(np.asarray(canvas,dtype=np.float32)/255)
    # Both channels share the union frame, preserving relative size and offset.
    return np.stack(channels)


def nearby(a,b,height):
    aa,bb=a.box,b.box
    dx=max(0,aa[0]-bb[2],bb[0]-aa[2]); dy=max(0,aa[1]-bb[3],bb[1]-aa[3])
    return (dx*dx+dy*dy)**.5 <= 1.5*height


def _collect_pairs(sources,output,settings):
    """Use official train/val ownership, excluding ALL held-out content parents."""
    from .notation import content_key
    data=sources/'crohme'
    rows={s:[__import__('json').loads(line) for line in (data/(s+'.jsonl')).read_text(encoding='utf-8').splitlines()]
          for s in ('train','val','test')}
    held={}
    for s in ('val','test'):
        held[s]=set()
        for r in rows[s]:
            try: held[s].add(content_key(r['tokens']))
            except ValueError: pass
    for r in read(sources/'mathwriting/manifest.json'):
        if r['split']!='train':
            try: held['test'].add(content_key(r['tokens']))
            except ValueError: pass
    progress=Progress(output,'merge-data'); rng=random.Random(settings.seed)
    result={}; audit={}
    with zipfile.ZipFile(data/'crohme2014.zip') as archive:
        for split in ('train','val'):
            xs,ys=[],[]; skipped=0
            for index,r in enumerate(rows[split]):
                try:
                    key=content_key(r['tokens'])
                    if key in held['test'] or split=='train' and key in held['val']: continue
                    with Image.open(resolve(data,r['image'])) as original:
                        original=original.convert('L'); image,mask=preprocess(original)
                        masks=stroke_masks(archive.read(r['source']),len(r['glyphs']),original.size)
                    masks=[m.resize(image.size,Image.Resampling.LANCZOS) for m in masks]
                    labels,groups=raw_components(mask); belonging=owners(labels,masks)
                    height=max(1,float(np.median([g.box[3]-g.box[1] for g in groups]))) if groups else 48
                    possible=[(a,b) for i,a in enumerate(groups) for b in groups[i+1:]
                        if a.ids[0] in belonging and b.ids[0] in belonging and nearby(a,b,height)]
                    positive=[p for p in possible if belonging[p[0].ids[0]]==belonging[p[1].ids[0]]]
                    negative=[p for p in possible if belonging[p[0].ids[0]]!=belonging[p[1].ids[0]]]
                    chosen=rng.sample(positive,min(len(positive),settings.merge_pairs_per_formula//2))
                    chosen+=rng.sample(negative,min(len(negative),max(2,len(chosen))))
                    for a,b in chosen:
                        xs.append((pair_input(labels,a,b)*255).astype('uint8'))
                        ys.append(belonging[a.ids[0]]==belonging[b.ids[0]])
                except (ValueError,KeyError): skipped+=1
                if index%100==0: progress.update('획 소속 기반 병합 '+split,index+1,len(rows[split]))
            if not xs or len(set(ys))<2: raise ValueError('Missing merge classes in '+split)
            result[split]=(np.stack(xs),np.array(ys,dtype=np.float32))
            audit[split]={'n':len(ys),'positive':int(sum(ys)),'skipped_ambiguous_or_unsupported':skipped}
    save(output/'merge_data_audit.json',audit)
    return result


def prepare_pairs(sources,output,settings):
    """Persist expensive stroke ownership; cache validity includes its exact code."""
    from .io import identity
    from .config import ROOT
    signature=identity({'sources':digest(sources/'sources_manifest.json'),
        'code':{n:digest(ROOT/n) for n in ('merge.py','components.py','annotations.py','notation.py')},
        'seed':settings.seed,'pairs':settings.merge_pairs_per_formula})
    marker=output/'pairs_cache.json';cache=output/'pairs_cache.npz'
    if marker.exists() and cache.exists():
        spec=read(marker)
        if spec['signature']==signature and digest(cache)==spec['sha256']:
            with np.load(cache,allow_pickle=False) as arrays:
                print('[merge-data] 기존 획 소속 캐시 재사용',flush=True)
                return {s:(arrays[s+'_x'],arrays[s+'_y']) for s in ('train','val')}
    result=_collect_pairs(sources,output,settings)
    temporary=output/'pairs_cache.tmp.npz'
    np.savez(temporary,**{s+'_'+k:v for s,values in result.items() for k,v in zip(('x','y'),values)})
    temporary.replace(cache);save(marker,{'signature':signature,'sha256':digest(cache)})
    return result


def network():
    return keras.Sequential([keras.Input((48,48,2)),L.Conv2D(16,3,padding='same',activation='relu'),L.MaxPool2D(),L.Conv2D(32,3,padding='same',activation='relu'),L.GlobalAveragePooling2D(),L.Dense(64,activation='relu'),L.Dense(1)])

class Merger:
    def __init__(self,path,rounds=3):
        self.net=keras.models.load_model(path,compile=False);self.threshold=read(Path(path).with_name('calibration.json'))['threshold'];self.rounds=rounds
        self._predict=tf.function(self._predict_impl,input_signature=[tf.TensorSpec((None,48,48,2),tf.float32)])
    def _predict_impl(self,x):
        # Both channel orders share ONE TensorFlow call, avoiding per-image tracing.
        n=tf.shape(x)[0];z=tf.sigmoid(self.net(tf.concat([x,tf.reverse(x,[-1])],0),training=False))[:,0]
        return (z[:n]+z[n:])/2
    def groups(self,labels,groups):
        return self.groups_many([(labels,groups)])[0]
    def groups_many(self,items):
        """Batch candidates across images; retain each image's greedy merge order."""
        states=[list(groups) for _,groups in items];active=set(range(len(items)))
        for _ in range(self.rounds):
            candidates=[];offsets={}
            for k in sorted(active):
                groups=states[k];h=max(1,float(np.median([g.box[3]-g.box[1] for g in groups]))) if groups else 48
                from .raw import separate_script
                pairs=[(i,j) for i in range(len(groups)) for j in range(i+1,len(groups))
                       if nearby(groups[i],groups[j],h) and not separate_script(groups[i],groups[j])]
                offsets[k]=(len(candidates),pairs);candidates += [(k,i,j) for i,j in pairs]
            if not candidates:break
            scores=[]
            for offset in range(0,len(candidates),512):
                part=candidates[offset:offset+512]
                x=np.stack([pair_input(items[k][0],states[k][i],states[k][j]).transpose(1,2,0) for k,i,j in part])
                scores.extend(self._predict(x).numpy().tolist())
            next_active=set()
            for k,(start,pairs) in offsets.items():
                groups=states[k];used=set();merged=[];p=scores[start:start+len(pairs)]
                for index in sorted(range(len(pairs)),key=lambda i:p[i],reverse=True):
                    i,j=pairs[index]
                    if p[index]>=self.threshold and i not in used and j not in used:
                        used.update((i,j));merged.append(combine(groups[i],groups[j]))
                if merged:
                    states[k]=[g for i,g in enumerate(groups) if i not in used]+merged;next_active.add(k)
            active=next_active
            if not active:break
        return [sorted(groups,key=lambda g:((g.box[0]+g.box[2])/2,(g.box[1]+g.box[3])/2)) for groups in states]

def fit(sources,output,s):
    output.mkdir(parents=True,exist_ok=True)
    path=output/'model.keras'
    if (output/'complete.json').exists():
        if digest(path)!=read(output/'complete.json')['sha256']:raise ValueError('Merge checkpoint changed.')
        return path
    dataset=prepare_pairs(sources,output,s);keras.utils.set_random_seed(s.seed)
    net=network();opt=keras.optimizers.AdamW(1e-3,weight_decay=s.weight_decay);net.compile(optimizer=opt)
    last=output/'last.keras';state=read(output/'state.json') if (output/'state.json').exists() else {'epoch':0,'best':1e9,'history':[]}
    if last.exists():net=keras.models.load_model(last);opt=net.optimizer
    rng=np.random.default_rng(s.seed);progress=Progress(output,'Keras merge')
    tx,ty=dataset['train'];vx,vy=dataset['val']
    @tf.function(reduce_retracing=True)
    def step(x,y):
        with tf.GradientTape() as tape:
            z=net(x,training=True)[:,0];loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=z))
        opt.apply_gradients(zip(tape.gradient(loss,net.trainable_variables),net.trainable_variables));return loss
    for epoch in range(state['epoch'],s.merge_epochs):
        order=np.random.default_rng(s.seed+epoch).permutation(len(ty));losses=[];start=time.monotonic()
        for offset in range(0,len(order),s.merge_batch):
            ids=order[offset:offset+s.merge_batch];x=tx[ids].transpose(0,2,3,1).astype('float32')/255
            swap=rng.random(len(x))<.5;x[swap]=x[swap,:,:,::-1]
            losses.append(float(step(x,ty[ids])))
            if offset%(s.merge_batch*10)==0:progress.update(f'병합 {epoch+1}/{s.merge_epochs}',offset+len(ids),len(order),stage_elapsed=time.monotonic()-start,loss=float(np.mean(losses)))
        probs=[]
        for off in range(0,len(vy),s.merge_batch):
            x=vx[off:off+s.merge_batch].transpose(0,2,3,1).astype('float32')/255
            probs.extend(((tf.sigmoid(net(x,training=False))+tf.sigmoid(net(x[:,:,:,::-1],training=False)))/2).numpy().ravel())
        p=np.clip(np.asarray(probs),1e-7,1-1e-7);val=float(np.mean(-vy*np.log(p)-(1-vy)*np.log(1-p)))
        if val<state['best']:
            state['best']=val;net.save(output/'best.keras')
        state['epoch']=epoch+1;state['history'].append({'epoch':epoch+1,'loss':float(np.mean(losses)),'val':val})
        net.save(last);save(output/'state.json',state)
    net=keras.models.load_model(output/'best.keras',compile=False);probs=[]
    for off in range(0,len(vy),s.merge_batch):
        x=vx[off:off+s.merge_batch].transpose(0,2,3,1).astype('float32')/255
        probs.extend(((tf.sigmoid(net(x,training=False))+tf.sigmoid(net(x[:,:,:,::-1],training=False)))/2).numpy().ravel())
    p=np.asarray(probs);choices=[]
    for threshold in np.linspace(.5,.99,50):
        pred=p>=threshold;tp=int(np.sum(pred&(vy==1)));fp=int(np.sum(pred&(vy==0)));precision=tp/max(1,tp+fp);recall=tp/max(1,np.sum(vy==1))
        if tp and precision>=s.merge_min_precision:choices.append((recall,precision,float(threshold),tp,fp))
    chosen=max(choices) if choices else (0.,1.,1.01,0,0)
    save(output/'calibration.json',dict(zip(('val_recall','val_precision','threshold','tp','fp'),chosen))|{'test_used_for_selection':False})
    net.save(path);save(output/'complete.json',{'sha256':digest(path)});return path
