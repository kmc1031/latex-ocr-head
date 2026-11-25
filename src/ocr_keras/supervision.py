"""Conservative GT matching; unmatched, split or mixed crops are ignored."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .config import CLASSES
RELATIONS=['None','Right','Sup','Sub','Above','Below','Inside']

def match(boxes,truth,owner=None,labels=None,groups=None):
    p=np.asarray(boxes);t=np.asarray([g['box'] for g in truth]);mapping={}
    if not len(p) or not len(t):return mapping
    lo=np.maximum(p[:,None,:2],t[None,:,:2]);hi=np.minimum(p[:,None,2:],t[None,:,2:])
    inter=np.maximum(0,hi-lo).prod(-1)
    union=(p[:,2:]-p[:,:2]).prod(-1)[:,None]+(t[:,2:]-t[:,:2]).prod(-1)[None]-inter
    iou=inter/np.maximum(union,1e-6);valid=iou>=.5
    if owner is not None:
        # Compare recall in the SAME thresholded foreground domain. Counting
        # antialias fringes from the original raster would reject complete crops.
        sizes=np.bincount(owner[labels>0],minlength=len(truth)+1)
        for i,g in enumerate(groups):
            region=np.isin(labels,g.ids);hits=np.bincount(owner[region],minlength=len(truth)+1)[1:len(truth)+1]
            valid[i] &= (hits/max(1,region.sum())>=.8)&(hits/np.maximum(1,sizes[1:len(truth)+1])>=.8)
    a,b=linear_sum_assignment(-(valid*1000+iou))
    return {int(i):int(j) for i,j in zip(a,b) if valid[i,j]}

def targets(boxes,truth,edges,tokens,anchors,mapping):
    n=len(boxes);classes=np.full(n,-1,np.int32);rel=np.full((n,n),-1,np.int32)
    attention=np.zeros((len(tokens)+1,n),np.float32)
    lookup={(int(a),int(b)):RELATIONS.index(r) for a,b,r in edges if r in RELATIONS}
    for i,a in mapping.items():
        label=truth[a]['label'];label='-' if label==r'\frac' else label
        if label in CLASSES:classes[i]=CLASSES.index(label)
        for j,b in mapping.items():
            if i!=j:rel[i,j]=lookup.get((a,b),0)
    reverse={v:k for k,v in mapping.items()}
    for row,ids in enumerate(anchors or [[] for _ in tokens]):
        # Partial function names/split symbols are not a reliable alignment target.
        if ids and all(i in reverse for i in ids):
            for i in ids:attention[row,reverse[i]]=1/len(ids)
    return classes,rel,attention
