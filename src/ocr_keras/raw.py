"""CPU-only raw components and merge patch geometry."""
import numpy as np
from scipy import ndimage
from PIL import Image
from .components import Group,combine

def raw_components(mask):
    labels, _ = ndimage.label(mask, np.ones((3,3)))
    areas = np.bincount(labels.ravel())
    groups, tiny = [], []
    for index, bounds in enumerate(ndimage.find_objects(labels), 1):
        if bounds:
            y,x = bounds
            group = Group((index,), (x.start,y.start,x.stop,y.stop))
            (groups if areas[index]>=3 else tiny).append(group)
    anchors = [g for g in groups if g.box[3]-g.box[1]>=6 and areas[g.ids[0]]>=9]
    for g in tiny:
        cx=(g.box[0]+g.box[2])/2
        cy=(g.box[1]+g.box[3])/2
        for x0,y0,x1,y1 in (a.box for a in anchors):
            height=y1-y0
            near_x=x0-.12*height<=cx<=x1+.45*height
            near_edge=y0-.32*height<=cy<=y0+.15*height or y1-.15*height<=cy<=y1+.32*height
            if near_x and near_edge:
                groups.append(g)
                break
    return labels, sorted(groups,key=lambda g:g.ids)


def separate_script(a,b):
    """Protect a small elevated or lowered glyph beside a larger base."""
    left,right=sorted((a.box,b.box),key=lambda box:(box[0]+box[2])/2)
    base_height=left[3]-left[1]
    script_height=right[3]-right[1]
    if base_height<6 or not .20*base_height<=script_height<=.78*base_height:
        return False
    if right[0]<left[2]-.04*base_height:
        return False
    dy=abs((right[1]+right[3]-left[1]-left[3])/2)
    return dy>=.22*base_height


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
