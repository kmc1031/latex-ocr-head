"""Ground-truth stroke ownership for merge supervision, never for inference."""
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def stroke_masks(raw, expected_count, size):
    root = ET.fromstring(raw)
    traces = {}
    for trace in root.findall('{*}trace'):
        points = np.array([[float(v) for v in p.split()[:2]]
                           for p in trace.text.strip().split(',')], dtype=np.float32)
        traces[trace.attrib['id']] = points
    groups = [[t.attrib['traceDataRef'] for t in g.findall('{*}traceView')]
              for g in root.findall('.//{*}traceGroup') if g.findall('{*}traceView')]
    if len(groups) != expected_count:
        raise ValueError('InkML and prepared glyph ordering disagree.')
    boxes = [np.concatenate([traces[k] for k in refs]) for refs in groups]
    points = np.concatenate(list(traces.values())); low = points.min(0)
    extent = np.maximum(points.max(0)-low, 1e-4)
    heights = [p[:, 1].max()-p[:, 1].min() for p in boxes if p[:, 1].max()>p[:, 1].min()]
    scale = min(48/max(np.median(heights),1e-4),1008/extent[0],240/extent[1])
    factor = 3
    result = []
    for refs in groups:
        canvas = Image.new('L', (size[0]*factor,size[1]*factor),0)
        draw = ImageDraw.Draw(canvas)
        for key in refs:
            xy = [tuple(v*factor) for v in (traces[key]-low)*scale+8]
            if len(xy)>1:
                draw.line(xy,fill=255,width=5,joint='curve')
            for x,y in (xy[0],xy[-1]):
                draw.ellipse((x-2.5,y-2.5,x+2.5,y+2.5),fill=255)
        result.append(canvas.resize(size,Image.Resampling.LANCZOS))
    return result


def owners(labels, masks):
    """Only unambiguous pixel ownership is labelled; mixed components ignored."""
    result = {}
    arrays=[np.asarray(m)>100 for m in masks]
    for index,bounds in enumerate(ndimage.find_objects(labels),1):
        if bounds is None: continue
        active = labels[bounds] == index
        if active.sum()<3:
            continue
        hits = np.array([np.count_nonzero(active & m[bounds]) for m in arrays])
        ranked = np.argsort(hits)
        if len(ranked) and hits[ranked[-1]]/active.sum()>=.8:
            second = hits[ranked[-2]] if len(ranked)>1 else 0
            if second/active.sum()<=.1:
                result[index] = int(ranked[-1])
    return result
