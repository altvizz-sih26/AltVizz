#!/usr/bin/env python3
"""Universal input-based 3D terrain/DSM generator.

Accepts RGB: jpg/jpeg/png/tif/tiff/webp/bmp
Accepts elevation: npy/npz/npz.zip

1 image + 1 elevation -> model.glb
2 images + 2 elevation files -> before.glb + during.glb
2 images + 1 pair-NPZ -> before.glb + during.glb

RGB is texture only; real height requires NPY/NPZ elevation data.
"""
from __future__ import annotations
import argparse, json, math, zipfile
from pathlib import Path
from typing import Any, Optional
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, median_filter

IMG_EXT={'.jpg','.jpeg','.png','.tif','.tiff','.webp','.bmp'}
ELEV_EXT={'.npy','.npz','.zip'}

def scalar(v):
    a=np.asarray(v)
    if a.size!=1: raise ValueError(f'Expected scalar, got {a.shape}')
    x=a.reshape(-1)[0]
    return x.item() if hasattr(x,'item') else x

def load_npz(path:Path):
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if not n.endswith('/')]
            npzs=[n for n in names if n.lower().endswith('.npz')]
            if npzs:
                with z.open(npzs[0]) as f:
                    with np.load(f,allow_pickle=True) as d:return {k:d[k] for k in d.files}
            npys=[n for n in names if n.lower().endswith('.npy')]
            if not npys: raise ValueError(f'{path} contains no NPZ/NPY data')
            out={}
            for n in npys:
                with z.open(n) as f:out[Path(n).stem]=np.load(f,allow_pickle=True)
            return out
    with np.load(path,allow_pickle=True) as d:return {k:d[k] for k in d.files}

def read_elevation(path:Path):
    if path.suffix.lower()=='.npy':
        a=np.load(path,allow_pickle=True)
        if a.ndim!=2 or not np.issubdtype(a.dtype,np.number): raise ValueError(f'{path.name} must be a numeric 2D elevation array')
        return {'kind':'single','before':a.astype(np.float32),'after':None,'mode':'unknown','noise':float('nan')}
    d=load_npz(path)
    if 'elevation_before' in d and 'elevation_after' in d:
        b=np.asarray(d['elevation_before'],np.float32); a=np.asarray(d['elevation_after'],np.float32)
        if b.shape!=a.shape or b.ndim!=2: raise ValueError(f'NPZ before/after shape mismatch: {b.shape}, {a.shape}')
        mode=str(scalar(d['mode'])) if 'mode' in d else 'unknown'
        if mode not in {'srtm_calibrated','height_above_ground_m','unknown'}: raise ValueError(f'Unsupported mode: {mode}')
        noise=float(scalar(d['noise_floor'])) if 'noise_floor' in d else float('nan')
        return {'kind':'pair','before':b,'after':a,'mode':mode,'noise':noise}
    for k in ('elevation','height','depth','elevation_map','final_elevation'):
        if k in d and np.asarray(d[k]).ndim==2:
            return {'kind':'single','before':np.asarray(d[k],np.float32),'after':None,'mode':str(scalar(d['mode'])) if 'mode' in d else 'unknown','noise':float(scalar(d['noise_floor'])) if 'noise_floor' in d else float('nan')}
    two_d=[np.asarray(v) for v in d.values() if np.asarray(v).ndim==2 and np.issubdtype(np.asarray(v).dtype,np.number)]
    if len(two_d)==1:return {'kind':'single','before':two_d[0].astype(np.float32),'after':None,'mode':'unknown','noise':float('nan')}
    raise ValueError(f'No usable elevation array found in {path}')

def fill(a):
    a=np.asarray(a,np.float32); bad=(~np.isfinite(a))|(a==-9999)
    if bad.all():raise ValueError('Elevation contains no valid pixels')
    if not bad.any():return a.copy()
    _,idx=distance_transform_edt(bad,return_indices=True)
    return a[tuple(idx)].astype(np.float32)

def prep(a,mode,med=3,sigma=1.0):
    a=median_filter(fill(a),size=med,mode='nearest').astype(np.float32)
    if sigma>0:a=gaussian_filter(a,sigma=sigma,mode='nearest').astype(np.float32)
    if mode=='height_above_ground_m':a=np.maximum(a,0)
    return a

def trim_bounds(shape,pct):
    h,w=shape; r=int(round(h*np.clip(pct,0,45)/100)); c=int(round(w*np.clip(pct,0,45)/100))
    if h-2*r<4 or w-2*c<4:raise ValueError('Border trim too large')
    return r,h-r,c,w-c

def crop(a,b):
    r0,r1,c0,c1=b; return a[r0:r1,c0:c1]

def down(a,step):
    if step==1:return a.astype(np.float32)
    h,w=a.shape; h=(h//step)*step; w=(w//step)*step
    return a[:h,:w].reshape(h//step,step,w//step,step).mean((1,3)).astype(np.float32)

def grid(h,w,spacing):
    x=(np.arange(w,dtype=np.float32)-(w-1)/2)*spacing
    z=(np.arange(h,dtype=np.float32)-(h-1)/2)*spacing
    X,Z=np.meshgrid(x,-z); return X.astype(np.float32),Z.astype(np.float32)

def faces(h,w):
    r=np.arange(h-1)[:,None]; c=np.arange(w-1)[None,:]; i=r*w+c
    a=np.stack([i,i+1,i+w],-1); b=np.stack([i+1,i+w+1,i+w],-1)
    return np.concatenate([a.reshape(-1,3),b.reshape(-1,3)]).astype(np.int64)

def mesh(height,X,Z,floor,texture):
    h,w=height.shape;n=h*w
    top=np.column_stack([X.ravel(),height.ravel(),Z.ravel()]).astype(np.float32)
    bot=np.column_stack([X.ravel(),np.full(n,floor,np.float32),Z.ravel()])
    v=np.vstack([top,bot]); tf=faces(h,w); bf=tf[:,[0,2,1]]+n
    def wall(edge,rev=False):
        a=edge[:-1];b=edge[1:];d=a+n;c=b+n
        q=np.stack([a,d,c,a,c,b] if rev else [a,b,c,a,c,d],1);return q.reshape(-1,3)
    front=np.arange(w);back=np.arange((h-1)*w,h*w);left=np.arange(0,n,w);right=np.arange(w-1,n,w)
    f=np.vstack([tf,wall(front),wall(back,1),wall(left,1),wall(right),bf]).astype(np.int64)
    m=trimesh.Trimesh(vertices=v,faces=f,process=False);trimesh.repair.fix_normals(m,multibody=True)
    uv=np.full((len(v),2),.5,np.float32);u=np.linspace(0,1,w);vv=np.linspace(1,0,h);U,V=np.meshgrid(u,vv);uv[:n]=np.column_stack([U.ravel(),V.ravel()])
    m.visual=trimesh.visual.texture.TextureVisuals(uv=uv,material=trimesh.visual.material.PBRMaterial(baseColorTexture=texture,metallicFactor=0,roughnessFactor=.95))
    return m

def slope95(a,spacing,scale):
    gy,gx=np.gradient(a*scale,spacing,spacing);s=np.degrees(np.arctan(np.hypot(gx,gy)));return float(np.percentile(s,95)),float(np.mean(s>60))

def choose_step(b,a,requested,gsd,auto):
    candidates=[requested]+([4,8,16,32] if auto else [])
    candidates=list(dict.fromkeys(x for x in candidates if x>=requested))
    last=(requested,1)
    for st in candidates:
        bd=down(b,st);ad=down(a,st) if a is not None else None;sp=float(st if gsd is None else st*gsd)
        vals=[]
        for x in (bd,ad):
            if x is not None:
                gy,gx=np.gradient(x,sp,sp);g=np.hypot(gx,gy);vals.append(float(np.percentile(g,95)))
        g=max(vals) if vals else 0
        scale=8.0 if g<=1e-8 else float(np.clip(math.tan(math.radians(35))/g,1,8))
        ok=True
        for x in (bd,ad):
            if x is not None:
                p,gt=slope95(x,sp,scale);ok &= p<=35 and gt<.02
        last=(st,scale)
        if ok:
            if st!=requested:print(f'[AUTO] step {requested} -> {st} to control slope')
            return last
    print('[WARN] slope limits not fully met; using last tested step')
    return last

def load_image(path,shape):
    im=Image.open(path).convert('RGB')
    if im.size!=(shape[1],shape[0]):raise ValueError(f'{path.name}: image {im.size} does not match elevation {(shape[1],shape[0])}; no silent resize')
    return im

def validate(m,topn,floor):
    return {'watertight':bool(m.is_watertight),'finite':bool(np.isfinite(m.vertices).all()),'flat_bottom':bool(np.ptp(m.vertices[topn:,1])<1e-5),'floor_match':abs(float(m.vertices[topn:,1].mean())-floor)<1e-4,'height_exists':bool(np.ptp(m.vertices[:topn,1])>1e-6)}

def generate(inputs,output='output',step=2,trim_pct=5,med=3,sigma=1,thickness_frac=.06,gsd=None,auto=True):
    paths=[]
    for raw in inputs:
        p=Path(raw)
        if not p.exists():raise FileNotFoundError(p)
        paths += sorted([x for x in p.iterdir() if x.is_file()]) if p.is_dir() else [p]
    imgs=[p for p in paths if p.suffix.lower() in IMG_EXT]
    elevs=[p for p in paths if p.suffix.lower() in ELEV_EXT]
    if not imgs:raise ValueError('No image supplied')
    if len(imgs)>2:raise ValueError('Maximum two images: before and during')
    if not elevs:raise ValueError('No NPY/NPZ elevation supplied. RGB alone cannot create real terrain height.')
    src=[read_elevation(p) for p in elevs]
    if len(src)>2:raise ValueError('Maximum two elevation sources')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # Single image -> one model. If a pair-NPZ is given, use its BEFORE elevation.
    if len(imgs)==1:
        s=src[0]; e=s['before'];mode=s['mode'];noise=s['noise']
        im=load_image(imgs[0],e.shape);bnd=trim_bounds(e.shape,trim_pct);e=crop(e,bnd);im=im.crop((bnd[2],bnd[0],bnd[3],bnd[1]));e=prep(e,mode,med,sigma)
        st,scale=choose_step(e,None,step,gsd,auto);ed=down(e,st);sp=float(st if gsd is None else st*gsd);y0=0.0 if mode=='height_above_ground_m' else float(ed.min());h=ed.shape;X,Z=grid(*h,sp);height=(ed-y0)*scale;floor=float(height.min()-max(1e-3,max(np.ptp(X),np.ptp(Z))*thickness_frac));m=mesh(height,X,Z,floor,im)
        meta={'mode':mode,'units':'metres above sea level' if mode=='srtm_calibrated' else 'metres above ground' if mode=='height_above_ground_m' else 'unknown','vertical_scale':scale,'y_zero':y0,'floor_y':floor,'step':st,'grid_shape':list(h),'border_trim_pct':trim_pct,'gsd_m':gsd,'image':str(imgs[0])}
        m.metadata['extras']=meta;m.export(out/'model.glb');(out/'model.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print('Created',out/'model.glb');print('Validation',validate(m,h[0]*h[1],floor));return
    # Pair mode: one pair-NPZ OR two single sources OR two pair sources.
    if len(src)==1:
        s=src[0]
        if s['kind']=='single':raise ValueError('Two images require two elevation grids, or one NPZ containing elevation_before/elevation_after')
        sb=sa=s
    elif len(src)==2:
        if src[0]['kind']=='pair' and src[1]['kind']=='pair':sb,sa=src
        elif src[0]['kind']=='single' and src[1]['kind']=='single':sb,sa=src
        else:raise ValueError('Use either one pair-NPZ, or two single elevation sources')
    else:raise ValueError('No elevation source')
    be=sb['before']; ae=sa['after'] if sa['kind']=='pair' else sa['before'];
    if be.shape!=ae.shape:raise ValueError(f'Before/during grids differ: {be.shape} vs {ae.shape}')
    mode=sb['mode'] if sb['mode']!='unknown' else sa['mode'];
    if sb['mode']!='unknown' and sa['mode']!='unknown' and sb['mode']!=sa['mode']:raise ValueError('Before/during modes differ')
    ib=load_image(imgs[0],be.shape);ia=load_image(imgs[1],ae.shape);bnd=trim_bounds(be.shape,trim_pct);ib=ib.crop((bnd[2],bnd[0],bnd[3],bnd[1]));ia=ia.crop((bnd[2],bnd[0],bnd[3],bnd[1]));bp=prep(crop(be,bnd),mode,med,sigma);ap=prep(crop(ae,bnd),mode,med,sigma)
    st,scale=choose_step(bp,ap,step,gsd,auto);bd=down(bp,st);ad=down(ap,st);sp=float(st if gsd is None else st*gsd);y0=0.0 if mode=='height_above_ground_m' else float(min(bd.min(),ad.min()));bh=(bd-y0)*scale;ah=(ad-y0)*scale;X,Z=grid(*bd.shape,sp);floor=float(min(bh.min(),ah.min())-max(1e-3,max(np.ptp(X),np.ptp(Z))*thickness_frac));mb=mesh(bh,X,Z,floor,ib);ma=mesh(ah,X,Z,floor,ia)
    common={'mode':mode,'units':'metres above sea level' if mode=='srtm_calibrated' else 'metres above ground' if mode=='height_above_ground_m' else 'unknown','vertical_scale':scale,'y_zero':y0,'floor_y':floor,'step':st,'grid_shape':list(bd.shape),'border_trim_pct':trim_pct,'gsd_m':gsd,'coordinate_system':'glTF X/Z horizontal, Y vertical'}
    mb.metadata['extras']={**common,'date':'before'};ma.metadata['extras']={**common,'date':'during'};mb.export(out/'before.glb');ma.export(out/'during.glb');(out/'before.json').write_text(json.dumps({**common,'date':'before'},indent=2),encoding='utf-8');(out/'during.json').write_text(json.dumps({**common,'date':'during'},indent=2),encoding='utf-8')
    n=bd.size;pair={'same_vertex_count':len(mb.vertices)==len(ma.vertices),'same_faces':np.array_equal(mb.faces,ma.faces),'same_X':np.allclose(mb.vertices[:n,0],ma.vertices[:n,0]),'same_Z':np.allclose(mb.vertices[:n,2],ma.vertices[:n,2]),'different_Y':not np.allclose(mb.vertices[:n,1],ma.vertices[:n,1])};print('Created',out/'before.glb',out/'during.glb');print('Pair validation',pair);print('Before validation',validate(mb,n,floor));print('During validation',validate(ma,n,floor))

def main():
    p=argparse.ArgumentParser(description='Universal automatic 3D terrain/DSM generator')
    p.add_argument('inputs',nargs='+',help='1-2 images plus 1-2 NPY/NPZ elevation files, or a directory containing them')
    p.add_argument('--output',default='output');p.add_argument('--step',type=int,default=2,choices=[1,2,4,8,16,32]);p.add_argument('--border-trim-pct',type=float,default=5);p.add_argument('--median-size',type=int,default=3,choices=[3,5]);p.add_argument('--smooth-sigma',type=float,default=1.0);p.add_argument('--base-thickness-fraction',type=float,default=.06);p.add_argument('--gsd-m',type=float,default=None);p.add_argument('--no-auto-step',action='store_true');a=p.parse_args();generate(a.inputs,a.output,a.step,a.border_trim_pct,a.median_size,a.smooth_sigma,a.base_thickness_fraction,a.gsd_m,not a.no_auto_step)
if __name__=='__main__':main()
