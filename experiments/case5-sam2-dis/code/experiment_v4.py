#!/usr/bin/env python3
"""Compare annotation-seeded phantom tracking methods, not for clinical use."""
import argparse,csv,json,time
from collections import Counter
from pathlib import Path
import cv2,numpy as np
from compare_tracking import W,H,polygon,mask,iou,features
cv2.setNumThreads(2)
DIS=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

def sample(p):
    xy=p.reshape(-1,2); out=[]
    for a,b in zip(xy,np.roll(xy,-1,axis=0)):
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/5))); out.extend(a+(b-a)*(i/n) for i in range(n))
    return np.asarray(out,dtype=np.float32).reshape(-1,1,2)

def advance(method,old,new,shape,pts,c):
    if method in ('dense_flow','dense_flow_checked','dense_dis'):
        a=cv2.resize(old,(W//2,H//2)); b=cv2.resize(new,(W//2,H//2)); flow=DIS.calc(b,a,None) if method=='dense_dis' else cv2.calcOpticalFlowFarneback(b,a,None,.5,3,21,3,5,1.2,0)
        yy,xx=np.indices(a.shape,dtype=np.float32); sm=cv2.resize(mask(shape),(W//2,H//2),interpolation=cv2.INTER_NEAREST)
        moved=cv2.remap(sm,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        if method=='dense_flow_checked':
            forward=cv2.calcOpticalFlowFarneback(a,b,None,.5,3,21,3,5,1.2,0)
            back_x=xx+flow[:,:,0]; back_y=yy+flow[:,:,1]
            fx=cv2.remap(forward[:,:,0],back_x,back_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT); fy=cv2.remap(forward[:,:,1],back_x,back_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            consistent=np.hypot(flow[:,:,0]+fx,flow[:,:,1]+fy)<1.5
            selected=moved>0; count=int(selected.sum()); reliable=int((consistent & selected).sum()); c['flow_mask_pixels']+=count; c['flow_consistent_pixels']+=reliable
            if count and reliable/count<0.5: c['lost_low_flow_consistency']+=1; return None,None
        contours,_=cv2.findContours(cv2.resize(moved,(W,H),interpolation=cv2.INTER_NEAREST),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours: c['lost_empty_mask']+=1; return None,None
        p=max(contours,key=cv2.contourArea)
        if len(p)<3 or cv2.contourArea(p)<25: c['lost_small_mask']+=1; return None,None
        return p.astype(np.float32),None
    if pts is None or len(pts)<3: c['lost_no_points']+=1; return None,None
    c['points_input']+=len(pts)
    q,s,_=cv2.calcOpticalFlowPyrLK(old,new,pts,None,winSize=(25,25),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.02))
    if q is None or s is None: c['forward_call_fail']+=len(pts); return None,None
    back,bs,_=cv2.calcOpticalFlowPyrLK(new,old,q,None,winSize=(25,25),maxLevel=3)
    if back is None or bs is None: c['backward_call_fail']+=len(pts); return None,None
    good=s.ravel()==1; c['forward_status_rejected']+=int((~good).sum())
    v=bs.ravel()==1; c['backward_status_rejected']+=int((good&~v).sum()); good &=v
    err=np.linalg.norm(back-pts,axis=2).ravel(); v=np.isfinite(err)&(err<3); c['forward_backward_error_rejected']+=int((good&~v).sum()); good &=v
    v=np.isfinite(q).all(axis=(1,2))&(q[:,0,0]>=0)&(q[:,0,0]<W)&(q[:,0,1]>=0)&(q[:,0,1]<H); c['out_of_bounds_rejected']+=int((good&~v).sum()); good &=v
    c['points_surviving']+=int(good.sum()); a,b=pts[good],q[good]
    if len(a)<3: c['lost_too_few_points']+=1; return None,None
    if method=='baseline': return b,b
    m,inliers=cv2.estimateAffinePartial2D(a.reshape(-1,2),b.reshape(-1,2),method=cv2.RANSAC,ransacReprojThreshold=3)
    if m is None or inliers is None or inliers.sum()<3: c['lost_affine']+=1; return None,None
    c['affine_inliers']+=int(inliers.sum()); c['affine_outliers']+=len(inliers)-int(inliers.sum()); shape=cv2.transform(shape,m)
    pts=sample(shape) if method=='dense_boundary' else features(new,shape)
    if pts is None or len(pts)<3: c['lost_no_features']+=1; return None,None
    return shape,pts

def run(video,rows,method,step,out,duration,video_output):
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or abs(fps-25)>0.1: raise RuntimeError('expected nominal 25 fps')
    fps=25.0  # Annotation int_frame, not presentation timestamps, defines matching.
    writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H)) if step==1 and video_output else None
    if writer and not writer.isOpened(): raise RuntimeError('writer failed')
    old=shape=pts=None; last_sec=-1; last_time=None; c=Counter(); checks=[]; diagnostics=[]; index=0; start=time.monotonic()
    while True:
        ok,frame=cap.read()
        if not ok: break
        fi=index; index+=1
        if fi%step: continue
        t=fi/fps
        if t>=duration: break
        before=c.copy(); sec=int(t); frame=cv2.resize(frame,(W,H)); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); gt=polygon(rows.get(sec)); anchor=sec!=last_sec
        if anchor:
            if last_sec>=0 and gt is not None and polygon(rows.get(sec-1)) is not None and rows[sec]['str_instrument']==rows[sec-1]['str_instrument']:
                if step==1 and shape is not None: shape,pts=advance(method,old,gray,shape,pts,c)
                pm=mask(shape,convex=method=='baseline'); checks.append({'target_second':sec,'prediction_time':t if step==1 else last_time,'had_prediction':bool(pm.any()),'iou':iou(pm,mask(gt))})
            shape=gt.copy() if gt is not None else None
            pts=None if shape is None or method in ('dense_flow','dense_flow_checked','dense_dis') else (shape.copy() if method=='baseline' else sample(shape) if method=='dense_boundary' else features(gray,shape)); last_sec=sec
        elif shape is not None: shape,pts=advance(method,old,gray,shape,pts,c)
        valid=shape is not None and len(shape)>=3; c['frames']+=1; c['visible_annotation_frames']+=int(gt is not None); c['visible_annotation_overlay_frames']+=int(valid and gt is not None); c['overlay_frames']+=int(valid)
        diagnostics.append({"frame":fi,"time":t,"anchor":anchor,"valid":valid,"tracked_points":len(pts) if pts is not None else 0,"mask_area_pixels":int(np.count_nonzero(mask(shape))) if valid else 0,"blur_laplacian_variance":float(cv2.Laplacian(gray,cv2.CV_64F).var()),"counter_delta":dict(c-before)})
        if writer:
            if valid:
                p=np.rint(shape).astype(np.int32); p=cv2.convexHull(p) if method=='baseline' else p
                overlay=frame.copy(); cv2.fillPoly(overlay,[p],(215,208,65)); frame=cv2.addWeighted(frame,.8,overlay,.2,0); cv2.polylines(frame,[p],True,(215,208,65),2)
            cv2.rectangle(frame,(0,0),(W,48),(15,23,28),-1); cv2.putText(frame,f'{method.upper()} | {t:05.2f}s | '+('ANCHOR' if anchor else 'TRACKED' if valid else 'TRACK LOST'),(10,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,235,235),1,cv2.LINE_AA); cv2.putText(frame,'ANNOTATION-SEEDED PHANTOM RESEARCH',(10,39),cv2.FONT_HERSHEY_SIMPLEX,.4,(230,235,235),1,cv2.LINE_AA); writer.write(frame)
        old=gray; last_time=t
    cap.release()
    if writer: writer.release()
    vals=[x['iou'] for x in checks]; surviving=[x['iou'] for x in checks if x['had_prediction']]
    return {'diagnostics':diagnostics,'method':method,'step':step,'elapsed_seconds':time.monotonic()-start,'counts':dict(c),'checks':checks,'mean_iou':float(np.mean(vals)),'median_iou':float(np.median(vals)),'iou_at_least_half':sum(x>=.5 for x in vals),'survived':len(surviving),'surviving_mean_iou':float(np.mean(surviving)) if surviving else None}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--video',type=Path,required=True); p.add_argument('--annotations',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--methods',nargs='+',default=['baseline','affine_features','dense_boundary','dense_flow']); p.add_argument('--case',default='11'); p.add_argument('--seconds',type=int,default=60); p.add_argument('--no-video',action='store_true'); p.add_argument('--steps',nargs='+',type=int,default=[1]); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    with a.annotations.open(newline='',encoding='utf8') as f: allrows=[r for r in csv.DictReader(f) if r['int_video']==a.case and r['int_second']]
    by_second={}
    for row in allrows: by_second.setdefault(int(row['int_second']),[]).append(row)
    rows={k:v[0] for k,v in by_second.items() if len(v)==1}
    print('excluded ambiguous multi-polygon seconds',[k for k,v in by_second.items() if len(v)>1],flush=True)
    assert all(int(row['int_frame'])==s*25 for s,row in rows.items() if s<a.seconds)
    results=[]
    for step in a.steps:
        for method in a.methods:
            result=run(a.video,rows,method,step,a.out/f'{method}-25fps.mp4',a.seconds,not a.no_video); results.append(result); (a.out/'results.json').write_text(json.dumps(results,indent=2)); print(method,step,result['counts'],result['survived'],result['mean_iou'],flush=True)
if __name__=='__main__': main()
