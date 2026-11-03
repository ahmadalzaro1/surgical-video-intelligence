#!/usr/bin/env python3
"""Resumable SAM 2.1 versus DIS, annotation-seeded, not clinical software."""
import argparse,csv,gc,json,os,time
from pathlib import Path
from collections import Counter
import cv2,numpy as np,torch
from compare_tracking import W,H,polygon,mask,iou
from experiment_v4 import advance
from sam2.build_sam import build_sam2_video_predictor
cv2.setNumThreads(2)
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--video',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path,default=Path('data/sam2.1_hiera_tiny.pt'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
with a.annotations.open(newline='',encoding='utf8') as f: allrows=[r for r in csv.DictReader(f) if r['int_video']==a.case and r['int_second']]
groups={}
for r in allrows: groups.setdefault(int(r['int_second']),[]).append(r)
rows={sec:v[0] for sec,v in groups.items() if len(v)==1}
assert all(int(r['int_frame'])==sec*25 for sec,r in rows.items())
cap=cv2.VideoCapture(str(a.video)); assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS)-25)<0.1
nframes=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); device='mps' if torch.backends.mps.is_available() else 'cpu'
if device!='mps':raise RuntimeError('MPS unavailable: refusing to silently run a multi-hour CPU job')
print('case',a.case,'source_frames',nframes,'device',device,'ambiguous_seconds',[k for k,v in groups.items() if len(v)>1],flush=True)
model=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(a.checkpoint.resolve()),device=device,apply_postprocessing=False)
sec=0;carry=None
while True:
 if carry is None:
  ok,carry=cap.read()
  if not ok:break
 frames=[cv2.resize(carry,(W,H))];carry=None
 for j in range(25):
  ok,frame=cap.read()
  if not ok:break
  frame=cv2.resize(frame,(W,H));frames.append(frame)
 if len(frames)==26:carry=frames[-1];display_count=25
 else:display_count=len(frames)
 segment=a.out/f'{sec:04d}';segment.mkdir(exist_ok=True); result_path=segment/'result.json'; video_path=segment/'comparison.mp4'
 if result_path.exists() and video_path.exists():
  r=json.loads(result_path.read_text())
  if r.get('completed') and r.get('display_frames')==display_count: print('resume skip',sec,flush=True);sec+=1;continue
 started=time.monotonic(); row=rows.get(sec); gt=polygon(row); nextrow=rows.get(sec+1); nextgt=polygon(nextrow)
 eligible=(len(frames)==26 and gt is not None and nextgt is not None and row['str_instrument']==nextrow['str_instrument'])
 pred_sam=[None]*len(frames);pred_dis=[None]*len(frames);counts=Counter()
 if gt is not None:
  jpg=segment/'frames';jpg.mkdir(exist_ok=True)
  for j,f in enumerate(frames):
   if not cv2.imwrite(str(jpg/f'{j:05d}.jpg'),f,[cv2.IMWRITE_JPEG_QUALITY,95]):raise RuntimeError('JPEG write failed')
  with torch.inference_mode():
   state=model.init_state(video_path=str(jpg),offload_video_to_cpu=True,offload_state_to_cpu=True)
   model.add_new_mask(state,frame_idx=0,obj_id=1,mask=mask(gt)>0)
   for j,ids,logits in model.propagate_in_video(state):pred_sam[j]=(logits[0,0]>0).cpu().numpy().astype(np.uint8)*255
  del state
  # Read the identical JPEGs used by SAM 2 to remove input compression as a comparison confound.
  shape=gt.copy();pts=None;pred_dis[0]=mask(shape);old=None
  for j in range(len(frames)):
   frame_j=cv2.imread(str(jpg/f'{j:05d}.jpg')); gray=cv2.cvtColor(frame_j,cv2.COLOR_BGR2GRAY)
   if j and shape is not None:shape,pts=advance('dense_dis',old,gray,shape,pts,counts)
   if j:pred_dis[j]=mask(shape)
   old=gray
  gc.collect();torch.mps.empty_cache()
 writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),25,(1280,360))
 if not writer.isOpened():raise RuntimeError('cannot create comparison video')
 for j in range(display_count):
  panels=[]
  for name,pred in [('IMPROVED DIS',pred_dis[j]),('SAM 2.1',pred_sam[j])]:
   im=frames[j].copy()
   if pred is not None and np.any(pred):
    lay=im.copy();lay[pred>0]=(215,208,65);im=cv2.addWeighted(im,.8,lay,.2,0)
    contours,_=cv2.findContours(pred.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,contours,-1,(215,208,65),2)
   cv2.rectangle(im,(0,0),(W,45),(15,23,28),-1);cv2.putText(im,f'CASE {a.case} | {name} | {sec+j/25:.2f}s',(10,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,235,235),1,cv2.LINE_AA);cv2.putText(im,'ANNOTATION SEEDED | PHANTOM RESEARCH',(10,38),cv2.FONT_HERSHEY_SIMPLEX,.38,(230,235,235),1,cv2.LINE_AA);panels.append(cv2.resize(im,(640,360)))
  writer.write(np.hstack(panels))
 writer.release()
 result={'completed':True,'case':a.case,'second':sec,'display_frames':display_count,'source_frames_available':len(frames),'anchor':gt is not None,'eligible':bool(eligible),'instrument':row['str_instrument'] if row else None,'elapsed_seconds':time.monotonic()-started,'dis_loss_counts':dict(counts)}
 if eligible:
  target=mask(nextgt);result['target_frame']=(sec+1)*25;result['dis_iou']=iou(pred_dis[25] if pred_dis[25] is not None else np.zeros((H,W),dtype=np.uint8),target);result['sam_iou']=iou(pred_sam[25] if pred_sam[25] is not None else np.zeros((H,W),dtype=np.uint8),target)
 tmp=segment/'result.tmp';tmp.write_text(json.dumps(result,indent=2));os.replace(tmp,result_path)
 print('completed second',sec,'/',(nframes+24)//25,'elapsed',round(result['elapsed_seconds'],1),'eligible',eligible,flush=True)
 sec+=1
cap.release();print('DONE',a.case,'seconds',sec,flush=True)
