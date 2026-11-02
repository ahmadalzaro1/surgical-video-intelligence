import cv2,numpy as np,csv,re,json,subprocess,pathlib,collections
root=pathlib.Path('.');out=root/'outputs/tracking-demo';src=pathlib.Path('data')
rows=[r for r in csv.DictReader(open(src/'annotations.csv')) if r['int_video']=='5'];by={int(r['int_second']):r for r in rows}; W,H=960,540
cap=cv2.VideoCapture(str(src/'video_05.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));duration=n/fps
cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','12.5','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'tracked-full.mp4')];enc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
colors={'blunt_dissector':(215,208,65),'kerrisons':(45,178,255),'pituitary_rongeurs':(231,122,222),'no_instrument_visible':(180,180,180)}
prev=None;pts=None;lastsec=-1;trail=[];records=[];frameidx=0;validframes=0;trackedframes=0
while True:
 ok,frame=cap.read()
 if not ok:break
 if frameidx%2:frameidx+=1;continue
 t=frameidx/fps;sec=int(t);frame=cv2.resize(frame,(W,H));gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);r=by.get(sec,rows[-1]);label=r['str_instrument'];color=colors.get(label,(200,200,200));reset=sec!=lastsec
 if reset:
  coords=re.findall(r'x=([\d.]+),y=([\d.]+)',r['ls_flt_coordinates']);pts=np.array([[float(x)*W,float(y)*H] for x,y in coords],dtype=np.float32).reshape(-1,1,2) if coords else None
  if by.get(lastsec,{}).get('str_instrument')!=label:trail=[]
  lastsec=sec
 elif pts is not None and prev is not None:
  new,status,err=cv2.calcOpticalFlowPyrLK(prev,gray,pts,None,winSize=(25,25),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.02))
  if new is not None:
   back,bst,be=cv2.calcOpticalFlowPyrLK(gray,prev,new,None,winSize=(25,25),maxLevel=3)
   good=(status.ravel()==1)&(bst.ravel()==1)&(np.linalg.norm(back-pts,axis=2).ravel()<3)&(new[:,0,0]>=0)&(new[:,0,0]<W)&(new[:,0,1]>=0)&(new[:,0,1]<H)
   pts=new[good].reshape(-1,1,2) if sum(good)>=3 else None
  else:pts=None
 if pts is not None and len(pts)>=3:
  poly=pts[:,0,:].astype(np.int32);hull=cv2.convexHull(poly);overlay=frame.copy();cv2.fillPoly(overlay,[hull],color);frame=cv2.addWeighted(frame,.80,overlay,.20,0);cv2.polylines(frame,[hull],True,color,2)
  x,y,w,h=cv2.boundingRect(poly);cv2.rectangle(frame,(x,y),(x+w,y+h),color,2);cx,cy=np.mean(poly,axis=0).astype(int);trail.append((cx,cy));trail=trail[-38:]
  if len(trail)>1:cv2.polylines(frame,[np.array(trail)],False,color,2)
  cv2.drawMarker(frame,(cx,cy),color,cv2.MARKER_CROSS,18,2)
  for px,py in poly[::max(1,len(poly)//12)]:cv2.circle(frame,(int(px),int(py)),2,color,-1)
  validframes+=1
  if not reset:trackedframes+=1
  state='ANNOTATION ANCHOR' if reset else 'OPTICAL FLOW TRACK'
 else:state='NO VISIBLE INSTRUMENT' if label=='no_instrument_visible' else 'TRACK LOST / WAITING FOR ANCHOR'
 cv2.rectangle(frame,(0,0),(W,65),(15,23,28),-1);cv2.putText(frame,'PITUITARY PHANTOM  /  CASE 05',(20,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(220,233,235),1,cv2.LINE_AA);cv2.putText(frame,f'{int(t)//60:02d}:{int(t)%60:02d}   {label.replace("_"," ").upper()}',(20,51),cv2.FONT_HERSHEY_SIMPLEX,.58,color,1,cv2.LINE_AA)
 cv2.rectangle(frame,(0,H-37),(W,H),(15,23,28),-1);cv2.putText(frame,state+'  |  ANNOTATION-SEEDED, NOT AUTOMATIC RECOGNITION',(16,H-13),cv2.FONT_HERSHEY_SIMPLEX,.44,(215,222,222),1,cv2.LINE_AA)
 if 169.9<t<170.1:cv2.imwrite(str(out/'preview.jpg'),frame)
 enc.stdin.write(frame.tobytes());prev=gray;frameidx+=1
 if frameidx%750==0:print('processed',round(t),flush=True)
cap.release();enc.stdin.close();assert enc.wait()==0
segments=[]
for r in rows:
 sec=int(r['int_second']);lab=r['str_instrument']
 if not segments or segments[-1]['label']!=lab:segments.append({'start':sec,'end':sec+1,'label':lab})
 else:segments[-1]['end']=sec+1
meta={'duration':duration,'method':'Annotation-seeded Lucas–Kanade optical flow; re-anchor at 1 Hz; forward/backward error rejection','recognition':'Dataset labels, not automatic model predictions','fps':12.5,'valid_overlay_frames':validframes,'propagated_frames':trackedframes,'segments':segments,'source':'UCL public pituitary phantom, DOI 10.5522/04/26511049.v1','license':'CC BY-NC-SA 4.0'}
(out/'tracking-evidence.json').write_text(json.dumps(meta,indent=2));print('DONE',duration,validframes,trackedframes)
