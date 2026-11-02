import subprocess,pathlib,json
root=pathlib.Path.cwd();p=root/'work/search-clips';p.mkdir(exist_ok=True);out=root/'outputs/tracking-demo/search';out.mkdir(exist_ok=True)
source='data/video_05.mp4';clips=[]
for i,start in enumerate(range(0,480,8)):
 end=min(start+12,484.72);name=f'clip-{i:03d}.mp4';f=p/name
 subprocess.run(['ffmpeg','-y','-v','error','-ss',str(start),'-i',source,'-t',str(end-start),'-vf','fps=1,scale=384:216','-an','-c:v','libx264','-crf','23',str(f)],check=True)
 subprocess.run(['ffmpeg','-y','-v','error','-ss',str(start+4),'-i',source,'-frames:v','1','-vf','scale=384:216',str(out/f'thumb-{i:03d}.jpg')],check=True)
 clips.append({'id':i,'start':start,'end':end,'path':str(f),'thumbnail':f'search/thumb-{i:03d}.jpg'})
(root/'work/search-clips.json').write_text(json.dumps(clips,indent=2));print('Prepared',len(clips),'unlabelled video-only windows')
