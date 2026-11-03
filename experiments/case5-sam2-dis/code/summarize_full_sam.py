#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--expected',type=int,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
all_paths=sorted(a.folder.glob('[0-9][0-9][0-9][0-9]/result.json')); data=[json.loads(x.read_text()) for x in all_paths if (x.parent/'comparison.mp4').exists()];indices=[x['second'] for x in data]
if indices!=list(range(len(indices))):raise ValueError('Missing or duplicated interval: refusing to combine')
if not a.allow_partial and len(data)!=a.expected:raise ValueError(f'expected {a.expected} intervals, got {len(data)}')
checks=[x for x in data if x.get('eligible')];
if not checks:raise ValueError('no eligible transitions yet')
res={'complete':len(data)==a.expected,'intervals_completed':len(data),'intervals_expected':a.expected,'display_frames':sum(x['display_frames'] for x in data),'eligible':len(checks)}
for method in ('dis','sam'):
 v=[x[method+'_iou'] for x in checks];res[method]={'mean_iou':float(np.mean(v)),'median_iou':float(np.median(v)),'iou_at_least_05':sum(x>=0.5 for x in v)}
d=[x['sam_iou']-x['dis_iou'] for x in checks];res['paired']={'sam_higher':sum(x>0 for x in d),'dis_higher':sum(x<0 for x in d),'equal':sum(x==0 for x in d)}
(a.folder/'summary.json').write_text(json.dumps(res,indent=2));manifest=a.folder/'segments.txt';manifest.write_text(''.join("file '"+str((x.parent/'comparison.mp4').resolve()).replace("'","'\\''")+"'\n" for x in all_paths))
name='comparison-full.mp4' if res['complete'] else 'comparison-preview.mp4'
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(manifest),'-c','copy','-movflags','+faststart',str(a.folder/name)],check=True)
print(json.dumps(res,indent=2))
