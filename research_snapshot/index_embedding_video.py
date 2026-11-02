import os,time,json,pathlib,hashlib
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,numpy as np
from sentence_transformers import SentenceTransformer
root=pathlib.Path.cwd()
out=root/'outputs/tracking-demo/search'
clips=json.loads((root/'work/search-clips.json').read_text())
device='mps' if torch.backends.mps.is_available() else 'cpu'
print('Loading model on',device,flush=True)
started=time.time()
model=SentenceTransformer(str(root/'work/embeddinggemma-2'),device=device,config_kwargs={'audio_config':None},model_kwargs={'torch_dtype':torch.float32})
print('Loaded in',time.time()-started,flush=True)
vecs=[]
for i,c in enumerate(clips):
    t=time.time()
    v=model.encode({'video':c['path']},normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    v=np.asarray(v,dtype=np.float32).reshape(-1)
    assert len(v)==768 and np.isfinite(v).all()
    vecs.append(v)
    print('Indexed',i+1,'/',len(clips),'seconds',round(time.time()-t,2),flush=True)
    np.save(out/'clip-vectors.npy',np.stack(vecs))
metadata=[{k:v for k,v in c.items() if k!='path'} for c in clips]
(out/'clips.json').write_text(json.dumps(metadata))
queries=['A blunt dissector in the endoscopic surgical view','Kerrison rongeurs in the endoscopic surgical view','Pituitary rongeurs in the endoscopic surgical view','An endoscopic view with no instrument visible']
results=[]
for q in queries:
    v=model.encode(q,prompt_name='SearchQuery',normalize_embeddings=True,convert_to_numpy=True).reshape(-1)
    scores=np.stack(vecs)@v
    ranked=[]
    for i in np.argsort(scores)[::-1]:
        c=metadata[int(i)]
        if any(abs(c['start']-x['start'])<12 for x in ranked):continue
        ranked.append({**c,'score':float(scores[i])})
        if len(ranked)==5:break
    results.append({'query':q,'results':ranked})
(out/'pilot-results.json').write_text(json.dumps(results,indent=2))
(out/'provenance.json').write_text(json.dumps({'model':'google/embeddinggemma-2','device':device,'dtype':'float32','clips':len(clips),'window_seconds':12,'stride_seconds':8,'source':'raw public phantom video, no annotation text or overlay','elapsed_seconds':time.time()-started,'dimensions':768,'weights_sha256':hashlib.file_digest(open(root/'work/embeddinggemma-2/model.safetensors','rb'),'sha256').hexdigest()},indent=2))
print('INDEX COMPLETE',flush=True)
