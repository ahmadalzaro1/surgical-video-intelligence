import os,json,time,re,pathlib,threading
from urllib.parse import urlparse,parse_qs
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,torch
from sentence_transformers import SentenceTransformer
root=pathlib.Path.cwd()
directory=root/'outputs/tracking-demo'
vectors=np.load(directory/'search/clip-vectors.npy')
clips=json.loads((directory/'search/clips.json').read_text())
assert len(clips)==len(vectors)==60 and np.isfinite(vectors).all()
device='mps' if torch.backends.mps.is_available() else 'cpu'
model=SentenceTransformer(str(root/'work/embeddinggemma-2'),device=device,config_kwargs={'audio_config':None},model_kwargs={'torch_dtype':torch.float32})
lock=threading.Lock()
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(directory),**kwargs)
    def do_GET(self):
        url=urlparse(self.path)
        if url.path!='/api/search':return super().do_GET()
        q=parse_qs(url.query).get('q',[''])[0].strip()
        if not q or len(q)>500:self.send_error(400,'Query must contain 1–500 characters');return
        start=time.time()
        try:
            with lock:
                v=model.encode(q,prompt_name='SearchQuery',normalize_embeddings=True,convert_to_numpy=True).reshape(-1)
            assert np.isfinite(v).all()
            scores=vectors@v
            ranked=np.argsort(scores)[::-1]
            picked=[]
            for i in ranked:
                c=clips[int(i)]
                if any(abs(c['start']-x['start'])<12 for x in picked):continue
                picked.append({**c,'score':float(scores[i])})
                if len(picked)==3:break
            data=json.dumps({'query':q,'results':picked,'elapsed_seconds':time.time()-start,'model':'google/embeddinggemma-2','labels_used':False}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',len(data));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        except Exception as e:self.send_error(500,str(e))
    def send_head(self):
        p=self.translate_path(self.path)
        if os.path.isfile(p) and 'Range' in self.headers:
            f=open(p,'rb');size=os.path.getsize(p);m=re.match(r'bytes=(\d+)-(\d*)',self.headers['Range'])
            if m:
                start=int(m[1]);end=min(int(m[2]) if m[2] else size-1,size-1)
                if start>=size:f.close();self.send_error(416);return None
                self.send_response(206);self.send_header('Content-Type',self.guess_type(p));self.send_header('Accept-Ranges','bytes');self.send_header('Content-Range',f'bytes {start}-{end}/{size}');self.send_header('Content-Length',end-start+1);self.end_headers();f.seek(start);self.remaining=end-start+1;return f
            f.close()
        self.remaining=None
        return super().send_head()
    def copyfile(self,source,destination):
        if self.remaining is None:return super().copyfile(source,destination)
        remaining=self.remaining
        while remaining:
            chunk=source.read(min(65536,remaining))
            if not chunk:break
            destination.write(chunk);remaining-=len(chunk)
print('SEARCH DEMO READY http://127.0.0.1:8771',flush=True)
ThreadingHTTPServer(('127.0.0.1',8771),Handler).serve_forever()
