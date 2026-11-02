"""Evaluate saved rankings after embedding. Labels never enter model inputs."""
import csv,json,pathlib,collections
root=pathlib.Path.cwd()
out=root/'outputs/tracking-demo/search'
rows=[r for r in csv.DictReader(open('data/annotations.csv')) if r['int_video']=='5']
by={int(r['int_second']):r['str_instrument'] for r in rows}
rankings=json.loads((out/'pilot-results.json').read_text())
targets=['blunt_dissector','kerrisons','pituitary_rongeurs','no_instrument_visible']
report=[]
for query,target in zip(rankings,targets):
    ranked=[]
    for result in query['results']:
        labels=[by[t] for t in range(int(result['start']),int(result['end'])) if t in by]
        share=labels.count(target)/len(labels) if labels else 0
        ranked.append({**result,'target_fraction':share,'relevant':share>=0.5})
    report.append({'query':query['query'],'target':target,'hit_at_1':ranked[0]['relevant'],'hit_at_3':any(r['relevant'] for r in ranked[:3]),'results':ranked})
summary={'queries':len(report),'hit_at_1':sum(x['hit_at_1'] for x in report)/len(report),'hit_at_3':sum(x['hit_at_3'] for x in report)/len(report),'relevance_rule':'target visible in at least half annotated seconds of the retrieved window','scope':'Four prespecified instrument-state queries on one public phantom video; not independent-case validation','labels_excluded_from_index':True,'results':report}
(out/'retrieval-evaluation.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))
