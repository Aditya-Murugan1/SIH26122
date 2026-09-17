import csv
import json
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from backend.app import connect, embed_text, extract_event, init_db, score_candidate

ROOT=Path(__file__).parents[1]; corpus=ROOT/'benchmark'/'generated'
if not (corpus/'held_out.jsonl').exists():
    raise SystemExit('Run python scripts/generate_corpus.py first')
with (corpus/'schedule.csv').open(encoding='utf-8') as handle: activities=list(csv.DictReader(handle))
reports=[json.loads(line) for line in (corpus/'held_out.jsonl').read_text(encoding='utf-8').splitlines() if line]
init_db()
terms=list(connect().execute('SELECT * FROM terminology_map'))
for activity in activities:
    activity['embedding']=json.dumps(embed_text(activity['description']))
results=[]; latencies=[]
for report in reports:
    started=time.perf_counter(); event=extract_event(report['text'],'benchmark',1); event['embedding']=embed_text(event['activity_terms']); ranked=[]
    for activity in activities:
        scores,evidence=score_candidate(event, activity, terms); ranked.append((scores['fused_score'], activity['activity_code'], scores))
    ranked.sort(reverse=True); top=ranked[0]; second=ranked[1]; margin=top[0]-second[0]
    decision='AUTO_MATCHED' if top[0]>=.82 and margin>=.12 else 'REVIEW_REQUIRED' if top[0]>=.45 else 'UNMATCHED'
    correct=report['truth'] == top[1]
    reason = 'genuine ambiguity' if margin < .12 else 'weak retrieval' if top[0] < .45 else 'weak semantic similarity' if top[2]['score_semantic'] < .5 else 'weak context' if top[2]['score_context'] < .5 else 'missing information' if not event['identifiers'] and not event['location_terms'] else 'conservative margin'
    results.append((decision, correct, report['truth'] is None, any(item[1]==report['truth'] for item in ranked[:5]), reason, margin))
    latencies.append((time.perf_counter()-started)*1000)
auto=[r for r in results if r[0]=='AUTO_MATCHED']; reasons={}
for result in results:
    if result[0] == 'REVIEW_REQUIRED': reasons[result[4]] = reasons.get(result[4], 0) + 1
print(f'Total cases: {len(reports)}'); print(f'Held-out cases: {len(results)}'); print(f'Auto-match precision: {sum(r[1] for r in auto)/len(auto)*100 if auto else 0:.1f}%'); print(f'Silent errors: {sum(not r[1] and r[0]=="AUTO_MATCHED" for r in results)}'); print(f'Auto-match rate: {sum(r[0]=="AUTO_MATCHED" for r in results)/len(results)*100:.1f}%'); print(f'Review rate: {sum(r[0]=="REVIEW_REQUIRED" for r in results)/len(results)*100:.1f}%'); print(f'Unmatched rate: {sum(r[0]=="UNMATCHED" for r in results)/len(results)*100:.1f}%'); print(f'Recall@5: {sum(r[3] for r in results)/len(results)*100:.1f}%'); print(f'Median latency: {statistics.median(latencies):.2f} ms'); print('Review breakdown:', json.dumps(reasons, sort_keys=True)); print('Margin distribution:', json.dumps({'min':round(min(r[5] for r in results),3),'median':round(statistics.median(r[5] for r in results),3),'max':round(max(r[5] for r in results),3)}))
