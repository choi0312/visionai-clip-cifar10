"""Audit saved logits against per-image CSV, confusion matrices and reported metrics."""
from pathlib import Path
import os, csv, json, hashlib
import numpy as np
ROOT=Path(os.environ.get('VISIONAI_ROOT', str(Path(__file__).resolve().parents[1])))
R=ROOT/'results'
z=np.load(R/'logits.npz');y=np.load(R/'labels.npy')
m=json.loads((R/'metrics.json').read_text())
cms=json.loads((R/'confusion_matrices.json').read_text())
rows=list(csv.DictReader((R/'predictions.csv').open()))
assert len(rows)==len(y)==10000 and np.bincount(y).tolist()==[1000]*10
def close(a,b): assert abs(float(a)-float(b))<1e-10,(a,b)
pred={}
for name in ['bare','photo','ensemble']:
    a=z[name].astype('float64');assert a.shape==(10000,10) and np.isfinite(a).all()
    p=a.argmax(1);pred[name]=p;conf=[];cm=np.zeros((10,10),int)
    # Scalar per-image accounting is deliberately separate from the plotting implementation.
    for i,(truth,guess,row) in enumerate(zip(y,p,rows)):
        assert int(row['index'])==i and int(row['label'])==truth and int(row[name+'_pred'])==guess
        score=1/sum(np.exp(a[i]-a[i,guess]));close(score,row[name+'_score']);conf.append(score)
        cm[truth,guess]+=1
    assert cm.tolist()==cms[name]
    close(cm.trace()/cm.sum(),m['conditions'][name]['accuracy'])
    assert int(cm.trace())==m['conditions'][name]['correct']
    f1=[]
    for c in range(10):
        tp=int(cm[c,c]);fp=int(cm[:,c].sum())-tp;fn=int(cm[c,:].sum())-tp
        f1.append(2*tp/(2*tp+fp+fn))
    close(sum(f1)/10,m['conditions'][name]['macro_f1'])
    top3=sum(int(int(t) in np.argpartition(a[i],-3)[-3:]) for i,t in enumerate(y))/len(y)
    close(top3,m['conditions'][name]['top3_accuracy'])
    assert sum(c>=.9 and g!=t for c,g,t in zip(conf,p,y))==m['conditions'][name]['wrong_score_ge_09']
for a,b in [('bare','photo'),('photo','ensemble')]:
    pair=m['paired'][a+'_to_'+b]
    assert sum(x!=t and v==t for x,v,t in zip(pred[a],pred[b],y))==pair['fixed_errors']
    assert sum(x==t and v!=t for x,v,t in zip(pred[a],pred[b],y))==pair['new_errors']
    assert int((pred[a]!=pred[b]).sum())==pair['prediction_changes']
o=m['candidate_omission'];allow=o['candidate_ids'];unknown=np.isin(y,[3,5])
assert int(unknown.sum())==2000 and o['unknown_forced_wrong_count']==2000
oscores=[];op=[]
for i,row in enumerate(rows):
    v=z['photo'][i,allow].astype(float);k=int(v.argmax());p=allow[k];s=1/sum(np.exp(v-v[k]))
    assert p==int(row['omitted_pred']);close(s,row['omitted_score']);op.append(p);oscores.append(s)
oscores=np.array(oscores);op=np.array(op)
close(np.mean(op[~unknown]==y[~unknown]),o['known_accuracy'])
for t in o['thresholds']:
    keep=oscores>=t['threshold']
    assert int((keep&unknown).sum())==t['unknown_accepted']
    assert int((keep&~unknown).sum())==t['known_accepted']
    assert int((keep&~unknown&(op==y)).sum())==t['known_accepted_correct']
examples=json.loads((R/'selected_examples.json').read_text());pixels=np.load(R/'example_pixels.npz')
assert len({e['index'] for e in examples})==8
assert all(pixels[str(e['index'])].shape==(32,32,3) for e in examples)
audit={'status':'passed','images':len(y),'conditions':list(pred),'checks':['CSV/logits consistency','confusion counts','top-1/top-3','macro F1 from TP/FP/FN','paired changes','8-candidate softmax and threshold retention','example records'],
    'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'logits.npz',R/'predictions.csv',R/'labels.npy']}}
(R/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
