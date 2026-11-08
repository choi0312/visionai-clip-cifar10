"""Compute metrics/figures from saved CLIP logits; no model fitting or selection."""
from pathlib import Path
import os,json,csv
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(os.environ.get('VISIONAI_ROOT','/content/visionai_week10'))
R=ROOT/'results';F=ROOT/'figures';F.mkdir(exist_ok=True)
protocol=json.loads((R/'protocol.json').read_text());classes=protocol['classes']
labels=np.load(R/'labels.npy');z=np.load(R/'logits.npz')
names=['bare','photo','ensemble'];display={'bare':'Class name','photo':'Photo sentence','ensemble':'5-prompt ensemble'}
colors=['#337da5','#d08232','#409467']
def write(name,obj):(R/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def softmax(x):
    exp=np.exp(x.astype(np.float64)-x.max(axis=1,keepdims=True));return exp/exp.sum(axis=1,keepdims=True)
probs={c:softmax(z[c]) for c in names};pred={c:z[c].argmax(axis=1) for c in names}
metrics={};matrices={}
for c in names:
    p=pred[c];conf=probs[c].max(axis=1);correct=p==labels
    cm=np.zeros((10,10),dtype=int);np.add.at(cm,(labels,p),1);matrices[c]=cm
    tp=cm.diagonal();f1=2*tp/(cm.sum(axis=0)+cm.sum(axis=1))
    metrics[c]={'n':len(labels),'correct':int(correct.sum()),'accuracy':float(correct.mean()),
        'top3_accuracy':float(np.mean(np.any(np.argsort(-z[c],axis=1)[:,:3]==labels[:,None],axis=1))),
        'macro_f1':float(f1.mean()),'errors':int((~correct).sum()),'per_class_recall':(tp/1000).tolist(),
        'per_class_f1':f1.tolist(),'mean_max_softmax':float(conf.mean()),
        'wrong_score_ge_09':int(np.sum((~correct)&(conf>=.9)))}
paired={}
for a,b in [('bare','photo'),('photo','ensemble')]:
    ca=pred[a]==labels;cb=pred[b]==labels
    paired[a+'_to_'+b]={'fixed_errors':int((~ca&cb).sum()),'new_errors':int((ca&~cb).sum()),
        'prediction_changes':int((pred[a]!=pred[b]).sum())}
allowed=np.array([i for i in range(10) if i not in [3,5]])
op=softmax(z['photo'][:,allowed]);omit_pred=allowed[op.argmax(axis=1)];omit_score=op.max(axis=1)
unknown=np.isin(labels,[3,5]);known=~unknown
omission={'removed_class_ids':[3,5],'candidate_ids':allowed.tolist(),'unknown_count':int(unknown.sum()),
    'known_count':int(known.sum()),'known_accuracy':float(np.mean(omit_pred[known]==labels[known])),
    'unknown_forced_wrong_count':int(np.sum(omit_pred[unknown]!=labels[unknown])),
    'unknown_mean_max_softmax':float(omit_score[unknown].mean()),
    'unknown_score_ge_09':int(np.sum(omit_score[unknown]>=.9)),
    'thresholds':[{ 'threshold':t,'unknown_accepted':int(np.sum(unknown&(omit_score>=t))),
        'known_accepted':int(np.sum(known&(omit_score>=t))),
        'known_accepted_correct':int(np.sum(known&(omit_score>=t)&(omit_pred==labels)))}
        for t in protocol['candidate_omission']['score_thresholds']]}
write('metrics.json',{'conditions':metrics,'paired':paired,'candidate_omission':omission,
    'definitions':{'accuracy':'correct top-1 / 10000','top3':'ground truth within the three highest logits / 10000',
    'macro_f1':'mean of 10 class F1 scores, 2TP/(2TP+FP+FN)',
    'omission':'same photo logits, columns cat/dog removed, softmax recomputed over 8 candidates; these classes are absent from candidates, not necessarily unseen in pretraining'}})
write('confusion_matrices.json',{c:matrices[c].tolist() for c in names})
with (R/'predictions.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['index','label',*[n+'_pred' for n in names],*[n+'_score' for n in names],'omitted_pred','omitted_score'])
    for i,y in enumerate(labels):w.writerow([i,int(y),*[int(pred[n][i]) for n in names],*[float(probs[n][i].max()) for n in names],int(omit_pred[i]),float(omit_score[i])])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190})
def save(fig,name):fig.savefig(F/name,bbox_inches='tight',pad_inches=.07);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,2.7))
bars=axs[0].bar(range(3),[metrics[c]['accuracy']*100 for c in names],color=colors)
axs[0].set(ylim=(0,100),ylabel='Top-1 accuracy (%)',title='Same 10,000 test images')
axs[0].set_xticks(range(3),[display[c] for c in names],fontsize=9)
for b,c in zip(bars,names):axs[0].text(b.get_x()+b.get_width()/2,b.get_height()+1,f"{metrics[c]['accuracy']*100:.2f}%",ha='center')
x=np.arange(2);v=list(paired.values())
axs[1].bar(x-.17,[r['fixed_errors'] for r in v],width=.34,color=colors[2],label='Wrong -> correct')
axs[1].bar(x+.17,[r['new_errors'] for r in v],width=.34,color=colors[1],label='Correct -> wrong')
axs[1].set_xticks(x,['Name -> sentence','Sentence -> ensemble']);axs[1].set(ylabel='Images',title='Both improvement and regression');axs[1].legend(fontsize=8)
fig.tight_layout();save(fig,'prompt_comparison.png')

fig,axs=plt.subplots(1,2,figsize=(10,3.5),gridspec_kw={'width_ratios':[1.15,1]})
cm=matrices['photo'];ax=axs[0];ax.imshow(cm/10,cmap='Blues',vmin=0,vmax=100)
for y in range(10):
    for x in range(10):
        if cm[y,x]>0:ax.text(x,y,str(cm[y,x]),ha='center',va='center',fontsize=6,color='white' if cm[y,x]>500 else '#222222')
ax.set_xticks(range(10),classes,rotation=45,ha='right',fontsize=7);ax.set_yticks(range(10),classes,fontsize=7);ax.set(xlabel='Prediction',ylabel='Ground truth',title='Photo sentence: confusion counts')
y=np.arange(10)
for offset,c,color in [(-.18,'photo',colors[1]),(.18,'ensemble',colors[2])]:axs[1].barh(y+offset,np.array(metrics[c]['per_class_recall'])*100,height=.36,label=display[c],color=color)
axs[1].set_yticks(y,classes);axs[1].set_ylim(11.1,-.6)
axs[1].set(xlim=(0,100),xlabel='Recall (%)',title='1,000 test images per class')
axs[1].legend(fontsize=7,loc='lower left',ncol=2)
fig.tight_layout();save(fig,'confusion.png')

fig,axs=plt.subplots(1,2,figsize=(10,2.65))
axs[0].hist(omit_score[known],bins=np.linspace(0,1,21),weights=np.ones(known.sum())/known.sum()*100,alpha=.65,label='In-candidate GT (8,000)',color=colors[0])
axs[0].hist(omit_score[unknown],bins=np.linspace(0,1,21),weights=np.ones(unknown.sum())/unknown.sum()*100,alpha=.6,label='Omitted cat/dog (2,000)',color=colors[1])
axs[0].set(xlabel='Max softmax over 8 candidates',ylabel='Images in group (%)',title='High score does not mean known class');axs[0].legend(fontsize=7)
ts=omission['thresholds']
axs[1].plot([t['threshold'] for t in ts],[t['known_accepted']/8000*100 for t in ts],'o-',label='In-candidate GT retained',color=colors[0])
axs[1].plot([t['threshold'] for t in ts],[t['unknown_accepted']/2000*100 for t in ts],'o-',label='Omitted classes retained (wrong)',color=colors[1])
axs[1].set(xlabel='Fixed score threshold',ylabel='Retained in each group (%)',title='Descriptive thresholds; none selected');axs[1].legend(fontsize=7);axs[1].grid(alpha=.2)
fig.tight_layout();save(fig,'candidate_omission.png')

chosen=[];used=set()
groups=[('Name wrong -> sentence correct',np.flatnonzero((pred['bare']!=labels)&(pred['photo']==labels))),
    ('Name correct -> sentence wrong',np.flatnonzero((pred['bare']==labels)&(pred['photo']!=labels))),
    ('Sentence + ensemble wrong',np.flatnonzero((pred['photo']!=labels)&(pred['ensemble']!=labels)))]
for reason,ids in groups:
    for i in ids:
        if int(i) not in used:chosen.append({'index':int(i),'reason':reason});used.add(int(i))
        if sum(r['reason']==reason for r in chosen)==2:break
for i in np.flatnonzero(unknown)[np.argsort(-omit_score[unknown])]:
    if int(i) not in used:chosen.append({'index':int(i),'reason':'GT excluded from candidates'});used.add(int(i))
    if len(chosen)>=8:break
assert len(chosen)==8
write('selected_examples.json',chosen)
image_path=ROOT/'data/test_images.npy'
if image_path.exists():imgs=np.load(image_path)
else:
    # A figure-only replay uses the tiny preserved examples, not the full original archive.
    selected_pixels=np.load(R/'example_pixels.npz');imgs={int(k):selected_pixels[k] for k in selected_pixels.files}
np.savez_compressed(R/'example_pixels.npz',**{str(r['index']):imgs[r['index']] for r in chosen})
fig,axs=plt.subplots(2,4,figsize=(10,4.3))
for ax,case in zip(axs.flat,chosen):
    i=case['index'];ax.imshow(imgs[i],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
    ax.set_title(f"#{i} | GT: {classes[labels[i]]}",fontsize=9)
    if case['reason']=='GT excluded from candidates':s=f"8 labels: {classes[omit_pred[i]]}\nscore={omit_score[i]:.3f} (wrong)"
    else:s='\n'.join(f"{a}: {classes[pred[c][i]]}" for a,c in [('Name','bare'),('Photo','photo'),('Ensemble','ensemble')])
    ax.set_xlabel(s,fontsize=8)
fig.tight_layout(h_pad=1.0);save(fig,'predictions.png')
print('METRICS',json.dumps({'conditions':metrics,'paired':paired,'candidate_omission':omission}),flush=True)
print('ANALYSIS_DONE',flush=True)
