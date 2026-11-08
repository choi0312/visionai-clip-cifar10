"""Frozen CLIP zero-shot inference; fixed prompts, complete CIFAR-10 test split."""
from pathlib import Path
import os, json, hashlib, tarfile, pickle, random, time, platform, importlib.metadata as md
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import numpy as np
import requests
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

ROOT=Path(os.environ.get('VISIONAI_ROOT','/content/visionai_week10'))
DATA=ROOT/'data';R=ROOT/'results'
DATA.mkdir(parents=True,exist_ok=True);R.mkdir(exist_ok=True)
MODEL='openai/clip-vit-base-patch32'
REVISION='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
URL='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
EXPECTED_MD5='c58f30108f718f92721af3b95e74349a'
CLASSES=['airplane','car','bird','cat','deer','dog','frog','horse','ship','truck']
ORIGINAL_NAMES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
BATCH=128;SEED=42
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available()
def write_json(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def noun(label):return ('an ' if label[0] in 'aeiou' else 'a ')+label
prompts={
    'bare':[[label] for label in CLASSES],
    'photo':[[f'a photo of {noun(label)}.'] for label in CLASSES],
    'ensemble':[[f'a photo of {noun(label)}.',f'a blurry photo of {noun(label)}.',
        f'a low resolution photo of {noun(label)}.',f'a close-up photo of {noun(label)}.',
        f'a photo of the {label}.'] for label in CLASSES]}
# Conditions are declared and saved before loading test labels or running inference.
write_json(R/'protocol.json',{'model':MODEL,'revision':REVISION,'seed':SEED,'classes':CLASSES,
    'original_class_names':ORIGINAL_NAMES,'prompts':prompts,'training_epochs':0,
    'candidate_omission':{'base_condition':'photo','removed_class_ids':[3,5],
        'removed_classes':['cat','dog'],'score_thresholds':[.5,.7,.8,.9,.95,.99]},
    'ensemble_rule':'L2-normalize each text vector, mean within class, L2-normalize class mean',
    'selection':'all 10000 test_batch images in the original order; no training batches loaded',
    'evaluation':'labels used only to compute metrics after scoring; no prompt selection or threshold tuning'})
archive=DATA/'cifar-10-python.tar.gz'
if not archive.exists():
    print('DOWNLOADING_CIFAR10',flush=True)
    resp=requests.get(URL,timeout=180);resp.raise_for_status();archive.write_bytes(resp.content)
raw=archive.read_bytes();assert hashlib.md5(raw).hexdigest()==EXPECTED_MD5,'Official archive MD5 mismatch'
with tarfile.open(archive,'r:gz') as tf:
    test_bytes=tf.extractfile('cifar-10-batches-py/test_batch').read()
    # Only the checksum-verified official data file is unpickled; no arbitrary input is accepted.
    test=pickle.loads(test_bytes,encoding='bytes')
images=np.asarray(test[b'data'],dtype=np.uint8).reshape(-1,3,32,32).transpose(0,2,3,1)
labels=np.asarray(test[b'labels'],dtype=np.int64)
assert images.shape==(10000,32,32,3) and np.bincount(labels).tolist()==[1000]*10
write_json(DATA/'manifest.json',{'source':URL,'source_final_url':str(resp.url) if 'resp' in globals() else URL,
    'archive_md5':EXPECTED_MD5,'archive_sha256':hashlib.sha256(raw).hexdigest(),
    'test_batch_sha256':hashlib.sha256(test_bytes).hexdigest(),'test_images':10000,'per_class':1000,
    'raw_shape':[32,32,3],'order':'official test_batch order, indices 0 through 9999',
    'original_classes':ORIGINAL_NAMES,'prompt_classes':CLASSES,'train_split_used':False})
np.save(R/'labels.npy',labels)
np.save(DATA/'test_images.npy',images)
print('DATA_READY',len(images),flush=True)
processor=CLIPProcessor.from_pretrained(MODEL,revision=REVISION,use_fast=False)
model=CLIPModel.from_pretrained(MODEL,revision=REVISION).cuda().eval().requires_grad_(False)
processor.save_pretrained(R/'processor')
assert model.config.projection_dim==512
model_config=model.config.to_dict();write_json(R/'model_config.json',model_config)
scale=float(model.logit_scale.exp().item())
weights={};text_vectors={};text_start=time.perf_counter()
with torch.inference_mode():
    for name,groups in prompts.items():
        texts=[s for group in groups for s in group]
        inputs=processor(text=texts,return_tensors='pt',padding=True).to('cuda')
        vectors=model.get_text_features(**inputs)
        vectors=vectors/vectors.norm(dim=-1,keepdim=True)
        text_vectors[name]=vectors.cpu().numpy()
        v=vectors.reshape(10,len(groups[0]),-1).mean(dim=1)
        weights[name]=v/v.norm(dim=-1,keepdim=True)
torch.cuda.synchronize();text_s=time.perf_counter()-text_start
features=[];prep_s=0.;forward_s=0.
def sync():torch.cuda.synchronize()
with torch.inference_mode():
    warm=processor(images=[Image.fromarray(images[0])],return_tensors='pt').to('cuda')
    model.get_image_features(**warm);sync()
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for first in range(0,len(images),BATCH):
        sync();t=time.perf_counter()
        inp=processor(images=[Image.fromarray(im) for im in images[first:first+BATCH]],return_tensors='pt').to('cuda')
        sync();prep_s+=time.perf_counter()-t
        t=time.perf_counter();f=model.get_image_features(**inp);sync();forward_s+=time.perf_counter()-t
        f=f/f.norm(dim=-1,keepdim=True);features.append(f.cpu())
        if (first//BATCH+1)%10==0:print('ENCODED',min(first+BATCH,len(images)),'/',len(images),flush=True)
    image_loop_s=time.perf_counter()-start
    image_features=torch.cat(features).cuda()
    assert float((image_features.norm(dim=1)-1).abs().max())<1e-5
    logits={name:(scale*image_features@w.T).cpu().numpy() for name,w in weights.items()}
    # Cross-check cached embeddings against the CLIP forward API on the same 8 images/texts.
    check=processor(images=[Image.fromarray(im) for im in images[:8]],
        text=[s[0] for s in prompts['photo']],return_tensors='pt',padding=True).to('cuda')
    direct=model(**check).logits_per_image.cpu().numpy()
    diff=float(np.max(np.abs(direct-logits['photo'][:8])))
    assert diff<.002,('Forward/cached mismatch',diff)
np.savez_compressed(R/'logits.npz',**logits)
np.savez_compressed(R/'text_features.npz',**text_vectors)
# Image embeddings are kept locally on the VM, not included in the public result archive.
np.save(R/'image_features.npy',image_features.cpu().numpy())
env={'model':MODEL,'revision':REVISION,'gpu':torch.cuda.get_device_name(0),'python':platform.python_version(),
    'packages':{k:md.version(k) for k in ['torch','torchvision','transformers','numpy','Pillow','matplotlib','requests']},
    'dtype':'float32','batch_size':BATCH,'seed':SEED,'image_count':10000,'class_count':10,
    'parameters':sum(p.numel() for p in model.parameters()),'projection_dim':512,'training_epochs':0,
    'logit_scale_exp':scale,'timings':{'text_encoding_s':text_s,'image_loop_s':image_loop_s,
        'image_preprocessing_s':prep_s,'image_gpu_forward_s':forward_s},
    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'direct_forward_check_max_abs_logit_error':diff,
    'preprocessing':processor.image_processor.to_dict(),
    'scope':'zero additional task-specific training; full CIFAR-10 test split, possible overlap with CLIP pretraining unknown'}
write_json(R/'environment.json',env)
print('INFERENCE_DONE',json.dumps(env),flush=True)
