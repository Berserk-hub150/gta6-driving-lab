import argparse, csv, json
from pathlib import Path
import numpy as np
from PIL import Image
def features(path):
    with Image.open(path) as im:
        w,h=im.size
        road=im.convert('L').crop((0,h//2,w,h)).resize((16,8),Image.Resampling.BOX)
        return np.append(np.asarray(road,dtype=float).ravel()/255,1.)
def train(manifest,model):
    with manifest.open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)<2: raise ValueError('Provide at least two training frames')
    x=np.stack([features(manifest.parent/r['frame']) for r in rows])
    y=np.array([float(r['steer']) for r in rows])
    if not np.all(np.isfinite(y)&(np.abs(y)<=1)): raise ValueError('Steering outside [-1,1]')
    penalty=np.eye(x.shape[1])*.05;penalty[-1,-1]=0
    weights=np.linalg.solve(x.T@x+penalty,x.T@y)
    model.write_text(json.dumps({'version':1,'weights':weights.tolist()}))
def predict(model,frame):
    weights=np.asarray(json.loads(model.read_text())['weights'])
    return float(np.clip(features(frame)@weights,-1,1))
if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('train');a.add_argument('manifest',type=Path);a.add_argument('model',type=Path)
    a=sub.add_parser('predict');a.add_argument('model',type=Path);a.add_argument('frame',type=Path);a=p.parse_args()
    if a.mode=='train': train(a.manifest,a.model)
    else: print(predict(a.model,a.frame))
