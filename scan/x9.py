import numpy as np,cv2,json,math,sys
from PIL import Image
sys.path.insert(0,'.')
from det9 import detect
from x8 import vec   # reuse vectorizer (x8 runs main on import -> guard)
S=4
done=json.load(open('s8/done.json'))
towns={}
for n,x,y in done: towns.setdefault(n,[]).append((x,y))
out={'wall':[],'fence':[],'bld':[]}
for t,shots in towns.items():
    X0=min(x for x,y in shots);Y0=min(y for x,y in shots)
    W=int((max(x for x,y in shots)-X0+342)*S)+2;H=int((max(y for x,y in shots)-Y0+152)*S)+2
    C={k:np.zeros((H,W),np.uint8) for k in ['bld','walls','fence']}
    for x,y in shots:
        a=np.array(Image.open(f's8/z8_{x:05d}_{y:05d}.jpg').convert('RGB')).astype(np.int16)
        d=detect(a)
        for k in C:
            m=d[k].copy();m[:4]=0;m[-4:]=0;m[:,:4]=0;m[:,-4:]=0
            ox,oy=int(round((x-X0)*S)),int(round((y-Y0)*S));h,w=m.shape
            np.maximum(C[k][oy:oy+h,ox:ox+w],m[:H-oy,:W-ox],out=C[k][oy:oy+h,ox:ox+w])
    tw=lambda q:[[round(X0+u/S,1),round(Y0+v/S,1)] for u,v in q]
    out['wall']+=[tw(q) for q in vec(C['walls'],12,1.5)]
    out['fence']+=[tw(q) for q in vec(C['fence'],12,1.5)]
    cs,_=cv2.findContours(C['bld'],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in cs:
        if cv2.contourArea(c)<6*S*S: continue
        p=cv2.approxPolyDP(c,1.8,True)
        out['bld'].append([[round(X0+q[0][0]/S,1),round(Y0+q[0][1]/S,1)] for q in p])
    print(t,{k:len(v) for k,v in out.items()},flush=True)
json.dump(out,open('x9.json','w'),separators=(',',':'))
