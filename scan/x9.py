import numpy as np,cv2,json,math,sys
from PIL import Image
sys.path.insert(0,'.')
from det9 import detect
from x8 import vec   # reuse vectorizer (x8 runs main on import -> guard)
S=4
E=lambda r:cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
def ortho(c,A):
    (cx,cy),(w,h),ang=cv2.minAreaRect(c)
    if A>=0.95*w*h:
        return cv2.boxPoints(((cx,cy),(w,h),ang)).tolist()
    th=np.deg2rad(ang);R=np.array([[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]])
    p=c[:,0,:].astype(float)-[cx,cy];q=p@R.T
    ap=cv2.approxPolyDP(q.astype(np.float32).reshape(-1,1,2),3.0,True)[:,0,:].astype(float)
    n=len(ap)
    if n<4: return cv2.boxPoints(((cx,cy),(w,h),ang)).tolist()
    # snap each edge to horizontal/vertical in rotated frame
    for k in range(n):
        a,b=ap[k],ap[(k+1)%n]
        dx_,dy_=abs(b[0]-a[0]),abs(b[1]-a[1])
        if min(dx_,dy_)>0.4*max(dx_,dy_): continue
        if dx_>=dy_: m=(a[1]+b[1])/2;ap[k][1]=m;ap[(k+1)%n][1]=m
        else: m=(a[0]+b[0])/2;ap[k][0]=m;ap[(k+1)%n][0]=m
    keep=[ap[k] for k in range(n) if np.hypot(*(ap[k]-ap[k-1]))>0.5]
    if len(keep)<4: return cv2.boxPoints(((cx,cy),(w,h),ang)).tolist()
    return (np.array(keep)@R+[cx,cy]).tolist()
done=json.load(open('s8/done.json'))
towns={}
for n,x,y in done: towns.setdefault(n,[]).append((x,y))
out={'wall':[],'fence':[],'bld':[]}
for t,shots in towns.items():
    X0=min(x for x,y in shots);Y0=min(y for x,y in shots)
    W=int((max(x for x,y in shots)-X0+342)*S)+2;H=int((max(y for x,y in shots)-Y0+152)*S)+2
    C={k:np.zeros((H,W),np.uint8) for k in ['bld','walls','fence']};IMG=np.zeros((H,W,3),np.uint8)
    for x,y in shots:
        a=np.array(Image.open(f's8/z8_{x:05d}_{y:05d}.jpg').convert('RGB')).astype(np.int16)
        d=detect(a)
        ox_,oy_=int(round((x-X0)*S)),int(round((y-Y0)*S));h_,w_=a.shape[:2];IMG[oy_:oy_+h_,ox_:ox_+w_]=a[:H-oy_,:W-ox_].astype(np.uint8)
        for k in C:
            m=d[k].copy();m[:4]=0;m[-4:]=0;m[:,:4]=0;m[:,-4:]=0
            ox,oy=int(round((x-X0)*S)),int(round((y-Y0)*S));h,w=m.shape
            np.maximum(C[k][oy:oy+h,ox:ox+w],m[:H-oy,:W-ox],out=C[k][oy:oy+h,ox:ox+w])
    tw=lambda q:[[round(X0+u/S,1),round(Y0+v/S,1)] for u,v in q]
    def straightok(q):
        L=sum(math.hypot(q[i][0]-q[i-1][0],q[i][1]-q[i-1][1]) for i in range(1,len(q)))
        return L>=40 or (L>=16 and math.hypot(q[-1][0]-q[0][0],q[-1][1]-q[0][1])>=0.9*L)
    out['wall']+=[tw(q) for q in vec(C['walls'],16,4) if straightok(q)]
    out['fence']+=[tw(q) for q in vec(cv2.morphologyEx(C['fence'],cv2.MORPH_CLOSE,E(9)),16,4) if straightok(q)]
    bm=cv2.morphologyEx(C['bld'],cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    bm=cv2.morphologyEx(bm,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    cs,_=cv2.findContours(bm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for c in cs:
        A=cv2.contourArea(c)
        if A<8*S*S: continue
        mk=np.zeros((H,W),np.uint8);cv2.drawContours(mk,[c],-1,1,-1);mk=cv2.erode(mk,np.ones((3,3),np.uint8))
        r,g,b=[float(v) for v in IMG[mk>0].mean(0)] if mk.any() else (100,100,100)
        cl=3 if (r-b>60 and r>115) else 2 if (r-b>40 and g>85 and r>95) else 1 if (r-b>15 and r<130) else 4 if r<55 else 0
        out['bld'].append([cl]+[[round(X0+u/S,1),round(Y0+v/S,1)] for u,v in ortho(c,A)])
    print(t,{k:len(v) for k,v in out.items()},flush=True)
# dedupe overlapping buildings (largest first)
G=np.zeros((16400,16400),np.uint8);keep=[]
for p in sorted(out['bld'],key=lambda p:-cv2.contourArea(np.array(p[1:],np.float32))):
    q=np.array(p[1:],np.int32);mk=np.zeros((1,1));x0_,y0_=q.min(0);x1_,y1_=q.max(0)+1
    sub=G[y0_:y1_,x0_:x1_];m=np.zeros_like(sub);cv2.fillPoly(m,[q-[x0_,y0_]],1)
    if m.sum()==0: continue
    if (sub&m).sum()>0.3*m.sum(): continue
    sub|=m;keep.append(p)
print('buildings',len(out['bld']),'->',len(keep));out['bld']=keep
json.dump(out,open('x9.json','w'),separators=(',',':'))
