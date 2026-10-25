import numpy as np,cv2,json
from PIL import Image
from x8 import vec,longcomp
E=lambda r:cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
S8=4;S=2
u=lambda m:m.astype(np.uint8)
def masks(a):
    R,G,B=a[...,0],a[...,1],a[...,2];s=a.sum(2)
    brown=u((R>=125)&(R<=185)&(G>=90)&(G<=140)&(B>=55)&(B<=115)&(R-B>=45)&(R-B<=95)&(R-G>=25))
    brown=cv2.morphologyEx(brown,cv2.MORPH_CLOSE,E(13))
    tan=u((R-B>=18)&(R-B<=48)&(R-G>=10)&(R>=170)&(R<=222)&(G<=R-8))
    wide=cv2.morphologyEx(tan,cv2.MORPH_OPEN,E(7))
    tan=tan&(1-cv2.dilate(wide,E(5)))
    tan=cv2.morphologyEx(tan,cv2.MORPH_CLOSE,E(5))
    yel=u((R>225)&(G>185)&(B<175)&(R-B>60));yel=cv2.morphologyEx(yel,cv2.MORPH_CLOSE,E(5))
    return brown,tan,yel
done=json.load(open('s8/done.json'));towns={}
for n,x,y in done: towns.setdefault(n,[]).append((x,y))
res={'trail':[],'hw':[]}
for t,shots in towns.items():
    X0=min(x for x,y in shots);Y0=min(y for x,y in shots)
    W=int((max(x for x,y in shots)-X0+342)*S)+2;H=int((max(y for x,y in shots)-Y0+152)*S)+2
    C=[np.zeros((H,W),np.uint8) for _ in range(3)]
    for x,y in shots:
        a=np.array(Image.open(f's8/z8_{x:05d}_{y:05d}.jpg').convert('RGB')).astype(np.int16)
        for k,m in enumerate(masks(a)):
            m[:3]=0;m[-3:]=0;m[:,:3]=0;m[:,-3:]=0
            d=m[:606,:1366].reshape(303,2,683,2).max(axis=(1,3))
            ox,oy=int(round((x-X0)*S)),int(round((y-Y0)*S));h,w=d.shape
            np.maximum(C[k][oy:oy+h,ox:ox+w],d[:H-oy,:W-ox],out=C[k][oy:oy+h,ox:ox+w])
    tw=lambda q:[[round(X0+u_/S,1),round(Y0+v_/S,1)] for u_,v_ in q]
    tr=longcomp(cv2.morphologyEx(C[0]|C[1],cv2.MORPH_CLOSE,E(5)),16,10)
    res['trail']+=[tw(q) for q in vec(tr,16,1.2)]
    yl=longcomp(C[2],40,40)
    res['hw']+=[tw(q) for q in vec(cv2.morphologyEx(yl,cv2.MORPH_OPEN,E(3)),40,1.5)]
    print(t,{k:len(v) for k,v in res.items()},flush=True)
json.dump(res,open('x10.json','w'))
