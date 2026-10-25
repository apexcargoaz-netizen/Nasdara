import numpy as np,cv2,json,math,sys
from PIL import Image
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
E=lambda r:cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
S8=4; S=2  # shot px/m, canvas px/m
def masks(a):
    R,G,B=a[...,0],a[...,1],a[...,2];s=a.sum(2)
    u=lambda m:m.astype(np.uint8)
    mauve=u((R-B>=10)&(R-B<=38)&(R-G>=6)&(R-G<=30)&(G>=B-4)&(R>=165)&(R<=218))
    mauve=cv2.morphologyEx(cv2.morphologyEx(mauve,cv2.MORPH_CLOSE,E(3)),cv2.MORPH_OPEN,E(7))
    dark=u((s<3*120)&(abs(R-B)<=16))
    blk=cv2.morphologyEx(dark,cv2.MORPH_OPEN,E(7))
    thin=dark&(1-cv2.dilate(blk,E(3)))
    lt=u((s>=3*236)&(abs(R-B)<=12)&(abs(G-R)<=10))
    def sh(m,dx,dy):
        o=np.zeros_like(m);h,w=m.shape
        o[max(dy,0):h+min(dy,0),max(dx,0):w+min(dx,0)]=m[max(-dy,0):h-max(dy,0),max(-dx,0):w-max(dx,0)];return o
    wh=np.zeros_like(lt)
    for dx,dy in [(1,0),(0,1),(1,1),(1,-1),(2,1),(1,2),(2,-1),(1,-2)]:
        n=max(abs(dx),abs(dy));P=np.zeros_like(lt);M=np.zeros_like(lt)
        for w in range(2,9):
            ox,oy=round(dx*w/n),round(dy*w/n);P|=sh(thin,ox,oy);M|=sh(thin,-ox,-oy)
        wh|=P&M
    wh&=lt
    orange=u((R>=200)&(R-B>=80)&(G>=130)&(G<=200)&(B<=150))
    orange=cv2.morphologyEx(orange,cv2.MORPH_CLOSE,E(3))
    grey=u((s>=3*85)&(s<=3*175)&(abs(R-B)<=10)&(abs(R-G)<=8))
    grey&=1-cv2.dilate(cv2.morphologyEx(grey,cv2.MORPH_OPEN,E(5)),E(5))   # thin only
    grey&=1-cv2.dilate(blk|mauve|wh,E(5))
    return mauve,wh,orange,grey
def longcomp(m,minlen,maxw):
    n,lab,st,_=cv2.connectedComponentsWithStats(m,8)
    sk=skeletonize(m>0);L=np.bincount(lab[sk],minlength=n)+1
    keep=(L>=minlen)&(st[:,4]/L<=maxw);keep[0]=False
    return keep[lab].astype(np.uint8)
def vec(m,minlen,eps):
    sk=skeletonize(m>0);nb=ndi.convolve(sk.astype(np.uint8),np.ones((3,3)),mode='constant')-1
    node=sk&(nb!=2);N8=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    ys,xs=np.nonzero(sk);SS=set(zip(ys.tolist(),xs.tolist()));NODE=set(map(tuple,np.argwhere(node).tolist()))
    nbs=lambda y,x:[(y+a,x+b) for a,b in N8 if (y+a,x+b) in SS]
    used=set();lines=[]
    def trace(p0,p1):
        p=[p0,p1];used.add((p0,p1));prev,cur=p0,p1
        while cur not in NODE:
            nx=[q for q in nbs(*cur) if q!=prev and q not in p[-3:]]
            if not nx:break
            prev,cur=cur,nx[0];p.append(cur)
            if cur==p0:break
        used.add((p[-1],p[-2]));return p
    for n0 in NODE:
        for q in nbs(*n0):
            if (n0,q) in used:continue
            lines.append(trace(n0,q))
    def dp(p,e):
        if len(p)<3:return p
        a,b=p[0],p[-1];dx,dy=b[0]-a[0],b[1]-a[1];nn=math.hypot(dx,dy) or 1
        d=[abs(dy*(q[0]-a[0])-dx*(q[1]-a[1]))/nn for q in p[1:-1]];k=max(range(len(d)),key=d.__getitem__)
        return dp(p[:k+2],e)[:-1]+dp(p[k+1:],e) if d[k]>e else [a,b]
    out=[]
    for p in lines:
        q=dp([(x,y) for y,x in p],eps)
        L=sum(math.hypot(q[i][0]-q[i-1][0],q[i][1]-q[i-1][1]) for i in range(1,len(q)))
        if L>=minlen: out.append(q)
    return out
if __name__=='__main__':
    done=json.load(open('s8/done.json'))
    towns={}
    for n,x,y in done: towns.setdefault(n,[]).append((x,y))
    res={'road':[],'street':[],'pipe':[],'thin':[],'cov':[]}
    for t,shots in towns.items():
        X0=min(x for x,y in shots);Y0=min(y for x,y in shots)
        W=int((max(x for x,y in shots)-X0+342)*S)+2;H=int((max(y for x,y in shots)-Y0+152)*S)+2
        C={k:np.zeros((H,W),np.uint8) for k in ['mauve','wh','orange','grey']}
        for x,y in shots:
            a=np.array(Image.open(f's8/z8_{x:05d}_{y:05d}.jpg').convert('RGB')).astype(np.int16)
            ms=masks(a)
            for k,m in zip(['mauve','wh','orange','grey'],ms):
                m[:3]=0;m[-3:]=0;m[:,:3]=0;m[:,-3:]=0
                f=S8//S;d=m[:606,:1366].reshape(303,f,683,f).max(axis=(1,3))
                ox,oy=int(round((x-X0)*S)),int(round((y-Y0)*S));h,w=d.shape
                np.maximum(C[k][oy:oy+h,ox:ox+w],d[:H-oy,:W-ox],out=C[k][oy:oy+h,ox:ox+w])
            res['cov'].append([x,y])
        road=longcomp(cv2.morphologyEx(C['mauve'],cv2.MORPH_CLOSE,E(5)),30,9)
        st=longcomp(cv2.morphologyEx(C['wh'],cv2.MORPH_CLOSE,E(5)),40,6)&(1-cv2.dilate(road,E(3)))
        pp=longcomp(cv2.morphologyEx(C['orange'],cv2.MORPH_CLOSE,E(3)),16,6)
        gr=C['grey']
        gr=longcomp(cv2.morphologyEx(gr,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))),24,4)
        lines=cv2.HoughLinesP(gr*255,1,np.pi/360,30,minLineLength=50,maxLineGap=14)
        thin=[]
        if lines is not None:
            for x1,y1,x2,y2 in lines.reshape(-1,4): thin.append([(x1,y1),(x2,y2)])
        tw=lambda q:[[round(X0+u/S,1),round(Y0+v/S,1)] for u,v in q]
        res['road']+= [tw(q) for q in vec(road,20,1.0)]
        res['street']+=[tw(q) for q in vec(st,20,1.0)]
        res['pipe']+= [tw(q) for q in vec(pp,10,1.0)]
        res['thin']+= [tw(q) for q in thin]
        print(t,{k:len(v) for k,v in res.items()})
    json.dump(res,open('x8.json','w'))
