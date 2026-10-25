import numpy as np,cv2,math
from skimage.morphology import skeletonize
E=lambda r:cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
u=lambda m:m.astype(np.uint8)
def sh(m,dx,dy):
    o=np.zeros_like(m);h,w=m.shape
    o[max(dy,0):h+min(dy,0),max(dx,0):w+min(dx,0)]=m[max(-dy,0):h-max(dy,0),max(-dx,0):w-max(dx,0)];return o
def comps(m,minlen,maxw):
    n,lab,st,_=cv2.connectedComponentsWithStats(m,8)
    sk=skeletonize(m>0);L=np.bincount(lab[sk],minlength=n)+1
    keep=(L>=minlen)&(st[:,4]/L<=maxw);keep[0]=False
    return keep[lab].astype(np.uint8),lab,keep
def detect(a):
    R,G,B=a[...,0],a[...,1],a[...,2];s=a.sum(2)
    neu=(abs(R-B)<=14)&(abs(R-G)<=12)
    # fills
    yel=u((R>225)&(G>185)&(B<175)&(R-B>60))
    mauve=u((R-B>=10)&(R-B<=38)&(R-G>=6)&(R-G<=30)&(G>=B-4)&(R>=165)&(R<=218))
    mauve=cv2.morphologyEx(cv2.morphologyEx(mauve,cv2.MORPH_CLOSE,E(3)),cv2.MORPH_OPEN,E(7))
    bgrey=u(neu&(s>=3*90)&(s<=3*140)); bgrey=cv2.morphologyEx(bgrey,cv2.MORPH_OPEN,E(7))
    brown=u((R-B>=12)&(R-B<=45)&(R<=125)&(R>=45)&(G<=R)); brown=cv2.morphologyEx(brown,cv2.MORPH_OPEN,E(5))
    olive=u((R>=95)&(R<=175)&(G>=85)&(G<=160)&(B<=95)&(R-B>=40)&(abs(R-G)<=35))
    oliveF=cv2.morphologyEx(olive,cv2.MORPH_OPEN,E(9))     # tents/blocks
    black=u(s<=3*45); blackF=cv2.morphologyEx(black,cv2.MORPH_OPEN,E(9))
    bld=bgrey|brown|oliveF|blackF
    # thin dark lines (fences + casings)
    bgm=cv2.medianBlur((s//3).astype(np.uint8),15).astype(np.int16)*3
    dk=u(neu&((s<3*175)|(s<bgm-55))); dk&=1-cv2.dilate(bld,E(3))
    dkthin=dk&(1-cv2.dilate(cv2.morphologyEx(dk,cv2.MORPH_OPEN,E(3)),E(3)))
    roadz=cv2.dilate(mauve|yel,E(15))
    # white street fill detection to find casings
    lt=u((s>=3*236)&neu)
    wh=np.zeros_like(lt)
    for dx,dy in [(1,0),(0,1),(1,1),(1,-1)]:
        P=np.zeros_like(lt);M=np.zeros_like(lt)
        for w in range(2,9):P|=sh(dk,dx*w,dy*w);M|=sh(dk,-dx*w,-dy*w)
        wh|=P&M
    wh&=lt
    whz=cv2.dilate(cv2.morphologyEx(wh,cv2.MORPH_CLOSE,E(5)),E(13))
    g8=(s//3).astype(np.uint8)
    bh=cv2.morphologyEx(g8,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_RECT,(11,11)))
    tl=u((bh>=16)&neu)&(1-roadz)&(1-whz)&(1-cv2.dilate(bld,E(3)))
    tl=cv2.morphologyEx(tl,cv2.MORPH_CLOSE,E(3))
    tl,tlab,_=comps(tl,16,5)
    dt=cv2.distanceTransform(tl,cv2.DIST_L2,3)
    skl=skeletonize(tl>0);n2=tlab.max()+1
    wsum=np.bincount(tlab[skl],weights=dt[skl],minlength=n2);cnt=np.bincount(tlab[skl],minlength=n2)+1e-9
    wid=wsum/cnt
    dsum=np.bincount(tlab[skl],weights=bh[skl].astype(float),minlength=n2)/cnt
    isw=(wid>=1.6)&(dsum>=60)
    st2=cv2.connectedComponentsWithStats(tl,8)[2] if False else None
    ys,xs=np.nonzero(tl);bx0=np.full(n2,1e9);bx1=np.zeros(n2);by0=np.full(n2,1e9);by1=np.zeros(n2)
    lb=tlab[ys,xs];np.minimum.at(bx0,lb,xs);np.maximum.at(bx1,lb,xs);np.minimum.at(by0,lb,ys);np.maximum.at(by1,lb,ys)
    diag=np.hypot(bx1-bx0,by1-by0)+1
    straight=(cnt/diag)<=1.9
    fence=((~isw)&straight&(wid<=1.9))[tlab].astype(np.uint8)&tl
    wthin=isw[tlab].astype(np.uint8)&tl
    # grey bead walls / rails (3-6px wide grey lines)
    gl=u(neu&(s>=3*80)&(s<=3*165)); gl&=1-cv2.dilate(bld,E(3)); gl&=1-roadz; gl&=1-whz
    gl=gl&(1-cv2.morphologyEx(gl,cv2.MORPH_OPEN,E(8)))
    gl=cv2.morphologyEx(gl,cv2.MORPH_CLOSE,E(5))
    wallg,lab,keep=comps(gl,24,7)
    # olive linear walls (sandbags / hesco)
    ol=olive&(1-cv2.dilate(oliveF,E(9)))&(1-cv2.dilate(brown,E(7))); ol=cv2.morphologyEx(ol|cv2.morphologyEx(olive,cv2.MORPH_CLOSE,E(3))&(1-cv2.dilate(oliveF,E(3))),cv2.MORPH_CLOSE,E(3))
    wallo,_,_=comps(ol,20,9)
    walls=wallg|wallo|wthin
    n,lab,st,_=cv2.connectedComponentsWithStats(walls,8)
    sk=skeletonize(walls>0);LL=np.bincount(lab[sk],minlength=n)
    big=np.maximum(st[:,2],st[:,3]);k=(big>=36)&~((big<80)&(LL>1.6*big));k[0]=False;walls=k[lab].astype(np.uint8)
    # drop small closed rectangles (vehicles/containers outlines)
    walls&=1-cv2.dilate(fence,E(1)) if False else walls
    return dict(bld=bld,walls=walls,fence=fence)
if __name__=='__main__':
    import sys
    from PIL import Image
    a=np.array(Image.open(sys.argv[1]).convert('RGB')).astype(np.int16)
    d=detect(a);im=(a.astype(np.uint8)*0.55+100).astype(np.uint8)
    im[d['bld']>0]=(120,120,255);im[d['walls']>0]=(255,0,0);im[d['fence']>0]=(0,170,0)
    Image.fromarray(im).save(sys.argv[2])
