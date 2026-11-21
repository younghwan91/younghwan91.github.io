import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
BG='#0b1020'; GRID='#1c2540'; INK='#d7def0'; CY='#38e1ff'; AM='#ffb020'; PK='#ff4f9a'; GR='#5cf28a'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,'axes.labelcolor':INK,'xtick.color':'#7f8bb0','ytick.color':'#7f8bb0','axes.edgecolor':GRID})
rng=np.random.default_rng(3)

# ---------- 1. GOALS: discontinuous mini-batch losses vs derivative sign change ----------
fig,(a1,a2)=plt.subplots(2,1,figsize=(8,5),dpi=100,sharex=True,gridspec_kw=dict(height_ratios=[1.25,1],hspace=.08))
fig.patch.set_facecolor(BG)
for ax in (a1,a2):
    ax.set_facecolor(BG); ax.grid(color=GRID,lw=.8); [s.set_visible(False) for s in ax.spines.values()]
al=np.linspace(0,4,800); astar=1.55; A0=0.035
full=0.11+A0*(al-astar)**2
seg=np.array_split(np.arange(al.size),26)
def batch(): return A0*(1+rng.normal(0,.12)), astar+rng.normal(0,.12), rng.normal(0,.012)
f=np.empty_like(al); dn=np.empty_like(al)
for sgi in seg:
    ab,cb,ob=batch(); f[sgi]=0.11+ob+ab*(al[sgi]-cb)**2; dn[sgi]=2*ab*(al[sgi]-cb)
a1.plot(al,full,color='#3a4a7a',lw=6,alpha=.6,solid_capstyle='round')
a1.plot(al,f,color=PK,lw=1.6)
a1.set_ylabel('loss  f(α)')
a1.text(.05,.97,'mini-batch losses jump,\nso function values mislead',transform=a1.transAxes,ha='left',va='top',fontsize=11,color=PK)
pts=np.linspace(.2,3.8,11); di=np.array([2*ab*(p-cb) for p,(ab,cb,_) in zip(pts,[batch() for _ in pts])])
GOALS_CHECK=dict(astar=astar,pts=pts,di=di)
a2.axhline(0,color='#7f8bb0',lw=1)
a2.plot(al,dn,color=PK,lw=1,alpha=.35)
A=np.polyfit(pts,di,1); a2.plot(al,np.polyval(A,al),color=CY,lw=2.4)
a2.scatter(pts,di,s=36,color=np.where(di>0,AM,CY),edgecolor=BG,zorder=3)
r=-A[1]/A[0]; a2.scatter([r],[0],s=150,marker='X',color=GR,edgecolor=BG,zorder=4)
a1.axvline(r,color=GR,lw=1.2,ls=(0,(4,3))); a2.axvline(r,color=GR,lw=1.2,ls=(0,(4,3)))
a2.text(.97,.08,'derivative sign change → step size α*',transform=a2.transAxes,ha='right',color=GR,fontsize=11)
a2.set_ylabel("slope  f′(α)"); a2.set_xlabel('learning rate α along the search direction')
for ax in (a1,a2): ax.set_yticklabels([])
fig.savefig('v_goals.png',facecolor=BG,bbox_inches='tight',pad_inches=.25,dpi=200); plt.close(fig)

# ---------- 2. Sub-dimensional surrogate on Ackley (map + 1-D slice) ----------
def ackley(x,y): return -20*np.exp(-.2*np.sqrt(.5*(x**2+y**2)))-np.exp(.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.e+20
n=500; x=np.linspace(-4,4,n); X,Y=np.meshgrid(x,x); Z=ackley(X,Y)
fig,(m,sl)=plt.subplots(1,2,figsize=(8,3.9),dpi=100,gridspec_kw=dict(width_ratios=[1,1.25],wspace=.12)); fig.patch.set_facecolor(BG)
m.imshow(Z,extent=(-4,4,-4,4),origin='lower',cmap='magma'); m.set_axis_off()
p0=np.array([3.1,-2.4]); dvec=-p0/np.linalg.norm(p0); t=np.linspace(0,7.6,400)
L=p0[None,:]+t[:,None]*dvec; zl=ackley(L[:,0],L[:,1])
ts=np.linspace(.3,7.2,8); Ls=p0[None,:]+ts[:,None]*dvec; zs=ackley(Ls[:,0],Ls[:,1])
c=np.polyfit(ts,zs,2); tm=-c[1]/(2*c[0]); pm=p0+tm*dvec
m.plot(L[:,0],L[:,1],color=CY,lw=2.2); m.scatter(Ls[:,0],Ls[:,1],s=34,color=AM,edgecolor=BG,zorder=3)
m.scatter([pm[0]],[pm[1]],s=140,marker='X',color=GR,edgecolor=BG,zorder=4)
m.text(-3.6,3.6,'Ackley: rugged,\nmany local minima',color='#ffffff',fontsize=10.5,va='top',bbox=dict(facecolor=BG,alpha=.55,edgecolor='none',pad=3))
sl.set_facecolor(BG); sl.grid(color=GRID,lw=.8); [q.set_visible(False) for q in sl.spines.values()]
sl.plot(t,zl,color=PK,lw=1.3,alpha=.9,label='true function along the subspace')
sl.plot(t,np.polyval(c,t),color=CY,lw=2.6,label='1-D quadratic surrogate')
sl.scatter(ts,zs,s=40,color=AM,edgecolor=BG,zorder=3,label='samples (the budget)')
sl.scatter([tm],[np.polyval(c,tm)],s=150,marker='X',color=GR,edgecolor=BG,zorder=4)
sl.set_xlabel('position along the chosen subspace'); sl.set_yticklabels([]); sl.set_xticklabels([])
sl.legend(loc='upper center',frameon=False,fontsize=9.5,labelcolor=INK,ncol=1)
sl.set_ylim(min(zl.min(),np.polyval(c,t).min())-.5,zl.max()+3.2)
fig.savefig('v_subdim.png',facecolor=BG,bbox_inches='tight',pad_inches=.2,dpi=200); plt.close(fig)

# ---------- 3. Sensor placement on a cantilever FEM mesh ----------
fig,ax=plt.subplots(figsize=(8,3.6),dpi=100); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
Lb,H,nx,ny=10,1.6,40,8
xs=np.linspace(0,Lb,nx+1); ys=np.linspace(-H/2,H/2,ny+1)
XX,YY=np.meshgrid((xs[:-1]+xs[1:])/2,(ys[:-1]+ys[1:])/2)
# Euler-Bernoulli beam: clamped at x=0, tip load P down at x=L, springs k1 at 0.6L and k2 at L
def beam_moment(Lb,ne=200,EI=1.0,P=1.0,k1=0.02,k2=0.01):
    le=Lb/ne; nd=ne+1; K=np.zeros((2*nd,2*nd)); F=np.zeros(2*nd)
    ke=EI/le**3*np.array([[12,6*le,-12,6*le],[6*le,4*le**2,-6*le,2*le**2],[-12,-6*le,12,-6*le],[6*le,2*le**2,-6*le,4*le**2]])
    for e in range(ne):
        idx=[2*e,2*e+1,2*e+2,2*e+3]; K[np.ix_(idx,idx)]+=ke
    xn=np.linspace(0,Lb,nd); i1=int(round(.6*ne)); K[2*i1,2*i1]+=k1; K[2*ne,2*ne]+=k2
    F[2*ne]=-P
    free=np.arange(2,2*nd); u=np.zeros(2*nd); u[free]=np.linalg.solve(K[np.ix_(free,free)],F[free])
    # moment at element midpoints from Hermite second derivative
    xm=(xn[:-1]+xn[1:])/2; M=np.empty(ne)
    for e in range(ne):
        w1,t1,w2,t2=u[2*e:2*e+4]; M[e]=EI*(0*w1/le**2 + (-1/le)*t1 + 0*w2 + (1/le)*t2)  # w'' at midpoint = (t2-t1)/le
    R1=k1*(-u[2*i1]); R2=k2*(-u[2*ne])
    return xm,M,(xn[i1],R1),(Lb,R2),u
xm,M,(x1,R1),(x2,R2),u=beam_moment(Lb)
# statics check: M(x) = -P(L-x) + R1*max(x1-x,0) + R2*(L-x)  (sagging +)
Mst=-(Lb-xm)*1.0 + R1*np.clip(x1-xm,0,None) + R2*(Lb-xm)
BEAM_CHECK=dict(err=float(np.max(np.abs(M-Mst))),R1=R1,R2=R2)
Mc=np.interp(XX[0],xm,M)
S=np.abs(Mc[None,:]*YY)
ax.pcolormesh(xs,ys,S,cmap='viridis',shading='flat',edgecolors='#0b1020',linewidth=.4)
NX,NY=np.meshgrid(xs,ys); ax.scatter(NX,NY,s=5,color='#9aa6c8',alpha=.55,lw=0)
sel=[(2,8),(2,0),(9,8),(16,0),(24,8),(31,0)]
for i,j in sel: ax.scatter(xs[i],ys[j],s=170,facecolor='none',edgecolor=AM,lw=2.2,zorder=4); ax.scatter(xs[i],ys[j],s=26,color=AM,zorder=5)
for yy in np.linspace(-H/2-.15,H/2+.15,9): ax.plot([-.35,0],[yy-.18,yy],color='#7f8bb0',lw=1)
ax.plot([0,0],[-H/2-.2,H/2+.2],color='#7f8bb0',lw=2)
def spring(x0,y0,y1,w=.18,k=7):
    yy=np.linspace(y0,y1,2*k+1); xx=x0+np.r_[0,np.tile([w,-w],k)[:2*k-1],0]; ax.plot(xx,yy,color=PK,lw=1.8)
spring(Lb,-H/2,-H/2-1.1); spring(Lb*.6,-H/2,-H/2-1.1)
for xb in (Lb,Lb*.6): ax.plot([xb-.3,xb+.3],[-H/2-1.1]*2,color='#7f8bb0',lw=2)
ax.annotate('',xy=(Lb-.1,H/2+.05),xytext=(Lb-.1,H/2+1.0),arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.6))
ax.text(Lb*.45,H/2+.6,'pick a handful of sensors out of every node: a combinatorial search',ha='center',color=INK,fontsize=11)
ax.set_xlim(-.6,Lb+.6); ax.set_ylim(-H/2-1.35,H/2+1.1); ax.set_aspect('equal'); ax.axis('off')
fig.savefig('v_sensors.png',facecolor=BG,bbox_inches='tight',pad_inches=.15,dpi=200); plt.close(fig)

# ---------- checks ----------
import math
g=GOALS_CHECK; A=np.polyfit(g['pts'],g['di'],1); r=-A[1]/A[0]
print('GOALS root',round(r,3),'true min',g['astar'],'| slope fit',A.round(4),'expected slope 2*A0=',2*0.035)
print('Ackley(0,0)=',round(float(ackley(0,0)),12),' min on grid',round(float(Z.min()),4))
print('subdim quad c0>0',c[0]>0,' surrogate argmin t',round(tm,3),' line passes origin at t',round(float(np.linalg.norm(p0)),3))
print('beam FE vs statics max |dM|',f"{BEAM_CHECK['err']:.2e}",' R1',round(BEAM_CHECK['R1'],4),' R2',round(BEAM_CHECK['R2'],4))
nn=(nx+1)*(ny+1); print('drawn mesh nodes',nn,' C(nodes,6)=',f"{math.comb(nn,6):.2e}",' thesis C(1010,11)=',f"{math.comb(1010,11):.2e}")
