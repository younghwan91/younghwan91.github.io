import numpy as np, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],'mathtext.fontset':'stix',
 'font.size':11,'axes.linewidth':.8,'axes.edgecolor':'#222','xtick.direction':'in','ytick.direction':'in',
 'xtick.top':True,'ytick.right':True,'legend.frameon':False,'savefig.dpi':220})
BLUE='#1f4e9c'; ORANGE='#d9822b'; GREY='#9a9a9a'; RED='#c0392b'; GREEN='#2e7d32'
rng=np.random.default_rng(3)
def plabel(ax,t): ax.text(-.11,1.02,t,transform=ax.transAxes,fontsize=12,fontweight='bold',va='bottom')

# (1) GOALS ------------------------------------------------------------
fig,(a1,a2)=plt.subplots(2,1,figsize=(6.4,4.6),sharex=True,gridspec_kw=dict(hspace=.12))
al=np.linspace(0,4,800); astar=1.55; A0=.035; full=.11+A0*(al-astar)**2
seg=np.array_split(np.arange(al.size),26)
def batch(): return A0*(1+rng.normal(0,.12)), astar+rng.normal(0,.12), rng.normal(0,.012)
f=np.empty_like(al)
for g in seg:
    ab,cb,ob=batch(); f[g]=.11+ob+ab*(al[g]-cb)**2
a1.plot(al,full,color=GREY,lw=2.2,label='full-batch loss')
a1.plot(al,f,color=BLUE,lw=1,label='dynamic mini-batch loss')
a1.set_ylabel(r'$\tilde{f}(\alpha)$'); a1.legend(loc='upper center',ncol=2,fontsize=9.5); a1.set_yticklabels([])
pts=np.linspace(.2,3.8,11); di=np.array([2*ab*(p-cb) for p,(ab,cb,_) in zip(pts,[batch() for _ in pts])])
A=np.polyfit(pts,di,1); r=-A[1]/A[0]
a2.axhline(0,color='#222',lw=.6)
a2.plot(al,np.polyval(A,al),color=ORANGE,lw=1.6,label='derivative-only surrogate')
a2.plot(pts,di,'o',ms=4.5,mfc='white',mec=BLUE,mew=1.2,label=r"sampled $\tilde{f}'(\alpha)$")
for ax in (a1,a2): ax.axvline(r,color=RED,lw=.9,ls='--')
a2.plot([r],[0],'x',color=RED,ms=8,mew=2)
a2.annotate(r'sign change: $\alpha^{*}$',xy=(r,0),xytext=(r+.35,-.075),fontsize=10,color=RED,arrowprops=dict(arrowstyle='->',color=RED,lw=.8))
a2.set_ylabel(r"$\tilde{f}'(\alpha)$"); a2.set_xlabel(r'learning rate $\alpha$'); a2.set_yticklabels([])
a2.legend(loc='upper left',fontsize=9.5)
plabel(a1,'(a)'); plabel(a2,'(b)')
fig.savefig('a_goals.png',bbox_inches='tight',pad_inches=.08,facecolor='white'); plt.close(fig)
print('GOALS root',round(r,3),'vs',astar,'slope',round(A[0],4),'vs',2*A0)

# (2) sub-dimensional surrogate on Ackley --------------------------------
def ackley(x,y): return -20*np.exp(-.2*np.sqrt(.5*(x**2+y**2)))-np.exp(.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+np.e+20
x=np.linspace(-4,4,500); X,Y=np.meshgrid(x,x); Z=ackley(X,Y)
fig,(m,sl)=plt.subplots(1,2,figsize=(7.6,3.3),gridspec_kw=dict(width_ratios=[1,1.3],wspace=.5))
cf=m.contourf(X,Y,Z,levels=24,cmap='viridis'); m.contour(X,Y,Z,levels=12,colors='k',linewidths=.25,alpha=.5)
p0=np.array([3.1,-2.4]); d=-p0/np.linalg.norm(p0); t=np.linspace(0,7.6,400); L=p0+t[:,None]*d; zl=ackley(L[:,0],L[:,1])
ts=np.linspace(.3,7.2,8); Ls=p0+ts[:,None]*d; zs=ackley(Ls[:,0],Ls[:,1]); c=np.polyfit(ts,zs,2); tm=-c[1]/(2*c[0]); pm=p0+tm*d
m.plot(L[:,0],L[:,1],color='white',lw=1.4); m.plot(Ls[:,0],Ls[:,1],'o',ms=4.5,mfc=ORANGE,mec='k',mew=.6)
m.plot(pm[0],pm[1],'x',color=RED,ms=8,mew=2)
m.set_xlabel('$x_1$'); m.set_ylabel('$x_2$'); m.set_aspect('equal')
cb=fig.colorbar(cf,ax=m,fraction=.046,pad=.03); cb.set_label('$f(x)$'); cb.outline.set_linewidth(.6)
sl.plot(t,zl,color=BLUE,lw=1,label='Ackley along subspace')
sl.plot(t,np.polyval(c,t),color=ORANGE,lw=1.6,label='1-D quadratic surrogate')
sl.plot(ts,zs,'o',ms=4.5,mfc='white',mec='k',mew=1,label='samples')
sl.plot(tm,np.polyval(c,tm),'x',color=RED,ms=8,mew=2,label='surrogate minimiser')
sl.set_xlabel('position along subspace, $t$'); sl.set_ylabel(r'$f(\mathbf{x}_0+t\,\mathbf{d})$'); sl.legend(fontsize=8.8,loc='upper center')
sl.set_ylim(-.5,zl.max()+4.5)
plabel(m,'(a)'); plabel(sl,'(b)')
fig.savefig('a_subdim.png',bbox_inches='tight',pad_inches=.08,facecolor='white'); plt.close(fig)
print('Ackley(0,0)',round(float(ackley(0,0)),12),'c0>0',c[0]>0,'t*',round(tm,3),'origin at',round(float(np.linalg.norm(p0)),3))

# (3) sensor placement on a cantilever with springs ----------------------
Lb,H,nx,ny=10,1.6,40,8
def beam_moment(Lb,ne=200,EI=1.,P=1.,k1=.02,k2=.01):
    le=Lb/ne; nd=ne+1; K=np.zeros((2*nd,2*nd)); F=np.zeros(2*nd)
    ke=EI/le**3*np.array([[12,6*le,-12,6*le],[6*le,4*le**2,-6*le,2*le**2],[-12,-6*le,12,-6*le],[6*le,2*le**2,-6*le,4*le**2]])
    for e in range(ne): i=[2*e,2*e+1,2*e+2,2*e+3]; K[np.ix_(i,i)]+=ke
    xn=np.linspace(0,Lb,nd); i1=int(round(.6*ne)); K[2*i1,2*i1]+=k1; K[2*ne,2*ne]+=k2; F[2*ne]=-P
    fr=np.arange(2,2*nd); u=np.zeros(2*nd); u[fr]=np.linalg.solve(K[np.ix_(fr,fr)],F[fr])
    xm=(xn[:-1]+xn[1:])/2; M=np.array([EI*(u[2*e+3]-u[2*e+1])/le for e in range(ne)])
    return xm,M,xn[i1],k1*(-u[2*i1]),k2*(-u[2*ne])
xm,M,x1,R1,R2=beam_moment(Lb)
Mst=-(Lb-xm)+R1*np.clip(x1-xm,0,None)+R2*(Lb-xm); print('beam FE vs statics',f'{np.max(np.abs(M-Mst)):.1e}')
xs=np.linspace(0,Lb,nx+1); ys=np.linspace(-H/2,H/2,ny+1); XX,YY=np.meshgrid((xs[:-1]+xs[1:])/2,(ys[:-1]+ys[1:])/2)
S=np.abs(np.interp(XX[0],xm,M)[None,:]*YY); S/=S.max()
fig,ax=plt.subplots(figsize=(7.4,2.6))
pc=ax.pcolormesh(xs,ys,S,cmap='Blues',shading='flat',edgecolors='#8a8a8a',linewidth=.25)
NX,NY=np.meshgrid(xs,ys); ax.plot(NX.ravel(),NY.ravel(),'.',ms=1.6,color='#555')
sel=[(2,8),(2,0),(9,8),(16,0),(24,8),(31,0)]
ax.plot([xs[i] for i,j in sel],[ys[j] for i,j in sel],'o',ms=8,mfc='none',mec=RED,mew=1.6,label='selected sensors')
for yy in np.linspace(-H/2,H/2,7): ax.plot([-.3,0],[yy-.2,yy],color='#222',lw=.7)
ax.plot([0,0],[-H/2-.1,H/2+.1],color='#222',lw=1.4)
def spring(x0,y0,y1,w=.15,k=6):
    yy=np.linspace(y0,y1,2*k+1); xx=x0+np.r_[0,np.tile([w,-w],k)[:2*k-1],0]; ax.plot(xx,yy,color='#222',lw=.9)
    ax.plot([x0-.25,x0+.25],[y1,y1],color='#222',lw=1.2)
spring(x1,-H/2,-H/2-.9); spring(Lb,-H/2,-H/2-.9)
ax.text(x1+.25,-H/2-.5,'$k_1$',fontsize=10); ax.text(Lb+.25,-H/2-.5,'$k_2$',fontsize=10)
ax.annotate('',xy=(Lb,H/2+.02),xytext=(Lb,H/2+.75),arrowprops=dict(arrowstyle='-|>',color='#222',lw=1)); ax.text(Lb+.15,H/2+.45,'$P$',fontsize=10)
cb=fig.colorbar(pc,ax=ax,fraction=.025,pad=.02); cb.set_label(r'$|\sigma_{xx}|$ (normalised)'); cb.outline.set_linewidth(.6)
ax.legend(loc='upper center',bbox_to_anchor=(.45,1.2),fontsize=9.5)
ax.set_xlim(-.5,Lb+.7); ax.set_ylim(-H/2-1.05,H/2+.85); ax.set_aspect('equal'); ax.axis('off')
fig.savefig('a_sensors.png',bbox_inches='tight',pad_inches=.06,facecolor='white'); plt.close(fig)
print('mesh nodes',(nx+1)*(ny+1),'C(n,6)',f'{math.comb((nx+1)*(ny+1),6):.1e}')
