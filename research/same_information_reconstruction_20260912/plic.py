"""Mass-preserving Parker-Youngs PLIC from coarse fractions only.

Algorithmic source: Pilliod & Puckett, JCP 199 (2004), discussion of the
Parker-Youngs reconstruction; author preprint section2.5. This implementation
uses the established weighted gradient with weight2, not ELVIRA or a new method.
The input interface deliberately excludes property geometry and reference labels.
"""
import numpy as np

def block_mean(x,size=16):
    x=np.asarray(x,dtype=float);h,w=x.shape[-2:]
    return x.reshape(*x.shape[:-2],size,h//size,size,w//size).mean((-3,-1))

def normals(p):
    p=np.asarray(p,float)
    z=np.pad(p,[(0,0)]*(p.ndim-2)+[(1,1),(1,1)],mode="edge")
    # Pair differences before adding, so an exactly constant transverse field
    # has exactly zero transverse gradient rather than cancellation roundoff.
    nx=(z[...,:-2,:-2]-z[...,:-2,2:])+2*(z[...,1:-1,:-2]-z[...,1:-1,2:])+(z[...,2:,:-2]-z[...,2:,2:])
    ny=(z[...,:-2,:-2]-z[...,2:,:-2])+2*(z[...,:-2,1:-1]-z[...,2:,1:-1])+(z[...,:-2,2:]-z[...,2:,2:])
    scale=abs(nx)+abs(ny);valid=scale>1e-12
    nx=np.divide(nx,scale,out=np.zeros_like(nx),where=valid)
    ny=np.divide(ny,scale,out=np.zeros_like(ny),where=valid)
    return nx,ny,valid

def square_cdf(z,a,b):
    """Area{a*x+b*y<=z} for unit square; a,b>=0, a+b=1.

    Symmetric piecewise expression avoids cancellation in a truncated-power CDF.
    """
    z,a,b=np.broadcast_arrays(z,a,b)
    zz=np.clip(z,0,1);t=np.minimum(zz,1-zz)
    small=np.minimum(a,b);big=np.maximum(a,b)
    triangle=np.divide(t*t,2*a*b,out=np.zeros_like(t),where=(a*b)>0)
    low=np.where(t<small,triangle,(t-small/2)/big)
    return np.clip(np.where(zz<=.5,low,1-low),0,1)

def intercept(p,nx,ny):
    """Exact inverse square-cut area; normal has L1 norm1."""
    a,b=abs(nx),abs(ny);small=np.minimum(a,b);big=np.maximum(a,b)
    q=np.minimum(p,1-p)
    z=np.where(q<=small/(2*big),np.sqrt(2*a*b*q),big*q+small/2)
    z=np.where(p<=.5,z,1-z)
    return z+np.minimum(nx,0)+np.minimum(ny,0)

def fractions(p,nx,ny,resolution=32):
    """Exact fractional areas of all fine sub-squares for M coarse cells."""
    p,nx,ny=map(lambda a:np.asarray(a,float).reshape(-1),[p,nx,ny])
    alpha=intercept(p,nx,ny)
    y,x=np.meshgrid(np.arange(resolution)/resolution,np.arange(resolution)/resolution,indexing="ij")
    z=(alpha[:,None]-nx[:,None]*x.ravel()-ny[:,None]*y.ravel())*resolution
    z-=np.minimum(nx,0)[:,None]+np.minimum(ny,0)[:,None]
    return square_cdf(z,abs(nx[:,None]),abs(ny[:,None])).reshape(-1,resolution,resolution)

def integrate(coarse,footprints,return_maps=False):
    """Fields (...,16,16), footprints (4,512,512); returns (...,4).

    Every arm has the full F512. Where F is constant inside a cell the integral
    is fixed by its conserved mass, so only cells with a mixed F need fine cuts.
    """
    p=np.asarray(coarse,float);f=np.asarray(footprints,float)
    if p.shape[-2:]!=(16,16) or f.shape!=(4,512,512):raise ValueError("Fixed interface dimensions")
    if np.any((p<0)|(p>1)) or not np.isfinite(p).all():raise ValueError("Invalid fractions")
    if not np.isin(f,[0,1]).all():raise ValueError("Binary reference footprint required")
    prefix=p.shape[:-2];p=p.reshape(-1,16,16)
    fl=block_mean(f);den=fl.sum((1,2))
    if np.any(den<=0):raise ValueError("Empty footprint")
    uniform=np.einsum("mij,kij->mk",p,fl)/den[None]
    nx,ny,valid=normals(p)
    mixed=(p>0)&(p<1)&valid
    mass_inverse=0.
    if mixed.any():
        alpha=intercept(p[mixed],nx[mixed],ny[mixed])
        area=square_cdf(alpha-np.minimum(nx[mixed],0)-np.minimum(ny[mixed],0),abs(nx[mixed]),abs(ny[mixed]))
        mass_inverse=float(abs(area-p[mixed]).max())
    query_mixed=((fl>0)&(fl<1)).any(0)
    active=mixed&query_mixed[None]
    if return_maps:active=mixed
    mi,yi,xi=np.where(active)
    reconstructed=uniform.copy();mass_grid=0.
    maps=np.repeat(np.repeat(p,32,axis=-2),32,axis=-1) if return_maps else None
    fblocks=f.reshape(4,16,32,16,32).transpose(1,3,0,2,4)
    # Bound working memory for noisy model fields without changing the computation.
    for start in range(0,len(mi),256):
        m,y,x=[v[start:start+256] for v in [mi,yi,xi]]
        fine=fractions(p[m,y,x],nx[m,y,x],ny[m,y,x])
        mass_grid=max(mass_grid,float(abs(fine.mean((1,2))-p[m,y,x]).max()))
        delta=np.einsum("qij,qkij->qk",fine-p[m,y,x,None,None],fblocks[y,x])/1024
        for k in range(4):np.add.at(reconstructed[:,k],m,delta[:,k]/den[k])
        if return_maps:
            for j in range(len(m)):maps[m[j],y[j]*32:(y[j]+1)*32,x[j]*32:(x[j]+1)*32]=fine[j]
    audit={"max_inverse_mass_error":mass_inverse,"max_fine_mass_error":mass_grid,
           "gradient_fallback_cells":int(((p>0)&(p<1)&~valid).sum()),"reconstructed_query_cells":len(mi)}
    if max(mass_inverse,mass_grid)>1e-10:raise AssertionError(audit)
    result=(uniform.reshape(*prefix,4),reconstructed.reshape(*prefix,4),audit)
    return (*result,maps.reshape(*prefix,512,512)) if return_maps else result
