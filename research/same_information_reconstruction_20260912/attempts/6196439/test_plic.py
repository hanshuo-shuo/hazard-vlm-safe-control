import unittest
import numpy as np
from plic import normals,intercept,fractions,square_cdf,integrate,block_mean

def polygon_area(nx,ny,alpha):
    """Independent Sutherland-Hodgman clipping, not the analytic CDF."""
    poly=[(0.,0.),(1.,0.),(1.,1.),(0.,1.)];out=[]
    for a,b in zip(poly,poly[1:]+poly[:1]):
        da=nx*a[0]+ny*a[1]-alpha;db=nx*b[0]+ny*b[1]-alpha
        if da<=0:out.append(a)
        if (da<=0)!=(db<=0):
            t=da/(da-db);out.append((a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])))
    if len(out)<3:return 0.
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(out,out[1:]+out[:1])))/2

class PLICTests(unittest.TestCase):
    def test_mass_matches_independent_polygon(self):
        for theta in [.0,.2,1.2,2.8,4.4,5.6]:
            nx,ny=np.cos(theta),np.sin(theta);s=abs(nx)+abs(ny);nx/=s;ny/=s
            for p in [1e-8,.01,.2,.5,.83,1-1e-8]:
                a=float(intercept(p,nx,ny))
                self.assertAlmostEqual(polygon_area(nx,ny,a),p,places=12)
                self.assertAlmostEqual(float(fractions([p],[nx],[ny]).mean()),p,places=12)

    def test_constant_lift_identity_and_area(self):
        rng=np.random.default_rng(2612091402)
        p=rng.random((2,16,16));f=(rng.random((4,512,512))>.5).astype(float)
        u,r,a,m=integrate(p,f,True)
        lifted=np.repeat(np.repeat(p,32,-1),32,-2)
        direct=np.einsum('cij,kij->ck',lifted,f)/f.sum((1,2))
        np.testing.assert_allclose(u,direct,atol=1e-13,rtol=0)
        np.testing.assert_allclose(block_mean(m),p,atol=1e-12,rtol=0)

    def test_uniform_fallback_no_invented_orientation(self):
        p=np.full((1,16,16),.37);f=np.zeros((4,512,512));f[:,:,:256]=1
        u,r,a=integrate(p,f)
        np.testing.assert_allclose(u,r,atol=1e-14,rtol=0)
        self.assertEqual(a['gradient_fallback_cells'],256)

    def test_complement_and_transposition(self):
        p=np.tile(np.linspace(0,1,16),(16,1))[None]
        f=np.zeros((4,512,512));f[:,100:430,57:310]=1
        _,r,_=integrate(p,f);_,c,_=integrate(1-p,f)
        _,t,_=integrate(p.swapaxes(-2,-1),f.swapaxes(-2,-1))
        np.testing.assert_allclose(r+c,1,atol=1e-12,rtol=0)
        np.testing.assert_allclose(r,t,atol=1e-12,rtol=0)

    def test_orientation_from_coarse_neighbors(self):
        p=np.tile(np.linspace(0,1,16),(16,1))[None]
        nx,ny,valid=normals(p)
        self.assertTrue(valid.all());np.testing.assert_allclose(nx,-1);np.testing.assert_allclose(ny,0)

    def test_full_or_empty_footprint_cells_ignore_subcell_orientation(self):
        rng=np.random.default_rng(31);p=rng.random((2,16,16))
        f=np.zeros((4,512,512));f[:,:,:256]=1
        u,r,_=integrate(p,f)
        np.testing.assert_allclose(u,r,atol=1e-13,rtol=0)

if __name__=='__main__':unittest.main()
