"""Unmodified scientific functions extracted from authenticated study code."""
from functools import lru_cache
import numpy as np
import torch
SEEDS = [71, 72, 73, 74, 75]
DATA_SEED = 2026092302

def input_key(kind,chunk):
    assert type(chunk) is int
    if kind=='bank':assert 0<=chunk<20;return [DATA_SEED,10,chunk],50000
    if kind=='panel':assert chunk==0;return [DATA_SEED,20,0],4096
    if kind=='pilot':assert chunk in (0,1);return [DATA_SEED,40,chunk],64
    raise ValueError(kind)


def target_key(kind,chunk,row,seed):
    assert type(seed) is int and seed in SEEDS and type(row) is int and type(chunk) is int
    if kind=='bank':assert 0<=chunk<20 and 0<=row<50000;return [DATA_SEED,30,chunk,row,seed]
    if kind=='pilot':assert chunk==0 and 0<=row<64;return [DATA_SEED,50,0,row,seed]
    raise ValueError('No supervised targets for evaluation inputs')


def valid_probability(a,storage=False):
    assert np.isfinite(a).all() and (a>=0).all() and (a<=1).all()
    assert float(np.max(np.abs(a.astype(np.float64).sum(-1)-1)))<=(5e-7 if storage else 1e-12)


@lru_cache(maxsize=3)
def quadrature(n):
    assert n in (32,128,256)
    from pfn_dag_verify.corrected_models import BIN_EDGES
    xi,wi=np.polynomial.legendre.leggauss(n);xt,wt=np.polynomial.legendre.leggauss(4*n)
    edges=BIN_EDGES[1:-1];u=(xt+1)/2;tw=wt/2/(1-u)**2
    values=[edges[0]-u/(1-u)];weights=[tw];bins=[np.zeros(4*n,np.int64)]
    for i in range(1,99):
        left,right=edges[i-1:i+1]
        values.append((right-left)*xi/2+(right+left)/2);weights.append(wi*(right-left)/2);bins.append(np.full(n,i,np.int64))
    values.append(edges[-1]+u/(1-u));weights.append(tw);bins.append(np.full(4*n,99,np.int64))
    out=np.concatenate(values),np.concatenate(bins),np.log(np.concatenate(weights))
    assert tuple(np.bincount(out[1]))==(4*n,)+(n,)*98+(4*n,)
    return out


def query_components(api,world,x,grid=32):
    import pfn_dag_verify.corrected_oracle as oracle
    oracle._Q=quadrature(grid)
    assert x.shape==(27,3) and x.dtype==np.float32
    x64=x.astype(np.float64)
    w=api.exact_joint_posterior(world,x64[:20])['w_lo']
    out=[]
    for query in x64[20:]:
        op=api.obs_query_operator(world,query[:2],2)
        a,q,Q=api.mixture(w,op.num,op.den)
        valid_probability(a);valid_probability(q);valid_probability(Q)
        out.append((a,q,Q))
    return w,out


def coupled_row(api,world,x,kind,chunk,row):
    w,parts=query_components(api,world,x)
    rngs=[np.random.default_rng(np.random.SeedSequence(target_key(kind,chunk,row,s))) for s in SEEDS]
    chosen=np.empty((5,7,7,100),np.float32);labels=np.empty((5,7,7,16),np.uint8)
    ids=np.empty((5,7,7),np.int64);uniforms=np.empty((5,7,7),np.float64)
    expectation=0.
    for j,(a,q,Q) in enumerate(parts):
        rounded=q.astype(np.float32);rounded=rounded/rounded.sum(-1,keepdims=True)
        teacher=Q.astype(np.float32);teacher=teacher/teacher.sum()
        expectation=max(expectation,float(np.max(np.abs(a@rounded.astype(np.float64)-teacher.astype(np.float64)))))
        c=np.cumsum(a);c[-1]=1.
        for si,rng in enumerate(rngs):
            u=rng.random(7);z=np.searchsorted(c,u,side='right')
            assert (z>=0).all() and (z<len(a)).all() and (a[z]>0).all()
            cdf=np.cumsum(q[z],axis=-1);cdf[:,-1]=1.
            y=(rng.random((7,16,1))>cdf[:,None,:]).sum(-1)
            assert (y>=0).all() and (y<100).all()
            chosen[si,:,j]=q[z].astype(np.float32);labels[si,:,j]=y.astype(np.uint8)
            ids[si,:,j]=z;uniforms[si,:,j]=u
    valid_probability(chosen,storage=True)
    assert expectation<=2e-6
    return dict(Q=np.stack([p[2] for p in parts]),chosen=chosen,labels=labels,
                ids=ids,uniforms=uniforms,expectation_max_abs=expectation,context_weights=w,grid=32)

def mixture(w, num, den):
    w=np.asarray(w).reshape(-1); num=np.asarray(num).reshape(len(w),-1); den=np.asarray(den).reshape(-1)
    assert np.isfinite(num).all() and np.isfinite(den).all() and (num>=0).all() and (den>=0).all()
    assert np.allclose(num.sum(-1),den,rtol=1e-10,atol=1e-300)
    a=w*den; assert a.sum()>0; a/=a.sum()
    q=np.divide(num,den[:,None],out=np.full_like(num,1/num.shape[-1]),where=den[:,None]>0)
    q/=q.sum(-1,keepdims=True)
    Q=a@q
    assert np.allclose(Q,(w@num)/(w@den),rtol=1e-10,atol=1e-12)
    assert np.allclose(Q.sum(),1) and np.allclose(q.sum(-1),1)
    return a,q,Q

def forward(model,X,dev):
    x=torch.as_tensor(X,device=dev,dtype=torch.float32)
    return model(x[:,:20],x[:,20:,:2],torch.full((len(x),),2,device=dev,dtype=torch.long))

def loss_for(logits, target, arm, dtype=None):
    import torch
    dtype = torch.float32 if dtype is None else dtype
    logp = logits.to(dtype).log_softmax(-1).reshape(-1, 100)
    Q, q, y = target
    def ce(prob):
        t = torch.as_tensor(prob, dtype=dtype, device=logp.device).detach().reshape(-1, 100)
        return -(t * logp).sum(-1).mean()
    def label_ce():
        t = torch.as_tensor(y, device=logp.device, dtype=torch.long).detach().reshape(-1, 1)
        return -logp.gather(1, t).mean()
    if arm == 'signed':
        return ce(Q) + label_ce() - ce(q)
    if arm == 'Q':
        return ce(Q)
    if arm == 'qz':
        return ce(q)
    assert arm == 'FreshY1'
    return label_ce()

