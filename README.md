import numpy as np, random

# --- Robot ---
class Robot:
    def __init__(self, n): self.p=np.random.rand(2)*n; self.v=np.random.randn(2)
    def move(self, a, b, nbrs):
        exp=a*np.random.randn(2)
        coh=b*(np.mean([r.p for r in nbrs],0)-self.p) if nbrs else 0
        self.v=np.clip(self.v+exp+coh,-1,1); self.p+=self.v

# --- Environment ---
class SwarmEnv:
    def __init__(self,N=20,S=40):
        self.S=S; self.R=[Robot(S) for _ in range(N)]
        self.C=np.zeros((S,S))
    def step(self,a,b):
        for r in self.R:
            n=[x for x in self.R if np.linalg.norm(x.p-r.p)<5 and x!=r]
            r.move(a,b,n); x,y=np.clip(r.p.astype(int),0,self.S-1)
            self.C[x,y]=1
    def score(self): return np.sum(self.C)/(self.S**2)

# --- Meta-Learner ---
a,b,lr=0.5,0.3,0.1
for ep in range(30):
    env=SwarmEnv()
    for _ in range(100): env.step(a,b)
    perf=env.score()
    a+=lr if perf<0.6 else -lr/2
    b+=lr if perf>0.6 else -lr/2
    a,b=np.clip(a,0.1,1),np.clip(b,0.1,1)
    print(f"Ep{ep+1}: Coverage={perf:.2f}")
