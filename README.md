# 🤖 Meta-Learning Based Swarm Robotics (Single-Slide Simulation)

This project demonstrates a **swarm of autonomous robots** performing an **area coverage task** in a **dynamic and uncertain environment**.  
A **meta-learning mechanism** adapts the swarm’s collective behavior based on past performance feedback.

---

## 🧠 Key Concepts
- Decentralized swarm coordination
- Local robot interactions
- Area coverage task
- Meta-learning (learning to adapt behavior parameters)
- Dynamic and uncertain environment

---

## 📌 Single-Slide Python Implementation

```python
import numpy as np, random

# Robot definition
class Robot:
    def __init__(self, size):
        self.pos = np.random.rand(2) * size
        self.vel = np.random.randn(2)

    def move(self, alpha, beta, neighbors):
        explore = alpha * np.random.randn(2)
        cohesion = beta * (np.mean([n.pos for n in neighbors], 0) - self.pos) if neighbors else 0
        self.vel = np.clip(self.vel + explore + cohesion, -1, 1)
        self.pos += self.vel

# Swarm environment
class SwarmEnv:
    def __init__(self, robots=20, size=40):
        self.size = size
        self.swarm = [Robot(size) for _ in range(robots)]
        self.coverage = np.zeros((size, size))

    def step(self, alpha, beta):
        for r in self.swarm:
            neighbors = [n for n in self.swarm if np.linalg.norm(n.pos - r.pos) < 5 and n != r]
            r.move(alpha, beta, neighbors)
            x, y = np.clip(r.pos.astype(int), 0, self.size - 1)
            self.coverage[x, y] = 1

    def performance(self):
        return np.sum(self.coverage) / (self.size ** 2)

# Meta-learning loop
alpha, beta, lr = 0.5, 0.3, 0.1
for episode in range(30):
    env = SwarmEnv()
    for _ in range(100):
        env.step(alpha, beta)
    score = env.performance()
    alpha += lr if score < 0.6 else -lr / 2
    beta  += lr if score > 0.6 else -lr / 2
    alpha, beta = np.clip(alpha, 0.1, 1), np.clip(beta, 0.1, 1)
    print(f"Episode {episode+1}: Coverage = {score:.2f}")
