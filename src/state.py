import numpy as np

class OUStateController:
    """
    Implements the Ornstein-Uhlenbeck (OU) state dynamics controller.
    
    Equation:
    X_{t+1} = X_t + theta * (mu - X_t) + D_t + epsilon_t
    
    where:
    - X_t: Current state vector (e.g., [Mood, Stress, Defense, ...])
    - theta: Mean reversion rate
    - mu: Long-term mean (trait anchor)
    - D_t: Bounded impulse from event e_t
    - epsilon_t: Noise term
    """
    
    def __init__(self, dim=5, theta=0.5, mu=0.5, sigma=0.05, dt=1.0):
        self.dim = dim
        self.theta = theta
        self.mu = np.full(dim, mu)
        self.sigma = sigma
        self.dt = dt
        self.state = np.copy(self.mu)

    def get_impulse(self, event_embedding):
        """
        Simulates the State Update Network (MLP) with tanh bounding.
        D_t = D_max * tanh(MLP(event))
        """
        np.random.seed(42) 
        W = np.random.randn(len(event_embedding), self.dim) * 0.1
        
        D_max = 0.2
        
        raw_impulse = np.dot(event_embedding, W)
        bounded_impulse = D_max * np.tanh(raw_impulse)
        return bounded_impulse

    def step(self, event_embedding=None):
        drift = self.theta * (self.mu - self.state) * self.dt
        
        impulse = np.zeros(self.dim)
        if event_embedding is not None:
            impulse = self.get_impulse(event_embedding)
            
        noise = np.random.normal(0, self.sigma, self.dim) * np.sqrt(self.dt)
        
        self.state = self.state + drift + impulse + noise
        
        self.state = np.clip(self.state, 0.0, 1.0)
        
        return self.state

    def get_state(self):
        return self.state

    def set_trait_anchor(self, new_mu):
        self.mu = np.array(new_mu)
