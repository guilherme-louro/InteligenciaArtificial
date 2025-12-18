import torch
import torch.nn as nn
import ltn

# ============================================================
# 1. MODELO NEURAL BASE PARA PREDICADOS
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, *xs):
        if len(xs) == 1:
            x = xs[0]
        else:
            x = torch.cat(xs, dim=-1)
        return self.net(x)


# ============================================================
# 2. DEFINICAO DOS PREDICADOS LTN
# ============================================================

# Predicados binarios: entrada = dois objetos concatenados (22)
LeftOf = ltn.Predicate(MLP(input_dim=22))
RightOf = ltn.Predicate(MLP(input_dim=22))
CloseTo = ltn.Predicate(MLP(input_dim=22))
Above = ltn.Predicate(MLP(input_dim=22))
Below = ltn.Predicate(MLP(input_dim=22))

# Predicado ternario: entrada = tres objetos concatenados (33)
InBetween = ltn.Predicate(MLP(input_dim=33))


# ============================================================
# 3. EXEMPLO DE USO COM DATASET
# ============================================================

if __name__ == "__main__":
    x = torch.rand(10, 22)
    y = torch.rand(10, 33)

    print("LeftOf:", LeftOf(x))
    print("InBetween:", InBetween(y))