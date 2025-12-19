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
# 2. PREDICADOS
# ============================================================

# Predicados unários: entrada = um objeto concatenado (11)
# Predicados binarios: entrada = dois objetos concatenados (22)

# Tarefa 1: Taxonomia e Formas
isCylinder = ltn.Predicate(MLP(input_dim=11))
isCone = ltn.Predicate(MLP(input_dim=11))
isTriangle = ltn.Predicate(MLP(input_dim=11))
isCircle = ltn.Predicate(MLP(input_dim=11))
isSquare = ltn.Predicate(MLP(input_dim=11))
isSmall = ltn.Predicate(MLP(input_dim=11))
isBig = ltn.Predicate(MLP(input_dim=11))

# Tarefa 2: Raciocínio Espacial
leftOf = ltn.Predicate(MLP(input_dim=22))
rightOf = ltn.Predicate(MLP(input_dim=22))
closeTo = ltn.Predicate(MLP(input_dim=22))

# Tarefa 3: Raciocínio Vertical
above = ltn.Predicate(MLP(input_dim=22))
below = ltn.Predicate(MLP(input_dim=22))

# Tarefa 4: Raciocínio Composto

isRed = ltn.Predicate(MLP(input_dim=11))
isGreen = ltn.Predicate(MLP(input_dim=11))
isBlue = ltn.Predicate(MLP(input_dim=11))