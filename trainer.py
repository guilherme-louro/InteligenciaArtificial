import torch
import ltn
from predicates import *

# ============================================================
# TREINAMENTO DE UMA BASE DE CONHECIMENTO LTN
# ============================================================

sat_agg = ltn.fuzzy_ops.SatAgg()

def train_ltn(axioms_func, objects, epochs=500, lr=0.01, verbose=False):
    """
    Treina uma lista de axiomas LTN maximizando a satisfatibilidade global.
    """
    
    # Coletar parâmetros de todos os predicados
    predicates = [isCylinder, isCone, isTriangle, isCircle, isSquare,
                  isSmall, isBig, leftOf, rightOf, closeTo, above, below,
                  isRed, isGreen, isBlue]
    all_params = []
    for predicate in predicates:
        all_params.extend(list(predicate.parameters()))
    
    # Otimizador
    optimizer = torch.optim.Adam(all_params, lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Recriar axiomas a cada época para evitar problemas de grafos
        axioms = axioms_func(objects)
        sat = sat_agg(*axioms)

        # queremos maximizar sat -> minimizar (1 - sat)
        loss = 1.0 - sat
        loss.backward()
        optimizer.step()

        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch:04d} | satAgg = {sat.item():.4f}")

    return sat.item()