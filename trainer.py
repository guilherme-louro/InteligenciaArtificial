import torch
import ltn
from predicates import LeftOf, RightOf, CloseTo, Above, Below, InBetween

# ============================================================
# TREINAMENTO DE UMA BASE DE CONHECIMENTO LTN
# ============================================================

def train_ltn(axioms_func, objects, epochs=500, lr=1e-3, verbose=False):
    """
    Treina uma lista de axiomas LTN maximizando a satisfatibilidade global.

    Parametros:
        axioms_func : função que retorna lista de tuplas (nome, axiom) 
        objects : tensor (N x 11) com os objetos do dominio
        epochs  : numero de epocas de treinamento
        lr      : learning rate
        verbose : se True, imprime progresso

    Retorna:
        sat_final : valor final de satisfatibilidade (float)
    """
    
    # Coletar parâmetros de todos os predicados
    predicates = [LeftOf, RightOf, CloseTo, Above, Below, InBetween]
    all_params = []
    for predicate in predicates:
        all_params.extend(list(predicate.parameters()))
    
    # Otimizador
    optimizer = torch.optim.Adam(all_params, lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Recriar axiomas a cada época para evitar problemas de grafos
        axioms = axioms_func(objects)
        axiom_list = [axiom for name, axiom in axioms]

        # Calcular satisfatibilidade como média dos axiomas
        sat_values = []
        for axiom in axiom_list:
            sat_values.append(axiom.value)
        
        # Satisfatibilidade global (média)
        sat = torch.stack(sat_values).mean()

        # queremos maximizar sat -> minimizar (1 - sat)
        loss = 1.0 - sat
        loss.backward()
        optimizer.step()

        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch:04d} | satAgg = {sat.item():.4f}")

    return sat.item()


# ============================================================
# EXEMPLO DE USO (para teste isolado)
# ============================================================

if __name__ == "__main__":
    print("Este modulo deve ser usado a partir do notebook experiments.ipynb")