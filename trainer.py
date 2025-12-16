import torch
import ltn

# ============================================================
# TREINAMENTO DE UMA BASE DE CONHECIMENTO LTN
# ============================================================
# Este modulo cuida apenas do treinamento:
# - maximizar a satisfatibilidade (satAgg)
# - atualizar os pesos dos predicados neurais


def train_ltn(kb, objects, epochs=500, lr=1e-3, verbose=False):
    """
    Treina uma Knowledge Base LTN maximizando a satisfatibilidade global.

    Parametros:
        kb      : ltn.KnowledgeBase com axiomas ja definidos
        objects : tensor (N x 11) com os objetos do dominio
        epochs  : numero de epocas de treinamento
        lr      : learning rate
        verbose : se True, imprime progresso

    Retorna:
        sat_final : valor final de satisfatibilidade (float)
    """

    # Variavel logica: dominio dos objetos
    x = ltn.Variable("x", objects)
    y = ltn.Variable("y", objects)
    z = ltn.Variable("z", objects)

    # Registrando as variaveis no KB
    kb.variables = {"x": x, "y": y, "z": z}

    # Otimizador: pega todos os parametros dos predicados
    optimizer = torch.optim.Adam(kb.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # satisfatibilidade global
        sat = kb.sat()

        # queremos maximizar sat -> minimizar (1 - sat)
        loss = 1.0 - sat
        loss.backward()
        optimizer.step()

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:04d} | satAgg = {sat.item():.4f}")

    return sat.item()


# ============================================================
# EXEMPLO DE USO (para teste isolado)
# ============================================================

if __name__ == "__main__":
    print("Este modulo deve ser usado a partir do notebook experiments.ipynb")