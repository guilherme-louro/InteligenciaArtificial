import torch
import random
import math

# ============================================================
# 1. GERACAO DE OBJETOS (CLEVR SIMPLIFICADO)
# ============================================================

# Cada objeto eh um vetor de tamanho 11:
# [0,1]   -> posicao x, y
# [2,3,4] -> cor (vermelho, verde, azul)
# [5..9]  -> forma (circulo, quadrado, cilindro, cone, triangulo)
# [10]    -> tamanho (0.0 pequeno, 1.0 grande)


def generate_objects(n=25, seed=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    objects = []

    for _ in range(n):
        obj = torch.zeros(11)

        # posicao
        obj[0] = random.random()  # x
        obj[1] = random.random()  # y

        # cor
        color = random.randint(0, 2)
        obj[2 + color] = 1.0

        # forma
        shape = random.randint(0, 4)
        obj[5 + shape] = 1.0

        # tamanho
        obj[10] = float(random.randint(0, 1))

        objects.append(obj)

    return torch.stack(objects)


# ============================================================
# 2. GROUND TRUTH GEOMETRICO (SEM LTN)
# ============================================================


def left_of(o1, o2):
    return o1[0] < o2[0]


def right_of(o1, o2):
    return o1[0] > o2[0]


def below(o1, o2):
    return o1[1] < o2[1]


def above(o1, o2):
    return o1[1] > o2[1]


def close_to(o1, o2, threshold=0.2):
    dx = o1[0] - o2[0]
    dy = o1[1] - o2[1]
    dist = math.sqrt(dx * dx + dy * dy)
    return dist < threshold


def in_between(o, y, z):
    return (left_of(y, o) and right_of(z, o)) or (left_of(z, o) and right_of(y, o))


# ============================================================
# 3. PERGUNTAS / DATASETS PARA TREINAMENTO E AVALIACAO
# ============================================================


def build_binary_relation_dataset(objects, relation_fn):
    """
    Cria dataset para relacoes binarias (LeftOf, RightOf, CloseTo, etc)
    Retorna:
        X -> pares de objetos concatenados
        y -> ground truth (0 ou 1)
    """
    X = []
    y = []

    n = len(objects)
    for i in range(n):
        for j in range(n):
            if i != j:
                X.append(torch.cat([objects[i], objects[j]]))
                y.append(float(relation_fn(objects[i], objects[j])))

    return torch.stack(X), torch.tensor(y)


def build_inbetween_dataset(objects):
    """
    Dataset para relacao ternaria InBetween(x, y, z)
    Entrada: concatenacao de tres objetos
    """
    X = []
    y = []

    n = len(objects)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i != j and i != k and j != k:
                    X.append(torch.cat([objects[i], objects[j], objects[k]]))
                    y.append(float(in_between(objects[i], objects[j], objects[k])))

    return torch.stack(X), torch.tensor(y)


# ============================================================
# 4. EXEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    objects = generate_objects(n=25, seed=42)

    X_left, y_left = build_binary_relation_dataset(objects, left_of)
    X_close, y_close = build_binary_relation_dataset(objects, close_to)
    X_between, y_between = build_inbetween_dataset(objects)

    print("Objetos:", objects.shape)
    print("LeftOf dataset:", X_left.shape, y_left.shape)
    print("CloseTo dataset:", X_close.shape, y_close.shape)
    print("InBetween dataset:", X_between.shape, y_between.shape)