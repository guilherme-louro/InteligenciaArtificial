import torch
import random
from metrics import compute_metrics
import math
import ltn

# ============================================================
# 1. GERACAO DE OBJETOS (CLEVR SIMPLIFICADO)
# ============================================================

# Cada objeto eh um vetor de tamanho 11:
# [0,1]   -> posicao x, y
# [2,3,4] -> cor (vermelho, verde, azul)
# [5..9]  -> forma (circulo, quadrado, cilindro, cone, triangulo)
# [10]    -> tamanho (<0.5 pequeno, >=0.5 grande, contínuo)

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
        obj[10] = random.random()

        objects.append(obj)

    return torch.stack(objects)


# ============================================================
# 2. GROUND TRUTH GEOMETRICO (SEM LTN)
# ============================================================

# Tarefa 1
def gt_is_circle(obj):
    return obj[5] == 1.0

def gt_is_square(obj):
    return obj[6] == 1.0

def gt_is_cylinder(obj):
    return obj[7] == 1.0

def gt_is_cone(obj):
    return obj[8] == 1.0

def gt_is_triangle(obj):
    return obj[9] == 1.0

def gt_is_small(obj):
    return obj[10] <= 0.5

def gt_is_big(obj):
    return obj[10] > 0.5


# Tarefa 2
def gt_left_of(o1, o2):
    return o1[0] < o2[0]

def gt_right_of(o1, o2):
    return o1[0] > o2[0]


# Tarefa 3
def gt_below(o1, o2):
    return o1[1] < o2[1]

def gt_above(o1, o2):
    return o1[1] > o2[1]

# Usamos a distância eucliana por ser mais simples de implementar do que a gaussiana
def gt_close_to(o1, o2, threshold=0.2):
    dx = o1[0] - o2[0]
    dy = o1[1] - o2[1]
    dist = math.sqrt(dx * dx + dy * dy)
    return dist < threshold


# Tarefa 4
def gt_is_red(obj):
    return obj[2] == 1.0

def gt_is_green(obj):
    return obj[3] == 1.0

def gt_is_blue(obj):
    return obj[4] == 1.0



# ============================================================
# 3. PERGUNTAS / DATASETS PARA TREINAMENTO E AVALIACAO
# ============================================================

def build_unary_relation_dataset(objects, relation_fn):
    """
    Cria dataset para relacoes binarias
    """
    X = []
    y = []

    n = len(objects)
    for i in range(n):
        X.append(objects[i])
        y.append(float(relation_fn(objects[i])))

    return torch.stack(X), torch.tensor(y)


def build_binary_relation_dataset(objects, relation_fn):
    """
    Cria dataset para relacoes binarias 
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


# ============================================================
# 4. FUNCOES AUXILIARES PARA AVALIACAO FINAL
# ============================================================

def eval_unary_predicate(pred, gt_fn, objects):
    X, y_true = build_unary_relation_dataset(objects, gt_fn)
    
    with torch.no_grad():
        ltn_input = ltn.Variable("dummy_unary", X)       

        ltn_output = pred(ltn_input)
        
        y_pred = ltn_output.value
        
    return compute_metrics(y_true.float(), y_pred)


def eval_binary_predicate(pred, gt_fn, objects):
    X, y_true = build_binary_relation_dataset(objects, gt_fn)
    
    with torch.no_grad():
        ltn_input = ltn.Variable("dummy_unary", X)    
        
        ltn_output = pred(ltn_input)
        
        y_pred = ltn_output.value
        
    return compute_metrics(y_true.float(), y_pred)