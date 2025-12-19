import ltn
import torch
from predicates import *
from data_generation import *

# ============================================================
# 1. OPERADORES FUZZY E QUANTIFICADORES
# ============================================================

And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier='f')
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier='e')

# ============================================================
# 2. QUERIES
# ============================================================

def q_composite_filtering(objects):
    """
    1. Filtragem Composta:
    Existe algum objeto Pequeno que esteja Abaixo de um Cilindro E à Esquerda de um Quadrado?
    Fórmula: ∃x(Small(x) ^ ∃y(Cyl(y) ^ Below(x,y)) ^ ∃z(Square(z) ^ Left(x,z)))
    """
    x = ltn.Variable("x", objects)
    y = ltn.Variable("y", objects)
    z = ltn.Variable("z", objects)

    return Exists(x, And(
        isSmall(x),
        And(
            Exists(y, And(isCylinder(y), below(x, y))),
            Exists(z, And(isSquare(z), leftOf(x, z)))
        )
    ))

def q_deduction_position(objects):
    """
    2. Dedução de Posição Absoluta:
    Existe um Cone Verde que está Entre (InBetween) dois outros objetos quaisquer?
    Fórmula: ∃x,y,z(Cone(x) ^ Green(x) ^ InBetween(x,y,z))
    """
    x = ltn.Variable("x", objects)
    y = ltn.Variable("y", objects)
    z = ltn.Variable("z", objects)

    # Lógica auxiliar de "Estar Entre" (X está entre Y e Z)
    # (Y < X < Z) OU (Z < X < Y)
    in_between_logic = Or(
        And(leftOf(y, x), leftOf(x, z)),
        And(leftOf(z, x), leftOf(x, y))
    )

    return Exists([x, y, z], And(
        isCone(x),
        And(
            isGreen(x),
            in_between_logic
        )
    ))

# ============================================================
# 3. GROUND TRUTH DAS QUERIES
# ============================================================

def gt_task4_filtragem_composta(objects):
    """
    Verifica a Query 1:
    ∃x(IsSmall(x) ^ ∃y(IsCylinder(y) ^ Below(x, y)) ^ ∃z(IsSquare(z) ^ LeftOf(x, z)))
    """
    # Procurar o candidato X
    for i, x in enumerate(objects):
        
        # O candidato deve ser Pequeno
        if not gt_is_small(x):
            continue
            
        # Se achou um pequeno, verifica se satisfaz as relações
        found_cylinder_above = False
        found_square_right = False
        
        # Sub-busca para Y (Cilindro acima de X)
        for j, y in enumerate(objects):
            if i == j: continue
            if gt_is_cylinder(y) and gt_below(x, y):
                found_cylinder_above = True
                break # Encontrou um, não precisa ver os outros
        
        # Se não tem cilindro acima, esse X já não serve. Próximo X.
        if not found_cylinder_above:
            continue
            
        # Sub-busca para Z (Quadrado à direita de X / X à esquerda de Z)
        for k, z in enumerate(objects):
            if i == k: continue
            if gt_is_square(z) and gt_left_of(x, z):
                found_square_right = True
                break
        
        # Se satisfez ambas as condições, a query é VERDADEIRA
        if found_cylinder_above and found_square_right:
            return True # 1.0

    # Se varreu todos os objetos e ninguém satisfez tudo
    return False # 0.0


def gt_task4_deducao_posicao(objects):
    """
    Verifica a Query 2:
    ∃x, y, z(IsCone(x) ^ IsGreen(x) ^ InBetween(x, y, z))
    """
    # Procurar o candidato X (O Cone Verde)
    for i, x in enumerate(objects):
        
        # Verifica predicados unários
        if not (gt_is_cone(x) and gt_is_green(x)):
            continue
            
        # Se achou o cone verde, procura um par (y, z) que o "sanduiche"
        for j, y in enumerate(objects):
            if i == j: continue
            
            for k, z in enumerate(objects):
                if k == i or k == j: continue
                
                condicao_1 = gt_left_of(y, x) and gt_right_of(z, x) # Y ... X ... Z
                condicao_2 = gt_left_of(z, x) and gt_right_of(y, x) # Z ... X ... Y
                
                if condicao_1 or condicao_2:
                    return True # Encontrou o trio perfeito

    return False