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
# 2. TAXONOMIA E FORMAS
# ============================================================

def ax_shape_unique(objects):
    """
    Garante que não seja TODAS as formas ao mesmo tempo.
    """
    x = ltn.Variable("x", objects)
    
    return Forall(x, Not(
        And(isCircle(x),
        And(isCone(x),
        And(isCylinder(x),
        And(isSquare(x), isTriangle(x))
    )))))

def ax_shape_coverage(objects):
    """Garante que o objeto seja pelo menos uma das formas conhecidas."""
    x = ltn.Variable("x", objects)
    
    return Forall(x,
        Or(isCircle(x),
        Or(isCone(x),
        Or(isCylinder(x),
        Or(isSquare(x), isTriangle(x))
    ))))

def create_axioms_taxonomia_e_formas(objects):
    return [
        ax_shape_unique(objects),
        ax_shape_coverage(objects)
    ]


# ============================================================
# 3. RACIOCÍNIO ESPACIAL
# ============================================================

def ax_spatial_irreflexive(objects):
    """Irreflexividade: ¬leftOf(x, x)"""
    x = ltn.Variable("x", objects)
    return Forall(x, Not(leftOf(x, x)))

def ax_spatial_asymmetry(objects):
    """Assimetria: leftOf(x, y) => ¬leftOf(y, x)"""
    x = ltn.Variable("x", objects)
    y = ltn.Variable("y", objects)
    return Forall([x, y], Implies(leftOf(x, y), Not(leftOf(y, x))))

def ax_spatial_transitivity(objects):
    """Transitividade: leftOf(x, y) ^ leftOf(y, z) => leftOf(x, z)"""
    x = ltn.Variable("x", objects)
    y = ltn.Variable("y", objects)
    z = ltn.Variable("z", objects)
    return Forall([x, y, z], 
        Implies(And(leftOf(x, y), leftOf(y, z)), leftOf(x, z)))

def ax_spatial_left_iff_right(objects):
    """Inverso L<->R: leftOf(x, y) <=> rightOf(y, x)"""
    x = ltn.Variable("x", objects)
    y = ltn.Variable("y", objects)
    # Implementação explícita do IFF: (A->B) ^ (B->A)
    return Forall([x, y], 
        And(
            Implies(leftOf(x, y), rightOf(y, x)), 
            Implies(rightOf(y, x), leftOf(x, y))
        )
    )

def ax_spatial_in_between(objects):
    x = ltn.Variable("x", objects)
    y = ltn.Variable("y", objects)
    z = ltn.Variable("z", objects)

    def logic_is_between(m, a, b):
        return Or(
            And(leftOf(a, m), leftOf(m, b)), # Sentido L -> R
            And(leftOf(b, m), leftOf(m, a))  # Sentido R -> L
        )

    return Forall([x, y, z],
        Or(
            Or(
                logic_is_between(x, y, z), 
                Or(
                    logic_is_between(y, x, z), # Y está no meio de X e Z
                    logic_is_between(z, x, y)  # Z está no meio de X e Y
                )
            ),
            Or(
                closeTo(x, y),
                Or(closeTo(y, z), closeTo(x, z))
            )
        )
    )


def ax_spatial_last_left(objects):
    x = ltn.Variable("x", objects)
    y = ltn.Variable("y", objects)
    
    return Exists(x, Forall(y, 
        Or(leftOf(x, y), closeTo(x, y)) 
    ))

def ax_spatial_last_right(objects):
    x = ltn.Variable("x", objects)
    y = ltn.Variable("y", objects)
    
    return Exists(x, Forall(y, 
        Or(rightOf(x, y), closeTo(x, y))
    ))


def create_axioms_raciocinio_espacial(objects):
    return [
        ax_spatial_irreflexive(objects),
        ax_spatial_asymmetry(objects),
        ax_spatial_transitivity(objects),
        ax_spatial_in_between(objects),
        ax_spatial_left_iff_right(objects),
        ax_spatial_last_left(objects),
        ax_spatial_last_right(objects)
    ]


# ============================================================
# 4. RACIOCÍNIO VERTICAL
# ============================================================

def ax_vertical_inverse(objects):
    """Inverso vertical: below(x, y) => above(y, x)"""
    x = ltn.Variable("x", objects)
    y = ltn.Variable("y", objects)
    return Forall([x, y], Implies(below(x, y), above(y, x)))

def ax_vertical_transitivity(objects):
    """Transitividade vertical: below(x, y) ^ below(y, z) => below(x, z)"""
    x = ltn.Variable("x", objects)
    y = ltn.Variable("y", objects)
    z = ltn.Variable("z", objects)
    return Forall([x, y, z], 
        Implies(And(below(x, y), below(y, z)), below(x, z)))

def create_axioms_raciocinio_vertical(objects):
    return [
        ax_vertical_inverse(objects),
        ax_vertical_transitivity(objects)
    ]

# ============================================================
# 4. Raciocínio Composto (somente a nova regra)
# ============================================================

# foi dito que isso é uma "nova regra", então incluo nos axiomas
def ax_proximity_restriction(objects):
    # Fórmula: ∀x,y ((Triangle(x) ^ Triangle(y) ^ CloseTo(x,y)) => SameSize(x,y))
    x = ltn.Variable("x", objects)
    y = ltn.Variable("y", objects)

    # Lógica de SameSize: É equivalente a dizer (isSmall(x) <-> isSmall(y))
    # Usamos o Iff composto: (Small(x) -> Small(y)) ^ (Small(y) -> Small(x))
    same_size_logic = And(
        Implies(isSmall(x), isSmall(y)),
        Implies(isSmall(y), isSmall(x))
    )

    # Antecedente: Triangulo(x) ^ Triangulo(y) ^ Perto(x,y)
    antecedent = And(
        isTriangle(x),
        And(
            isTriangle(y), 
            closeTo(x, y)
        )
    )

    return Forall([x, y], Implies(antecedent, same_size_logic))

# Função Agregadora
def create_axioms_raciocinio_composto(objects):
    return [
        ax_proximity_restriction(objects)
    ]

# ============================================================
# 5. FATOS
# ============================================================

def create_supervision_facts(objects):
    """
    Cria fatos (axiomas de supervisão) para todas as tarefas baseados no Ground Truth.
    """
    facts = []

    # Iteramos sobre os objetos para criar predicados Unários
    for i, x_data in enumerate(objects):
        
        # Transformamos o dado bruto em uma Constante LTN (não treinável)
        x = ltn.Constant(x_data, trainable=False)

        # --- TAREFA 1: FORMAS (Unários) ---
        if gt_is_circle(x_data):
            facts.append(isCircle(x))
            facts.append(Not(isSquare(x)))
            facts.append(Not(isCylinder(x)))
            facts.append(Not(isCone(x)))
            facts.append(Not(isTriangle(x)))
        
        if gt_is_square(x_data):
            facts.append(Not(isCircle(x)))
            facts.append(isSquare(x))
            facts.append(Not(isCylinder(x)))
            facts.append(Not(isCone(x)))
            facts.append(Not(isTriangle(x)))
            
        if gt_is_cylinder(x_data):
            facts.append(Not(isCircle(x)))
            facts.append(Not(isSquare(x)))
            facts.append(isCylinder(x))
            facts.append(Not(isCone(x)))
            facts.append(Not(isTriangle(x)))
            
        if gt_is_cone(x_data):
            facts.append(Not(isCircle(x)))
            facts.append(Not(isSquare(x)))
            facts.append(Not(isCylinder(x)))
            facts.append(isCone(x))
            facts.append(Not(isTriangle(x)))
            
        if gt_is_triangle(x_data):
            facts.append(Not(isCircle(x)))
            facts.append(Not(isSquare(x)))
            facts.append(Not(isCylinder(x)))
            facts.append(Not(isCone(x)))
            facts.append(isTriangle(x))

        # --- TAREFA 1: TAMANHO (Unários) ---
        if gt_is_small(x_data):
            facts.append(isSmall(x))
            facts.append(Not(isBig(x)))
            
        if gt_is_big(x_data):
            facts.append(Not(isSmall(x)))
            facts.append(isBig(x))

        # --- TAREFA 4: CORES (Unários) ---
        if gt_is_red(x_data):
            facts.append(isRed(x))
            facts.append(Not(isGreen(x)))
            facts.append(Not(isBlue(x)))
            
        if gt_is_green(x_data):
            facts.append(Not(isRed(x)))
            facts.append(isGreen(x))
            facts.append(Not(isBlue(x)))

        if gt_is_blue(x_data):
            facts.append(Not(isRed(x)))
            facts.append(Not(isGreen(x)))
            facts.append(isBlue(x))
        
        # --- RELAÇÕES BINÁRIAS (Loop Aninhado) ---
        for j, y_data in enumerate(objects):
            y = ltn.Constant(y_data, trainable=False)

            # --- TAREFA 2: RACIOCÍNIO ESPACIAL ---
            
            if gt_left_of(x_data, y_data):
                facts.append(leftOf(x, y))
            else:
                facts.append(Not(leftOf(x, y))) 
                
            if gt_right_of(x_data, y_data):
                facts.append(rightOf(x, y))
            else:
                facts.append(Not(rightOf(x, y)))
            
            if gt_close_to(x_data, y_data):
                facts.append(closeTo(x, y))
            else:
                facts.append(Not(closeTo(x, y)))

            # --- TAREFA 3: RACIOCÍNIO VERTICAL ---
            
            if gt_below(x_data, y_data):
                facts.append(below(x, y))
            else:
                facts.append(Not(below(x, y)))
                
            if gt_above(x_data, y_data):
                facts.append(above(x, y))
            else:
                facts.append(Not(above(x, y)))
                
    return facts

# ============================================================
# 6. BASE DE CONHECIMENTO
# ============================================================

def create_knowledge_base(objects):
    """Cria base de conhecimento com objetos específicos"""
    axioms = []
    axioms += create_axioms_taxonomia_e_formas(objects)
    axioms += create_axioms_raciocinio_espacial(objects)
    axioms += create_axioms_raciocinio_vertical(objects)
    axioms += create_axioms_raciocinio_composto(objects)
    axioms += create_supervision_facts(objects) 
    
    return axioms
