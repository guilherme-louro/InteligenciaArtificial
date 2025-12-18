import ltn
import torch
from predicates import LeftOf, RightOf, Above, Below

# ============================================================
# 1. OPERADORES FUZZY E QUANTIFICADORES
# ============================================================

And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

# Compor IFF como (A→B) ∧ (B→A)
def Iff(a, b):
    return And(Implies(a, b), Implies(b, a))

Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier='f')
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier='e')

# ============================================================
# 2. VARIAVEIS LOGICAS
# ============================================================

# Variáveis placeholder - serão substituídas por dados reais
x = ltn.Variable("x", torch.zeros(1, 11))
y = ltn.Variable("y", torch.zeros(1, 11))
z = ltn.Variable("z", torch.zeros(1, 11))

# ============================================================
# 3. AXIOMAS DE RACIOCINIO ESPACIAL
# ============================================================

def create_axioms(objects):
    """Cria axiomas com objetos específicos"""
    x_var = ltn.Variable("x", objects)
    y_var = ltn.Variable("y", objects)
    z_var = ltn.Variable("z", objects)
    
    axioms = []
    
    # Irreflexividade: ¬LeftOf(x, x)
    axiom_irreflexive = Forall(x_var, Not(LeftOf(x_var, x_var)))
    axioms.append(("Irreflexividade", axiom_irreflexive))
    
    # Assimetria: LeftOf(x, y) => ¬LeftOf(y, x)
    axiom_asymmetry = Forall([x_var, y_var], Implies(LeftOf(x_var, y_var), Not(LeftOf(y_var, x_var))))
    axioms.append(("Assimetria", axiom_asymmetry))
    
    # Transitividade: LeftOf(x, y) ∧ LeftOf(y, z) => LeftOf(x, z)
    axiom_transitivity = Forall([x_var, y_var, z_var], 
        Implies(And(LeftOf(x_var, y_var), LeftOf(y_var, z_var)), LeftOf(x_var, z_var)))
    axioms.append(("Transitividade", axiom_transitivity))
    
    # Inverso L->R simples: LeftOf(x, y) => RightOf(y, x)
    axiom_lr_implies = Forall([x_var, y_var], Implies(LeftOf(x_var, y_var), RightOf(y_var, x_var)))
    axioms.append(("LeftOf => RightOf", axiom_lr_implies))
    
    # Inverso R->L simples: RightOf(x, y) => LeftOf(y, x)
    axiom_rl_implies = Forall([x_var, y_var], Implies(RightOf(x_var, y_var), LeftOf(y_var, x_var)))
    axioms.append(("RightOf => LeftOf", axiom_rl_implies))
    
    # Inverso vertical: Below(x, y) => Above(y, x)
    axiom_below_above = Forall([x_var, y_var], Implies(Below(x_var, y_var), Above(y_var, x_var)))
    axioms.append(("Below => Above", axiom_below_above))
    
    # Transitividade vertical: Below(x, y) ∧ Below(y, z) => Below(x, z)
    axiom_vertical_transitivity = Forall([x_var, y_var, z_var], 
        Implies(And(Below(x_var, y_var), Below(y_var, z_var)), Below(x_var, z_var)))
    axioms.append(("Transitividade Vertical", axiom_vertical_transitivity))
    
    return axioms

# ============================================================
# 4. BASE DE CONHECIMENTO
# ============================================================

def create_knowledge_base(objects):
    """Cria base de conhecimento com objetos específicos"""
    return create_axioms(objects)

# ============================================================
# 5. EXEMPLO DE AVALIACAO
# ============================================================

if __name__ == "__main__":
    print("=== Teste com dados dummy ===")
    dummy_objects = torch.rand(4, 11)  # 4 objetos com 11 features cada
    axioms = create_knowledge_base(dummy_objects)
    print(f"Número de axiomas na KB: {len(axioms)}")
    
    # Testar se os axiomas são construídos corretamente
    for name, axiom in axioms:
        try:
            valor = float(axiom.value.detach())  # Use .detach() para evitar warnings de gradient
            print(f"{name}: OK (valor = {valor:.4f})")
        except Exception as e:
            print(f"{name}: ERRO - {e}")
            
    print(f"\nTeste concluído! Base de conhecimento funcional com {len(axioms)} axiomas.")