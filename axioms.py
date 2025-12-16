import ltn
import torch
from predicates import LeftOf, RightOf, CloseTo, InBetween, Above, Below

# ============================================================
# 1. OPERADORES FUZZY E QUANTIFICADORES
# ============================================================

And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
Iff = ltn.Connective(ltn.fuzzy_ops.IffReichenbach())

Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier='forall')
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier='exists')

# ============================================================
# 2. VARIAVEIS LOGICAS
# ============================================================

x = ltn.Variable("x", torch.empty(0))
y = ltn.Variable("y", torch.empty(0))
z = ltn.Variable("z", torch.empty(0))

# ============================================================
# 3. AXIOMAS DE RACIOCINIO ESPACIAL
# ============================================================

# Irreflexividade: ¬LeftOf(x, x)
axiom_irreflexive = Forall(x, Not(LeftOf(x, x)))

# Assimetria: LeftOf(x, y) => ¬LeftOf(y, x)
axiom_asymmetry = Forall((x, y), Implies(LeftOf(x, y), Not(LeftOf(y, x))))

# Inverso esquerda-direita: LeftOf(x, y) <=> RightOf(y, x)
axiom_inverse = Forall((x, y), Iff(LeftOf(x, y), RightOf(y, x)))

# Transitividade: LeftOf(x, y) ∧ LeftOf(y, z) => LeftOf(x, z)
axiom_transitivity = Forall((x, y, z), Implies(And(LeftOf(x, y), LeftOf(y, z)), LeftOf(x, z)))

# Definicao de InBetween
axiom_inbetween = Forall((x, y, z),
    Iff(
        InBetween(x, y, z),
        Or(
            And(LeftOf(y, x), RightOf(z, x)),
            And(LeftOf(z, x), RightOf(y, x))
        )
    )
)

# Inverso vertical: Below(x, y) <=> Above(y, x)
axiom_vertical_inverse = Forall((x, y), Iff(Below(x, y), Above(y, x)))

# Transitividade vertical: Below(x, y) ∧ Below(y, z) => Below(x, z)
axiom_vertical_transitivity = Forall((x, y, z), Implies(And(Below(x, y), Below(y, z)), Below(x, z)))

# ============================================================
# 4. BASE DE CONHECIMENTO
# ============================================================

axioms = [
    axiom_irreflexive,
    axiom_asymmetry,
    axiom_inverse,
    axiom_transitivity,
    axiom_inbetween,
    axiom_vertical_inverse,
    axiom_vertical_transitivity
]

kb = ltn.KnowledgeBase(axioms)

# ============================================================
# 5. EXEMPLO DE AVALIACAO
# ============================================================

if __name__ == "__main__":
    print("Numero de axiomas na KB:", len(kb.axioms))