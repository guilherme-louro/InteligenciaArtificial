# Entrega 4 - Resultados dos Experimentos LTN

## 1. Neuro-Symbolic AI e Logic Tensor Networks

### Neuro-Symbolic AI (NeSy)
A Inteligência Artificial Neuro-Simbólica combina o poder do aprendizado profundo com raciocínio lógico estruturado, oferecendo:
- **Interpretabilidade**: Decisões baseadas em regras lógicas compreensíveis
- **Robustez**: Incorporação de conhecimento de domínio através de axiomas
- **Generalização**: Capacidade de raciocinar sobre conceitos não vistos durante treinamento

### Logic Tensor Networks (LTN)
LTN é um framework que integra redes neurais com lógica de primeira ordem através de:
- **Constantes**: Tensores representando objetos do domínio
- **Predicados**: Redes neurais que retornam valores de verdade fuzzy [0,1]
- **Axiomas**: Fórmulas lógicas implementadas como funções de perda diferenciáveis
- **Satisfação**: Otimização para maximizar a satisfação dos axiomas lógicos

## 2. Dataset CLEVR Simplificado

### Descrição do Dataset
- **Objetos**: 25 objetos por geração com propriedades aleatórias
- **Propriedades**: Coordenadas (x,y), cor, forma, tamanho
- **Cores**: red, blue, green, yellow, gray, purple, cyan, brown
- **Formas**: circle, square, triangle
- **Tamanhos**: small, large

### Exemplo de Objetos Gerados
```
Object 1: red square large at (1, 3)
Object 2: blue circle small at (5, 2) 
Object 3: green triangle large at (2, 7)
... (22 objetos adicionais por execução)
```

### Ground Truth Computado
- **Predicados Unários**: `isCircle`, `isSquare`, `isTriangle`, `isRed`, `isBlue`, etc.
- **Predicados Binários**: `leftOf`, `rightOf`, `frontOf`, `behind`, `above`, `below`
- **Relacionamentos Espaciais**: Baseados em coordenadas (x,y) dos objetos

## 3. Valores de Satisfação das Fórmulas Específicas

### Categorias de Axiomas Avaliados

#### 3.1 Taxonomia e Formas (sat_tax)
```python
# Axiomas implementados:
# ∀x: isCircle(x) → Shape(x, circle)
# ∀x: isSquare(x) → Shape(x, square)  
# ∀x: isTriangle(x) → Shape(x, triangle)
```
**Satisfação média**: ~96-99% (excelente performance)

#### 3.2 Raciocínio Espacial (sat_spatial)  
```python
# Axiomas implementados:
# ∀x,y: leftOf(x,y) → ¬rightOf(x,y)
# ∀x,y: above(x,y) → ¬below(x,y)
# ∀x,y: frontOf(x,y) → ¬behind(x,y)
```
**Satisfação média**: ~62-65% (desafio significativo)

#### 3.3 Raciocínio Vertical (sat_vertical)
```python
# Axiomas implementados:
# ∀x,y: above(x,y) ∧ isLarge(x) → ¬below(y,x)
# Relacionamentos verticais com restrições de tamanho
```
**Satisfação média**: ~90-93% (boa performance)

## 4. Resultados das 5 Execuções com Datasets Aleatórios

### Configuração Experimental
- **Seeds utilizados**: 1, 21, 41, 61, 81
- **Objetos por dataset**: 25
- **Épocas de treinamento**: 1000
- **Taxa de aprendizado**: 0.01

### Resultados Detalhados

#### Execução 1 (seed = 1)
```
Satisfação Final: 0.7179
Categorias:
  - Taxonomia e Formas: sat = 0.961
  - Raciocínio espacial: sat = 0.653  
  - Raciocínio vertical: sat = 0.909

Queries Complexas:
  - q1 (Filtragem Composta): 0.003/1.0
  - q2 (Dedução de Posição Absoluta): 0.017/1.0

Métricas Predicados Unários:
  - Accuracy: 1.000
  - Precision: 0.999
  - Recall: 0.999
  - F1 Score: 0.999

Métricas Predicados Binários:
  - Accuracy: 0.932
  - Precision: 0.927
  - Recall: 0.937
  - F1 Score: 0.932
```

#### Execução 2 (seed = 21)
```
Satisfação Final: 0.7421
Categorias:
  - Taxonomia e Formas: sat = 0.979
  - Raciocínio espacial: sat = 0.625
  - Raciocínio vertical: sat = 0.924

Queries Complexas:
  - q1 (Filtragem Composta): 0.002/1.0
  - q2 (Dedução de Posição Absoluta): 0.009/1.0

Métricas Predicados Unários:
  - Accuracy: 1.000
  - Precision: 0.999
  - Recall: 0.999
  - F1 Score: 0.999

Métricas Predicados Binários:
  - Accuracy: 0.960
  - Precision: 0.976
  - Recall: 0.943
  - F1 Score: 0.959
```

#### Execução 3 (seed = 41)
```
Satisfação Final: 0.7444
Categorias:
  - Taxonomia e Formas: sat = 0.999
  - Raciocínio espacial: sat = 0.652
  - Raciocínio vertical: sat = 0.927

Queries Complexas:
  - q1 (Filtragem Composta): 0.000/0.0
  - q2 (Dedução de Posição Absoluta): 0.014/1.0

Métricas Predicados Unários:
  - Accuracy: 1.000
  - Precision: 0.999
  - Recall: 0.999
  - F1 Score: 0.999

Métricas Predicados Binários:
  - Accuracy: 0.960
  - Precision: 0.969
  - Recall: 0.950
  - F1 Score: 0.960
```

#### Execução 4 (seed = 61)
```
Satisfação Final: 0.7597
Categorias:
  - Taxonomia e Formas: sat = 0.969
  - Raciocínio espacial: sat = 0.652
  - Raciocínio vertical: sat = 0.931

Queries Complexas:
  - q1 (Filtragem Composta): 0.002/1.0
  - q2 (Dedução de Posição Absoluta): 0.012/1.0

Métricas Predicados Unários:
  - Accuracy: 1.000
  - Precision: 0.999
  - Recall: 0.999
  - F1 Score: 0.999

Métricas Predicados Binários:
  - Accuracy: 0.958
  - Precision: 0.963
  - Recall: 0.953
  - F1 Score: 0.958
```

#### Execução 5 (seed = 81)
```
Satisfação Final: 0.7501
Categorias:
  - Taxonomia e Formas: sat = 0.958
  - Raciocínio espacial: sat = 0.647
  - Raciocínio vertical: sat = 0.907

Queries Complexas:
  - q1 (Filtragem Composta): 0.005/1.0
  - q2 (Dedução de Posição Absoluta): 0.000/0.0

Métricas Predicados Unários:
  - Accuracy: 1.000
  - Precision: 0.999
  - Recall: 0.999
  - F1 Score: 0.999

Métricas Predicados Binários:
  - Accuracy: 0.933
  - Precision: 0.939
  - Recall: 0.927
  - F1 Score: 0.933
```

## Análise dos Resultados

### Métricas Agregadas (5 execuções)
| Métrica | Predicados Unários | Predicados Binários |
|---------|-------------------|---------------------|
| **Accuracy** | 1.000 ± 0.000 | 0.949 ± 0.012 |
| **Precision** | 0.999 ± 0.000 | 0.951 ± 0.018 |
| **Recall** | 0.999 ± 0.000 | 0.942 ± 0.010 |
| **F1 Score** | 0.999 ± 0.000 | 0.948 ± 0.012 |

### Satisfação por Categoria
| Categoria | Satisfação Média | Desvio Padrão |
|-----------|------------------|---------------|
| **Taxonomia e Formas** | 0.973 | ±0.014 |
| **Raciocínio Espacial** | 0.646 | ±0.011 |
| **Raciocínio Vertical** | 0.920 | ±0.009 |

### Performance das Queries Complexas
| Query | Tipo | Performance Média | Observação |
|-------|------|------------------|------------|
| **q1** | Filtragem Composta | 0.002/1.0 | Muito baixa confiança |
| **q2** | Dedução Posicional | 0.010/1.0 | Baixa confiança |

## Conclusões

1. **Performance Excelente em Predicados Unários**: O modelo alcançou praticamente 100% de accuracy (1.000) em predicados unários, demonstrando excelente capacidade de aprender propriedades básicas (cor, forma, tamanho).

2. **Boa Performance em Predicados Binários**: Com accuracy média de ~95% nos predicados binários, o modelo mostra capacidade sólida para relacionamentos espaciais básicos (leftOf, rightOf, above, below).

3. **Desafio nas Queries Complexas**: As queries q1 e q2 apresentaram valores muito baixos (0.002 e 0.010), indicando dificuldade significativa em raciocínio composto e filtragen múltipla.

4. **Taxonomia e Formas Dominada**: Satisfação média de 97% nos axiomas de taxonomia confirma que o modelo aprendeu perfeitamente os relacionamentos entre objetos e suas propriedades.

5. **Raciocínio Espacial Moderado**: Satisfação de ~65% no raciocínio espacial, embora menor que taxonomia, ainda representa aprendizado significativo dos relacionamentos posicionais.

6. **Consistência Entre Execuções**: Baixos desvios padrão demonstram robustez e reprodutibilidade do modelo através de diferentes seeds aleatórios.

7. **Oportunidade de Melhoria**: O principal foco para melhorias futuras deveria ser o raciocínio em queries complexas, possivelmente através de arquiteturas hierárquicas ou decomposição de problemas.