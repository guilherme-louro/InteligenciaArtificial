# Trabalho Final de FIA: Raciocınio Espacial Neuro-Simbólico com LTNtorch

## Integrantes do Projeto
- André Yudji Silva Okimoto
- Carolina Falabelo Maycá
- Fernando Lucas Almeida Nascimento
- Guilherme Dias Correa
- Guilherme Louro de Salignac Souza
- Luiza da Costa Caxeixa
- Nicolas Mady Correa Gomes
- Sofia de Castro Sato

## Descrição sobre o repositório

- `predicates.py`: Define os predicados na especificação. Ele também tem a classe de MLP que cria as redes neurais. Eles possuem uma camada oculta de 32 neurônios, uma camada oculta de 16 neurônios e uma camada de saída com 1 neurônio.
- `axioms.py`: Define os axiomas na especificação.
- `metrics.py`: Implementa as métricas exigidas (acurácia, precisão, recall e f1) no trabalho).
- `data_generation.py`: Código que implementa a função de criar um dataset aleatório. Ele também possui algumas perguntas ground truth de predicados e possui funções de preparação para avaliação dos predicados.
- `queries.py`: Define duas consultas da parte "Tarefa 4: Raciocínio Composto" da especificação, junto de suas funções ground truth.
- `trainer.py`: Implementa a função que faz o treinamento do LTN.
- `experiments.ipynb`: Faz a execução do trabalho. Treina o modelo com o dataset feito a mão em sala e testa com 5 datasets aleatórios.

## Algumas definições importantes: 

### Inteligência Artificial Neuro-Simbólica (NeSy)

A Inteligência Artificial Neuro-Simbólica busca integrar duas vertentes tradicionais da IA:
* **Abordagens neurais**, responsáveis por aprender padrões a partir de dados, geralmente de forma contínua e diferenciável;
* **Abordagens simbólicas**, baseadas em lógica e regras explícitas, capazes de representar conhecimento estruturado e realizar inferências.

Essa integração permite superar limitações de cada abordagem isoladamente, combinando a capacidade de aprendizado das redes neurais com a precisão e interpretabilidade do raciocínio lógico.

### 3. Logic Tensor Networks (LTN)

As **Logic Tensor Networks** estendem a lógica de primeira ordem para o domínio contínuo, permitindo que predicados lógicos sejam representados como funções diferenciáveis. Dessa forma, fórmulas lógicas deixam de ser avaliadas apenas como verdadeiras ou falsas e passam a assumir valores de verdade contínuos no intervalo ([0,1]).

Em LTN:

* **Predicados** são implementados como redes neurais;
* **Constantes** representam objetos do domínio;
* **Fórmulas lógicas** são traduzidas em expressões diferenciáveis;
* O grau de satisfação de uma fórmula indica o quanto ela é respeitada pelo modelo.

O treinamento consiste em maximizar a **satisfatibilidade global** das fórmulas definidas na base de conhecimento.

## Passo a passo de como rodar

Nós tivemos muito problema com as dependências de biblioteca que rodem para um ou outro da equipe (você deve saber que python é uma linguagem terrível quando se fala em dependências de biblioteca). Por isso, deixamos duas formas de baixar as dependências. A primeira usando  `conda` que recomendamos mais por usar uma versão python `3.10`, e a segunda usando o `venv` que pode ser mais difícil de rodar por depender da versão do python instalada na máquina.

### Usando `conda`

1. Se já tiver o conda instalado, basta criar o ambiente:
```bash
conda env create -f ./environments.yml
```

2. Use o interpretador python do ambiente conda que criamos chamado de `ltn_env` no notebook `experiments.ipynb`


## Usando `venv`

1. Crie um ambiente virtual python com as versões da biblioteca que utilizamos. Pode renomear a venv com outro nome:
```bash
python -m venv myenv
```

2. Ative o ambiente virtual:
```bash
source myenv/bin/activate
```

3. Se apareceu algo como `(myenv)` no seu terminal, está ativo. Agora instale as bibliotecas para o ambiente virtual:
```bash
pip install -r requirements.txt
```

4. Após isso, você pode executar nosso trabalho no notebook `experiments.ipynb` usando o interpretador python do `myenv`
