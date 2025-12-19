# Trabalho Final de FIA: Raciocınio Espacial Neuro-Simbólico com LTNtorch

## Integrantes do Projeto
-André Yudji Silva Okimoto
-Carolina Falabelo Maycá
-Fernando Lucas Almeida Nascimento
-Guilherme Dias Correa
-Guilherme Louro de Salignac Souza
-Luiza da Costa Caxeixa
-Nicolas Mady Correa Gomes
-Sofia de Castro Sato

## Descrição sobre o repositório

- `predicates.py`: Define os predicados na especificação. Ele também tem a classe de MLP que cria as redes neurais. Eles possuem uma camada oculta de 32 neurônios, uma camada oculta de 16 neurônios e uma camada de saída com 1 neurônio.
- `axioms.py`: Define os axiomas na especificação.
- `metrics.py`: Implementa as métricas exigidas (acurácia, precisão, recall e f1) no trabalho).
- `data_generation.py`: Código que implementa a função de criar um dataset aleatório. Ele também possui algumas perguntas ground truth de predicados e possui funções de preparação para avaliação dos predicados.
- `queries.py`: Define duas consultas da parte "Tarefa 4: Raciocínio Composto" da especificação, junto de suas funções ground truth.
- `trainer.py`: Implementa a função que faz o treinamento do LTN.
- `experiments.ipynb`: Faz a execução do trabalho. Treina o modelo com o dataset feito a mão em sala e testa com 5 datasets aleatórios.

## Passo a passo de como rodar

1. Crie um ambiente virtual python com as versões da biblioteca que utilizamos. Isso pode ser crucial pra que o código rode de maneira correta na sua máquina. Pode renomear a venv com outro
```bash
python -m venv myvenv
```

2. Ative o ambiente virtual:
```bash
source/myenv/bin/activate
```

3. Se apareceu algo como `(myenv)` no seu terminal, está ativo. Agora instale as bibliotecas para o ambiente virtual:
```bash
pip install -r requirements.txt
```

4. Após isso, você pode executar nosso trabalho no notebook `experiments.ipynb`
