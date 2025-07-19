# Solver de Programação Inteira Mista (Branch and Cut)

## Visão Geral

**Solver\_MILP\_JP** é uma implementação acadêmica de um solver para problemas de Programação Inteira Mista (MILP), desenvolvido em Python. O projeto foi construído a partir de princípios fundamentais para explorar e demonstrar as técnicas que formam a base de solvers de otimização modernos.

O núcleo do solver é um framework de **Branch and Cut**, que combina a enumeração sistemática da árvore de Branch and Bound com a geração dinâmica de planos de corte para fortalecer a formulação do problema e acelerar a convergência. A biblioteca `gurobipy` é utilizada como um motor de baixo nível, especificamente para resolver as relaxações de programação linear em cada nó da árvore.

## Principais Funcionalidades

O solver integra um conjunto robusto de técnicas avançadas para otimização:

-   **🧠 Pré-processamento (Presolve):** Rotinas para simplificar o modelo antes da resolução, incluindo propagação de bounds, remoção de linhas redundantes, fortalecimento de coeficientes e geração de cortes de clique.
-   **🌳 Estratégias de Ramificação (Branching):** Múltiplas opções para a seleção da variável de ramificação:
    -   `most_infeasible`: Escolhe a variável com a parte fracionária mais próxima de 0.5.
    -   `strong`: Avalia candidatas simulando os ramos e medindo o impacto no limitante dual.
    -   `pseudocost`: Usa um histórico de ramificações passadas para estimar o impacto de forma mais barata.
-   **✂️ Geração de Planos de Corte:**
    -   **Cover Cuts:** Para restrições de mochila (knapsack).
    -   **Gomory Mixed-Integer Cuts:** Cortes de propósito geral derivados a partir do tableau do Simplex.
-   **💡 Heurísticas Primais:** Algoritmos para encontrar soluções inteiras de alta qualidade rapidamente:
    -   **Diving Heuristic:** "Mergulha" na árvore de busca de forma inteligente para encontrar uma solução inicial.
    -   **Feasibility Pump:** Itera para encontrar uma solução viável quando o arredondamento simples falha.
    -   **RINS (Relaxation Induced Neighborhood Search):** Usa informações da solução incumbente e da relaxação LP para refinar a melhor solução encontrada.
-   **🌲 Estratégia de Busca Híbrida:** Inicia a exploração da árvore com Busca em Profundidade (DFS) para encontrar uma incumbente rapidamente e pode alternar para Best-Bound para focar em provar a otimalidade.

## Arquitetura do Projeto

O código é organizado de forma modular para facilitar a manutenção e a extensibilidade:

-   `main.py`: Ponto de entrada da aplicação, lida com os argumentos de linha de comando.
-   `solver.py`: Contém a classe `MILPSolver`, o orquestrador principal do framework Branch and Cut.
-   `tree_elements.py`: Define as estruturas de dados `Node` e `Tree` para o gerenciamento da árvore de B&B.
-   `strategies.py`: Implementa as diferentes estratégias de ramificação.
-   `cut_generator.py`: Contém a lógica para a geração de Cover Cuts e Gomory Cuts.
-   `heuristics.py`: Implementa as heurísticas primais (Diving, RINS, Feasibility Pump).
-   `presolve.py`: Contém as rotinas de pré-processamento.
-   `simplex.py`: Uma implementação numérica do algoritmo Simplex, usada para gerar os cortes de Gomory.
-   `utils.py`: Utilitários, incluindo o `StatisticsLogger` para formatar a saída do progresso.

## Pré-requisitos

Para executar o solver, você precisará de:

-   Python (versão 3.7 ou superior)
-   Gurobi Optimizer (e uma licença válida, que pode ser a licença acadêmica gratuita)
-   Biblioteca `gurobipy` (geralmente instalada com o Gurobi)
-   NumPy

```bash
pip install numpy
```

## Como Executar

O solver é executado a partir da linha de comando através do arquivo `main.py`.

### Sintaxe Básica

```bash
python main.py [caminho_do_problema.mps] [opções...]
```

### Exemplos de Uso

1.  **Execução Simples (Estratégia Padrão):**
    Resolve o problema `mas76.mps` usando a estratégia `most_infeasible`.

    ```bash
    python main.py ./tests/mas76.mps
    ```

2.  **Usando Strong Branching e Heurísticas:**

    ```bash
    python main.py ./tests/instance_0003.mps --strategy strong --use-heuristics
    ```

3.  **Execução Completa (Todas as Técnicas Ativadas):**
    Esta é a configuração mais poderosa, ativando presolve, cortes, heurísticas e a estratégia de pseudocusto.

    ```bash
    python main.py [arquivo.mps] --strategy pseudocost --use-presolve --use-cuts --use-heuristics --rins-freq 100
    ```

## Entendendo os Parâmetros

| Parâmetro         | Descrição                                                                                             | Opções/Padrão                               |
| ----------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `problem_file`    | Caminho para o arquivo do problema no formato `.mps` ou `.lp`.                                        | (Obrigatório)                               |
| `--strategy`      | Define a estratégia de ramificação a ser usada.                                                       | `most_infeasible`, `strong`, `pseudocost`   |
| `--use-presolve`  | Ativa a fase de pré-processamento para simplificar o modelo.                                          | (Flag, desativado por padrão)               |
| `--use-cuts`      | Ativa a geração de Cover e Gomory cuts no nó raiz e durante a busca.                                  | (Flag, desativado por padrão)               |
| `--use-heuristics`| Ativa as heurísticas primais (Diving/Feasibility Pump) para encontrar uma incumbente inicial.         | (Flag, desativado por padrão)               |
| `--rins-freq`     | Frequência (em número de nós) para acionar a heurística RINS. `0` desativa.                           | Padrão: `0`                                 |
| `--dfs-limit`     | Número de nós a explorar em modo DFS antes de trocar para Best-Bound.                                 | Padrão: `9999999`                           |
| `--cut-frequency` | Frequência (em número de nós) para tentar gerar cortes locais na árvore. `0` desativa.                | Padrão: `0`                                 |
| `--cut-depth`     | Profundidade máxima na árvore para tentar gerar cortes locais.                                        | Padrão: `5`                                 |
