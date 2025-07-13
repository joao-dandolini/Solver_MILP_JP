# tree_elements.py
# Contém as estruturas de dados fundamentais para a árvore de Branch and Bound.

from typing import Dict, Any, Optional, List, Tuple
import heapq # --- NOVA IMPORTAÇÃO ---

class Node:
    """
    Representa um único nó na árvore de busca do Branch and Bound.
    """
    _id_counter = 0

    def __init__(self, parent_id: Optional[int], depth: int, local_bounds: Dict[str, Any]):
        self.id: int = Node._id_counter
        Node._id_counter += 1
        
        self.parent_id: Optional[int] = parent_id
        self.depth: int = depth
        self.local_bounds: Dict[str, Any] = local_bounds

        self.lp_status: Optional[str] = None
        self.lp_objective_value: Optional[float] = None
        self.variable_values: Optional[Dict[str, float]] = None
        self.is_integer: bool = False

    def __repr__(self) -> str:
        return (
            f"Node(id={self.id}, "
            f"depth={self.depth}, "
            f"obj={self.lp_objective_value}, "
            f"status='{self.lp_status}')"
        )

    def set_lp_solution(self, status: str, objective_value: Optional[float], 
                        variable_values: Optional[Dict[str, float]], is_integer: bool):
        self.lp_status = status
        self.lp_objective_value = objective_value
        self.variable_values = variable_values
        self.is_integer = is_integer

# --- CLASSE TREE TOTALMENTE REESCRITA ---

class Tree:
    """
    Gerencia a coleção de nós abertos na árvore de Branch and Bound.
    Implementa uma estratégia híbrida que começa com DFS (pilha) e pode
    trocar para Best-Bound (fila de prioridade).
    """
    def __init__(self, root_node: Node):
        """
        Inicializa a árvore com o nó raiz, começando no modo DFS.
        """
        self.mode: str = 'DFS'
        
        # Estruturas de dados para os dois modos
        self._dfs_stack: List[Node] = [root_node]
        self._best_bound_queue: List[Tuple[float, int, Node]] = []

    def add_nodes(self, nodes: List[Node]):
        """
        Adiciona uma lista de novos nós à coleção apropriada, dependendo do modo.
        """
        if self.mode == 'DFS':
            self._dfs_stack.extend(nodes)
        else: # Modo BEST_BOUND
            for node in nodes:
                # Usamos uma tupla (valor_objetivo, id_do_no, no) para a fila de prioridade.
                # O id serve como desempate para o heapq.
                heapq.heappush(self._best_bound_queue, (node.lp_objective_value, node.id, node))

    def get_next_node(self) -> Optional[Node]:
        """
        Recupera e remove o próximo nó a ser explorado, de acordo com o modo atual.
        """
        if self.is_empty():
            return None

        if self.mode == 'DFS':
            return self._dfs_stack.pop()
        else: # Modo BEST_BOUND
            # --- ADICIONE ESTA LINHA DE DEPURAÇÃO ---
            #print(f"[DEBUG-BestBound] Selecionando nó com melhor Obj: {self._best_bound_queue[0][0]:.2f}")
            # --- FIM DA LINHA DE DEPURAÇÃO ---
            
            # heapq.heappop retorna a tupla com o menor valor de objetivo
            return heapq.heappop(self._best_bound_queue)[2] # Retorna o objeto Node

    def is_empty(self) -> bool:
        """
        Verifica se ainda existem nós a serem explorados em qualquer uma das estruturas.
        """
        return not self._dfs_stack and not self._best_bound_queue

    def switch_to_best_bound_mode(self):
        """
        Converte a árvore do modo DFS para o modo Best-Bound.
        Move todos os nós da pilha para a fila de prioridade.
        """
        if self.mode == 'DFS':
            print("INFO: [Tree] Trocando estratégia de busca para BEST-BOUND.")
            self.mode = 'BEST_BOUND'
            
            # Transfere todos os nós da pilha para a fila de prioridade
            for node in self._dfs_stack:
                if node.lp_objective_value is not None:
                    heapq.heappush(self._best_bound_queue, (node.lp_objective_value, node.id, node))
            
            self._dfs_stack = [] # Esvazia a pilha

    def get_current_best_bound(self) -> float:
        """
        Calcula o best bound atual de forma eficiente, dependendo do modo.
        """
        if self.is_empty():
            return float('inf')

        if self.mode == 'DFS':
            # No modo DFS, precisamos varrer todos os nós abertos
            return min(node.lp_objective_value for node in self._dfs_stack if node.lp_objective_value is not None)
        else: # Modo BEST_BOUND
            # No modo Best-Bound, o melhor nó é sempre o primeiro da fila
            return self._best_bound_queue[0][0]