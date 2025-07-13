# tree_elements.py
# Contém as estruturas de dados fundamentais para a árvore de Branch and Bound.

from typing import Dict, Any, Optional, List, Tuple
import heapq

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
        
        self._dfs_stack: List[Node] = [root_node]
        self._best_bound_queue: List[Tuple[float, int, Node]] = []

        # --- ALTERAÇÃO ---
        # Rastreia o best bound em modo DFS para evitar varreduras O(N).
        root_obj = root_node.lp_objective_value
        self._dfs_best_bound: float = root_obj if root_obj is not None else float('inf')

    def add_nodes(self, nodes: List[Node]):
        """
        Adiciona uma lista de novos nós à coleção apropriada, dependendo do modo.
        """
        if self.mode == 'DFS':
            self._dfs_stack.extend(nodes)
            # --- ALTERAÇÃO ---
            # Atualiza o best bound de forma eficiente ao adicionar nós.
            min_obj_in_new_nodes = min(
                (n.lp_objective_value for n in nodes if n.lp_objective_value is not None),
                default=float('inf')
            )
            self._dfs_best_bound = min(self._dfs_best_bound, min_obj_in_new_nodes)
        else: # Modo BEST_BOUND
            for node in nodes:
                heapq.heappush(self._best_bound_queue, (node.lp_objective_value, node.id, node))

    def get_next_node(self) -> Optional[Node]:
        """
        Recupera e remove o próximo nó a ser explorado, de acordo com o modo atual.
        """
        if self.is_empty():
            return None

        if self.mode == 'DFS':
            node = self._dfs_stack.pop()
            # --- ALTERAÇÃO ---
            # Se o nó removido era o que definia o best bound, precisamos recalcular.
            # Isso é necessário para correção, mas só ocorre quando o melhor nó é explorado.
            if node.lp_objective_value is not None and abs(node.lp_objective_value - self._dfs_best_bound) < 1e-9:
                self._recalculate_dfs_best_bound()
            return node
        else: # Modo BEST_BOUND
            return heapq.heappop(self._best_bound_queue)[2]

    def is_empty(self) -> bool:
        """
        Verifica se ainda existem nós a serem explorados.
        """
        return not self._dfs_stack and not self._best_bound_queue

    def switch_to_best_bound_mode(self):
        """
        Converte a árvore do modo DFS para o modo Best-Bound.
        """
        if self.mode == 'DFS':
            print("INFO: [Tree] Trocando estratégia de busca para BEST-BOUND.")
            self.mode = 'BEST_BOUND'
            
            for node in self._dfs_stack:
                if node.lp_objective_value is not None:
                    heapq.heappush(self._best_bound_queue, (node.lp_objective_value, node.id, node))
            
            self._dfs_stack = []

    def get_current_best_bound(self) -> float:
        """
        Calcula o best bound atual de forma eficiente, dependendo do modo.
        Para problemas de minimização, este é o menor valor de objetivo de todos os nós abertos.
        """
        if self.is_empty():
            # Se a árvore está vazia, o problema foi resolvido. O bound pode ser considerado infinito
            # ou, mais corretamente, o valor da melhor solução encontrada.
            # No contexto de um gap, inf funciona.
            return float('inf') 

        if self.mode == 'DFS':
            # --- ALTERAÇÃO ---
            # Retorna o valor rastreado em O(1).
            return self._dfs_best_bound
        else: # Modo BEST_BOUND
            return self._best_bound_queue[0][0]

    # --- NOVO MÉTODO ---
    def _recalculate_dfs_best_bound(self):
        """
        Varre a pilha DFS para encontrar o novo best bound.
        Chamado apenas quando o nó que definia o best bound é removido da pilha.
        """
        if not self._dfs_stack:
            self._dfs_best_bound = float('inf')
        else:
            self._dfs_best_bound = min(
                (n.lp_objective_value for n in self._dfs_stack if n.lp_objective_value is not None),
                default=float('inf')
            )