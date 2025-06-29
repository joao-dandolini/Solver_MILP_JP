# tree_node.py
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Node:
    """Representa um nó na árvore de Branch and Bound."""
    extra_bounds: List[Tuple[str, str, float]] = field(default_factory=list)
    parent_objective: Optional[float] = None
    branch_variable: Optional[str] = None
    branch_direction: Optional[str] = None
    depth: int = 0
    
    def __lt__(self, other: 'Node') -> bool:
        """
        Compara dois nós para a fila de prioridade (heapq).
        Prioriza o nó com o melhor 'parent_objective'.
        Em caso de empate, prioriza o nó mais profundo.
        """
        if self.parent_objective is None: return False
        if other.parent_objective is None: return True
        
        if self.parent_objective != other.parent_objective:
            return self.parent_objective < other.parent_objective
        
        return self.depth > other.depth