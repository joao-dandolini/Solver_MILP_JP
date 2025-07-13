# strategies.py
# Contém as diferentes estratégias de branching que o solver pode utilizar.

from abc import ABC, abstractmethod
from typing import Optional, Dict, List
import math
import gurobipy as gp
from gurobipy import GRB

from tree_elements import Node

# --- INTERFACE ATUALIZADA ---
class BranchingStrategy(ABC):
    """
    Classe base abstrata (interface) para todas as estratégias de branching.
    """
    @abstractmethod
    def select_variable(self, node: Node, model: gp.Model) -> Optional[str]:
        """ Seleciona a melhor variável para o branching. """
        raise NotImplementedError

    def update_scores(self, parent_node: Node, child_nodes: List[Node], branch_var: str):
        """
        [ATUALIZADO] Opcional: Atualiza os scores internos da estratégia após um branching.
        Útil para estratégias que aprendem, como pseudo-custos.
        """
        pass # A implementação padrão continua não fazendo nada, apenas aceita o argumento.

# ... (MostInfeasibleStrategy e StrongBranchingStrategy permanecem as mesmas) ...
class MostInfeasibleStrategy(BranchingStrategy):
    def select_variable(self, node: Node, model: gp.Model) -> Optional[str]:
        # ... código inalterado ...
        if not node.variable_values: return None
        branching_variable = None
        max_infeasibility = -1.0
        TOLERANCE = 1e-6
        for var_name, var_value in node.variable_values.items():
            if abs(var_value - round(var_value)) > TOLERANCE:
                infeasibility = 0.5 - abs((var_value - math.floor(var_value)) - 0.5)
                if infeasibility > max_infeasibility:
                    max_infeasibility = infeasibility
                    branching_variable = var_name
        return branching_variable

class StrongBranchingStrategy(BranchingStrategy):
    def __init__(self, candidate_limit: int = 10):
        self.candidate_limit = candidate_limit
    def _solve_lookahead_lp(self, base_model: gp.Model, node_bounds: Dict, branch_var: str, val_to_branch: float, direction: str) -> float:
        # ... código inalterado ...
        temp_model = base_model.copy()
        #temp_model.setParam('OutputFlag', 0)
        temp_model.setParam(GRB.Param.OutputFlag, 0)   # Já tínhamos, mas é bom manter aqui
        temp_model.setParam(GRB.Param.Presolve, 0)      # Desliga o pré-processamento do modelo
        temp_model.setParam(GRB.Param.Cuts, 0)          # Desliga todos os geradores de cortes automáticos
        temp_model.setParam(GRB.Param.Heuristics, 0)    # Desliga todas as heurísticas automáticas (FP, RINS, etc.)
        temp_model.setParam(GRB.Param.Symmetry, 0) 

        for var_name, bound_info in node_bounds.items():
            var = temp_model.getVarByName(var_name)
            if bound_info['type'] == '<=': var.ub = bound_info['value']
            else: var.lb = bound_info['value']
        var_to_branch = temp_model.getVarByName(branch_var)
        if direction == 'down': var_to_branch.ub = math.floor(val_to_branch)
        else: var_to_branch.lb = math.ceil(val_to_branch)
        temp_model.optimize()
        if temp_model.Status == GRB.OPTIMAL: return temp_model.ObjVal
        return float('inf')
    def select_variable(self, node: Node, model: gp.Model) -> Optional[str]:
        # ... código inalterado ...
        if not node.variable_values: return None
        TOLERANCE = 1e-6
        all_candidates = []
        for var_name, var_value in node.variable_values.items():
            if abs(var_value - round(var_value)) > TOLERANCE:
                infeasibility = 0.5 - abs((var_value - math.floor(var_value)) - 0.5)
                all_candidates.append({'name': var_name, 'value': var_value, 'score': infeasibility})
        if not all_candidates: return None
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        limited_candidates = all_candidates[:self.candidate_limit]
        best_score = -1.0
        best_var = None
        for cand in limited_candidates:
            var_name, var_value = cand['name'], cand['value']
            obj_down = self._solve_lookahead_lp(model, node.local_bounds, var_name, var_value, 'down')
            obj_up = self._solve_lookahead_lp(model, node.local_bounds, var_name, var_value, 'up')
            score = min(obj_down, obj_up)
            if score > best_score:
                best_score = score
                best_var = var_name
        return best_var

# --- NOSSA NOVA ESTRATÉGIA INTELIGENTE ---

class PseudoCostStrategy(StrongBranchingStrategy):
    """
    Implementa Reliability Branching usando Pseudo-custos.
    
    Usa Strong Branching para 'aprender' sobre as variáveis no início
    e depois muda para uma estimativa de custo barata para o resto da busca.
    Herda de StrongBranching para reutilizar o método _solve_lookahead_lp.
    """
    def __init__(self, original_vars: Dict[str, str], reliability_threshold: int = 2, candidate_limit: int = 10):
        super().__init__(candidate_limit=candidate_limit)
        self.original_vars = original_vars # Armazena o dicionário
        self.reliability_threshold = reliability_threshold
        # Estruturas para armazenar o aprendizado
        self.pseudo_costs_sum_degradation = {} # {var_name: {'down': float, 'up': float}}
        self.pseudo_costs_sum_fractionality = {}
        self.pseudo_costs_reliability_count = {}

    def select_variable(self, node: Node, model: gp.Model, debug: bool = False) -> Optional[str]:
            # Verificação de segurança 1: Se o nó for inviável, não há o que fazer.
            if node.lp_objective_value is None:
                return None
                
            # Verificação de segurança 2: Se não houver valores de variáveis.
            if not node.variable_values:
                return None

            # Passo 1: Filtra apenas as variáveis fracionárias para análise.
            TOLERANCE = 1e-6
            fractional_vars = {
                k: v for k, v in node.variable_values.items() 
                if self.original_vars.get(k) != GRB.CONTINUOUS and abs(v - round(v)) > TOLERANCE
            }

            if not fractional_vars:
                return None # Nenhuma variável fracionária para ramificar.

            # Passo 2: Pré-seleciona as candidatas mais promissoras com a heurística barata.
            all_candidates = []
            for var_name, var_value in fractional_vars.items():
                infeasibility = 0.5 - abs((var_value - math.floor(var_value)) - 0.5)
                all_candidates.append({'name': var_name, 'value': var_value, 'score': infeasibility})
            
            all_candidates.sort(key=lambda x: x['score'], reverse=True)
            limited_candidates = all_candidates[:self.candidate_limit]

            # Passo 3: Calcula o score para as candidatas (usando strong ou pseudo-custo).
            best_score = -1.0
            best_var = None

            for cand in limited_candidates:
                var_name = cand['name']
                var_value = cand['value']
                reliability = self.pseudo_costs_reliability_count.get(var_name, 0)

                if reliability < self.reliability_threshold:
                    # Usa strong branching para aprender
                    obj_down = self._solve_lookahead_lp(model, node.local_bounds, var_name, var_value, 'down')
                    obj_up = self._solve_lookahead_lp(model, node.local_bounds, var_name, var_value, 'up')
                    score = min(obj_down, obj_up)
                else:
                    # Usa a estimativa de pseudo-custo
                    frac = var_value - math.floor(var_value)
                    pc_down_sum_deg = self.pseudo_costs_sum_degradation[var_name]['down']
                    pc_down_sum_frac = self.pseudo_costs_sum_fractionality[var_name]['down']
                    pc_up_sum_deg = self.pseudo_costs_sum_degradation[var_name]['up']
                    pc_up_sum_frac = self.pseudo_costs_sum_fractionality[var_name]['up']

                    pc_down = (pc_down_sum_deg / pc_down_sum_frac) if pc_down_sum_frac > TOLERANCE else 0
                    pc_up = (pc_up_sum_deg / pc_up_sum_frac) if pc_up_sum_frac > TOLERANCE else 0

                    estimated_obj_down = node.lp_objective_value + pc_down * frac
                    estimated_obj_up = node.lp_objective_value + pc_up * (1 - frac)
                    score = min(estimated_obj_down, estimated_obj_up)

                if score > best_score:
                    best_score = score
                    best_var = var_name
            
            return best_var

    def update_scores(self, parent_node: Node, child_nodes: List[Node], branch_var: str):
            """
            [CORRIGIDO] Atualiza os dados de pseudo-custo, lidando com ramos inviáveis.
            """
            # Verificação de segurança: se o nó pai não tiver um valor, não há o que aprender.
            if parent_node.lp_objective_value is None:
                return

            parent_obj = parent_node.lp_objective_value
            parent_val = parent_node.variable_values[branch_var]
            frac = parent_val - math.floor(parent_val)

            if branch_var not in self.pseudo_costs_sum_degradation:
                self.pseudo_costs_sum_degradation[branch_var] = {'down': 0.0, 'up': 0.0}
                self.pseudo_costs_sum_fractionality[branch_var] = {'down': 0.0, 'up': 0.0}
                self.pseudo_costs_reliability_count[branch_var] = 0

            # Verifica o ramo 'down' antes de fazer contas
            child_down = next((n for n in child_nodes if branch_var in n.local_bounds and n.local_bounds[branch_var]['type'] == '<='), None)
            if child_down and child_down.lp_objective_value is not None:
                degradation = max(0, child_down.lp_objective_value - parent_obj)
                self.pseudo_costs_sum_degradation[branch_var]['down'] += degradation
                self.pseudo_costs_sum_fractionality[branch_var]['down'] += frac

            # Verifica o ramo 'up' antes de fazer contas
            child_up = next((n for n in child_nodes if branch_var in n.local_bounds and n.local_bounds[branch_var]['type'] == '>='), None)
            if child_up and child_up.lp_objective_value is not None:
                degradation = max(0, child_up.lp_objective_value - parent_obj)
                self.pseudo_costs_sum_degradation[branch_var]['up'] += degradation
                self.pseudo_costs_sum_fractionality[branch_var]['up'] += (1 - frac)

            self.pseudo_costs_reliability_count[branch_var] += 1