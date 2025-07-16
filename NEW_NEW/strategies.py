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
    def select_variable(self, node: Node, model: gp.Model, original_vars: Dict[str, str]) -> Optional[str]: # <--- ADICIONE original_vars
        """ Seleciona a melhor variável para o branching. """
        raise NotImplementedError

    def update_scores(self, parent_node: Node, child_nodes: List[Node], branch_var: str):
        # ...
        # Esta linha vai falhar, pois 'self' (a estratégia) não tem 'model'
        is_minimization = self.model.ModelSense == GRB.MINIMIZE

class MostInfeasibleStrategy(BranchingStrategy):
    # A assinatura do método agora corresponde à interface
    def select_variable(self, node: Node, model: gp.Model, original_vars: Dict[str, str]) -> Optional[str]:
        if not node.variable_values: return None
        
        branching_variable = None
        max_infeasibility = -1.0
        TOLERANCE = 1e-6
        
        for var_name, var_value in node.variable_values.items():
            # --- A CORREÇÃO CRÍTICA ESTÁ AQUI ---
            # Só consideramos ramificar em variáveis que NÃO são contínuas.
            if original_vars.get(var_name, GRB.CONTINUOUS) != GRB.CONTINUOUS:
                if abs(var_value - round(var_value)) > TOLERANCE:
                    frac = var_value - math.floor(var_value)
                    infeasibility = 0.5 - abs(frac - 0.5)
                    #infeasibility = min(frac, 1.0 - frac)
                    #infeasibility = 0.5 - abs((var_value - math.floor(var_value)) - 0.5)
                    if infeasibility > max_infeasibility:
                        max_infeasibility = infeasibility
                        branching_variable = var_name
                        
        return branching_variable
    
    def update_scores(self, parent_node: Node, child_nodes: List[Node], branch_var: str, model: gp.Model):
        # Esta estratégia não aprende, então o método não faz nada.
        # Ele só precisa existir para ser compatível com a interface da classe base.
        pass

class StrongBranchingStrategy(BranchingStrategy):
    """
    Implementa a estratégia de Strong Branching.

    Esta estratégia avalia um subconjunto de variáveis candidatas fracionárias
    resolvendo temporariamente o LP para os ramos "up" e "down" de cada uma.
    A variável que promete o maior progresso no dual bound é selecionada.
    """

    def __init__(self, candidate_limit: int = 5, epsilon: float = 1e-6):
        """
        Inicializa a estratégia de Strong Branching.

        Args:
            candidate_limit (int): O número máximo de variáveis candidatas a serem
                                 avaliadas em cada nó. Limitar isso é crucial para
                                 controlar o tempo de execução.
            epsilon (float): Uma pequena constante para estabilidade numérica na
                             função de score, evitando multiplicação por zero.
        """
        super().__init__()
        self.candidate_limit = candidate_limit
        self.epsilon = epsilon
        print(f"INFO: StrongBranchingStrategy inicializada com limite de {candidate_limit} candidatos.")

    def _solve_lookahead_lp(self, base_model: gp.Model, node: Node, branch_var: str, branch_val: float, direction: str) -> Optional[float]:
        temp_model = base_model.copy()
        temp_model.setParam(GRB.Param.OutputFlag, 0)
        temp_model.setParam(GRB.Param.Presolve, 0)
        temp_model.setParam(GRB.Param.Cuts, 0)
        
        # --- ALTERAÇÃO PRINCIPAL AQUI ---
        # Não precisamos da solução ótima, apenas de uma boa estimativa do bound.
        # Limitamos o número de iterações do Simplex.
        temp_model.setParam(GRB.Param.IterationLimit, 1000) # Valor inicial para teste

        # ... (resto da função para aplicar os bounds)
        for var_name, bound_info in node.local_bounds.items():
            var = temp_model.getVarByName(var_name)
            if bound_info['type'] == '<=':
                var.ub = bound_info['value']
            else:
                var.lb = bound_info['value']

        var_to_branch_obj = temp_model.getVarByName(branch_var)
        if direction == 'down':
            var_to_branch_obj.ub = math.floor(branch_val)
        else:
            var_to_branch_obj.lb = math.ceil(branch_val)
        
        temp_model.optimize()

        # O resultado ainda é válido mesmo que o Gurobi pare pelo limite de iterações.
        # Se for um problema de MIN, o ObjVal será um lower bound válido para o nó filho.
        if temp_model.Status in [GRB.OPTIMAL, GRB.ITERATION_LIMIT]:
            return temp_model.ObjVal
        
        return float('inf') if base_model.ModelSense == GRB.MINIMIZE else -float('inf')

    def select_variable(self, node: Node, model: gp.Model, original_vars: Dict[str, str]) -> Optional[str]:
        if not node.variable_values or node.lp_objective_value is None:
            return None

        # 1. Coletar todas as variáveis candidatas (que são inteiras e fracionárias)
        TOLERANCE = 1e-6
        all_candidates = []
        for var_name, var_value in node.variable_values.items():
            if original_vars.get(var_name) != GRB.CONTINUOUS and abs(var_value - round(var_value)) > TOLERANCE:
                # Usa a "infeasibility" (proximidade de 0.5) como um pré-score para selecionar os melhores candidatos
                frac = var_value - math.floor(var_value)
                infeasibility_score = 0.5 - abs(frac - 0.5)
                #infeasibility_score = min(frac, 1.0 - frac)
                all_candidates.append({'name': var_name, 'value': var_value, 'pre_score': infeasibility_score})

        if not all_candidates:
            return None

        # Ordena para pegar os candidatos mais promissores primeiro
        all_candidates.sort(key=lambda x: x['pre_score'], reverse=True)
        
        # 2. Limitar o número de candidatos a serem avaliados (REQUISITO 1)
        candidates_to_evaluate = all_candidates[:self.candidate_limit]

        best_var = None
        # O score inicial depende do objetivo (maximizar o score)
        best_score = -1.0 
        parent_obj_val = node.lp_objective_value

        is_minimization = model.ModelSense == GRB.MINIMIZE

        for cand in candidates_to_evaluate:
            var_name = cand['name']
            var_value = cand['value']
            
            # Realiza o lookahead para ambos os ramos
            obj_down = self._solve_lookahead_lp(model, node, var_name, var_value, 'down')
            obj_up = self._solve_lookahead_lp(model, node, var_name, var_value, 'up')
            
            # Calcula o ganho (degradação do objetivo) para cada ramo
            # Para MIN, um ganho maior (mais positivo) é melhor. Para MAX, um ganho menor (mais negativo) é melhor.
            if is_minimization:
                gain_down = obj_down - parent_obj_val
                gain_up = obj_up - parent_obj_val
            else: # Maximização
                gain_down = parent_obj_val - obj_down
                gain_up = parent_obj_val - obj_up

            # 3. Calcula o score usando a função produto (REQUISITO 2)
            # Referência: Tese de T. Achterberg, Seção 5.4, Equação 5.2
            # Esta função de score favorece variáveis que produzem bons ganhos em AMBOS os ramos.
            score = max(gain_down, self.epsilon) * max(gain_up, self.epsilon)
            
            if score > best_score:
                best_score = score
                best_var = var_name
                
        return best_var

    def update_scores(self, parent_node: Node, child_nodes: List[Node], branch_var: str, model: gp.Model):
        # Esta estratégia não aprende por si só; a lógica de aprendizado
        # está na classe filha PseudoCostStrategy.
        pass
    
class PseudoCostStrategy(StrongBranchingStrategy):
    """
    Implementa Reliability Branching usando Pseudo-custos.

    Esta é uma estratégia híbrida que equilibra o custo e o benefício do
    strong branching. Ela usa o custoso strong branching para "aprender"
    sobre o impacto de uma variável (fase de inicialização) e, uma vez
    que a informação é considerada confiável, passa a usar uma estimativa
    matemática barata (o pseudo-custo) para o resto da busca.

    Herda de StrongBranchingStrategy para reutilizar a lógica de lookahead.
    """
    def __init__(self, original_vars: Dict[str, str],
                 candidate_limit: int = 10,
                 reliability_threshold: int = 2):
        """
        Inicializa a estratégia de Pseudo-custo.

        Args:
            original_vars (Dict): Mapa de nomes de variáveis para seus tipos.
            candidate_limit (int): Limite de candidatos para a fase de strong branching.
            reliability_threshold (int): Número de vezes que uma variável precisa ser
                                         avaliada com strong branching antes de ser
                                         considerada "confiável" para usar pseudo-custos.
        """
        super().__init__(candidate_limit=candidate_limit)
        
        self.original_vars = original_vars
        self.reliability_threshold = reliability_threshold
        
        # Estruturas de dados para armazenar o histórico de aprendizado
        self.pseudo_costs_sum_degradation: Dict[str, Dict[str, float]] = {}
        self.pseudo_costs_sum_fractionality: Dict[str, Dict[str, float]] = {}
        self.pseudo_costs_reliability_count: Dict[str, int] = {}
        
        print(f"INFO: PseudoCostStrategy inicializada com reliability_threshold={self.reliability_threshold}.")

    def select_variable(self, node: Node, model: gp.Model, original_vars: Dict[str, str]) -> Optional[str]:
        if not node.variable_values or node.lp_objective_value is None:
            return None

        # 1. Coletar e pré-selecionar os melhores candidatos
        TOLERANCE = 1e-6
        all_candidates = []
        for var_name, var_value in node.variable_values.items():
            if self.original_vars.get(var_name) != GRB.CONTINUOUS and abs(var_value - round(var_value)) > TOLERANCE:
                frac = var_value - math.floor(var_value)
                infeasibility_score = 0.5 - abs(frac - 0.5)
                #infeasibility_score = min(frac, 1.0 - frac)
                all_candidates.append({'name': var_name, 'value': var_value, 'pre_score': infeasibility_score})
        
        if not all_candidates: return None
        all_candidates.sort(key=lambda x: x['pre_score'], reverse=True)
        candidates_to_evaluate = all_candidates[:self.candidate_limit]

        # 2. Avaliar candidatos com a lógica híbrida
        best_var = None
        best_score = -1.0
        parent_obj_val = node.lp_objective_value
        is_minimization = model.ModelSense == GRB.MINIMIZE

        for cand in candidates_to_evaluate:
            var_name = cand['name']
            var_value = cand['value']
            
            reliability_count = self.pseudo_costs_reliability_count.get(var_name, 0)

            # --- O CORAÇÃO DA ESTRATÉGIA HÍBRIDA ---
            if reliability_count < self.reliability_threshold:
                # VARIÁVEL NÃO CONFIÁVEL: Usa Strong Branching para aprender
                obj_down = self._solve_lookahead_lp(model, node, var_name, var_value, 'down')
                obj_up = self._solve_lookahead_lp(model, node, var_name, var_value, 'up')
                
                gain_down = (obj_down - parent_obj_val) if is_minimization else (parent_obj_val - obj_down)
                gain_up = (obj_up - parent_obj_val) if is_minimization else (parent_obj_val - obj_up)
            else:
                # VARIÁVEL CONFIÁVEL: Usa a estimativa de pseudo-custo (rápida)
                sum_deg = self.pseudo_costs_sum_degradation.get(var_name, {'down': 0.0, 'up': 0.0})
                sum_frac = self.pseudo_costs_sum_fractionality.get(var_name, {'down': 0.0, 'up': 0.0})

                # CORREÇÃO: Adicionar self.epsilon ao denominador para garantir estabilidade numérica.
                # Se a soma da fracionalidade for 0, o pseudo-custo resultante será 0.
                pc_down = sum_deg['down'] / (sum_frac['down'] + self.epsilon)
                pc_up   = sum_deg['up']   / (sum_frac['up']   + self.epsilon)
                
                frac = var_value - math.floor(var_value)
                gain_down = pc_down * frac
                gain_up = pc_up * (1.0 - frac)

            # A função de score produto é usada em ambos os casos
            score = max(gain_down, self.epsilon) * max(gain_up, self.epsilon)

            if score > best_score:
                best_score = score
                best_var = var_name
                
        return best_var

    def update_scores(self, parent_node: Node, child_nodes: List[Node], branch_var: str, model: gp.Model):
        """
        Após um branch ser realizado, esta função é chamada para atualizar o histórico
        da variável `branch_var`, alimentando o aprendizado dos pseudo-custos.
        """
        if parent_node.lp_objective_value is None or not parent_node.variable_values:
            return

        parent_obj = parent_node.lp_objective_value
        parent_val = parent_node.variable_values[branch_var]
        frac = parent_val - math.floor(parent_val)

        self.pseudo_costs_sum_degradation.setdefault(branch_var, {'down': 0.0, 'up': 0.0})
        self.pseudo_costs_sum_fractionality.setdefault(branch_var, {'down': 0.0, 'up': 0.0})
        self.pseudo_costs_reliability_count.setdefault(branch_var, 0)
        
        # CORREÇÃO: Usa o 'model' passado como parâmetro, não 'self.model'
        is_minimization = model.ModelSense == GRB.MINIMIZE
        
        child_down = next((n for n in child_nodes if n.local_bounds.get(branch_var, {}).get('type') == '<='), None)
        child_up = next((n for n in child_nodes if n.local_bounds.get(branch_var, {}).get('type') == '>='), None)

        # O resto da função permanece igual...
        if child_down and child_down.lp_objective_value is not None:
            degradation = (child_down.lp_objective_value - parent_obj) if is_minimization else (parent_obj - child_down.lp_objective_value)
            self.pseudo_costs_sum_degradation[branch_var]['down'] += max(0, degradation)
            self.pseudo_costs_sum_fractionality[branch_var]['down'] += frac
            
        if child_up and child_up.lp_objective_value is not None:
            degradation = (child_up.lp_objective_value - parent_obj) if is_minimization else (parent_obj - child_up.lp_objective_value)
            self.pseudo_costs_sum_degradation[branch_var]['up'] += max(0, degradation)
            self.pseudo_costs_sum_fractionality[branch_var]['up'] += (1.0 - frac)

        self.pseudo_costs_reliability_count[branch_var] += 1

#