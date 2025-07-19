# solver.py
# Contém a classe principal do solver que orquestra todo o processo.

import gurobipy as gp
from gurobipy import GRB, quicksum
from typing import Dict, Any, Optional
import math

from tree_elements import Node, Tree
from strategies import BranchingStrategy, MostInfeasibleStrategy, StrongBranchingStrategy, PseudoCostStrategy
from utils import StatisticsLogger
from cut_generator import find_cover_cuts, find_gomory_cuts
from heuristics import run_feasibility_pump, run_rins, run_diving_heuristic
from presolve import run_presolve

class MILPSolver:
    """
    A classe principal do solver de Programação Inteira Mista.
    Orquestra o processo de Branch and Bound utilizando os módulos de suporte.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config if config else {}
        self.logger = StatisticsLogger()
        self.branching_strategy: Optional[BranchingStrategy] = None
        self.dfs_node_limit = self.config.get('dfs_limit', 9999999)

        self.rins_frequency = self.config.get('rins_frequency', 0)
        
        self.model: gp.Model = None
        self.original_milp_model: Optional[gp.Model] = None
        self.original_vars: Dict[str, str] = {}
        self.incumbent_value: float = float('inf')
        self.incumbent_solution: Optional[Dict[str, float]] = None
        self.best_bound: float = -float('inf')
        self.node_count: int = 0
        self.lp_model_for_relaxations: Optional[gp.Model] = None
        self.tree: Tree = None
        self.added_cuts_pool = set()

    def _get_branching_strategy(self) -> BranchingStrategy:
        """
        Instancia a estratégia de branching com base na configuração.
        """
        strategy_name = self.config.get('branching_strategy', 'most_infeasible')
        print(f"INFO: Usando a estratégia de branching: '{strategy_name}'")
        if strategy_name == 'strong':
            return StrongBranchingStrategy()
        if strategy_name == 'pseudocost':
            return PseudoCostStrategy(self.original_vars, reliability_threshold=2, candidate_limit=10)
        return MostInfeasibleStrategy()
  
    def _is_solution_integer(self, var_values: Dict[str, float]) -> bool:
        """
        Verifica se uma solução satisfaz as condições de integralidade originais.
        """
        TOLERANCE = 1e-6
        for var_name, var_type in self.original_vars.items():
            if var_type != GRB.CONTINUOUS:
                if abs(var_values[var_name] - round(var_values[var_name])) > TOLERANCE:
                    return False
        return True

    def _solve_lp_relaxation(self, node: Node, base_lp_model: gp.Model) -> None:
        """
        Resolve a relaxação de LP a partir de um modelo LP base limpo.
        """
        node_model = base_lp_model.copy()
        for var_name, bound_info in node.local_bounds.items():
            var = node_model.getVarByName(var_name)
            if bound_info['type'] == '<=': var.ub = bound_info['value']
            else: var.lb = bound_info['value']

        node_model.setParam('OutputFlag', 0)
        node_model.optimize()

        status = node_model.Status
        if status == GRB.OPTIMAL:
            TOLERANCE = 1e-6
            all_vars = {v.VarName: v.X for v in node_model.getVars()}
            is_integer = all(
                self.original_vars.get(v_name) == GRB.CONTINUOUS or abs(v_val - round(v_val)) < TOLERANCE 
                for v_name, v_val in all_vars.items()
            )
            node.set_lp_solution('OPTIMAL', node_model.ObjVal, all_vars, is_integer)
        else:
            node.set_lp_solution('INFEASIBLE', None, None, False)

    def _run_cut_generation_for_node(self, node: Node):
            """
            Executa uma rodada de geração de cortes para um nó específico,
            buscando tanto Cover Cuts quanto Gomory Cuts.
            Retorna True se algum corte foi adicionado, False caso contrário.
            """
            print(f"INFO: [Node {node.id}] Tentando gerar cortes...")
            
            # Cria um modelo LP específico para este nó, com seus limites de branching
            node_lp_model = self._create_lp_for_node(node)
            
            cuts_added_this_round = 0
            
            # --- 1. LÓGICA PARA COVER CUTS ---
            if node.variable_values:
                cover_cut_recipes = find_cover_cuts(node_lp_model, node.variable_values, self.original_vars)
                
                for cut_vars, cut_rhs in cover_cut_recipes:
                    cut_signature = (tuple(sorted(cut_vars)), '<=', cut_rhs)
                    
                    if cut_signature in self.added_cuts_pool:
                        continue

                    print(f"INFO: [Cuts] Adicionando Cover Cut ao modelo: {' + '.join(cut_vars)} <= {cut_rhs}")

                    self.added_cuts_pool.add(cut_signature)
                    
                    expr_main = quicksum(self.model.getVarByName(v) for v in cut_vars)
                    expr_lp = quicksum(self.lp_model_for_relaxations.getVarByName(v) for v in cut_vars)
                    
                    cut_name = f"cover_{len(self.added_cuts_pool)}"
                    self.model.addConstr(expr_main <= cut_rhs, name=cut_name)
                    self.lp_model_for_relaxations.addConstr(expr_lp <= cut_rhs, name=cut_name)
                    cuts_added_this_round += 1

            # --- 2. LÓGICA PARA GOMORY CUTS ---
            gomory_cut_recipes = find_gomory_cuts(node_lp_model, self.original_vars)
            if gomory_cut_recipes:
                for coeffs, sense, rhs in gomory_cut_recipes:
                    cut_signature_items = tuple(sorted(coeffs.items()))
                    cut_signature = (cut_signature_items, sense, rhs)
                    
                    if cut_signature in self.added_cuts_pool:
                        continue

                    print(f"INFO: [Cuts] Gomory Cut encontrado para o Nó {node.id}.")
                    self.added_cuts_pool.add(cut_signature)
                    
                    expr_main = gp.LinExpr()
                    expr_lp = gp.LinExpr()
                    for var_name, coeff_val in coeffs.items():
                        expr_main.addTerms(coeff_val, self.model.getVarByName(var_name))
                        expr_lp.addTerms(coeff_val, self.lp_model_for_relaxations.getVarByName(var_name))
                    
                    cut_name = f"gomory_{len(self.added_cuts_pool)}"
                    if sense == GRB.GREATER_EQUAL:
                        self.model.addConstr(expr_main >= rhs, name=cut_name)
                        self.lp_model_for_relaxations.addConstr(expr_lp >= rhs, name=cut_name)
                    elif sense == GRB.LESS_EQUAL:
                        self.model.addConstr(expr_main <= rhs, name=cut_name)
                        self.lp_model_for_relaxations.addConstr(expr_lp <= rhs, name=cut_name)
                    else: # GRB.EQUAL
                        self.model.addConstr(expr_main == rhs, name=cut_name)
                        self.lp_model_for_relaxations.addConstr(expr_lp == rhs, name=cut_name)
                    
                    cuts_added_this_round += 1
            
            if cuts_added_this_round > 0:
                print(f"INFO: [Node {node.id}] {cuts_added_this_round} cortes novos adicionados no total.")
                self.model.update()
                self.lp_model_for_relaxations.update()
                return True
                
            return False

    def _create_lp_for_node(self, node: Node) -> gp.Model:
            """
            Cria uma cópia do modelo LP de relaxação e aplica os limites de
            branching específicos de um determinado nó.
            """
            # Cria uma cópia fresca do modelo LP da raiz
            node_lp_model = self.lp_model_for_relaxations.copy()
            
            # Aplica as restrições de branching deste nó
            for var_name, bound_info in node.local_bounds.items():
                var = node_lp_model.getVarByName(var_name)
                if bound_info['type'] == 'LOWER':
                    var.lb = max(var.lb, bound_info['value'])
                else: # UPPER
                    var.ub = min(var.ub, bound_info['value'])
            
            node_lp_model.update()
            return node_lp_model

    def _run_local_cut_and_solve(self, node: Node):
            """
            Cria um LP temporário para o nó, fortalece-o com cortes locais em um loop,
            e usa o resultado para obter um limitante dual mais forte para o nó.
            NÃO modifica os modelos globais.
            """
            print(f"INFO: [Node {node.id}] Tentando gerar cortes LOCAIS para apertar o limitante...")
            
            # 1. Cria e resolve o LP do nó pela primeira vez para obter um ponto de partida
            local_lp = self._create_lp_for_node(node)
            local_lp.setParam(GRB.Param.OutputFlag, 0)
            local_lp.optimize()
            
            # Se não houver uma solução ótima para começar, não há o que fazer.
            if local_lp.Status != GRB.OPTIMAL:
                print(f"AVISO: [Node {node.id}] LP local inicial não é ótimo. Abortando cortes para este nó.")
                return

            original_node_bound = local_lp.ObjVal

            # 2. Inicia o loop de geração e adição de cortes
            MAX_LOCAL_CUT_ROUNDS = 5
            total_cuts_added_in_session = 0

            for i in range(MAX_LOCAL_CUT_ROUNDS):
                current_lp_solution = {v.VarName: v.X for v in local_lp.getVars()}
                
                cuts_in_this_round = 0

                # 3. Adiciona os Cover Cuts encontrados
                cover_recipes = find_cover_cuts(local_lp, current_lp_solution, self.original_vars)
                for cut_vars, cut_rhs in cover_recipes:
                    cut_signature = (tuple(sorted(cut_vars)), '<=', cut_rhs)
                    if cut_signature not in self.added_cuts_pool:
                        self.added_cuts_pool.add(cut_signature)
                        expr = quicksum(local_lp.getVarByName(v) for v in cut_vars)
                        local_lp.addConstr(expr <= cut_rhs)
                        cuts_in_this_round += 1

                # 4. Adiciona os Gomory Cuts encontrados
                gomory_recipes = find_gomory_cuts(local_lp, self.original_vars)
                for coeffs, sense, rhs in gomory_recipes:
                    cut_signature = (tuple(sorted(coeffs.items())), sense, rhs)
                    if cut_signature not in self.added_cuts_pool:
                        self.added_cuts_pool.add(cut_signature)
                        expr = gp.LinExpr()
                        for var_name, coeff in coeffs.items():
                            expr.addTerms(coeff, local_lp.getVarByName(var_name))
                        
                        if sense == GRB.GREATER_EQUAL: local_lp.addConstr(expr >= rhs)
                        elif sense == GRB.LESS_EQUAL: local_lp.addConstr(expr <= rhs)
                        else: local_lp.addConstr(expr == rhs)
                        cuts_in_this_round += 1
                
                # 5. Verifica se o loop deve continuar
                if cuts_in_this_round == 0:
                    print(f"INFO: [Node {node.id}] Nenhuma fonte de corte nova encontrada na rodada {i+1}.")
                    break

                total_cuts_added_in_session += cuts_in_this_round
                print(f"INFO: [Node {node.id}] Rodada {i+1}: {cuts_in_this_round} cortes adicionados ao LP local. Re-resolvendo...")
                
                local_lp.update()
                local_lp.optimize()

                if local_lp.Status != GRB.OPTIMAL:
                    print(f"AVISO: [Node {node.id}] LP local tornou-se inviável após adição de cortes.")
                    break

            # 6. Após o loop, atualiza o limitante do nó principal se ele realmente melhorou
            if total_cuts_added_in_session > 0 and local_lp.Status == GRB.OPTIMAL:
                new_bound = local_lp.ObjVal
                is_minimization = self.model_sense == GRB.MINIMIZE
                
                # Verifica se o novo bound é realmente uma melhoria (considerando tolerâncias)
                if (is_minimization and new_bound > original_node_bound + 1e-6) or \
                (not is_minimization and new_bound < original_node_bound - 1e-6):
                    print(f"INFO: [Node {node.id}] Limitante dual melhorado com cortes locais: {original_node_bound:.4f} -> {new_bound:.4f}")
                    node.lp_objective_value = new_bound

    def is_node_promising(self, node: Node) -> bool:
        """
        Verifica se um nó é promissor para exploração (poda por bound).

        Um nó não é promissor se seu limitante dual (lp_objective_value)
        já for pior que a melhor solução inteira encontrada (incumbent_value).

        Args:
            node: O nó a ser verificado.

        Returns:
            True se o nó for promissor, False caso contrário.
        """
        if node.lp_objective_value is None:
            # Se o nó for inviável ou não resolvido, não é promissor para branching.
            return False

        # Para problemas de MINIMIZAÇÃO:
        if self.model_sense == GRB.MINIMIZE:
            # O nó só é promissor se seu lower bound for MENOR que o incumbent.
            # Usamos uma pequena tolerância para evitar problemas de ponto flutuante.
            return node.lp_objective_value < self.incumbent_value - 1e-9
        
        # Para problemas de MAXIMIZAÇÃO:
        else:
            # O nó só é promissor se seu upper bound for MAIOR que o incumbent.
            return node.lp_objective_value > self.incumbent_value + 1e-9


    def solve(self, problem_path: str):
        """
        Ponto de entrada principal para resolver um problema MILP, com a arquitetura
        correta de geração de cortes globais e locais.
        """
        # --- 1. SETUP INICIAL ---
        self.logger.start_solver()
        self.model = gp.read(problem_path)
        self.original_milp_model = self.model.copy()
        
        # Desliga as funcionalidades automáticas do Gurobi
        self.model.setParam(GRB.Param.OutputFlag, 0)
        self.model.setParam(GRB.Param.Presolve, 0)
        self.model.setParam(GRB.Param.Cuts, 0)
        self.model.setParam(GRB.Param.Heuristics, 0)
        
        self.original_vars = {v.VarName: v.VType for v in self.model.getVars()}
        self.branching_strategy = self._get_branching_strategy()
        
        self.model_sense = self.model.ModelSense
        if self.model_sense == GRB.MAXIMIZE:
            self.incumbent_value = -float('inf')
            self.best_bound = float('inf')
            print("INFO: Problema de MAXIMIZAÇÃO detectado.")
        else: # GRB.MINIMIZE
            self.incumbent_value = float('inf')
            self.best_bound = -float('inf')
            print("INFO: Problema de MINIMIZAÇÃO detectado.")
            
        # --- 2. PRESOLVE (OPCIONAL) ---
        if self.config.get('use_presolve', False):
            run_presolve(self.model, self.original_vars)
            self.model.update()
            self.original_vars = {v.VarName: v.VType for v in self.model.getVars()}
            self.original_milp_model = self.model.copy()

        # --- 3. CRIAÇÃO DO MODELO LP BASE ---
        print("INFO: Criando modelo de relaxação LP para o B&B...")
        self.lp_model_for_relaxations = self.model.copy()
        for v in self.lp_model_for_relaxations.getVars():
            if v.VType != GRB.CONTINUOUS:
                v.VType = GRB.CONTINUOUS
        self.lp_model_for_relaxations.update()

        # --- 4. HEURÍSTICA INICIAL (OPCIONAL) ---
        if self.config.get('use_heuristics', False):
            heuristic_result = run_diving_heuristic(
                self.original_milp_model, 
                self.original_vars
            )
            
            if heuristic_result:
                self.incumbent_value = heuristic_result['objective']
                self.incumbent_solution = heuristic_result['solution']
                print(f"INFO: [Solver] Heurística inteligente encontrou solução: {self.incumbent_value:.4f}")
                self.logger.log_event(0, self.incumbent_value, self.best_bound, force_log=True)

        # --- 5. RESOLUÇÃO E PREPARAÇÃO DA RAIZ ---
        root_node = Node(parent_id=None, depth=0, local_bounds={})
        self._solve_lp_relaxation(root_node, self.lp_model_for_relaxations)
        if root_node.lp_status == 'INFEASIBLE':
            self.logger.log_summary('Infeasible', 0, None, None)
            return
        self.best_bound = root_node.lp_objective_value

        # --- 6. GERAÇÃO DE CORTES GLOBAIS (APENAS NA RAIZ, OPCIONAL) ---
        if self.config.get('use_cuts', False):
            print("-" * 60)
            print("INFO: Iniciando fase de geração de cortes GLOBAIS no nó raiz...")
            if self._run_cut_generation_for_node(root_node):
                self._solve_lp_relaxation(root_node, self.lp_model_for_relaxations)
                self.best_bound = root_node.lp_objective_value
            print("-" * 60)
            
        self.tree = Tree(root_node)

        if root_node.is_integer:
            is_better = (root_node.lp_objective_value < self.incumbent_value) if self.model_sense == GRB.MINIMIZE else (root_node.lp_objective_value > self.incumbent_value)
            if is_better:
                 self.incumbent_value = root_node.lp_objective_value
                 self.incumbent_solution = root_node.variable_values

        while not self.tree.is_empty():
            self.node_count += 1
            if self.tree.mode == 'DFS' and self.node_count > self.dfs_node_limit:
                self.tree.switch_to_best_bound_mode()

            current_node = self.tree.get_next_node()

            # --- 8. GERAÇÃO DE CORTES LOCAIS (ITERATIVO, OPCIONAL) ---
            cut_freq = self.config.get('cut_frequency', 0)
            max_cut_depth = self.config.get('cut_depth', 5)

            if self.config.get('use_cuts', False) and cut_freq > 0 and \
               (self.node_count % cut_freq == 0) and (current_node.depth > 0) and \
               (current_node.depth <= max_cut_depth):
                
                self._run_local_cut_and_solve(current_node)

            # --- GATILHO DA HEURÍSTICA RINS ---
            if self.rins_frequency > 0 and self.incumbent_solution is not None and (self.node_count % self.rins_frequency == 0):
                
                rins_model = self.original_milp_model.copy()
                new_solution = run_rins(
                    rins_model, 
                    self.incumbent_solution, 
                    current_node.variable_values,
                    self.incumbent_value
                )

                if new_solution:
                    is_better = (new_solution['objective'] < self.incumbent_value) if self.model_sense == GRB.MINIMIZE else (new_solution['objective'] > self.incumbent_value)
                    if is_better:
                        print(f"INFO: [Solver] Heurística RINS melhorou o incumbente para: {new_solution['objective']:.4f}")
                        self.incumbent_value = new_solution['objective']
                        self.incumbent_solution = new_solution['solution']
                        self.logger.log_event(self.node_count, self.incumbent_value, self.best_bound, force_log=True)

            if current_node.lp_status != 'OPTIMAL':
                continue
            can_be_better = (current_node.lp_objective_value < self.incumbent_value) if self.model_sense == GRB.MINIMIZE else (current_node.lp_objective_value > self.incumbent_value)
            if not can_be_better:
                continue

            if current_node.is_integer:
                is_better = (current_node.lp_objective_value < self.incumbent_value) if self.model_sense == GRB.MINIMIZE else (current_node.lp_objective_value > self.incumbent_value)
                if is_better:
                    self.incumbent_value = current_node.lp_objective_value
                    self.incumbent_value = current_node.lp_objective_value
                    self.incumbent_solution = current_node.variable_values
                    self.logger.log_event(self.node_count, self.incumbent_value, self.best_bound, force_log=True)
                continue
            
            branch_var_name = self.branching_strategy.select_variable(current_node, self.model, self.original_vars)
            
            
            if branch_var_name is None:
                continue

            val = current_node.variable_values[branch_var_name]
            var_obj = self.lp_model_for_relaxations.getVarByName(branch_var_name)

            original_lb = var_obj.LB
            original_ub = var_obj.UB

            new_nodes_to_add = []
            child1 = Node(parent_id=current_node.id, depth=current_node.depth + 1, local_bounds={**current_node.local_bounds, branch_var_name: {'type': '<=', 'value': math.floor(val)}})
            child2 = Node(parent_id=current_node.id, depth=current_node.depth + 1, local_bounds={**current_node.local_bounds, branch_var_name: {'type': '>=', 'value': math.ceil(val)}})

            try:
                # --- Processa o filho 1 (ramo 'down') ---
                var_obj.UB = math.floor(val)
                self.lp_model_for_relaxations.optimize()
                
                status = self.lp_model_for_relaxations.Status
                if status == GRB.OPTIMAL:
                    all_vars = {v.VarName: v.X for v in self.lp_model_for_relaxations.getVars()}
                    is_int = self._is_solution_integer(all_vars)
                    child1.set_lp_solution('OPTIMAL', self.lp_model_for_relaxations.ObjVal, all_vars, is_int)
                    new_nodes_to_add.append(child1)
                else:
                    child1.set_lp_solution('INFEASIBLE', None, None, False)

            finally:
                # Este bloco é EXECUTADO SEMPRE, garantindo a limpeza do estado
                var_obj.UB = original_ub

            try:
                # --- Processa o filho 2 (ramo 'up') ---
                var_obj.LB = math.ceil(val)
                self.lp_model_for_relaxations.optimize()
                
                status = self.lp_model_for_relaxations.Status
                if status == GRB.OPTIMAL:
                    all_vars = {v.VarName: v.X for v in self.lp_model_for_relaxations.getVars()}
                    is_int = self._is_solution_integer(all_vars)
                    child2.set_lp_solution('OPTIMAL', self.lp_model_for_relaxations.ObjVal, all_vars, is_int)
                    new_nodes_to_add.append(child2)
                else:
                    child2.set_lp_solution('INFEASIBLE', None, None, False)
            
            finally:
                # Garante a limpeza final do estado
                var_obj.LB = original_lb
            
            self.branching_strategy.update_scores(parent_node=current_node, child_nodes=[child1, child2], branch_var=branch_var_name, model=self.model)
            
            # Adiciona os nós promissores à árvore
            promising_nodes = []
            if child1.lp_status == 'OPTIMAL' and self.is_node_promising(child1):
                promising_nodes.append(child1)
            if child2.lp_status == 'OPTIMAL' and self.is_node_promising(child2):
                promising_nodes.append(child2)

            self.tree.add_nodes(promising_nodes)
            
            if not self.tree.is_empty():
                self.best_bound = self.tree.get_current_best_bound()
            else:
                self.best_bound = self.incumbent_value

            self.logger.log_event(self.node_count, self.incumbent_value, self.best_bound)

        self.logger.log_summary('Optimal solution found' if self.incumbent_value != float('inf') else 'No solution found', 
                                self.node_count, self.incumbent_value, self.best_bound)