# solver.py
# Contém a classe principal do solver que orquestra todo o processo.

import gurobipy as gp
from gurobipy import GRB, quicksum
from typing import Dict, Any, Optional
import math # Importado para o branching

from tree_elements import Node, Tree
from strategies import BranchingStrategy, MostInfeasibleStrategy, StrongBranchingStrategy, PseudoCostStrategy
from utils import StatisticsLogger
from cut_generator import find_cover_cuts
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
        self.dfs_node_limit = self.config.get('dfs_limit', 9999999) # Padrão alto

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
        # --- FIM DA NOVA OPÇÃO ---
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

    def solve(self, problem_path: str):
        """
        Ponto de entrada principal para resolver um problema MILP, com a lógica final e correta.
        """
        self.logger.start_solver()
        self.model = gp.read(problem_path)
        self.original_milp_model = self.model.copy()
        self.model.setParam(GRB.Param.OutputFlag, 0)   # Já tínhamos, mas é bom manter aqui
        self.model.setParam(GRB.Param.Presolve, 0)      # Desliga o pré-processamento do modelo
        self.model.setParam(GRB.Param.Cuts, 0)          # Desliga todos os geradores de cortes automáticos
        self.model.setParam(GRB.Param.Heuristics, 0)    # Desliga todas as heurísticas automáticas (FP, RINS, etc.)
        self.model.setParam(GRB.Param.Symmetry, 0) 
        
        self.original_vars = {v.VarName: v.VType for v in self.model.getVars()}

        self.branching_strategy = self._get_branching_strategy()

        # --- ALTERAÇÃO 1: DETECTAR O SENTIDO E AJUSTAR OS LIMITES ---
        # Detectamos se o problema é de MIN ou MAX e ajustamos o incumbente inicial.
        self.model_sense = self.model.ModelSense
        print(f"[DEBUG-Sense] Gurobi reporta ModelSense: {self.model_sense} (Onde MIN={GRB.MINIMIZE} e MAX={GRB.MAXIMIZE})")
        if self.model_sense == GRB.MAXIMIZE:
            self.incumbent_value = -float('inf')
            self.best_bound = float('inf')
            print("INFO: Problema de MAXIMIZAÇÃO detectado.")
        else: # GRB.MINIMIZE
            # Os valores padrão do __init__ já são para minimização.
            print("INFO: Problema de MINIMIZAÇÃO detectado.")
        # --- FIM DA ALTERAÇÃO 1 ---
            
        if self.config.get('use_presolve', False):
            run_presolve(self.model, self.original_vars)
            # É uma boa prática chamar update() após modificar o modelo
            self.model.update() 
            print("INFO: [Solver] Atualizando o dicionário de variáveis pós-presolve.")
            self.original_vars = {v.VarName: v.VType for v in self.model.getVars()}
            self.original_milp_model = self.model.copy()

        print("INFO: Criando modelo de relaxação LP para o B&B...")
        self.lp_model_for_relaxations = self.model.copy()
        for v in self.lp_model_for_relaxations.getVars():
            if v.VType != GRB.CONTINUOUS:
                v.VType = GRB.CONTINUOUS
        self.lp_model_for_relaxations.update()

        if self.config.get('use_heuristics', False):
            # A chamada agora usa nossa nova e mais rápida heurística de mergulho
            heuristic_result = run_diving_heuristic(self.original_milp_model, self.original_vars)
            if heuristic_result:
                self.incumbent_value = heuristic_result['objective']
                self.incumbent_solution = heuristic_result['solution']
                print(f"INFO: [Solver] Heurística de Mergulho encontrou solução viável com objetivo: {self.incumbent_value:.4f}")
                self.logger.log_event(0, self.incumbent_value, self.best_bound, force_log=True)
                
        # Em solve()
        root_node = Node(parent_id=None, depth=0, local_bounds={})
        # Passe o modelo LP como o segundo argumento
        self._solve_lp_relaxation(root_node, self.lp_model_for_relaxations)


        if self.config.get('use_cuts', False):
            # --- NOVO: LOOP DE GERAÇÃO DE CORTES NA RAIZ (COM VERIFICAÇÃO DE DUPLICATAS) ---
            MAX_CUT_ROUNDS = 10
            print("-" * 60)
            print("INFO: Iniciando fase de geração de cortes no nó raiz...")
            
            # 1. INICIALIZAMOS NOSSO "POOL" DE CORTES
            added_cuts_pool = set()
            
            for i in range(MAX_CUT_ROUNDS):
                if root_node.is_integer:
                    print("INFO: Solução da raiz tornou-se inteira. Parando a geração de cortes.")
                    break

                new_cuts_data = find_cover_cuts(self.model, root_node.variable_values, self.original_vars, debug=True)

                if not new_cuts_data:
                    print("INFO: Nenhum novo Cover Cut encontrado. Finalizando a geração de cortes.")
                    break
                
                cuts_added_this_round = 0
                for cut_vars, cut_rhs in new_cuts_data:
                    
                    # 2. CRIAMOS UMA "ASSINATURA" ÚNICA PARA O CORTE
                    cut_signature = (tuple(sorted(cut_vars)), cut_rhs)

                    # 3. VERIFICAMOS SE O CORTE JÁ ESTÁ NO NOSSO POOL
                    if cut_signature not in added_cuts_pool:
                        # Se for novo, adicionamos ao modelo e ao pool
                        gurobi_vars = [self.model.getVarByName(v_name) for v_name in cut_vars]
                        # Usamos o tamanho do pool para garantir um nome único
                        cut_name = f"cover_cut_{len(added_cuts_pool)}" 
                        self.model.addConstr(quicksum(gurobi_vars) <= cut_rhs, name=cut_name)
                        
                        added_cuts_pool.add(cut_signature)
                        cuts_added_this_round += 1
                
                # 4. SE NÃO ADICIONAMOS NENHUM CORTE NOVO, PARAMOS
                if cuts_added_this_round == 0:
                    print("INFO: Nenhum corte NOVO encontrado nesta rodada. Finalizando a geração de cortes.")
                    break

                self.model.update()
                self._solve_lp_relaxation(root_node, self.lp_model_for_relaxations)
                print(f"INFO: Rodada {i+1} - Cortes novos adicionados: {cuts_added_this_round}. 'Best Bound' da raiz atualizado para: {root_node.lp_objective_value:.4f}")

            print("-" * 60)
            # --- FIM DO LOOP DE CORTES ---
            
        if root_node.lp_status == 'INFEASIBLE':
            self.logger.log_summary('Infeasible', 0, None, None)
            return

        # FASE 3: SETUP E LOOP PRINCIPAL DO BRANCH AND BOUND
        self.best_bound = root_node.lp_objective_value
        self.tree = Tree(root_node)
        #self.node_count = 0 # Inicializamos o contador aqui, antes do loop

        if root_node.is_integer:
            is_better = (root_node.lp_objective_value < self.incumbent_value) if self.model_sense == GRB.MINIMIZE else (root_node.lp_objective_value > self.incumbent_value)
            if is_better:
                 self.incumbent_value = root_node.lp_objective_value
                 self.incumbent_solution = root_node.variable_values

        while not self.tree.is_empty():
            self.node_count += 1
            # --- NOVO BLOCO DE DEPURAÇÃO DA TROCA ---
            #print(f"[DEBUG-Switch] Checando: Modo='{self.tree.mode}', "
            #      f"Contagem={self.node_count}, Limite={self.dfs_node_limit}. "
            #      f"Condição de troca: {self.tree.mode == 'DFS' and self.node_count > self.dfs_node_limit}")
            # --- FIM DO BLOCO DE DEPURAÇÃO ---

            if self.tree.mode == 'DFS' and self.node_count > self.dfs_node_limit:
                self.tree.switch_to_best_bound_mode()

            # O resto do loop continua como antes...
            current_node = self.tree.get_next_node()
            #print(f"[DEBUG-Busca] Processando Nó ID: {current_node.id}, ...")

            # --- GATILHO DA HEURÍSTICA RINS ---
            if self.rins_frequency > 0 and self.incumbent_solution is not None and (self.node_count % self.rins_frequency == 0):
                
                # --- CHAMADA CORRIGIDA E SIMPLIFICADA ---
                rins_model = self.original_milp_model.copy()
                new_solution = run_rins(
                    rins_model, 
                    self.incumbent_solution, 
                    current_node.variable_values,
                    self.incumbent_value # Passa o valor do incumbente diretamente
                )
                # --- FIM DA CORREÇÃO ---

                if new_solution:
                    # A lógica para atualizar o incumbente permanece a mesma
                    is_better = (new_solution['objective'] < self.incumbent_value) if self.model_sense == GRB.MINIMIZE else (new_solution['objective'] > self.incumbent_value)
                    if is_better:
                        print(f"INFO: [Solver] Heurística RINS melhorou o incumbente para: {new_solution['objective']:.4f}")
                        self.incumbent_value = new_solution['objective']
                        self.incumbent_solution = new_solution['solution']
                        self.logger.log_event(self.node_count, self.incumbent_value, self.best_bound, force_log=True)
            # --- FIM DO GATILHO DA RINS ---

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
            
            # Extrai apenas as vars fracionárias para a estratégia
            #fractional_vars = {k: v for k, v in current_node.variable_values.items() 
            #                   if self.original_vars.get(k) != GRB.CONTINUOUS and abs(v - round(v)) > 1e-6}
             
            # Cria um nó temporário apenas com as vars fracionárias para a estratégia
            #temp_node_for_strategy = Node(current_node.id, current_node.depth, current_node.local_bounds)
            #temp_node_for_strategy.variable_values = fractional_vars
            
            #branch_var_name = self.branching_strategy.select_variable(temp_node_for_strategy, self.model)
            branch_var_name = self.branching_strategy.select_variable(current_node, self.model, self.original_vars)
            
            
            if branch_var_name is None:
                continue

            val = current_node.variable_values[branch_var_name]

            # --- PRINT DE DEPURAÇÃO 1 ---
            #print("\n" + "="*20 + " DEBUG BRANCH " + "="*20)
            #print(f"Nó Pai ID: {current_node.id}, Obj: {current_node.lp_objective_value:.4f}")
            #print(f"Variável de Branching: '{branch_var_name}', Valor: {val:.4f}")
            #print(f"-> Criando filho 1 com bound: {branch_var_name} <= {math.floor(val)}")
            #print(f"-> Criando filho 2 com bound: {branch_var_name} >= {math.ceil(val)}")
            # --- FIM DO PRINT 1 ---
            
            bounds1 = {**current_node.local_bounds, branch_var_name: {'type': '<=', 'value': math.floor(val)}}
            child1 = Node(parent_id=current_node.id, depth=current_node.depth + 1, local_bounds=bounds1)
            # Passe o modelo LP
            self._solve_lp_relaxation(child1, self.lp_model_for_relaxations)

            bounds2 = {**current_node.local_bounds, branch_var_name: {'type': '>=', 'value': math.ceil(val)}}
            child2 = Node(parent_id=current_node.id, depth=current_node.depth + 1, local_bounds=bounds2)
            # Passe o modelo LP
            self._solve_lp_relaxation(child2, self.lp_model_for_relaxations)

            self.branching_strategy.update_scores(parent_node=current_node, child_nodes=[child1, child2], branch_var=branch_var_name)
            
            # --- PRINT DE DEPURAÇÃO 2 ---
            #print(f"Status Filho 1 (<=): {child1.lp_status}, Obj: {child1.lp_objective_value}")
            #print(f"Status Filho 2 (>=): {child2.lp_status}, Obj: {child2.lp_objective_value}")
            #print("="*54 + "\n")
            # --- FIM DO PRINT 2 ---

            new_nodes = []
            if child1.lp_status == 'OPTIMAL':
                is_promising1 = (child1.lp_objective_value < self.incumbent_value) if self.model_sense == GRB.MINIMIZE else (child1.lp_objective_value > self.incumbent_value)
                if is_promising1: new_nodes.append(child1)
            
            if child2.lp_status == 'OPTIMAL':
                is_promising2 = (child2.lp_objective_value < self.incumbent_value) if self.model_sense == GRB.MINIMIZE else (child2.lp_objective_value > self.incumbent_value)
                if is_promising2: new_nodes.append(child2)
            
            self.tree.add_nodes(new_nodes)
            
            if not self.tree.is_empty():
                self.best_bound = self.tree.get_current_best_bound()
            else:
                self.best_bound = self.incumbent_value

            self.logger.log_event(self.node_count, self.incumbent_value, self.best_bound)

        # FINALIZAÇÃO
        self.logger.log_summary('Optimal solution found' if self.incumbent_value != float('inf') else 'No solution found', 
                                self.node_count, self.incumbent_value, self.best_bound)