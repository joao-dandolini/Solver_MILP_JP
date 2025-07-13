# solver/milp_solver.py (VERSÃO INTEGRADA E CORRIGIDA)

import logging
import numpy as np
import math
import time
from typing import List

from gurobipy import GRB

from .problem import Problema
from .lp_interface import solve_lp_gurobi
from .cuts.gomory import gerar_cortes_gomory
from .cuts.cover import gerar_cover_cuts
# A heurística de mergulho está neste arquivo, mas importamos as outras
import heapq
from .heuristics import heuristica_de_arredondamento, heuristica_feasibility_pump

# A classe Node permanece a mesma
class Node:
    def __init__(self, node_id: int, estimativa: float, solucao_lp: dict, base_info: dict,
                 local_constraints: list, profundidade: int, status: str = 'PENDENTE'):
        self.id = node_id
        self.estimativa = estimativa
        self.solucao_lp = solucao_lp
        self.base_info = base_info
        self.local_constraints = local_constraints
        self.profundidade = profundidade
        self.status = status

    def __repr__(self) -> str:
        return (f"Nó(id={self.id}, est={self.estimativa:.2f}, "
                f"profundidade={self.profundidade}, constraints={len(self.local_constraints)})")

class MilpSolver:
    # __init__ e as outras funções auxiliares (_log_header, _checar_inteireza, etc.) permanecem as mesmas
    # A única mudança é no método `solve()`

    def __init__(self, problema: Problema, time_limit_seconds: int = 60):
        self.problema_raiz = problema
        self.melhor_solucao = None
        
        if self.problema_raiz.model.ModelSense == GRB.MAXIMIZE:
            self.melhor_limite_global = -float('inf')
        else:
            self.melhor_limite_global = float('inf')
        
        self.contador_nos = 0
        self.fronteira_aberta = [] # Agora será um heap
        self.fase_de_busca = 'DFS' 
        self.time_limit = time_limit_seconds
        self.pseudo_costs_up = {}
        self.pseudo_costs_down = {}
        self.pseudo_cost_counts = {}
        logging.info(f"Classe MilpSolver inicializada. Sentido: {'MAXIMIZE' if self.problema_raiz.model.ModelSense == GRB.MAXIMIZE else 'MINIMIZE'}.")
        
        self.indices_cover_candidates = []
        for i, constr in enumerate(self.problema_raiz.model.getConstrs()):
            if constr.Sense == GRB.LESS_EQUAL:
                linha = self.problema_raiz.model.getRow(constr)
                if linha.size() > 0:
                    vars_na_linha_nomes = {linha.getVar(k).VarName for k in range(linha.size())}
                    if all(v in self.problema_raiz.variaveis_inteiras_nomes for v in vars_na_linha_nomes):
                        self.indices_cover_candidates.append(i)
        logging.info(f"Análise estrutural encontrou {len(self.indices_cover_candidates)} restrições candidatas para Cover Cuts.")

    def _log_header(self):
        header = (
            "   Nodes    |    Current Node    |      Objective Bounds        |     Work\n"
            " Expl Unexpl |  Obj   Depth IntInf | Incumbent     BestBd    Gap | It/Node Time"
        )
        logging.info(header)

    def _log_progress(self, start_time, current_node, is_heuristic=False):
        # Esta função permanece a mesma da sua versão original
        expl, unexpl = self.contador_nos, len(self.fronteira_aberta)
        incumbent = self.melhor_limite_global
        
        if not self.fronteira_aberta:
            best_bound = incumbent
        else:
            # Com o heap, o melhor nó é sempre o primeiro item. Isso é muito mais rápido.
            if self.problema_raiz.model.ModelSense == GRB.MINIMIZE:
                # A prioridade (item [0]) na tupla já é a estimativa correta.
                best_bound = self.fronteira_aberta[0][0]
            else: # MAXIMIZE
                # Lembre-se que para MAXIMIZE, armazenamos a prioridade como -estimativa.
                # Portanto, precisamos inverter o sinal para obter o valor real.
                best_bound = -self.fronteira_aberta[0][0]

        gap = float('inf')
        if abs(incumbent) > 1e-9 and incumbent not in [float('inf'), -float('inf')]:
             gap = abs(incumbent - best_bound) / (abs(incumbent) + 1e-9)

        if is_heuristic:
            log_line = f"H{expl:<5}{unexpl:<7} {'':21s} {incumbent:<12.4f} {best_bound:<8.4f} {gap:6.2%}"
        else:
            incumbent_str = f"{incumbent:.4f}" if incumbent not in [float('inf'), -float('inf')] else " - "
            gap_str = f"{gap:6.2%}" if gap != float('inf') else " - "
            log_line = (f"{expl:>5} {unexpl:<5} {current_node.estimativa:>12.4f} {current_node.profundidade:>4d} {'':7s} {incumbent_str:>12s} {best_bound:>8.4f} {gap_str:>5s} {'':7s} {time.time() - start_time:4.0f}s")
        
        logging.info(log_line)

    def _checar_inteireza(self, solucao: dict):
        TOLERANCIA = 1e-6
        for var_nome in self.problema_raiz.variaveis_inteiras_nomes:
            valor = solucao.get(var_nome, 0.0)
            if abs(valor - round(valor)) > TOLERANCIA:
                return (var_nome, valor) # Retorna a primeira que encontrar
        return (None, None)

    def _select_branching_variable_hybrid(self, solucao_lp: dict, local_constraints: list, obj_pai: float):
        # Esta função permanece a mesma da sua versão original
        TOLERANCIA, RELIABILITY_THRESHOLD = 1e-6, 3
        candidatas = [ (vn, solucao_lp.get(vn,0.0)) for vn in self.problema_raiz.variaveis_inteiras_nomes if abs(solucao_lp.get(vn,0.0) - round(solucao_lp.get(vn,0.0))) > TOLERANCIA ]
        if not candidatas: return None
        
        melhor_var, melhor_pontuacao = None, -1.0
        for var_nome, valor_fracionario in candidatas:
            contagem = self.pseudo_cost_counts.get(var_nome, 0)
            if contagem < RELIABILITY_THRESHOLD:
                lp_res_baixo = solve_lp_gurobi(self.problema_raiz, local_constraints + [(var_nome, '<=', math.floor(valor_fracionario))])
                lp_res_cima = solve_lp_gurobi(self.problema_raiz, local_constraints + [(var_nome, '>=', math.ceil(valor_fracionario))])
                degradacao_baixo = abs(lp_res_baixo['objective'] - obj_pai) if lp_res_baixo['status'] == 'OPTIMAL' else self.pseudo_costs_down.get(var_nome, 1.0)
                degradacao_cima = abs(lp_res_cima['objective'] - obj_pai) if lp_res_cima['status'] == 'OPTIMAL' else self.pseudo_costs_up.get(var_nome, 1.0)
                self.pseudo_costs_down[var_nome] = (self.pseudo_costs_down.get(var_nome, 0) * contagem + degradacao_baixo) / (contagem + 1)
                self.pseudo_costs_up[var_nome] = (self.pseudo_costs_up.get(var_nome, 0) * contagem + degradacao_cima) / (contagem + 1)
                self.pseudo_cost_counts[var_nome] = contagem + 1
            
            f_j = valor_fracionario - math.floor(valor_fracionario)
            pc_baixo, pc_cima = self.pseudo_costs_down.get(var_nome, 1e-6), self.pseudo_costs_up.get(var_nome, 1e-6)
            pontuacao = (1 - f_j) * pc_baixo + f_j * pc_cima
            if pontuacao > melhor_pontuacao:
                melhor_pontuacao, melhor_var = pontuacao, (var_nome, valor_fracionario)
        return melhor_var

    def _heuristica_mergulho_dfs(self, local_constraints: list, profundidade_max: int = 50):
        # Esta função permanece a mesma da sua versão original
        if len(local_constraints) > profundidade_max: return None
        lp_resultado = solve_lp_gurobi(self.problema_raiz, local_constraints)
        if lp_resultado['status'] != 'OPTIMAL': return None

        var_para_ramificar, valor_frac = self._checar_inteireza(lp_resultado['solution'])
        if var_para_ramificar is None: return lp_resultado
        
        # Tenta arredondar para o lado mais próximo primeiro
        if (valor_frac - math.floor(valor_frac)) < 0.5:
            # Primeiro mergulha para baixo
            resultado = self._heuristica_mergulho_dfs(local_constraints + [(var_para_ramificar, '<=', math.floor(valor_frac))], profundidade_max)
            if resultado: return resultado
            # Se falhar, tenta para cima
            return self._heuristica_mergulho_dfs(local_constraints + [(var_para_ramificar, '>=', math.ceil(valor_frac))], profundidade_max)
        else:
            # Primeiro mergulha para cima
            resultado = self._heuristica_mergulho_dfs(local_constraints + [(var_para_ramificar, '>=', math.ceil(valor_frac))], profundidade_max)
            if resultado: return resultado
            # Se falhar, tenta para baixo
            return self._heuristica_mergulho_dfs(local_constraints + [(var_para_ramificar, '<=', math.floor(valor_frac))], profundidade_max)


    def solve(self):
        start_time = time.time()
        logging.info("="*25 + " INICIANDO PROCESSO DE BRANCH AND CUT " + "="*25)

        # 1. RESOLVER O NÓ RAIZ
        lp_resultado_raiz = solve_lp_gurobi(self.problema_raiz, [])
        if lp_resultado_raiz['status'] != "OPTIMAL":
            logging.error(f"Problema inviável ou ilimitado na raiz. Status: {lp_resultado_raiz['status']}.")
            return None, None

        # 2. HEURÍSTICA INICIAL (USANDO A SUA ESTRATÉGIA DE MERGULHO EFICAZ)
        logging.info("Executando heurística de mergulho (Diving) para encontrar solução inicial...")
        resultado_heuristico = self._heuristica_mergulho_dfs([])
        if resultado_heuristico and resultado_heuristico['status'] == 'OPTIMAL':
            self.melhor_solucao = resultado_heuristico['solution']
            self.melhor_limite_global = resultado_heuristico['objective']
            logging.info(f"Solução inicial encontrada pela heurística de mergulho! Obj: {self.melhor_limite_global:.4f}")

        # 3. SETUP DO LOOP PRINCIPAL
        no_raiz = Node(self.contador_nos, lp_resultado_raiz['objective'], lp_resultado_raiz['solution'], lp_resultado_raiz['base_info'], [], 0)
        self.contador_nos += 1
        # Adiciona o nó raiz ao heap usando o formato correto (prioridade, contador, objeto)
        prioridade_raiz = no_raiz.estimativa if self.problema_raiz.model.ModelSense == GRB.MINIMIZE else -no_raiz.estimativa
        # Usamos o contador_nos atual (que é 1 neste ponto) como desempatador
        heapq.heappush(self.fronteira_aberta, (prioridade_raiz, self.contador_nos, no_raiz))
        self._log_header()
        self._log_progress(start_time, no_raiz, is_heuristic=(self.melhor_solucao is not None))

        last_log_time = time.time()
        
        # 4. LOOP PRINCIPAL DE BRANCH & CUT
        while self.fronteira_aberta:
            if time.time() - start_time > self.time_limit:
                logging.warning(f"Limite de tempo de {self.time_limit}s atingido.")
                break

# --- SELEÇÃO DE NÓ CORRIGIDA E OTIMIZADA COM HEAPQ ---
            if not self.fronteira_aberta:
                break

            # A fila de prioridade (heap) já faz a busca pelo melhor nó para nós.
            # Simplesmente retiramos o nó mais promissor e desempacotamos a tupla.
            _, _, current_node = heapq.heappop(self.fronteira_aberta)
            # --- FIM DA CORREÇÃO ---

            # Poda por Limite
            if (self.problema_raiz.model.ModelSense == GRB.MINIMIZE and current_node.estimativa >= self.melhor_limite_global):
                continue

            # Fase de Cortes
            for _ in range(5): # Loop de cortes
                var_frac, _ = self._checar_inteireza(current_node.solucao_lp)
                if var_frac is None: break

                cortes_gomory = gerar_cortes_gomory(self.problema_raiz, current_node.solucao_lp, current_node.base_info)
                cortes_cover = gerar_cover_cuts(self.problema_raiz, current_node.solucao_lp, self.indices_cover_candidates)
                todos_potenciais_cortes = cortes_gomory + cortes_cover
                
                # --- FILTRO DE VIOLAÇÃO RESTAURADO ---
                cortes_uteis = []
                # Pega a solução atual do nó para checar violação
                solucao_atual_dict = current_node.solucao_lp
                
                for corte in todos_potenciais_cortes:
                    # Calcula o valor do lado esquerdo (LHS) do corte com a solução atual
                    lhs_val = corte.get('coeffs', np.zeros(len(solucao_atual_dict))) @ np.array(list(solucao_atual_dict.values()))

                    violacao = 0.0
                    if corte['sentido'] == '<=':
                        violacao = lhs_val - corte['rhs']
                    else: # >=
                        violacao = corte['rhs'] - lhs_val
                    
                    # Adiciona apenas se o corte for violado por uma tolerância
                    if violacao > 1e-6:
                        cortes_uteis.append(corte)
                # --- FIM DO FILTRO ---
                
                if not cortes_uteis: break

                lp_res_cortado = solve_lp_gurobi(self.problema_raiz, current_node.local_constraints, cortes_uteis)
                if lp_res_cortado['status'] == 'OPTIMAL':
                    current_node.estimativa = lp_res_cortado['objective']
                    current_node.solucao_lp = lp_res_cortado['solution']
                    current_node.base_info = lp_res_cortado['base_info']
                else:
                    current_node.status = 'PODADO_CORTE'
                    break
            if current_node.status == 'PODADO_CORTE': continue

            # Verificação de Inteireza
            var_para_ramificar, valor_fracionario = self._checar_inteireza(current_node.solucao_lp)
            if var_para_ramificar is None:
                if (self.problema_raiz.model.ModelSense == GRB.MINIMIZE and current_node.estimativa < self.melhor_limite_global):
                    self.melhor_limite_global = current_node.estimativa
                    self.melhor_solucao = current_node.solucao_lp
                    self._log_progress(start_time, current_node, is_heuristic=True)
                continue

            # Ramificação com a regra avançada
            var_branch_tupla = self._select_branching_variable_hybrid(current_node.solucao_lp, current_node.local_constraints, current_node.estimativa)
            if not var_branch_tupla: continue # Não achou em quem ramificar
            var_para_ramificar, valor_fracionario = var_branch_tupla

            # Cria os nós filhos
            for i in range(2):
                sentido, valor_limite = ('<=', math.floor(valor_fracionario)) if i == 0 else ('>=', math.ceil(valor_fracionario))
                restricoes_filho = current_node.local_constraints + [(var_para_ramificar, sentido, valor_limite)]
                
                # Resolvendo o LP do filho com WARM START
                lp_res_filho = solve_lp_gurobi(self.problema_raiz, restricoes_filho, warm_start_info=current_node.base_info)
                
                if lp_res_filho['status'] == 'OPTIMAL':
                    if (self.problema_raiz.model.ModelSense == GRB.MINIMIZE and lp_res_filho['objective'] < self.melhor_limite_global):
                        novo_no = Node(self.contador_nos, lp_res_filho['objective'], lp_res_filho['solution'], lp_res_filho['base_info'], restricoes_filho, current_node.profundidade + 1)
                        # Adiciona ao heap com prioridade. Para MIN, a prioridade é a estimativa.
                        # Para MAX, usamos o negativo para que o menor valor (maior estimativa) saia primeiro.
                        prioridade = novo_no.estimativa if self.problema_raiz.model.ModelSense == GRB.MINIMIZE else -novo_no.estimativa
                        heapq.heappush(self.fronteira_aberta, (prioridade, self.contador_nos, novo_no))
                        self.contador_nos += 1
            
            # Log de progresso periódico
            if self.contador_nos % 100 == 0 and time.time() - last_log_time > 2.0:
                self._log_progress(start_time, current_node)
                last_log_time = time.time()

        logging.info("Execução finalizada.")
        return self.melhor_solucao, self.melhor_limite_global