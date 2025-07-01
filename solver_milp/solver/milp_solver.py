# solver/milp_solver.py (VERSÃO FINAL CORRIGIDA - COM GOMORY E COVER CUTS)

import logging
import numpy as np
import math
import heapq
from dataclasses import dataclass, field
from gurobipy import GRB
import time
from typing import List

# Importações de módulos do nosso próprio projeto
from .problem import Problema
from .lp_interface import solve_lp_gurobi
from .cuts.gomory import gerar_cortes_gomory
from .cuts.cover import gerar_cover_cuts
from .heuristics import heuristica_de_arredondamento, heuristica_feasibility_pump
from .problem import Problema

class Node:
    def __init__(self, node_id: int, estimativa: float, solucao_lp: dict, base_info: dict,
                 local_constraints: list, profundidade: int, status: str = 'PENDENTE'):

        self.id = node_id
        self.estimativa = estimativa
        
        # A MUDANÇA ESTÁ AQUI: Guardamos a solução junto com o nó.
        self.solucao_lp = solucao_lp
        self.base_info = base_info
        
        self.local_constraints = local_constraints
        self.profundidade = profundidade
        self.status = status

    def __repr__(self) -> str:
        """Representação em string para facilitar a depuração."""
        return (f"Nó(id={self.id}, est={self.estimativa:.2f}, "
                f"profundidade={self.profundidade}, constraints={len(self.local_constraints)})")

class MilpSolver:

    def __init__(self, problema: Problema, time_limit_seconds: int = 60):
        self.problema_raiz = problema
        self.melhor_solucao = None
        
        if self.problema_raiz.model.ModelSense == GRB.MAXIMIZE:
            self.melhor_limite_global = -float('inf')
        else:
            self.melhor_limite_global = float('inf')
        
        self.contador_nos = 0
        self.fronteira_aberta: List[Node] = []
        self.fase_de_busca = 'DFS' 
        self.time_limit = time_limit_seconds

        # Memória para o Branching por Confiabilidade (se estiver usando)
        self.pseudo_costs_up = {}
        self.pseudo_costs_down = {}
        self.pseudo_cost_counts = {}

        logging.info(f"Classe MilpSolver inicializada. Sentido: {'MAXIMIZE' if self.problema_raiz.model.ModelSense == GRB.MAXIMIZE else 'MINIMIZE'}.")
        
        # Analisa o problema em busca de restrições candidatas para Cover Cuts
        self.indices_cover_candidates = []
        for i, constr in enumerate(self.problema_raiz.model.getConstrs()):
            if constr.Sense == GRB.LESS_EQUAL:
                linha = self.problema_raiz.model.getRow(constr)
                # Garante que temos variáveis na linha para evitar erro no .getVar()
                if linha.size() > 0:
                    vars_na_linha_nomes = {linha.getVar(k).VarName for k in range(linha.size())}
                    if all(v in self.problema_raiz.variaveis_inteiras_nomes for v in vars_na_linha_nomes):
                        self.indices_cover_candidates.append(i)
        
        logging.info(f"Análise estrutural encontrou {len(self.indices_cover_candidates)} restrições candidatas para Cover Cuts.")

    def _log_header(self):
        header = (
            "   Nodes    |    Current Node    |     Objective Bounds      |     Work\n"
            " Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time"
        )
        logging.info(header)

    def _log_progress(self, start_time, current_node, is_heuristic=False):
        expl, unexpl = self.contador_nos, len(self.fronteira_aberta)
        incumbent = self.melhor_limite_global
        
        if not self.fronteira_aberta:
            best_bound = incumbent
        else:
            best_bound = min(n.estimativa for n in self.fronteira_aberta) if self.problema_raiz.model.ModelSense == GRB.MINIMIZE else max(n.estimativa for n in self.fronteira_aberta)

        gap = float('inf')
        if abs(incumbent) > 1e-9 and incumbent != float('inf') and incumbent != -float('inf'):
             gap = abs(incumbent - best_bound) / abs(incumbent)

        if is_heuristic:
            log_line = f"H{expl:<5}{unexpl:<7} {'':21s} {incumbent:<12.4f} {best_bound:<8.4f} {gap:6.2%}"
        else:
            incumbent_str = f"{incumbent:.4f}" if incumbent != float('inf') and incumbent != -float('inf') else " - "
            gap_str = f"{gap:6.2%}" if gap != float('inf') else " - "
            log_line = (f"{expl:>5} {unexpl:<5} {current_node.estimativa:>12.4f} {current_node.profundidade:>4d} {'':7s} {incumbent_str:>12s} {best_bound:>8.4f} {gap_str:>5s} {'':7s} {time.time() - start_time:4.0f}s")
        
        logging.info(log_line)

    def _checar_inteireza(self, solucao: dict):
        """Versão rápida (most fractional) para a heurística."""
        TOLERANCIA = 1e-6; melhor_var = None; max_frac = -1.0
        for var_nome in self.problema_raiz.variaveis_inteiras_nomes:
            valor = solucao.get(var_nome, 0.0)
            if abs(valor - round(valor)) > TOLERANCIA:
                proximidade = 0.5 - abs((valor - math.floor(valor)) - 0.5)
                if proximidade > max_frac:
                    max_frac = proximidade
                    melhor_var = (var_nome, valor)
        return melhor_var if melhor_var else (None, None)

    def _select_branching_variable_hybrid(self, solucao_lp: dict, local_constraints: list, obj_pai: float):
        """Estratégia híbrida: Strong Branching para aprender, Pseudocustos para agir."""
        TOLERANCIA, RELIABILITY_THRESHOLD = 1e-6, 5
        candidatas = [ (vn, solucao_lp.get(vn,0.0)) for vn in self.problema_raiz.variaveis_inteiras_nomes if abs(solucao_lp.get(vn,0.0) - round(solucao_lp.get(vn,0.0))) > TOLERANCIA ]
        if not candidatas: return None, None

        melhor_var, melhor_pontuacao = None, -1.0
        for var_nome, valor_fracionario in candidatas:
            contagem = self.pseudo_cost_counts.get(var_nome, 0)
            if contagem < RELIABILITY_THRESHOLD:
                #logging.debug(f"Fazendo Strong Branching em '{var_nome}' (contagem={contagem})...")
                lp_res_baixo = solve_lp_gurobi(self.problema_raiz, local_constraints + [(var_nome, '<=', math.floor(valor_fracionario))])
                lp_res_cima = solve_lp_gurobi(self.problema_raiz, local_constraints + [(var_nome, '>=', math.ceil(valor_fracionario))])
                degradacao_baixo = abs(lp_res_baixo['objective'] - obj_pai) if lp_res_baixo['status'] == 'OPTIMAL' else self.pseudo_costs_down.get(var_nome, 0)
                degradacao_cima = abs(lp_res_cima['objective'] - obj_pai) if lp_res_cima['status'] == 'OPTIMAL' else self.pseudo_costs_up.get(var_nome, 0)
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
        if len(local_constraints) > profundidade_max: return None
        lp_resultado = solve_lp_gurobi(self.problema_raiz, local_constraints)
        if lp_resultado['status'] != 'OPTIMAL': return None
        var_para_ramificar, _ = self._checar_inteireza(lp_resultado['solution'])
        if var_para_ramificar is None: return lp_resultado
        valor_fracionario = lp_resultado['solution'][var_para_ramificar]
        resultado = self._heuristica_mergulho_dfs(local_constraints + [(var_para_ramificar, '<=', math.floor(valor_fracionario))], profundidade_max)
        if resultado: return resultado
        return self._heuristica_mergulho_dfs(local_constraints + [(var_para_ramificar, '>=', math.ceil(valor_fracionario))], profundidade_max)

# CÓDIGO NOVO E COMPLETO para o método solve() em MilpSolver

    def solve(self):
        start_time = time.time()
        logging.info("========================= INICIANDO PROCESSO DE BRANCH AND CUT =========================")
        
        # Heurística inicial para obter um primeiro incumbente
        resultado_heuristico = self._heuristica_mergulho_dfs([])
        if resultado_heuristico:
            self.melhor_solucao = resultado_heuristico['solution']
            self.melhor_limite_global = resultado_heuristico['objective']
        
        # Nó Raiz
        lp_resultado_raiz = solve_lp_gurobi(self.problema_raiz, [])
        if lp_resultado_raiz['status'] != "OPTIMAL":
            logging.error(f"O problema não pode ser resolvido porque a relaxação LP da raiz é '{lp_resultado_raiz['status']}'.")
            logging.error("Isso geralmente significa que o problema original não tem solução viável ou é mal formulado.")
            return None, None # Encerra a execução
        
        no_raiz = Node(
            node_id=self.contador_nos, 
            estimativa=lp_resultado_raiz['objective'], 
            solucao_lp=lp_resultado_raiz['solution'],
            base_info=lp_resultado_raiz['base_info'], # <-- ADICIONE AQUI
            local_constraints=[], 
            profundidade=0
        )
        self.contador_nos += 1; self.fronteira_aberta.append(no_raiz)
        
        # Log Inicial
        self._log_header()
        self._log_progress(start_time, no_raiz, is_heuristic=bool(resultado_heuristico))

        last_log_time = time.time()
        LOG_INTERVAL_SECONDS = 2.0 # Imprime uma atualização a cada 2 segundos

        # --- LOOP PRINCIPAL DE BRANCH & CUT ---
        while self.fronteira_aberta:
            if time.time() - start_time > self.time_limit:
                logging.warning(f"Limite de tempo de {self.time_limit}s atingido.")
                break
            
            # Seleção de Nó
            #current_node = self.fronteira_aberta.pop() # Começa com DFS

            # Seleção de Nó
            if self.fase_de_busca == 'DFS':
                current_node = self.fronteira_aberta.pop()
            else:
                if not self.fronteira_aberta: break
                best_idx = min(range(len(self.fronteira_aberta)), key=lambda i: self.fronteira_aberta[i].estimativa) if self.problema_raiz.model.ModelSense == GRB.MINIMIZE else max(range(len(self.fronteira_aberta)), key=lambda i: self.fronteira_aberta[i].estimativa)
                current_node = self.fronteira_aberta.pop(best_idx)

            if time.time() - last_log_time > LOG_INTERVAL_SECONDS:
                self._log_progress(start_time, current_node)
                last_log_time = time.time()

            # Poda por Limite
            if (self.problema_raiz.model.ModelSense == GRB.MINIMIZE and current_node.estimativa >= self.melhor_limite_global): continue
            
            # --- FASE DE CORTES ---
            MAX_RODADAS_CORTE, VIOLATION_TOLERANCE = 5, 1e-4
            node_foi_podado_nos_cortes = False

            for rodada in range(MAX_RODADAS_CORTE):
                solucao_atual_lp = current_node.solucao_lp
                
                var_frac, _ = self._checar_inteireza(solucao_atual_lp)
                if var_frac is None: break

                # --- GERAÇÃO DE CORTES (GOMORY + COVER) ---
                # A chamada para Gomory agora é possível e eficiente
                potenciais_cortes_gomory = gerar_cortes_gomory(self.problema_raiz, solucao_atual_lp, current_node.base_info)
                potenciais_cortes_cover = gerar_cover_cuts(self.problema_raiz, solucao_atual_lp, self.indices_cover_candidates)
                todos_potenciais_cortes = potenciais_cortes_gomory + potenciais_cortes_cover
                
                # Filtra cortes pela utilidade
                cortes_uteis = []
                solucao_array = np.array(list(solucao_atual_lp.values()))
                
                for corte in todos_potenciais_cortes:
                    lhs_val = np.dot(corte['coeffs'], solucao_array)
                    violacao = (lhs_val - corte['rhs']) if corte['sentido'] == '<=' else (corte['rhs'] - lhs_val)
                    if violacao > VIOLATION_TOLERANCE:
                        cortes_uteis.append(corte)

                if not cortes_uteis: break
                
                logging.debug(f"Nó {current_node.id}, Rodada {rodada + 1}: Adicionando {len(cortes_uteis)} corte(s).")
                
                lp_res_cortado = solve_lp_gurobi(self.problema_raiz, current_node.local_constraints, cortes_uteis)
                
                if lp_res_cortado['status'] == 'OPTIMAL':
                    current_node.estimativa = lp_res_cortado['objective']
                    current_node.solucao_lp = lp_res_cortado['solution']
                    current_node.base_info = lp_res_cortado['base_info'] # Atualiza a base também!
                    if (self.problema_raiz.model.ModelSense == GRB.MINIMIZE and current_node.estimativa >= self.melhor_limite_global):
                        node_foi_podado_nos_cortes = True; break
                else:
                    node_foi_podado_nos_cortes = True; break
            
            if node_foi_podado_nos_cortes:
                logging.debug(f"Nó {current_node.id} podado na fase de cortes.")
                continue

            # --- VERIFICAÇÃO DE INTEIREZA E RAMIFICAÇÃO ---
            var_para_ramificar, valor_fracionario = self._checar_inteireza(current_node.solucao_lp)
            
            if var_para_ramificar is None: # Solução é inteira
                
                # Calculamos o valor do objetivo manualmente, como já fizemos antes.
                objetivo_expr = self.problema_raiz.model.getObjective()
                valor_obj_inteiro = objetivo_expr.getConstant()
                for i in range(objetivo_expr.size()):
                    var_name = objetivo_expr.getVar(i).VarName
                    # Usamos a solução do nó atual que já sabemos ser inteira
                    var_value = current_node.solucao_lp.get(var_name, 0.0) 
                    valor_obj_inteiro += objetivo_expr.getCoeff(i) * var_value

                # Se a nova solução for melhor, atualizamos o incumbente
                if (self.problema_raiz.model.ModelSense == GRB.MINIMIZE and valor_obj_inteiro < self.melhor_limite_global):
                    self.melhor_limite_global = valor_obj_inteiro
                    self.melhor_solucao = current_node.solucao_lp
                    self._log_progress(start_time, current_node, is_heuristic=True) # Loga a nova solução
                
                # Poda por solução. Já que é inteira, não precisa ramificar.
                continue

            # Ramificação (usando a regra mais simples 'most fractional' por enquanto)
            for i in range(2):
                sentido, valor_limite = ('<=', math.floor(valor_fracionario)) if i == 0 else ('>=', math.ceil(valor_fracionario))
                restricoes_filho = current_node.local_constraints + [(var_para_ramificar, sentido, valor_limite)]
                lp_res_filho = solve_lp_gurobi(self.problema_raiz, restricoes_filho)
                if lp_res_filho['status'] == 'OPTIMAL':
                    novo_no = Node(
                        node_id=self.contador_nos, 
                        estimativa=lp_res_filho['objective'],
                        solucao_lp=lp_res_filho['solution'],
                        base_info=lp_res_filho['base_info'], # <-- ADICIONE AQUI
                        local_constraints=restricoes_filho,
                        profundidade=current_node.profundidade + 1
                    )
                    self.fronteira_aberta.append(novo_no)
                    self.contador_nos += 1
        
        logging.info("Execução finalizada.")
        return self.melhor_solucao, self.melhor_limite_global