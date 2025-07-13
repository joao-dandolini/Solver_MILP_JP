# solver/heuristics.py (VERSÃO REFATORADA PARA COMPATIBILIDADE)

import logging
import gurobipy as gp
from gurobipy import GRB
from typing import Tuple, Optional, Dict

# MUDANÇA: Não precisamos mais de numpy aqui, vamos trabalhar com dicionários e Gurobi.
from .problem import Problema
from .lp_interface import solve_lp_gurobi

# --- FUNÇÃO AUXILIAR REFATORADA ---
def _checar_viabilidade_heuristica(problema: Problema, solucao_dict: Dict[str, float]) -> bool:
    """
    Função auxiliar refatorada para verificar se uma solução inteira candidata (em formato de dicionário) é viável.
    """
    TOLERANCIA = 1e-6
    
    # Itera sobre todas as restrições do modelo
    for constr in problema.model.getConstrs():
        # Pega a expressão linear do lado esquerdo da restrição
        lhs_expr = problema.model.getRow(constr)
        lhs_val = lhs_expr.getConstant() # Começa com a constante da expressão, se houver

        # Calcula o valor da expressão com a solução candidata
        for i in range(lhs_expr.size()):
            var = lhs_expr.getVar(i)
            coeff = lhs_expr.getCoeff(i)
            lhs_val += coeff * solucao_dict.get(var.VarName, 0.0)

        rhs = constr.RHS
        sense = constr.Sense

        # Compara com o lado direito, respeitando o sentido da restrição
        if (sense == GRB.LESS_EQUAL and lhs_val > rhs + TOLERANCIA) or \
           (sense == GRB.GREATER_EQUAL and lhs_val < rhs - TOLERANCIA) or \
           (sense == GRB.EQUAL and abs(lhs_val - rhs) > TOLERANCIA):
            return False # Se qualquer restrição for violada, a solução é inviável
            
    return True # Se todas as restrições forem satisfeitas, a solução é viável

# --- HEURÍSTICA DE ARREDONDAMENTO REFATORADA ---
def heuristica_de_arredondamento(problema: Problema, solucao_lp_dict: Dict[str, float]) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    """Tenta encontrar uma solução inteira viável arredondando uma solução de LP (dicionário)."""
    if not solucao_lp_dict:
        return None, None

    logging.debug("Heurística: Tentando arredondamento simples...")
    solucao_arredondada = {name: round(val) for name, val in solucao_lp_dict.items()}
    
    if _checar_viabilidade_heuristica(problema, solucao_arredondada):
        # Se for viável, calcula o valor do objetivo original
        obj_expr = problema.model.getObjective()
        valor_obj = obj_expr.getConstant()
        for i in range(obj_expr.size()):
            var = obj_expr.getVar(i)
            coeff = obj_expr.getCoeff(i)
            valor_obj += coeff * solucao_arredondada.get(var.VarName, 0.0)
            
        logging.info(f"Heurística de arredondamento encontrou solução viável com valor {valor_obj:.4f}.")
        return solucao_arredondada, valor_obj
    else:
        logging.debug("Heurística: Solução arredondada não é viável.")
        return None, None

# --- FEASIBILITY PUMP REFATORADA ---
def heuristica_feasibility_pump(problema: Problema, solucao_lp_dict: Dict[str, float], max_iter: int = 10) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    """Implementa a heurística Feasibility Pump, agora compatível com a arquitetura Gurobi."""
    if not solucao_lp_dict:
        return None, None

    x_k_dict = solucao_lp_dict.copy()
    nomes_vars_inteiras = problema.variaveis_inteiras_nomes

    for k in range(max_iter):
        # Arredonda a solução atual para o inteiro mais próximo
        x_int_k_dict = {name: round(val) for name, val in x_k_dict.items()}
        
        # Checa a viabilidade da solução inteira
        if _checar_viabilidade_heuristica(problema, x_int_k_dict):
            logging.info(f"Heurística (Pump) encontrou uma solução viável na iteração {k+1}!")
            # Calcula o valor do objetivo original
            obj_expr = problema.model.getObjective()
            valor_obj = obj_expr.getConstant()
            for i in range(obj_expr.size()):
                var = obj_expr.getVar(i)
                coeff = obj_expr.getCoeff(i)
                valor_obj += coeff * x_int_k_dict.get(var.VarName, 0.0)

            return x_int_k_dict, valor_obj
            
        # --- MUDANÇA CENTRAL: CONSTRUÇÃO DO SUBPROBLEMA DE PROJEÇÃO ---
        model_copy = problema.model.copy()
        model_copy.setParam('OutputFlag', 0)
        
        # Cria a nova função objetivo para minimizar a distância L1
        nova_obj = gp.LinExpr()
        for var_nome in nomes_vars_inteiras:
            var = model_copy.getVarByName(var_nome)
            # Se nossa tentativa arredondada é 0, queremos minimizar x. Coeficiente = 1.
            # Se nossa tentativa é 1, queremos minimizar 1-x, que é o mesmo que maximizar x,
            # ou minimizar -x. Coeficiente = -1.
            if x_int_k_dict.get(var_nome, 0.0) < 0.5:
                nova_obj += 1.0 * var
            else:
                nova_obj += -1.0 * var
        
        # Define o novo objetivo no modelo copiado
        model_copy.setObjective(nova_obj, GRB.MINIMIZE)
        
        # Cria um problema temporário para passar para a interface
        problema_projecao = Problema(problema_input=model_copy, nome_problema="feasibility_pump_proj")
        
        # Resolve o LP de projeção
        resultado_projecao = solve_lp_gurobi(problema_projecao, [])
        
        if resultado_projecao['status'] != "OPTIMAL":
            logging.warning("Heurística (Pump): LP de projeção falhou.")
            return None, None
        
        # A nova solução LP se torna o ponto de partida para a próxima iteração
        x_k_dict = resultado_projecao['solution']
        
        # Limpa o modelo copiado da memória
        model_copy.dispose()

    logging.info("Heurística (Pump): Máximo de iterações atingido sem encontrar solução.")
    return None, None