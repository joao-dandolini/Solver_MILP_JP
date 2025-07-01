# CÓDIGO FINAL E CORRETO para a função em solver/lp_interface.py

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import List, Tuple, Dict, Any

from .problem import Problema

def solve_lp_gurobi(problema: Problema, 
                   local_constraints: List[Tuple[str, str, float]], 
                   general_cuts: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Resolve a relaxação LP, com suporte para planos de corte gerais e retornando
    a informação da base do simplex.
    """
    model_copy = None
    try:
        model_copy = problema.model.copy()
        model_copy.setParam('OutputFlag', 0) 

        for var in model_copy.getVars():
            if var.VType in [GRB.BINARY, GRB.INTEGER]:
                var.VType = GRB.CONTINUOUS
        
        for var_name, sense, rhs in local_constraints:
            var = model_copy.getVarByName(var_name)
            if sense == ">=": model_copy.addConstr(var >= rhs)
            elif sense == "<=": model_copy.addConstr(var <= rhs)

        if general_cuts:
            vars_modelo = model_copy.getVars()
            for corte in general_cuts:
                expr = gp.LinExpr(corte['coeffs'], vars_modelo)
                sentido = corte['sentido']
                rhs = corte['rhs']
                
                if sentido == ">=": model_copy.addConstr(expr >= rhs)
                elif sentido == "<=": model_copy.addConstr(expr <= rhs)

        model_copy.optimize()

        status_code = model_copy.status
        if status_code == GRB.OPTIMAL:
            # --- BLOCO ADICIONADO ---
            # Coletamos a informação de quais variáveis (vbasis) e restrições (cbasis)
            # estão na base da solução ótima do Simplex.
            base_info = {
                'vbasis': model_copy.getAttr('VBasis', model_copy.getVars()),
                'cbasis': model_copy.getAttr('CBasis', model_copy.getConstrs())
            }
            # --- FIM DO BLOCO ADICIONADO ---
            
            return {
                'status': 'OPTIMAL',
                'objective': model_copy.ObjVal,
                'solution': {v.VarName: v.X for v in model_copy.getVars()},
                'base_info': base_info # <-- Adicionamos a informação ao retorno
            }
        elif status_code == GRB.INFEASIBLE:
            return { 'status': 'INFEASIBLE', 'objective': None, 'solution': None, 'base_info': None }
        elif status_code == GRB.UNBOUNDED:
            return { 'status': 'UNBOUNDED', 'objective': None, 'solution': None, 'base_info': None }
        else:
            return { 'status': f'OTHER_{status_code}', 'objective': None, 'solution': None, 'base_info': None }

    except gp.GurobiError as e:
        print(f"Erro do Gurobi na interface: {e}")
        return {'status': 'ERROR', 'objective': None, 'solution': None, 'base_info': None}
    
    finally:
        if model_copy:
            model_copy.dispose()