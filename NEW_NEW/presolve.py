# presolve.py
# Contém as rotinas de pré-processamento para simplificar o modelo.

from typing import Dict, Any
import gurobipy as gp
from gurobipy import GRB
import math

def run_bound_strengthening(
    model: gp.Model, 
    original_vars: Dict[str, str]
) -> bool:
    """
    Executa o pré-processamento de Fortalecimento de Limites no modelo.
    
    Itera sobre as restrições para apertar os limites das variáveis até
    que nenhum aperto adicional seja possível (ponto fixo).

    Args:
        model: O modelo Gurobi a ser modificado in-loco.
        original_vars: Dicionário com os tipos de variáveis originais.

    Returns:
        True se algum limite foi alterado, False caso contrário.
    """
    print("-" * 60)
    print("INFO: [Presolve] Iniciando fase de Fortalecimento de Limites...")
    
    total_bounds_changed = 0
    
    # O motor de ponto fixo: continua enquanto fizermos progressos
    while True:
        bounds_changed_this_iteration = 0
        
        # Itera sobre todas as restrições do modelo
        for constr in model.getConstrs():
            # Por enquanto, focamos em restrições '<='
            if constr.Sense != GRB.LESS_EQUAL:
                continue

            lhs_expr = model.getRow(constr)
            rhs = constr.RHS
            
            # --- Lógica de Bound Strengthening (baseada no artigo fp2.pdf) --- [cite: 117, 121]
            
            # Calcula a atividade mínima da restrição com os limites atuais
            min_activity = 0.0
            for i in range(lhs_expr.size()):
                var = lhs_expr.getVar(i)
                coeff = lhs_expr.getCoeff(i)
                # L_min = sum(a_j * l_j) para a_j > 0 e sum(a_j * u_j) para a_j < 0 [cite: 128]
                if coeff > 0:
                    min_activity += coeff * var.LB
                else:
                    min_activity += coeff * var.UB
            
            # Tenta apertar o limite de cada variável na restrição
            for i in range(lhs_expr.size()):
                var = lhs_expr.getVar(i)
                coeff = lhs_expr.getCoeff(i)
                
                if abs(coeff) < 1e-9: continue

                # Calcula a folga disponível para a variável atual
                folga = rhs - (min_activity - coeff * (var.LB if coeff > 0 else var.UB))
                
                if coeff > 0:
                    # Tenta apertar o limite superior (Upper Bound)
                    new_bound = folga / coeff
                    if new_bound < var.UB - 1e-6:
                        # Para variáveis inteiras, podemos arredondar para baixo [cite: 136]
                        new_ub = math.floor(new_bound) if original_vars.get(var.VarName) != GRB.CONTINUOUS else new_bound
                        var.UB = new_ub
                        bounds_changed_this_iteration += 1
                else: # coeff < 0
                    # Tenta apertar o limite inferior (Lower Bound)
                    new_bound = folga / coeff
                    if new_bound > var.LB + 1e-6:
                        # Para variáveis inteiras, podemos arredondar para cima [cite: 136]
                        new_lb = math.ceil(new_bound) if original_vars.get(var.VarName) != GRB.CONTINUOUS else new_bound
                        var.LB = new_lb
                        bounds_changed_this_iteration += 1
        
        # Após uma rodada completa por todas as restrições...
        if bounds_changed_this_iteration == 0:
            # Se não fizemos nenhuma alteração, o sistema está estável. Saia do loop.
            break
        else:
            total_bounds_changed += bounds_changed_this_iteration
            print(f"INFO: [Presolve] Rodada de fortalecimento concluída. {bounds_changed_this_iteration} limites apertados.")
            model.update()

    print(f"INFO: [Presolve] Fase concluída. Total de {total_bounds_changed} limites alterados.")
    print("-" * 60)
    return total_bounds_changed > 0