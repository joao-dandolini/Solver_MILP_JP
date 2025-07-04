# NOVO ARQUIVO: solver/presolve/aplicador.py

import logging
from gurobipy import GRB, LinExpr
import math
from functools import reduce

# Importa a nossa nova classe Problema para type hinting
from ..problem import Problema

# -------------------------------------------------
# 1. A NOVA VERSÃO DO 'substitute_singletons'
# -------------------------------------------------
def presolve_substitute_singletons(problema: Problema) -> int:
    """
    Procura por restrições de igualdade com uma única variável (singletons),
    e fixa o valor dessa variável, modificando o modelo Gurobi diretamente.

    Retorna o número de variáveis fixadas.
    """
    vars_fixadas = 0
    # Usamos uma cópia da lista de restrições, pois podemos modificá-las
    for constr in problema.model.getConstrs():
        
        # A lógica se aplica apenas a restrições de igualdade
        if constr.Sense != GRB.EQUAL:
            continue
            
        # Pega a expressão da linha da restrição
        linha = problema.model.getRow(constr)
        
        # Se a linha tem apenas um termo, encontramos um singleton!
        if linha.size() == 1:
            var = linha.getVar(0)
            coeff = linha.getCoeff(0)
            rhs = constr.RHS
            
            if abs(coeff) > 1e-9:
                valor_fixo = rhs / coeff
                
                # VERIFICAÇÃO DE CONSISTÊNCIA: o valor fixo deve respeitar os limites existentes
                if valor_fixo < var.LB or valor_fixo > var.UB:
                    logging.error(f"Presolve: Inviabilidade detectada! Var {var.VarName} fixada em {valor_fixo} fora dos limites [{var.LB}, {var.UB}].")
                    # Em um solver real, isso provaria inviabilidade. Lançamos um erro.
                    raise ValueError("Inviabilidade detectada no Presolve (Singleton).")

                # A MÁGICA: Em vez de substituir em outras equações, nós simplesmente
                # fixamos os limites da variável. O Gurobi cuida do resto internamente.
                var.setAttr('LB', valor_fixo)
                var.setAttr('UB', valor_fixo)
                
                logging.debug(f"Presolve (Singleton): Variável {var.VarName} fixada em {valor_fixo}.")
                vars_fixadas += 1
                
    # O modelo do Gurobi precisa ser atualizado após mudanças estruturais
    problema.model.update()
    logging.info(f"  -> Presolve (Singletons): {vars_fixadas} variáveis fixadas.")
    return vars_fixadas

def presolve_euclidean_reduction(problema: Problema) -> int:
    """
    Simplifica restrições de igualdade com variáveis e coeficientes inteiros
    usando o Máximo Divisor Comum (MDC/GCD).
    Modifica o modelo Gurobi diretamente.
    """
    linhas_reduzidas = 0
    
    for constr in problema.model.getConstrs():
        # Aplica-se apenas a restrições de igualdade
        if constr.Sense != GRB.EQUAL:
            continue
            
        linha = problema.model.getRow(constr)
        
        # Pega as variáveis e coeficientes da linha
        vars_na_linha = [linha.getVar(i) for i in range(linha.size())]
        coeffs_na_linha = [linha.getCoeff(i) for i in range(linha.size())]
        
        # CONDIÇÃO 1: Todas as variáveis na restrição devem ser inteiras
        if not all(v.VType != GRB.CONTINUOUS for v in vars_na_linha):
            continue

        # CONDIÇÃO 2: Todos os coeficientes devem ser inteiros
        if not all(abs(c - round(c)) < 1e-9 for c in coeffs_na_linha):
            continue
        
        # Se as condições foram atendidas, calcula o MDC
        if not coeffs_na_linha: continue
        coefs_int = [int(round(c)) for c in coeffs_na_linha]
        gcd = reduce(math.gcd, [abs(c) for c in coefs_int])

        if gcd > 1:
            rhs = constr.RHS
            # Se o lado direito não for divisível pelo MDC, o problema é inviável
            if abs(round(rhs) - rhs) > 1e-9 or int(round(rhs)) % gcd != 0:
                raise ValueError(f"Inviabilidade detectada (Euclidiana): RHS {rhs} não é divisível pelo MDC {gcd}.")

            # Se for divisível, podemos simplificar a restrição
            logging.debug(f"Presolve (Euclidiana): Reduzindo restrição '{constr.ConstrName}' por fator {gcd}.")
            
            # Divide cada coeficiente na restrição pelo MDC
            for i in range(len(vars_na_linha)):
                var = vars_na_linha[i]
                novo_coeff = coeffs_na_linha[i] / gcd
                problema.model.chgCoeff(constr, var, novo_coeff)
            
            # Divide o lado direito (RHS) pelo MDC
            constr.setAttr('RHS', rhs / gcd)
            
            linhas_reduzidas += 1

    if linhas_reduzidas > 0:
        problema.model.update()
    
    logging.info(f"  -> Presolve (Euclidiana): {linhas_reduzidas} restrições reduzidas.")
    return linhas_reduzidas


def _calculate_activity_bounds(linha: LinExpr) -> tuple:
    """Função auxiliar para calcular a atividade mínima e máxima de uma restrição."""
    min_act, max_act = 0.0, 0.0
    
    # Adiciona a constante da restrição, se houver
    min_act += linha.getConstant()
    max_act += linha.getConstant()

    for i in range(linha.size()):
        coeff = linha.getCoeff(i)
        var = linha.getVar(i)
        lb, ub = var.LB, var.UB

        if coeff > 0:
            min_act += coeff * lb if lb != -GRB.INFINITY else -GRB.INFINITY
            max_act += coeff * ub if ub != GRB.INFINITY else GRB.INFINITY
        else:
            min_act += coeff * ub if ub != GRB.INFINITY else -GRB.INFINITY
            max_act += coeff * lb if lb != -GRB.INFINITY else GRB.INFINITY
            
    return min_act, max_act


def presolve_propagate_bounds(problema: Problema) -> int:
    """
    Usa as restrições para deduzir e propagar limites mais apertados para as variáveis.
    Modifica o modelo Gurobi diretamente.
    """
    bounds_apertados = 0
    mudou = True
    
    # Continua em loop enquanto conseguirmos apertar algum limite
    while mudou:
        mudou = False
        
        for constr in problema.model.getConstrs():
            linha = problema.model.getRow(constr)
            rhs = constr.RHS
            
            # Pula restrições vazias
            if linha.size() == 0: continue

            # Itera sobre cada variável na restrição para tentar apertar seus limites
            for i in range(linha.size()):
                var_alvo = linha.getVar(i)
                coeff_alvo = linha.getCoeff(i)
                
                # Cria uma cópia da linha e remove a variável alvo para calcular a "atividade do resto"
                outras_vars_expr = linha.copy()
                outras_vars_expr.remove(var_alvo)
                
                min_act_outras, max_act_outras = _calculate_activity_bounds(outras_vars_expr)

                # Salva os limites antigos para checar se houve mudança
                old_lb, old_ub = var_alvo.LB, var_alvo.UB

                # Deriva os novos limites baseado na equação: coeff * x <= rhs - outras_atividades
                if constr.Sense == GRB.LESS_EQUAL:
                    if coeff_alvo > 0:
                        # Aperta o limite superior (UB)
                        if max_act_outras != GRB.INFINITY:
                            new_ub = (rhs - min_act_outras) / coeff_alvo
                            if new_ub < var_alvo.UB: var_alvo.setAttr('UB', new_ub)
                    else: # coeff_alvo < 0
                        # Aperta o limite inferior (LB)
                        if min_act_outras != -GRB.INFINITY:
                            new_lb = (rhs - max_act_outras) / coeff_alvo
                            if new_lb > var_alvo.LB: var_alvo.setAttr('LB', new_lb)

                elif constr.Sense == GRB.GREATER_EQUAL:
                    if coeff_alvo > 0:
                        # Aperta o limite inferior (LB)
                        if min_act_outras != -GRB.INFINITY:
                            new_lb = (rhs - max_act_outras) / coeff_alvo
                            if new_lb > var_alvo.LB: var_alvo.setAttr('LB', new_lb)
                    else: # coeff_alvo < 0
                        # Aperta o limite superior (UB)
                        if max_act_outras != GRB.INFINITY:
                            new_ub = (rhs - min_act_outras) / coeff_alvo
                            if new_ub < var_alvo.UB: var_alvo.setAttr('UB', new_ub)

                # Se a variável é inteira, arredonda o novo limite para o inteiro mais próximo e válido
                if var_alvo.VType != GRB.CONTINUOUS:
                    var_alvo.setAttr('LB', math.ceil(var_alvo.LB))
                    var_alvo.setAttr('UB', math.floor(var_alvo.UB))

                # Checa se os limites se cruzaram (inviabilidade)
                if var_alvo.LB > var_alvo.UB + 1e-9:
                    raise ValueError(f"Inviabilidade detectada (Propagação): Var {var_alvo.VarName} com limites [{var_alvo.LB}, {var_alvo.UB}].")

                # Se algum limite mudou, marca para continuar o loop
                if var_alvo.LB > old_lb or var_alvo.UB < old_ub:
                    mudou = True
                    bounds_apertados += 1

    problema.model.update()
    logging.info(f"  -> Presolve (Prop. Limites): {bounds_apertados} limites apertados.")
    return bounds_apertados

def presolve_coefficient_tightening(problema: Problema) -> int:
    """
    Tenta reduzir os coeficientes de variáveis binárias em restrições,
    usando os limites das outras variáveis para fortalecer a formulação.
    """
    coeffs_apertados = 0
    
    for constr in problema.model.getConstrs():
        linha = problema.model.getRow(constr)
        rhs = constr.RHS
        
        for i in range(linha.size()):
            var_alvo = linha.getVar(i)
            
            # Esta técnica foca em variáveis binárias
            if var_alvo.VType != GRB.BINARY:
                continue

            coeff_alvo = linha.getCoeff(i)
            
            # Separa a expressão para calcular a atividade do "resto"
            outras_vars_expr = linha.copy()
            outras_vars_expr.remove(var_alvo)
            min_act_outras, max_act_outras = _calculate_activity_bounds(outras_vars_expr)

            # Lógica para restrições do tipo <=
            if constr.Sense == GRB.LESS_EQUAL:
                if coeff_alvo > 0 and min_act_outras != -GRB.INFINITY:
                    # Novo coeficiente potencial quando y=1: a' <= b - L
                    novo_coeff = rhs - min_act_outras
                    if novo_coeff < coeff_alvo:
                        problema.model.chgCoeff(constr, var_alvo, novo_coeff)
                        coeffs_apertados += 1
                        logging.debug(f"Presolve (Coeff. Tighten): Coeficiente de {var_alvo.VarName} em '{constr.ConstrName}' reduzido para {novo_coeff:.2f}.")

            # Lógica para restrições do tipo >=
            elif constr.Sense == GRB.GREATER_EQUAL:
                if coeff_alvo < 0 and max_act_outras != GRB.INFINITY:
                    # Novo coeficiente potencial quando y=1: a' >= b - U
                    novo_coeff = rhs - max_act_outras
                    if novo_coeff > coeff_alvo:
                        problema.model.chgCoeff(constr, var_alvo, novo_coeff)
                        coeffs_apertados += 1
                        logging.debug(f"Presolve (Coeff. Tighten): Coeficiente de {var_alvo.VarName} em '{constr.ConstrName}' aumentado para {novo_coeff:.2f}.")

    if coeffs_apertados > 0:
        problema.model.update()

    logging.info(f"  -> Presolve (Coeff. Tighten): {coeffs_apertados} coeficientes apertados.")
    return coeffs_apertados

def presolve_propagate_bounds(problema: Problema) -> tuple:
    """
    Usa as restrições para deduzir limites mais apertados.
    Retorna (sucesso, numero_de_mudancas). Versão com lógica de arredondamento corrigida.
    """
    bounds_apertados = 0
    mudou = True
    
    try:
        while mudou:
            mudou = False
            for constr in problema.model.getConstrs():
                linha = problema.model.getRow(constr)
                if linha.size() == 0: continue

                for i in range(linha.size()):
                    var_alvo = linha.getVar(i)
                    coeff_alvo = linha.getCoeff(i)
                    
                    outras_vars_expr = linha.copy()
                    outras_vars_expr.remove(var_alvo)
                    min_act_outras, max_act_outras = _calculate_activity_bounds(outras_vars_expr)

                    old_lb, old_ub = var_alvo.LB, var_alvo.UB
                    
                    # Lógica para derivação de novos limites (continua a mesma)
                    if constr.Sense == GRB.LESS_EQUAL:
                        if coeff_alvo > 0 and min_act_outras != -GRB.INFINITY:
                            new_ub = (constr.RHS - min_act_outras) / coeff_alvo
                            if new_ub < old_ub: var_alvo.setAttr('UB', new_ub)
                        elif coeff_alvo < 0 and max_act_outras != GRB.INFINITY:
                            new_lb = (constr.RHS - min_act_outras) / coeff_alvo
                            if new_lb > old_lb: var_alvo.setAttr('LB', new_lb)
                            
                    elif constr.Sense == GRB.GREATER_EQUAL:
                        if coeff_alvo > 0 and max_act_outras != GRB.INFINITY:
                            new_lb = (constr.RHS - max_act_outras) / coeff_alvo
                            if new_lb > old_lb: var_alvo.setAttr('LB', new_lb)
                        elif coeff_alvo < 0 and min_act_outras != -GRB.INFINITY:
                            new_ub = (constr.RHS - max_act_outras) / coeff_alvo
                            if new_ub < old_ub: var_alvo.setAttr('UB', new_ub)

                    # --- LÓGICA DE ARREDONDAMENTO CORRIGIDA E MAIS SEGURA ---
                    if var_alvo.VType != GRB.CONTINUOUS:
                        # Arredonda o limite inferior para cima, para o próximo inteiro
                        novo_lb_arredondado = math.ceil(var_alvo.LB - 1e-9) # Usamos tolerância para dentro
                        if novo_lb_arredondado > old_lb:
                            var_alvo.setAttr('LB', novo_lb_arredondado)

                        # Arredonda o limite superior para baixo, para o próximo inteiro
                        novo_ub_arredondado = math.floor(var_alvo.UB + 1e-9) # Usamos tolerância para dentro
                        if novo_ub_arredondado < old_ub:
                             var_alvo.setAttr('UB', novo_ub_arredondado)
                    # --- FIM DA CORREÇÃO ---

                    if var_alvo.LB > var_alvo.UB + 1e-9:
                        logging.error(f"--> Inviabilidade Falsa em '{constr.ConstrName}' para '{var_alvo.VarName}': LB={var_alvo.LB} > UB={var_alvo.UB}")
                        return False, bounds_apertados

                    if var_alvo.LB > old_lb or var_alvo.UB < old_ub:
                        mudou = True; bounds_apertados += 1
        
        problema.model.update()
        return True, bounds_apertados
    except Exception:
        # Se qualquer erro ocorrer, consideramos que a propagação falhou
        return False, bounds_apertados

def presolve_probing(problema: Problema, max_candidates: int = 25) -> int:
    """
    Testa fixar variáveis binárias em 0 e 1, selecionando as candidatas
    mais "impactantes" para sondar.
    """
    vars_fixadas_total = 0
    rodada_probing = 1
    
    while True:
        logging.debug(f"--- Probing (Rodada {rodada_probing}) ---")
        mudou_nesta_rodada = False
        
        # --- LÓGICA DE SELEÇÃO INTELIGENTE ---
        
        # 1. Pega todos os candidatos possíveis (não fixos)
        candidatos_full = [v for v in problema.model.getVars() if v.VType == GRB.BINARY and v.LB < 0.5 and v.UB > 0.5]
        
        if not candidatos_full: break

        # 2. Calcula uma pontuação de impacto para cada um
        candidatos_com_pontuacao = []
        for v in candidatos_full:
            # Pega a coluna para saber de quantas restrições a variável participa
            coluna = problema.model.getCol(v)
            num_constrs_participa = coluna.size()
            
            # Pontuação = |Custo no Objetivo| + (fator de peso * Número de Restrições)
            # O fator de peso ajuda a balancear a importância dos dois termos
            pontuacao = abs(v.Obj) + 0.5 * num_constrs_participa
            candidatos_com_pontuacao.append((pontuacao, v))

        # 3. Ordena os candidatos pela pontuação, do maior para o menor
        candidatos_com_pontuacao.sort(key=lambda item: item[0], reverse=True)
        
        # 4. Seleciona os melhores 'max_candidates' para sondar
        candidatos_para_sondar = [var for pontuacao, var in candidatos_com_pontuacao[:max_candidates]]
        
        nomes_dos_sondados = [v.VarName for v in candidatos_para_sondar]
        logging.info(f"  -> Candidatos selecionados para sondagem: {nomes_dos_sondados}")

        logging.debug(f"Encontrados {len(candidatos_full)} candidatos, sondando os {len(candidatos_para_sondar)} mais impactantes.")
        # --- FIM DA LÓGICA INTELIGENTE ---

        for var in candidatos_para_sondar:
            # O resto da lógica de sondagem continua a mesma de antes...
            if var.LB > 0.5 or var.UB < 0.5: continue
            
            # Teste para var=0
            model_copy_0 = problema.model.copy()
            model_copy_0.setParam('OutputFlag', 0)
            var_copy_0 = model_copy_0.getVarByName(var.VarName)
            var_copy_0.setAttr('UB', 0.0)
            status_0, _ = presolve_propagate_bounds(Problema(problema_input=model_copy_0))
            model_copy_0.dispose()

            # Teste para var=1
            model_copy_1 = problema.model.copy()
            model_copy_1.setParam('OutputFlag', 0)
            var_copy_1 = model_copy_1.getVarByName(var.VarName)
            var_copy_1.setAttr('LB', 1.0)
            status_1, _ = presolve_propagate_bounds(Problema(problema_input=model_copy_1))
            model_copy_1.dispose()

            if not status_0 and not status_1:
                raise ValueError(f"Inviabilidade (Probing): Var {var.VarName} inviável em ambos os ramos.")
            
            if not status_0:
                if var.LB < 1.0:
                    var.setAttr('LB', 1.0); vars_fixadas_total += 1; mudou_nesta_rodada = True
                    logging.info(f"Presolve (Probing): Variável {var.VarName} fixada em 1.")
            
            if not status_1:
                if var.UB > 0.0:
                    var.setAttr('UB', 0.0); vars_fixadas_total += 1; mudou_nesta_rodada = True
                    logging.info(f"Presolve (Probing): Variável {var.VarName} fixada em 0.")
        
        if mudou_nesta_rodada:
            presolve_propagate_bounds(problema)
            rodada_probing += 1
        else:
            break

    logging.info(f"  -> Presolve (Probing): {vars_fixadas_total} variáveis fixadas.")
    return vars_fixadas_total

# -------------------------------------------------
# 2. A FUNÇÃO ORQUESTRADORA (POR ENQUANTO, SÓ CHAMA UMA TÉCNICA)
# -------------------------------------------------

# CÓDIGO FINAL para aplicar_presolve em aplicador.py

def aplicar_presolve(problema: Problema) -> Problema:
    """
    Orquestra a aplicação de todas as técnicas de presolve em um loop
    até que nenhuma nova mudança seja detectada em uma rodada completa.
    """
    logging.info("="*20 + " INICIANDO FASE DE PRESOLVE (CUSTOMIZADO) " + "="*20)
    
    rodada = 1
    max_rodadas = 10 # Um limite de segurança para evitar loops infinitos
    
    while rodada <= max_rodadas:
        mudancas_totais_na_rodada = 0
        logging.info(f"--- Iniciando rodada {rodada} de Presolve ---")

        # 1. Propagação de Limites (A mais fundamental)
        sucesso, mudancas = presolve_propagate_bounds(problema)
        if not sucesso: raise ValueError("Inviabilidade detectada durante a propagação de limites.")
        if mudancas > 0: logging.info(f"  - Propagate Bounds encontrou {mudancas} mudanças.")
        mudancas_totais_na_rodada += mudancas

        # 2. Remoção de Colunas Singleton (Simples e segura)
        mudancas = presolve_substitute_singletons(problema)
        if mudancas > 0: logging.info(f"  - Singleton Columns encontrou {mudancas} mudanças.")
        mudancas_totais_na_rodada += mudancas

        # 3. Redundância Euclidiana (Segura)
        mudancas = presolve_euclidean_reduction(problema)
        if mudancas > 0: logging.info(f"  - Euclidean Redundancy encontrou {mudancas} mudanças.")
        mudancas_totais_na_rodada += mudancas

        # 4. Aperto de Coeficientes (Mais avançada)
        mudancas = presolve_coefficient_tightening(problema)
        if mudancas > 0: logging.info(f"  - Coefficient Tightening encontrou {mudancas} mudanças.")
        mudancas_totais_na_rodada += mudancas

        # 5. Probing (A mais poderosa e perigosa, por último)
        # Descomente a linha abaixo para reativar o probing quando estivermos confiantes
        mudancas = presolve_probing(problema)
        if mudancas > 0: logging.info(f"  - Probing encontrou {mudancas} mudanças.")
        mudancas_totais_na_rodada += mudancas

        logging.info(f"Rodada de Presolve {rodada} concluída com {mudancas_totais_na_rodada} mudanças totais.")

        # Se uma rodada inteira não produziu nenhuma mudança, o presolve terminou.
        if mudancas_totais_na_rodada == 0:
            logging.info("Nenhuma mudança detectada na rodada. Presolve estabilizado.")
            break
        
        rodada += 1

    logging.info("="*22 + " PRESOLVE CONCLUÍDO " + "="*22)
    problema.model.update()
    return problema

'''
def aplicar_presolve(problema: Problema) -> Problema:
    """
    Aplica uma sequência de técnicas de presolve em loop até que nenhuma
    outra simplificação seja possível.
    """
    logging.info("="*20 + " INICIANDO FASE DE PRESOLVE (CUSTOMIZADO) " + "="*20)
    
    rodada = 1
    mudancas_na_rodada = 1
    while mudancas_na_rodada > 0:
        logging.info(f"--- Iniciando rodada {rodada} de Presolve ---")
        mudancas_na_rodada = 0
        
        # --- CORREÇÃO NA CHAMADA E NA LÓGICA ---
        # Desempacotamos a tupla (sucesso, num_mudancas) retornada pela função
        sucesso, num_mudancas = presolve_propagate_bounds(problema)
        if not sucesso:
            # Se a propagação de limites falhou, o problema é inviável
            raise ValueError("Inviabilidade detectada durante a propagação de limites no Presolve.")
        mudancas_na_rodada += num_mudancas
        
        mudancas_na_rodada += presolve_substitute_singletons(problema)
        mudancas_na_rodada += presolve_euclidean_reduction(problema)
        mudancas_na_rodada += presolve_coefficient_tightening(problema)
        mudancas_na_rodada += presolve_probing(problema)
        # --- FIM DA CORREÇÃO ---
        
        logging.info(f"Rodada de Presolve {rodada} concluída com {mudancas_na_rodada} mudanças.")
        
        # Se uma rodada inteira não produziu nenhuma mudança, podemos parar.
        if mudancas_na_rodada == 0:
            break
        rodada += 1

    logging.info("="*22 + " PRESOLVE CONCLUÍDO " + "="*22)
    problema.model.update()
    return problema
'''