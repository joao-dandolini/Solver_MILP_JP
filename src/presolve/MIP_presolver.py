import math
from functools import reduce

class MIPPresolver:
    """
    Classe para pré-processamento de problemas de Programação Inteira Mista
    Implementa técnicas avançadas de presolve baseadas no artigo
    """
    
    def __init__(self, variables, constraints):
        self.variables = variables          # Dicionário {nome: {lb, ub, type}}
        self.constraints = constraints      # Lista de {'coeffs', 'sense', 'rhs'}
        self._build_var_to_constraints()
        
    def _build_var_to_constraints(self):
        """Constrói mapa de variáveis para restrições associadas"""
        self.var_to_constraints = {var: [] for var in self.variables}
        for idx, constr in enumerate(self.constraints):
            for var in constr['coeffs']:
                self.var_to_constraints[var].append(idx)

    # ================== Técnicas de Propagação de Limites ==================
    def bound_propagation(self):
        """Propagação de limites com tratamento especial para variáveis inteiras"""
        changed = True
        while changed:
            changed = False
            for constr in self.constraints.copy():
                min_act, max_act = self._calculate_activity_bounds(constr)
                
                # Verificar inviabilidade
                if constr['sense'] == '<=' and min_act > constr['rhs']:
                    raise ValueError(f"Inviabilidade detectada em {constr}")
                if constr['sense'] == '>=' and max_act < constr['rhs']:
                    raise ValueError(f"Inviabilidade detectada em {constr}")
                if constr['sense'] == '==' and not (min_act <= constr['rhs'] <= max_act):
                    raise ValueError(f"Inviabilidade detectada em {constr}")
                
                # Atualizar limites das variáveis
                changed |= self._update_bounds_for_constraint(constr, min_act, max_act)
        return self

    def _update_bounds_for_constraint(self, constr, min_act, max_act):
        """Atualiza limites das variáveis para uma restrição específica"""
        changed = False
        
        for var, coeff in constr['coeffs'].items():
            if coeff == 0:
                continue  # Ignorar coeficientes zero
                
            var_info = self.variables[var]
            original_lb = var_info['lb']
            original_ub = var_info['ub']
            
            # ========== Tratamento para restrições <= ==========
            if constr['sense'] == '<=':
                if coeff > 0:  # Variável com coeficiente positivo
                    # Novo limite superior: (RHS - (min_act - coeff * lb)) / coeff
                    new_ub = (constr['rhs'] - (min_act - coeff * var_info['lb'])) / coeff
                    new_ub = self._adjust_for_integer(var_info, new_ub, 'ub')
                    
                    if new_ub < var_info['ub']:
                        var_info['ub'] = new_ub
                        changed = True
                        
                else:  # Variável com coeficiente negativo
                    # Novo limite inferior: (RHS - (min_act - coeff * ub)) / coeff
                    new_lb = (constr['rhs'] - (min_act - coeff * var_info['ub'])) / coeff
                    new_lb = self._adjust_for_integer(var_info, new_lb, 'lb')
                    
                    if new_lb > var_info['lb']:
                        var_info['lb'] = new_lb
                        changed = True

            # ========== Tratamento para restrições >= ==========
            elif constr['sense'] == '>=':
                if coeff > 0:  # Variável com coeficiente positivo
                    # Novo limite inferior: (RHS - (max_act - coeff * ub)) / coeff
                    new_lb = (constr['rhs'] - (max_act - coeff * var_info['ub'])) / coeff
                    new_lb = self._adjust_for_integer(var_info, new_lb, 'lb')
                    
                    if new_lb > var_info['lb']:
                        var_info['lb'] = new_lb
                        changed = True
                        
                else:  # Variável com coeficiente negativo
                    # Novo limite superior: (RHS - (max_act - coeff * lb)) / coeff
                    new_ub = (constr['rhs'] - (max_act - coeff * var_info['lb'])) / coeff
                    new_ub = self._adjust_for_integer(var_info, new_ub, 'ub')
                    
                    if new_ub < var_info['ub']:
                        var_info['ub'] = new_ub
                        changed = True

            # ========== Tratamento para restrições == ==========
            elif constr['sense'] == '==':
                # Calcular intervalo viável para a variável
                other_min = min_act - (coeff * var_info['lb'] if coeff > 0 else coeff * var_info['ub'])
                other_max = max_act - (coeff * var_info['ub'] if coeff > 0 else coeff * var_info['lb'])
                
                if coeff > 0:
                    new_lb = (constr['rhs'] - other_max) / coeff
                    new_ub = (constr['rhs'] - other_min) / coeff
                else:
                    new_lb = (constr['rhs'] - other_min) / coeff
                    new_ub = (constr['rhs'] - other_max) / coeff
                
                # Ajustar para variáveis inteiras
                new_lb = self._adjust_for_integer(var_info, new_lb, 'lb')
                new_ub = self._adjust_for_integer(var_info, new_ub, 'ub')
                
                # Aplicar novos limites
                if new_lb > var_info['lb']:
                    var_info['lb'] = new_lb
                    changed = True
                if new_ub < var_info['ub']:
                    var_info['ub'] = new_ub
                    changed = True

            # Verificar consistência dos limites após atualização
            if var_info['lb'] > var_info['ub']:
                raise ValueError(f"Inviabilidade em {var}: lb > ub")
                
            # Verificar se houve mudança real
            changed |= (original_lb != var_info['lb']) or (original_ub != var_info['ub'])
            
        return changed

    # ================== Técnicas de Aperto de Coeficientes ==================
    def coefficient_tightening(self):
        """Aperto de coeficientes para restrições com variáveis inteiras"""
        for constr in self.constraints.copy():
            if any(self.variables[var]['type'] == 'int' for var in constr['coeffs']):
                self._tighten_coefficients(constr)
        return self

    def _tighten_coefficients(self, constr):
        """Implementa o aperto de coeficientes estilo Chvatal-Gomory"""
        for var in constr['coeffs']:
            if self.variables[var]['type'] != 'int':
                continue
                
            # Calcular atividade máxima das outras variáveis
            other_activity = sum(
                c * (self.variables[v]['ub'] if c > 0 else self.variables[v]['lb'])
                for v, c in constr['coeffs'].items() if v != var
            )
            
            # Aplicar fórmula do artigo para coeficiente strengthening
            if constr['sense'] == '<=':
                d = constr['rhs'] - other_activity - constr['coeffs'][var] * (self.variables[var]['ub'] - 1)
                if d > 0 and constr['coeffs'][var] > d:
                    new_coeff = constr['coeffs'][var] - d
                    constr['coeffs'][var] = new_coeff
                    constr['rhs'] -= d * self.variables[var]['ub']

    # ================== Técnicas de Redução Euclidiana ==================
    def _apply_gcd_reduction(self, constr):
        """Simplifica restrições usando GCD dos coeficientes"""
        try:
            coeffs = list(constr['coeffs'].values())
            # Converter coeficientes para inteiros (caso sejam floats com valor inteiro)
            int_coeffs = [int(c) if c.is_integer() else c for c in coeffs]
            
            # Verificar se todos os coeficientes são inteiros
            if all(isinstance(c, int) for c in int_coeffs):
                gcd = reduce(math.gcd, map(abs, int_coeffs))
                
                if gcd > 1:
                    # Atualizar coeficientes e RHS
                    for var in constr['coeffs']:
                        constr['coeffs'][var] = int(constr['coeffs'][var] / gcd)
                    
                    # Verificar se o RHS pode ser dividido pelo GCD
                    if constr['rhs'] % gcd == 0:
                        constr['rhs'] = int(constr['rhs'] / gcd)
                    else:
                        # Se não for divisível, não podemos aplicar a redução
                        return
        except (TypeError, ValueError):
            # Se ocorrer algum erro (como coeficientes não inteiros), simplesmente ignorar
            return
    def euclidean_reduction(self):
        """Aplica redução euclidiana para restrições inteiras de igualdade"""
        for constr in self.constraints.copy():
            if (constr['sense'] == '==' and 
                all(self.variables[var]['type'] == 'int' for var in constr['coeffs'])):
                self._apply_gcd_reduction(constr)
        return self

    def _apply_gcd_reduction(self, constr):
        """Simplifica restrições usando GCD dos coeficientes"""
        coeffs = list(constr['coeffs'].values())
        gcd = reduce(math.gcd, map(abs, coeffs))
        
        if gcd > 1:
            # Atualizar coeficientes e RHS
            for var in constr['coeffs']:
                constr['coeffs'][var] = int(constr['coeffs'][var] / gcd)
            constr['rhs'] = math.ceil(constr['rhs'] / gcd)
            
            # Verificar consistência
            if constr['rhs'] * gcd != constr['rhs_original']:
                raise ValueError("Restrição não pode ser satisfeita com valores inteiros")

    # ================== Técnicas de Substituição ==================
    def substitute_singletons(self):
        """Substitui variáveis singleton em equações"""
        for constr in self.constraints.copy():
            if constr['sense'] == '==' and len(constr['coeffs']) == 1:
                var, coeff = next(iter(constr['coeffs'].items()))
                self._substitute_variable(var, constr['rhs']/coeff, constr)
        return self

    def _substitute_variable(self, var, value, constr):
        """Substitui variável em todas as restrições"""
        # Atualizar limites da variável
        self.variables[var]['lb'] = max(self.variables[var]['lb'], value)
        self.variables[var]['ub'] = min(self.variables[var]['ub'], value)
        
        # Substituir em outras restrições
        for c_idx in self.var_to_constraints[var].copy():
            if c_idx != self.constraints.index(constr):
                c = self.constraints[c_idx]
                if var in c['coeffs']:
                    term = c['coeffs'].pop(var) * value
                    c['rhs'] -= term
                    
        # Remover restrição original e variável
        self.constraints.remove(constr)
        del self.variables[var]

    # ================== Técnicas de Probing Avançado ==================
    def probing(self, max_vars=5):
        """Técnica de probing avançada com rollback eficiente"""
        binaries = [v for v, info in self.variables.items() if self._is_binary(info)]
        change_stack = []  # Pilha para armazenar alterações temporárias
        
        for var in binaries[:max_vars]:
            try:
                # ========== Testar x = 0 ==========
                self._push_changes(change_stack)
                self.variables[var]['lb'] = self.variables[var]['ub'] = 0
                self.bound_propagation()
                feasible_0 = True
            except ValueError:
                feasible_0 = False
            finally:
                self._pop_changes(change_stack)
                
            # ========== Testar x = 1 ==========
            try:
                self._push_changes(change_stack)
                self.variables[var]['lb'] = self.variables[var]['ub'] = 1
                self.bound_propagation()
                feasible_1 = True
            except ValueError:
                feasible_1 = False
            finally:
                self._pop_changes(change_stack)
                
            # ========== Aplicar fixação permanente ==========
            if not feasible_0 and not feasible_1:
                raise ValueError(f"Probing: {var} torna problema inviável em ambos valores")
            elif not feasible_0:
                self.variables[var]['lb'] = self.variables[var]['ub'] = 1
                print(f"Probing: {var} fixado em 1 (x=0 inviável)")
                self.bound_propagation()
            elif not feasible_1:
                self.variables[var]['lb'] = self.variables[var]['ub'] = 0
                print(f"Probing: {var} fixado em 0 (x=1 inviável)")
                self.bound_propagation()
                
        return self

    # ========== Funções auxiliares para probing ==========
    def _push_changes(self, stack):
        """Armazena o estado atual das variáveis"""
        stack.append({k: (v['lb'], v['ub']) for k, v in self.variables.items()})
        
    def _pop_changes(self, stack):
        """Restaura o estado anterior das variáveis"""
        if stack:
            state = stack.pop()
            for var, (lb, ub) in state.items():
                self.variables[var]['lb'] = lb
                self.variables[var]['ub'] = ub

    # ================== Utilitários ==================
    def _calculate_activity_bounds(self, constr):
        """Calcula atividade mínima e máxima para uma restrição"""
        min_act = max_act = 0
        for var, coeff in constr['coeffs'].items():
            info = self.variables[var]
            if coeff > 0:
                min_act += coeff * info['lb']
                max_act += coeff * info['ub']
            else:
                min_act += coeff * info['ub']
                max_act += coeff * info['lb']
        return min_act, max_act

    def _adjust_for_integer(self, var_info, value, bound_type):
        """Ajusta valores para variáveis inteiras"""
        if var_info['type'] == 'int':
            return math.floor(value) if bound_type == 'ub' else math.ceil(value)
        return value

    def _is_binary(self, var_info):
        """Verifica se variável é binária"""
        return (var_info['type'] == 'int' and 
                var_info['lb'] == 0 and 
                var_info['ub'] == 1)