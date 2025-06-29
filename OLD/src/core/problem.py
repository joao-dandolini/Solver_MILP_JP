"""
Define a estrutura de dados central para representar um problema de otimização.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set

import numpy as np
from scipy.sparse import csr_matrix

class ObjectiveSense(Enum):
    """Define o sentido da otimização (maximizar ou minimizar)."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

class ConstraintSense(Enum):
    """Define o sentido de uma restrição (<=, >=, ==)."""
    LTE = "<="
    GTE = ">="
    EQ = "=="

@dataclass
class Problem:
    """
    Representa um problema de Programação Linear Mista-Inteira (MILP)
    no formato:

    minimize/maximize c^T * x
    sujeito a:
        A * x (<=, >=, ==) b
        lb <= x <= ub
        x_i são inteiros para i em I

    Atributos:
        name (str): Nome do problema.
        objective_sense (ObjectiveSense): Sentido da função objetivo.
        objective_coeffs (np.ndarray): Vetor de custos 'c'.
        constraint_matrix (csr_matrix): Matriz de restrições 'A' em formato esparso.
        rhs_vector (np.ndarray): Vetor 'b' do lado direito das restrições.
        constraint_senses (List[ConstraintSense]): Lista com os sentidos de cada restrição.
        variable_names (List[str]): Nomes das variáveis de decisão.
        lower_bounds (np.ndarray): Limites inferiores das variáveis.
        upper_bounds (np.ndarray): Limites superiores das variáveis.
        integer_variables (Set[int]): Um conjunto com os *índices* das variáveis que são inteiras.
    """
    name: str
    objective_sense: ObjectiveSense
    objective_coeffs: np.ndarray
    constraint_matrix: csr_matrix
    rhs_vector: np.ndarray
    constraint_senses: List[ConstraintSense]

    # Usamos 'field' para permitir valores padrão mais complexos
    variable_names: List[str] = field(default_factory=list)
    lower_bounds: np.ndarray = field(default_factory=np.array)
    upper_bounds: np.ndarray = field(default_factory=np.array)
    integer_variables: Set[int] = field(default_factory=set)

    def __post_init__(self):
        """Validações pós-inicialização para garantir a consistência dos dados."""
        num_vars = len(self.objective_coeffs)
        num_constraints = len(self.rhs_vector)

        if self.constraint_matrix.shape != (num_constraints, num_vars):
            raise ValueError("Dimensões da matriz de restrição A são inconsistentes.")

        if len(self.constraint_senses) != num_constraints:
            raise ValueError("Número de 'senses' de restrição é inconsistente.")

        # Inicializa nomes e limites se não forem fornecidos
        if not self.variable_names:
            self.variable_names = [f"x{i}" for i in range(num_vars)]
        if self.lower_bounds.size == 0:
            self.lower_bounds = np.zeros(num_vars)
        if self.upper_bounds.size == 0:
            self.upper_bounds = np.full(num_vars, np.inf)

# Cole este método inteiro dentro da sua classe Problem, substituindo o antigo __str__
    def __str__(self):
        """Gera uma representação matemática legível do problema."""
        parts = []
        
        # 1. Função Objetivo
        obj_sense_str = "Maximize" if self.objective_sense == ObjectiveSense.MAXIMIZE else "Minimize"
        obj_terms = []
        # Constrói a string da função objetivo termo a termo
        for i, coeff in enumerate(self.objective_coeffs):
            if coeff != 0:
                is_first_term = not obj_terms
                sign = ""
                if coeff > 0 and not is_first_term:
                    sign = "+ "
                elif coeff < 0:
                    sign = "- "
                
                abs_coeff = abs(coeff)
                term_str = f"{abs_coeff} {self.variable_names[i]}"
                if abs_coeff == 1:
                    term_str = self.variable_names[i]
                
                obj_terms.append(f"{sign}{term_str}")
        
        parts.append(f"{obj_sense_str}: {' '.join(obj_terms)}")
        parts.append("\nSubject To:")

        # 2. Restrições
        A_dense = self.constraint_matrix.toarray()
        for i in range(A_dense.shape[0]):
            lhs_terms = []
            for j, coeff in enumerate(A_dense[i, :]):
                if coeff != 0:
                    is_first_term = not lhs_terms
                    sign = ""
                    if coeff > 0 and not is_first_term:
                        sign = "+ "
                    elif coeff < 0:
                        sign = "- "
                    
                    abs_coeff = abs(coeff)
                    term_str = f"{abs_coeff} {self.variable_names[j]}"
                    if abs_coeff == 1:
                        term_str = self.variable_names[j]
                    
                    lhs_terms.append(f"{sign}{term_str}")
            
            sense_str = self.constraint_senses[i].value
            rhs_str = self.rhs_vector[i]
            # Adiciona um nome para a restrição se for possível, ex: "c1: ..."
            # (Poderíamos adicionar nomes de restrições à classe Problem no futuro)
            parts.append(f"  {' '.join(lhs_terms)} {sense_str} {rhs_str}")

        # 3. Limites (Bounds)
        parts.append("\nBounds:")
        for i, name in enumerate(self.variable_names):
            lb = self.lower_bounds[i]
            up = self.upper_bounds[i]
            
            # Não imprime o bound padrão de x >= 0, é implicito
            if lb == 0 and up == float('inf'):
                continue
            
            parts.append(f"  {lb} <= {name} <= {up}")

        # 4. Variáveis Inteiras e Binárias (Lógica atualizada)
        general_integers = []
        binary_variables = []

        if self.integer_variables:
            for i in self.integer_variables:
                # Uma variável é binária se for inteira e seus limites são 0 e 1
                if self.lower_bounds[i] == 0 and self.upper_bounds[i] == 1:
                    binary_variables.append(self.variable_names[i])
                else:
                    general_integers.append(self.variable_names[i])

        if general_integers:
            parts.append("\nIntegers:")
            parts.append(f"  {' '.join(general_integers)}")

        if binary_variables:
            parts.append("\nBinary:")
            parts.append(f"  {' '.join(binary_variables)}")
            
        return "\n".join(parts)