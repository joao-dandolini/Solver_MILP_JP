# CÓDIGO NOVO E MAIS FLEXÍVEL para solver/problem.py

import gurobipy as gp
from gurobipy import GRB
from typing import List, Union
import logging

class Problema:
    """
    Representa um problema MIP.
    Pode ser inicializado a partir de um arquivo ou de um objeto modelo Gurobi já existente.
    """
    def __init__(self, problema_input: Union[str, gp.Model], nome_problema: str = "MIP"):
        """
        Args:
            problema_input (Union[str, gp.Model]): O caminho para o arquivo (.mps) ou um objeto gp.Model.
        """
        self.nome = nome_problema
        
        if isinstance(problema_input, str):
            # Se for uma string, é um caminho de arquivo. Usamos gp.read().
            logging.info(f"Lendo modelo do arquivo: {problema_input}")
            self.model: gp.Model = gp.read(problema_input)
            self.model.setParam('OutputFlag', 0)
        elif isinstance(problema_input, gp.Model):
            # Se for um objeto Model, nós o usamos diretamente.
            logging.info("Carregando modelo Gurobi já existente.")
            self.model: gp.Model = problema_input
            self.model.setParam('OutputFlag', 0)
        else:
            raise TypeError("Input do problema deve ser um caminho de arquivo (str) ou um objeto gurobipy.Model")

        # A lógica para encontrar as variáveis inteiras continua a mesma
        self.variaveis_inteiras_nomes: List[str] = []
        for v in self.model.getVars():
            if v.VType == GRB.INTEGER or v.VType == GRB.BINARY:
                self.variaveis_inteiras_nomes.append(v.VarName)
        
        logging.info(f"Problema '{self.nome}' carregado. Encontradas {len(self.model.getVars())} variáveis e {len(self.model.getConstrs())} restrições.")

    def __repr__(self) -> str:
        return f"Problema(nome='{self.nome}')"