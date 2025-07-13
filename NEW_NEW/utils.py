# utils.py
# Contém classes e funções de utilidade para o solver.

import time
import math

class StatisticsLogger:
    """
    Trata do rastreamento e da impressão das estatísticas do solver
    durante o processo de Branch and Bound.

    Formata a saída de forma semelhante a solvers comerciais como o Gurobi.
    """
    def __init__(self, log_frequency_seconds: float = 5.0):
        """
        Inicializa o logger.

        Args:
            log_frequency_seconds: A frequência em segundos para imprimir logs periódicos.
        """
        self._start_time = 0.0
        self._last_log_time = 0.0
        self._log_frequency = log_frequency_seconds

    def _calculate_gap(self, incumbent: float, best_bound: float) -> float:
        if incumbent is None:
            return float('inf')

        # Adiciona uma pequena tolerância para estabilidade numérica
        denominator = 1e-10 + abs(incumbent) 

        return abs(incumbent - best_bound) / denominator

    def start_solver(self):
        """
        Registra o tempo de início e imprime o cabeçalho da tabela de log.
        """
        self._start_time = time.perf_counter()
        self._last_log_time = self._start_time
        print(f"{'Nodes':>7s} {'Incumbent':>15s} {'Best Bound':>15s} {'Gap (%)':>10s} {'Time (s)':>10s}")
        print("-" * 60)

    def log_event(self, node_count: int, incumbent: float, best_bound: float, force_log: bool = False):
        """
        Registra uma linha de log.

        Pode ser forçado (ex: ao encontrar um novo incumbent) ou baseado em frequência.
        """
        current_time = time.perf_counter()
        if not force_log and (current_time - self._last_log_time < self._log_frequency):
            return

        elapsed_time = current_time - self._start_time
        gap = self._calculate_gap(incumbent, best_bound)
        
        # Formatação
        incumbent_str = f"{incumbent:15.4f}" if incumbent is not None else "    -    "
        best_bound_str = f"{best_bound:15.4f}" if best_bound is not None else "    -    "
        gap_str = f"{gap*100:8.2f}%" if gap != float('inf') else "   inf   "
        
        print(f"{node_count:7d} {incumbent_str} {best_bound_str} {gap_str} {elapsed_time:10.2f}s")
        self._last_log_time = current_time

    def log_summary(self, status: str, node_count: int, incumbent: float, best_bound: float):
        """
        Imprime o resumo final da execução do solver.
        """
        total_time = time.perf_counter() - self._start_time
        final_gap = self._calculate_gap(incumbent, best_bound)
        
        print("-" * 60)
        print(f"Status: {status}")
        print(f"Total Nodes Explored: {node_count}")
        print(f"Total Time: {total_time:.4f} seconds")
        if incumbent is not None:
            print(f"Best Solution Found: {incumbent:.6f}")
            print(f"Final Gap: {final_gap*100:.4f}%")
        print("-" * 60)