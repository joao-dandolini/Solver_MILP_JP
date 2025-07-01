# utils/logger_config.py

import logging
import sys

def setup_logger():
    """
    Configura o logger principal para o projeto, direcionando a saída para o console.
    """
    # Define o formato da mensagem de log
    log_format = "[%(asctime)s] [%(levelname)-8s] [%(module)-15s] - %(message)s"
    
    # Pega o logger raiz
    logger = logging.getLogger()
    
    # Define o nível mínimo de log a ser exibido (DEBUG é o mais detalhado)
    logger.setLevel(logging.DEBUG)
    
    # Evita adicionar múltiplos handlers se a função for chamada mais de uma vez
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Cria um handler para enviar os logs para o console (stdout)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    
    # Cria um formatador com o formato que definimos
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Adiciona o formatador ao handler
    handler.setFormatter(formatter)
    
    # Adiciona o handler ao logger
    logger.addHandler(handler)
    
    logging.info("Logger configurado com sucesso.")