# CÓDIGO NOVO E CORRIGIDO para utils/logger_config.py

import logging
import sys

def setup_logger():
    """
    Configura o logger principal para o projeto de forma flexível.
    """
    log_format = "[%(asctime)s] [%(levelname)-8s] [%(module)-15s] - %(message)s"
    logger = logging.getLogger()
    
    # --- MUDANÇA 1 ---
    # O comportamento PADRÃO do logger será mostrar mensagens INFO ou mais importantes.
    # Este é o modo "limpo" que você geralmente vai querer.
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    handler = logging.StreamHandler(sys.stdout)
    
    # --- MUDANÇA 2 ---
    # O handler (o "carteiro") estará sempre pronto para o nível mais detalhado.
    # Ele só mostrará o que o logger principal (o "chefe") permitir.
    handler.setLevel(logging.DEBUG) 
    
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logging.info("Logger configurado com sucesso.")