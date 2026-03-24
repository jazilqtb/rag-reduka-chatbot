import logging
import sys

def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

if __name__ == "__main__":
    logger = get_logger("LoggerTest")

    print("mulai")
    logger.info("mulai")

    print("hitung")
    a=1
    b=2
    c = a+b
    print(c)

    try:
        d = a+"c"
    except Exception as e:
        logger.error(e)
    logger.info("Program selesai")
    print("mulai selesai")