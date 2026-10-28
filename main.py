from pyfedappwrap.engine.runtime import FedDBEngine

from app import UciFetchToCsvAPP

engine = FedDBEngine()

engine.register(UciFetchToCsvAPP())

if __name__ == "__main__":
    engine.start()
    engine.wait_until_stop()
