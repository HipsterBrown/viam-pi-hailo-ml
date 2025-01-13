import asyncio
from viam.module.module import Module


import hailort_vision


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
