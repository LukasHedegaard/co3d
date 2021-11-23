import os
from platform import uname
from typing import Any, Dict

from cpuinfo import get_cpu_info
from GPUtil import getGPUs
from psutil import cpu_count, cpu_freq, virtual_memory

NUM_CPU = os.cpu_count() or 1


def get_machine_info() -> Dict[str, Any]:
    sys = uname()
    cpu = get_cpu_info()
    svmem = virtual_memory()
    gpus = getGPUs()

    return {
        "system": {"system": sys.system, "node": sys.node, "release": sys.release},
        "cpu": {
            "model": cpu["brand_raw"],
            "architecture": cpu["arch_string_raw"],
            "cores": {
                "physical": cpu_count(logical=False),
                "total": cpu_count(logical=True),
            },
            "frequency": f"{(cpu_freq().max / 1000):.2f} GHz",
        },
        "memory": {
            "total": get_size(svmem.total),
            "used": get_size(svmem.used),
            "available": get_size(svmem.available),
        },
        "gpus": (
            [{"name": g.name, "memory": f"{g.memoryTotal} MB"} for g in gpus]
            if gpus
            else None
        ),
    }


def get_size(bytes, suffix="B"):
    """Scale bytes to its proper format, e.g. 1253656 => '1.20MB'"""
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f} {unit}{suffix}"
        bytes /= factor
