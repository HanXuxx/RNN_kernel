#!/usr/bin/env python3
"""从 Nsight Systems SQLite 导出 CUDA kernel 调度摘要。"""

import argparse
import sqlite3
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CUDA kernel launch table from nsys SQLite.")
    parser.add_argument("sqlite_path", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = """
    select s.value as name,
           count(*) as instances,
           round(avg((k.end - k.start) / 1000.0), 3) as avg_us,
           round(min((k.end - k.start) / 1000.0), 3) as min_us,
           round(max((k.end - k.start) / 1000.0), 3) as max_us,
           k.gridX, k.gridY, k.gridZ,
           k.blockX, k.blockY, k.blockZ,
           k.registersPerThread,
           k.dynamicSharedMemory,
           k.staticSharedMemory
    from CUPTI_ACTIVITY_KIND_KERNEL k
    join StringIds s on s.id = k.demangledName
    group by s.value,
             k.gridX, k.gridY, k.gridZ,
             k.blockX, k.blockY, k.blockZ,
             k.registersPerThread,
             k.dynamicSharedMemory,
             k.staticSharedMemory
    order by avg((k.end - k.start)) desc
    limit ?
    """
    with sqlite3.connect(args.sqlite_path) as connection:
        rows = connection.execute(query, (args.limit,)).fetchall()

    header = (
        "instances",
        "avg_us",
        "min_us",
        "max_us",
        "grid",
        "block",
        "regs",
        "dyn_smem",
        "static_smem",
        "name",
    )
    print(",".join(header))
    for row in rows:
        (
            name,
            instances,
            avg_us,
            min_us,
            max_us,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            registers,
            dynamic_smem,
            static_smem,
        ) = row
        print(
            f"{instances},{avg_us},{min_us},{max_us},"
            f"{grid_x}x{grid_y}x{grid_z},"
            f"{block_x}x{block_y}x{block_z},"
            f"{registers},{dynamic_smem},{static_smem},"
            f"{name}"
        )


if __name__ == "__main__":
    main()
