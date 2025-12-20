from pathlib import Path
from multiprocessing import Pool
from functools import partial
from selector import *
from loader import *


def process_one_file(
    input_file: Path,
    output_dir: Path,
    tree_path: str,
    cfg_select_path: str,
    cfg_io_path: str,
):
    """
    单文件处理函数（给 multiprocessing 用）
    """
    output_file = output_dir / (input_file.stem + ".parquet")

    reduce_raw(
        input_path=str(input_file),
        tree_path=tree_path,
        cfg_select_path=cfg_select_path,
        cfg_io_path=cfg_io_path,
        output_path=str(output_file),
    )

    return str(output_file)


def process_one_directory(
    input_dir: Path,
    output_root: Path,
    tree_path: str,
    cfg_select_path: str,
    cfg_io_path: str,
    n_workers: int,
    suffix: str = ".root",
):
    """
    处理一个 input directory（目录内文件并行）
    """
    input_dir = Path(input_dir)
    assert input_dir.is_dir(), f"{input_dir} is not a directory"

    # 为每个 input dir 建立独立输出目录
    output_dir = output_root / input_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob(f"*{suffix}"))
    if len(input_files) == 0:
        print(f"[WARN] no {suffix} files in {input_dir}")
        return

    print(f"[INFO] processing {input_dir}, files={len(input_files)}")

    worker = partial(
        process_one_file,
        output_dir=output_dir,
        tree_path=tree_path,
        cfg_select_path=cfg_select_path,
        cfg_io_path=cfg_io_path,
    )

    with Pool(processes=n_workers) as pool:
        for out in pool.imap_unordered(worker, input_files):
            print(f"[OK] {out}")


def main():

    cfg_select_path = "/nishome/kangye/ppfemto/code/cfg/config_select.json"
    cfg_io_path = "/nishome/kangye/ppfemto/code/cfg/config_io.json"

    cfg_io = load_config(cfg_io_path)

    input_dirs = cfg_io.data.rawdata_dirs
    output_dir = cfg_io.data.reduced_path
    n_workers = cfg_io.data.numCPU
    tree_path = cfg_io.data.tree_path
    # --------------------------------------------------

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for input_dir in input_dirs:
        process_one_directory(
            input_dir=Path(input_dir),
            output_root=output_root,
            tree_path=tree_path,
            cfg_select_path=cfg_select_path,
            cfg_io_path=cfg_io_path,
            n_workers=n_workers,
        )


if __name__ == "__main__":
    main()
