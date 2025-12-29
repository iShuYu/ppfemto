import os
from multiprocessing import Pool, cpu_count

from loader import *
from generator import generate


def run_one(args):
    """
    Worker：处理一个 reduced 子目录
    """
    name, base_input_dir, base_output_dir, m1, m2, mixing_time, min_angle, mother = args

    input_dir = os.path.join(base_input_dir, name)

    output_same_dir = os.path.join(base_output_dir, "same", name)
    output_same_kstar = os.path.join(output_same_dir, "kstar.parquet")
    output_same_costheta = os.path.join(output_same_dir, "costheta.parquet")

    output_mix_dir = os.path.join(base_output_dir, "mix", name)

    print(f"[INFO] Processing: {name}")

    generate(
        input_dir=input_dir,
        output_same_kstar=output_same_kstar,
        output_same_costheta=output_same_costheta,
        output_mix_kstar_dir=output_mix_dir,
        mixing_time=mixing_time,
        m1=m1,
        m2=m2,
        min_angle=min_angle,
        mother=mother,
    )


def main():
    cfg_io = load_config("/nishome/kangye/ppfemto/code/cfg/config_io.json")
    base_input_dir = cfg_io.data.reduced_path
    base_output_dir = cfg_io.data.output_dir
    mother = cfg_io.data.mother
    m1 = cfg_io.kstar_info.m1
    m2 = cfg_io.kstar_info.m2
    mixing_time = cfg_io.kstar_info.mixing_time
    min_angle = cfg_io.kstar_info.angle_threshold

    subdirs = sorted(
        d
        for d in os.listdir(base_input_dir)
        if os.path.isdir(os.path.join(base_input_dir, d))
    )

    nproc = min(len(subdirs), cfg_io.data.numCPU)

    tasks = [
        (name, base_input_dir, base_output_dir, m1, m2, mixing_time, min_angle, mother)
        for name in subdirs
    ]

    with Pool(processes=nproc) as pool:
        pool.map(run_one, tasks)


if __name__ == "__main__":
    main()
