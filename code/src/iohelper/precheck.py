import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from .iohelper import load_tree
from matplotlib.backends.backend_pdf import PdfPages


def compare_cut(
    file1_path: str = "/st0/lhcb/wangjq/raw_data/charm_h_magup25c1/00326684_00000002_1.data_25c1.root",
    file2_path: str = "/st0/lhcb/wangjq/raw_data/charm_h_magdown25c2/00326692_00000001_1.data_25c2.root",
    file3_path: str = "/st0/lhcb/wangjq/raw_data/charm_h_magup25c3/00326686_00000001_1.data_25c3.root",
    file4_path: str = "/st0/lhcb/wangjq/raw_data/charm_h_magdown25c4/00326690_00000002_1.data_25c4.root",
    tree_path: str = "Dp2kkpi/DecayTree",
    output_dir: str = "/nishome/kangye/ppfemto/plot/pre_check/cut_difference",
    combined_pdf_name: str = "ALL_BRANCHES.pdf",
):

    tree_head = tree_path.split("/")[0]
    output_dir = os.path.join(output_dir, tree_head)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading four files...")
    f1 = load_tree(file1_path, tree_path, only_scalar=True)
    f2 = load_tree(file2_path, tree_path, only_scalar=True)
    f3 = load_tree(file3_path, tree_path, only_scalar=True)
    f4 = load_tree(file4_path, tree_path, only_scalar=True)

    branches = f1.fields
    print(f"Total scalar branches: {len(branches)}")

    # 新增：总 PDF（多页）
    combined_pdf = PdfPages(os.path.join(output_dir, combined_pdf_name))

    # 四文件标签
    labels = ["c1", "c2", "c3", "c4"]

    for br in branches:
        print(f"Plotting {br}...")

        try:
            # 数据读取
            x = [
                ak.to_numpy(f1[br]),
                ak.to_numpy(f2[br]),
                ak.to_numpy(f3[br]),
                ak.to_numpy(f4[br]),
            ]
            x = [arr[~np.isnan(arr)] for arr in x]

            # 判空
            if sum(len(arr) for arr in x) == 0:
                continue

            bins = 60

            # ========== 画图 ==========
            plt.figure(figsize=(8, 6))
            for arr, lab in zip(x, labels):
                plt.hist(
                    arr,
                    bins=bins,
                    density=True,
                    histtype="step",
                    label=lab,
                    alpha=1,
                    linewidth=1.2,
                )

            plt.yscale("log")
            plt.xlabel(br)
            plt.ylabel("Density (log)")
            plt.title(f"Branch: {br}", fontsize=15)
            plt.legend()

            # ---- 各格式输出 ----
            plt.savefig(os.path.join(output_dir, f"{br}.pdf"))
            plt.savefig(os.path.join(output_dir, f"{br}.png"))
            plt.savefig(os.path.join(output_dir, f"{br}.jpg"))
            plt.savefig(os.path.join(output_dir, f"{br}.eps"), transparent=False)

            # ---- 写入总 PDF ----
            combined_pdf.savefig()

            # ---- 输出 ROOT .C 文件 ----
            with open(os.path.join(output_dir, f"{br}.C"), "w") as fC:
                fC.write(
                    f"""void {br}() {{
                        TCanvas *c = new TCanvas("c","c",800,600);
                        // NOTE: .C 版本仅创建空 Canvas，你可手动 Fill 内容或用 ROOT 绘图
                        c->SaveAs("{br}.pdf");
                        }}
                    """
                )

            plt.close()

        except Exception as err:
            print(f"Skipping {br}: {err}")

    combined_pdf.close()
    print("\nAll done!")
    print(f"Combined PDF located at: {os.path.join(output_dir, combined_pdf_name)}")


def compare_D0_variable(
    data_path: str = "/st0/lhcb/wangjq/DK_correlation/workflow_pp_2025/output_root/D0h_data/data_down25c1_001.root",
    mc_path: str = "/st0/lhcb/wangjq/DK_correlation/workflow_pp_2025/output_root/D0h_MC/MC.root",
    tree_path: str = "DecayTree",
    select_branch: list[str] = [
        "logD0FDCHI2",
        "logD0DIRA",
        "logPipIPCHI2",
        "logKmIPCHI2",
        "logPipIP",
        "logKmIP",
        "logPipGP",
        "logKmGP",
        "logSumPT",
        "logSumP",
        "sqrtD0VTXCHI2",
        "sqrtD0SDOCA",
        "sqrtD0SDOCACHI2",
        "sqrtD0VDRHO",
        "sqrtD0TAU",
    ],
    D0_MASS: float = 1864.84,
    window: float = 65,
    bins: int = 60,
    output_dir: str = "/nishome/kangye/ppfemto/plot/pre_check/data_mc_distdiff/D0",
    combined_pdf_name: str = "MC_DATA_COMPARE.pdf",
    max_data_files: int = 20,
):

    print("Loading data and MC trees ...")
    base_dir = os.path.dirname(data_path)
    base_name = os.path.basename(data_path)  # e.g. data_down25c1_001.root

    prefix, suffix = base_name.split("001")  # prefix="data_down25c1_", suffix=".root"

    data_list = [
        load_tree(
            root_path=os.path.join(base_dir, f"{prefix}{i:03d}{suffix}"),
            tree_path=tree_path,
            select_branch=select_branch + ["D0_MASS"],
        )
        for i in range(1, max_data_files)
        if os.path.exists(os.path.join(base_dir, f"{prefix}{i:03d}{suffix}"))
    ]

    if len(data_list) == 0:
        raise RuntimeError("No data files found in range 001–020.")

    # concat all data trees
    data = ak.concatenate(data_list, axis=0)
    mc = load_tree(root_path=mc_path, tree_path=tree_path, select_branch=select_branch)

    output_dir = os.path.join(output_dir, tree_path)
    os.makedirs(output_dir, exist_ok=True)

    # --- mass window ---
    mass = ak.to_numpy(data["D0_MASS"])
    mask_sig = np.abs(mass - D0_MASS) < window
    mask_sb = ~mask_sig

    # ==== combined pdf ====
    combined_pdf = PdfPages(os.path.join(output_dir, combined_pdf_name))

    for br in select_branch:
        print(f"Plotting {br} ...")

        try:
            # ========= load arrays =========
            mc_arr = ak.to_numpy(mc[br])
            data_arr = ak.to_numpy(data[br])

            # NAN remove
            mc_arr = mc_arr[~np.isnan(mc_arr)]
            data_arr = data_arr[~np.isnan(data_arr)]

            dsig = data_arr[mask_sig[: len(data_arr)]]
            dsb = data_arr[mask_sb[: len(data_arr)]]

            if len(mc_arr) == 0 and len(dsig) == 0 and len(dsb) == 0:
                print(f"Skipping {br}, empty.")
                continue

            # ================= Plot ======================
            plt.figure(figsize=(8, 6))

            plt.hist(
                mc_arr,
                bins=bins,
                density=True,
                histtype="step",
                label="MC",
                linewidth=1.3,
            )

            if len(dsig) > 0:
                plt.hist(
                    dsig,
                    bins=bins,
                    density=True,
                    histtype="step",
                    label="data signal",
                    linewidth=1.2,
                )

            if len(dsb) > 0:
                plt.hist(
                    dsb,
                    bins=bins,
                    density=True,
                    histtype="step",
                    label="data sideband",
                    linewidth=1.2,
                )

            # plt.yscale("log")
            plt.xlabel(br)
            plt.ylabel("Density")
            plt.title(f"Branch: {br}")
            plt.legend()

            # ================= save ======================
            plt.savefig(os.path.join(output_dir, f"{br}.pdf"))
            plt.savefig(os.path.join(output_dir, f"{br}.png"))
            plt.savefig(os.path.join(output_dir, f"{br}.jpg"))
            plt.savefig(os.path.join(output_dir, f"{br}.eps"), transparent=False)

            combined_pdf.savefig()
            plt.close()

            # ================= ROOT .C ====================
            with open(os.path.join(output_dir, f"{br}.C"), "w") as fC:
                fC.write(
                    f"""void {br}() {{
                        TCanvas *c = new TCanvas("c","c",800,600);
                        // NOTE: You can load MC/data histograms here manually
                        c->SaveAs("{br}.pdf");
                    }}
                    """
                )

        except Exception as e:
            print(f"Skipping {br}: {e}")

    combined_pdf.close()
    print("\nAll plots saved.")
    print(f"Combined PDF: {os.path.join(output_dir, combined_pdf_name)}")
