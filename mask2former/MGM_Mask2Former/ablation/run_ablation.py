import os
import glob
import threading
import subprocess
import socket
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = "/home/remote1/zhanghaonan/projects/mask2former"
TRAIN_SCRIPT = f"{PROJECT_ROOT}/MGM_Mask2Former/train_net_mgm_0831.py"
ABLATION_CONFIG_DIR = f"{PROJECT_ROOT}/MGM_Mask2Former/configs/ablation"
CONDA_ENV = "mask2former"

RESULTS_ROOT = f"{PROJECT_ROOT}/output/ablation_runs/0825_10K_RGB_NOISE"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_ROOT = os.path.join(RESULTS_ROOT, timestamp)

GPU_PAIRS = [(0,),(1,)]
SEEDS = [42]
OVERRIDE_NUM_WORKERS = 4


def tail(path, n=80):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 1024
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                step = min(block, size)
                f.seek(-step, os.SEEK_CUR)
                data = f.read(step) + data
                f.seek(-step, os.SEEK_CUR)
                size -= step
            return data.decode(errors="ignore").splitlines()[-n:]
    except Exception:
        return ["<no log or failed to read>"]

def pick_free_port():
    # 申请一个临时空闲端口
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def run_one_config_on_pair(config_file: str, gpu_pair: tuple[int, int], job_slot_idx: int):
    """
    在指定两张 GPU 上，顺序跑完该 config 的多个 seed。
    job_slot_idx: 0..3，对应本批次的第几个并行实验，用于生成稳定端口范围。
    """
    exp_name = Path(config_file).stem  # e.g. overlay_rgb_only
    gpu_str = ",".join(str(x) for x in gpu_pair)

    # 为该并行槽预留一段端口区间，避免不同并行槽互撞
    base_port = 29500 + job_slot_idx * 50

    for i, seed in enumerate(SEEDS):
        out_dir = os.path.join(RESULTS_ROOT, f"_{exp_name}_{seed}")
        os.makedirs(out_dir, exist_ok=True)
        log_file = os.path.join(out_dir, "train.log")

        # 为每个 seed 选一个不冲突端口
        port = base_port + i
        # 双保险：如果被占用，再临时找一个空闲端口
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                port = pick_free_port()

        dist_url = f"tcp://127.0.0.1:{port}"

        # 强制覆盖 OUTPUT_DIR / SEED / NUM_WORKERS / dist-url
        # 使用 conda run 更稳；也可用 sys.executable 替代（你在同一环境中运行脚本也行）
        cmd = [
            "conda", "run", "-n", CONDA_ENV,
            "python", TRAIN_SCRIPT,
            "--num-gpus", str(len(GPU_PAIRS[0])),
            "--dist-url", dist_url,
            "--config-file", config_file,
            "OUTPUT_DIR", out_dir,
            "SEED", str(seed),
        ]
        if OVERRIDE_NUM_WORKERS is not None:
            cmd += ["DATALOADER.NUM_WORKERS", str(OVERRIDE_NUM_WORKERS)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        # 可选：限制线程，避免 CPU 飙高
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")

        print(f"[LAUNCH] {exp_name} | seed={seed} | GPUs={gpu_str} | port={port} | out={out_dir}")
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
            ret = proc.wait()

        if ret != 0:
            print(f"[ERROR] {exp_name} seed={seed} 失败。日志末尾：{log_file}")
            print("\n".join(tail(log_file, 80)))
        else:
            print(f"[OK] {exp_name} seed={seed} 完成。日志：{log_file}")

def chunked(iterable, n):
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            pass
        if not chunk:
            break
        yield chunk

def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    yaml_files = sorted(glob.glob(os.path.join(ABLATION_CONFIG_DIR, "*.yaml")))
    if not yaml_files:
        raise RuntimeError(f"未找到 yaml：{ABLATION_CONFIG_DIR}")
    # 只取前 len(GPU_PAIRS) 个做一波；多于 4 个会自动分波次并行
    for batch in chunked(yaml_files, len(GPU_PAIRS)):
        threads = []
        for slot_idx, (cfg, gpu_pair) in enumerate(zip(batch, GPU_PAIRS)):
            t = threading.Thread(
                target=run_one_config_on_pair,
                args=(cfg, gpu_pair, slot_idx),
                daemon=False,
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    print("\n[ALL DONE] 全部 ablation 已跑完。根目录：", RESULTS_ROOT)

if __name__ == "__main__":
    main()
