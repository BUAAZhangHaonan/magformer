"""Keep a GPU occupied by holding memory until killed."""
import torch, time, signal, sys

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
hold_gb = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0

device = f'cuda:{gpu_id}'
# Allocate tensor to hold ~hold_gb of VRAM
n_elements = int(hold_gb * 1e9 / 4)  # float32 = 4 bytes
shape = (n_elements // 1000, 1000)
print(f'[{time.strftime("%H:%M:%S")}] Allocating {hold_gb:.1f} GB on {device}...')
x = torch.randn(*shape, device=device, dtype=torch.float32)
actual_gb = x.nelement() * 4 / 1e9
print(f'[{time.strftime("%H:%M:%S")}] Holding {actual_gb:.1f} GB on {device}. PID={__import__("os").getpid()}')
print(f'[{time.strftime("%H:%M:%S")}] Kill this process to release GPU.', flush=True)

running = True
def handler(sig, frame):
    global running
    print(f'[{time.strftime("%H:%M:%S")}] Signal {sig} received, releasing GPU...')
    running = False

signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

while running:
    time.sleep(30)
    # Touch the tensor to show GPU activity
    x.mul_(1.0)

print(f'[{time.strftime("%H:%M:%S")}] GPU holder exiting.', flush=True)
