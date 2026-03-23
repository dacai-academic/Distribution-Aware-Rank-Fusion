import torch

old_ckpt_path = "/home/cai/project/bm25_embedding/checkpoint/mix_m3_86.4.pt"      # 原 checkpoint
new_ckpt_path = "/home/cai/project/bm25_embedding/checkpoint/bge_zh_m3_86.4.pt"          # 新 checkpoint

state = torch.load(old_ckpt_path, map_location="cpu")

# 如果是 trainer 保存的，可能在 state["state_dict"]
if "state_dict" in state:
    state_dict = state["state_dict"]
else:
    state_dict = state

new_state_dict = {}

for k, v in state_dict.items():
    new_k = k.replace("conan", "m3")
    new_state_dict[new_k] = v

torch.save(new_state_dict, new_ckpt_path)

print(f"✅ 已保存新权重到 {new_ckpt_path}")