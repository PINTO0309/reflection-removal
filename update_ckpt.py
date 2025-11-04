import torch

src = "ckpts/reflection_removal_dinov3_vits16.pt"
dst = "ckpts/reflection_removal_dinov3_vits16_residual_init.pt"

state = torch.load(src, map_location="cpu")

if "state_dict" in state:
    generator_state = state["state_dict"]
elif "generator" in state:
    generator_state = state["generator"]
else:
    raise KeyError("generator/state_dict not found in checkpoint.")

for idx in range(8):
    generator_state[f"residual_scales.{idx}"] = torch.tensor(0.1)

state["state_dict"] = generator_state
state["generator_variant"] = "residual_skips"

torch.save(state, dst)
print(f"saved to {dst}")