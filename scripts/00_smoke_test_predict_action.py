import numpy as np
from PIL import Image
import torch

from vla_mech_monitor.vla.openvla_hf_adapter import OpenVLAHFConfig, OpenVLAPolicyHF


def main():
    cfg = OpenVLAHFConfig(
        model_id="openvla/openvla-7b",
        device="cuda" if torch.cuda.is_available() else "cpu",
        precision="bf16",
        unnorm_key="bridge_orig",
        hook_last_n_mlp=4,
    )
    pol = OpenVLAPolicyHF(cfg)
    pol.load()

    # Replace with any local image
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))

    instruction = "pick up the object on the left and place it in the bowl"
    out = pol.act({"image": img}, instruction)

    print("Action shape:", out.action.shape)
    print("Action:", out.action)

    site_vecs = pol.get_site_vectors()
    print("Captured sites:", len(site_vecs))
    for k, v in list(site_vecs.items())[:2]:
        print("  ", k, v.shape, np.linalg.norm(v))

    # Dummy steering directions: random vector with correct dimension
    # We infer dim from one captured site vector
    d = next(iter(site_vecs.values())).shape[0]
    dirs = {"wrong_object": np.random.randn(d).astype(np.float32)}
    pol.register_steering_directions(dirs)

    # Apply steering and call act again (this changes internal activations)
    pol.set_steering("wrong_object", alpha=0.5)
    out2 = pol.act({"image": img}, instruction)
    pol.clear_steering()

    print("Action2:", out2.action)


if __name__ == "__main__":
    main()
