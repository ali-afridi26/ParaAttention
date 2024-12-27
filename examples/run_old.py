import torch
import torch.distributed as dist
from diffusers import FluxPipeline

dist.init_process_group()

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    cache_dir = "/runware/sd-base-api/train_model/cache"
).to(f"cuda:{dist.get_rank()}")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

parallelize_pipe(
    pipe,
    mesh=init_context_parallel_mesh(
        pipe.device.type,
        max_ring_dim_size=2,
    ),
)

torch._inductor.config.reorder_for_compute_comm_overlap = True
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
import time
while True:
    st = time.perf_counter()
    image = pipe(
        "A cat holding a sign that says hello world",
        num_inference_steps=28,
        height= 1024,
        width=1024,
        output_type="pil" if dist.get_rank() == 0 else "latent",
    ).images[0]
    print(time.perf_counter()-st)
if dist.get_rank() == 0:
    print("Saving image to flux.png")
    image.save("flux.png")

dist.destroy_process_group()
