import torch
import torch.distributed as dist
from diffusers import FluxPipeline

dist.init_process_group()

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
).to(f"cuda:{dist.get_rank()}")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

mesh = init_context_parallel_mesh(
    pipe.device.type,
    max_ring_dim_size=2,
)
parallelize_pipe(
    pipe,
    mesh=mesh,
)
parallelize_vae(pipe.vae, mesh=mesh._flatten())

# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())

torch._inductor.config.reorder_for_compute_comm_overlap = True
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
import time
start= time.perf_counter()
image = pipe(
    "A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    num_inference_steps=4,
    output_type="pil" if dist.get_rank() == 0 else "pt",
).images[0]
print(time.perf_counter()-start)
start= time.perf_counter()
image = pipe(
    "A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    num_inference_steps=4,
    output_type="pil" if dist.get_rank() == 0 else "pt",
).images[0]
print(time.perf_counter()-start)
if dist.get_rank() == 0:
    print("Saving image to flux.png")
    image.save("flux.png")

dist.destroy_process_group()
