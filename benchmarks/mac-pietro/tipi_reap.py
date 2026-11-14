"""File dei tipi per llama-quantize: expert (_exps) al tipo dato, ogni altro tensore al tipo che ha gia (copiato identico)."""
import sys
from gguf import GGUFReader
src, exp_t, out = sys.argv[1], sys.argv[2], sys.argv[3]
r = GGUFReader(src); n = 0
with open(out, "w") as f:
    for t in r.tensors:
        ty = exp_t if "_exps." in t.name else t.tensor_type.name.lower()
        f.write(f"{t.name}={ty}\n"); n += 1
print("tipi scritti", n)
