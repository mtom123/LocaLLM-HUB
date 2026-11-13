"""Costruisce un Agents-A1 per MLX che entra in un Mac da 16 GB: potatura degli expert + quantizzazione mista per strato.
Sorgente: un checkpoint MLX quantizzato (es. mlx-community/Agents-A1-8bit, quasi senza perdite). Ogni tensore viene
de-quantizzato e ri-quantizzato con i bit scelti; gli expert tenuti sono gli stessi di ONYX-224r (conteggi dell'imatrix
di llama.cpp, a parita attivazioni piu forti). Per 2 e 3 bit usa la ricerca del range per gruppo di mlx-optiq
(`optiq.core.lowbit.quantize_best`), che a 2 bit evita il collasso dei pesi su un solo codice.

Uso: python mlx_onyx.py SORGENTE_MLX IMATRIX.gguf|PROFILO_REAP.json USCITA --keep=224 --piano=onyx [--gguf-py=<llama.cpp>/gguf-py]
Piani (bit degli expert per strato, gruppo):
  onyx : strati 3,7,11,27,34-39 a 3 bit g64 (ONYX li ha a IQ3_XXS), gli altri a 2 bit g128; resto del modello 4 bit g64,
         router 8 bit, lm_head 6 bit.
  r3   : tutti gli expert a 3 bit g64 (per sorgenti gia potate, es. REAP-19B; IMATRIX = - e --keep=all)
  r3g128: tutti gli expert a 3 bit g128 (con potatura: --keep=160 o simili)
  onyx3: come onyx ma gli strati che ONYX ha a IQ2_S passano a 3 bit g128 (piu qualita, piu memoria).
Il vision tower resta com'e (non viene caricato nel solo testo).
"""
import glob, json, os, re, shutil, sys
import numpy as np
import mlx.core as mx

opt = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
SRC, IMX, DST = sys.argv[1], sys.argv[2], sys.argv[3]
K = opt("keep", "224")
PIANO = opt("piano", "onyx")
if opt("gguf-py"):
    sys.path.insert(0, opt("gguf-py"))
from gguf import GGUFReader  # noqa: E402
try:
    from optiq.core.lowbit import quantize_best
except ImportError:  # senza optiq: quantizzazione standard
    quantize_best = lambda x, g, b: mx.quantize(x, group_size=g, bits=b)

SENS = {3, 7, 11, 27, 34, 35, 36, 37, 38, 39}          # IQ3_XXS in ONYX-224r
MEDI = {4, 15, 18, 19, 23, 26, 30, 31, 32, 33}         # IQ2_S in ONYX-224r


def bits_for(path):
    m = re.search(r"layers\.(\d+)\.", path)
    l = int(m.group(1)) if m else -1
    if ".switch_mlp." in path:
        if PIANO == "r3":
            return 3, 64
        if PIANO == "r3g128":
            return 3, 128
        if l in SENS:
            return 3, 64
        if PIANO == "onyx3" and l in MEDI:
            return 3, 128
        return 2, 128
    if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
        return 8, 64
    if "lm_head" in path:
        return 6, 64
    return 4, 64


cfg = json.load(open(os.path.join(SRC, "config.json")))
tc = cfg.get("text_config", cfg)
n_exp = tc["num_experts"]
qsrc = cfg["quantization"]

# ---- expert da tenere (stessa regola di prune_experts.py)
score = {}
for t in (GGUFReader(IMX).tensors if IMX not in ("-",) and not IMX.endswith(".json") else []):
    for kind in ("ffn_down_exps", "ffn_up_exps", "ffn_gate_exps"):
        if f".{kind}." in t.name and t.name.startswith("blk."):
            l = int(t.name.split(".")[1]); a = np.asarray(t.data, dtype=np.float64)
            if t.name.endswith(".counts") and a.size >= n_exp:
                score.setdefault(l, {})["counts"] = a.reshape(-1)[:n_exp]
            elif t.name.endswith(".in_sum2") and a.size % n_exp == 0:
                score.setdefault(l, {})["sum2"] = a.reshape(n_exp, -1).sum(axis=1)
keep = {}
K = n_exp if K == "all" else int(K)
if IMX.endswith(".json"):  # profilo REAP di `optiq prune-experts --profile-only`: tiene i K expert con punteggio REAP piu alto
    prof = json.load(open(IMX))["layers"]
    for l, r in prof.items():
        v = np.array([-1e30 if x is None else x for x in r["reap"]], dtype=np.float64)
        keep[int(l)] = np.sort(np.argsort(-v, kind="stable")[:K])
    score = {}
for l, s in (score.items() if K < n_exp else []):
    order = np.lexsort((-s.get("sum2", np.zeros(n_exp)), -s["counts"]))
    keep[l] = np.sort(order[:K])
print(f"expert tenuti: {K} su {n_exp} in {len(keep)} strati", flush=True)


def src_q(prefix):
    q = qsrc.get(prefix)
    if isinstance(q, dict):
        return q["bits"], q["group_size"]
    return qsrc["bits"], qsrc["group_size"]


os.makedirs(DST, exist_ok=True)
qout = {"group_size": 64, "bits": 4, "mode": "affine"}
shards = sorted(glob.glob(os.path.join(SRC, "model*.safetensors")))
tot_bytes, out_idx, n_out = 0, {}, 0
for si, sh in enumerate(shards):
    W = mx.load(sh)
    prefixes = sorted({k[: -len(".weight")] for k in W if k.endswith(".weight")})
    out = {}
    for k, v in W.items():  # tensori non quantizzati (norme, conv, A_log, ...) e non-.weight senza scales
        base = k.rsplit(".", 1)[0]
        if k.endswith((".scales", ".biases")) and base + ".weight" in W and base + ".scales" in W:
            continue
        if k.endswith(".weight") and base + ".scales" in W:
            continue
        out[k] = v
    for p in prefixes:
        if p + ".scales" not in W:
            continue
        b, g = src_q(p)
        w = mx.dequantize(W[p + ".weight"], W[p + ".scales"], W.get(p + ".biases"), group_size=g, bits=b).astype(mx.bfloat16)
        m = re.search(r"layers\.(\d+)\.", p)
        l = int(m.group(1)) if m else -1
        if ".switch_mlp." in p and l in keep:
            w = w[mx.array(keep[l])]
        elif p.endswith("mlp.gate") and l in keep:           # router: righe = expert
            w = w[mx.array(keep[l])]
        if p.startswith("vision_tower"):
            nb, ng = b, g
        else:
            nb, ng = bits_for(p)
        qw, qs, qb = quantize_best(w, ng, nb) if nb <= 3 else mx.quantize(w, group_size=ng, bits=nb)
        out[p + ".weight"], out[p + ".scales"], out[p + ".biases"] = qw, qs, qb
        qout[p] = {"bits": nb, "group_size": ng}
        mx.eval(qw, qs, qb)
    name = f"model-{si + 1:05d}-of-{len(shards):05d}.safetensors"
    mx.save_safetensors(os.path.join(DST, name), out, metadata={"format": "mlx"})
    for k, v in out.items():
        out_idx[k] = name; tot_bytes += v.nbytes
    print(f"{name}: {len(out)} tensori, totale finora {tot_bytes / 1e9:.2f} GB", flush=True)
    del W, out
    mx.clear_cache()

json.dump({"metadata": {"total_size": tot_bytes}, "weight_map": out_idx}, open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=1)
tc["num_experts"] = K
cfg["quantization"] = qout
cfg["quantization_config"] = qout
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=1)
for f in os.listdir(SRC):
    if not f.startswith("model") and not f.startswith(".") and os.path.isfile(os.path.join(SRC, f)) and f != "config.json":
        shutil.copy(os.path.join(SRC, f), DST)
print(f"FATTO: {DST} {tot_bytes / 1e9:.2f} GB, piano {PIANO}, {K} expert")
