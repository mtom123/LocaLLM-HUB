# Dreaming 17/11/2026 — come costruire la nuova gem (sintesi di 5 ricerche + piano)

## 1. Verdetto onesto sul sogno GLM-5.3-Flash su questa macchina

- `GLM-5.3-Flash-Coder-RCO-152E-IQ2_XXS`: **47,4 GB decimali (44,2 GiB)**, 170B parametri, 152 esperti/layer di 288, top-8, 45 layer (3 dense + 34 KDA + 11 MLA/DSA). La ricetta ufficiale dichiara **24 GB VRAM + 32 GB RAM** come minimo.
- La nostra macchina: **12 GB VRAM + 16 GB RAM**. Il file gira solo via mmap/NVMe e decide ~0,6 t/s. I 5 agenti concordano: con la memoria attuale **non è un utilizzo quotidiano**, a prescindere dai flag.
- Le evidenze sulla potatura spinta sono **negative**: gli autori RCO misurano KL 0,88 e loop infiniti già a **120 esperti** (di 288), KL 1,07 a 96. Il nostro K=24 (di 152) è fuori da ogni evidenza di qualità: lo testeremo come **esperimento di velocità**, non come gem.
- Il file RCO in nostro possesso **non contiene la testa MTP/NEXTN** (verificato: 0 tensori `nextn`/`mtp`), quindi niente speculative gratuito su quel file.

## 2. Le 5 leve che la ricerca ha portato (con fonti)

| Leva | Evidenza | Applicabile qui? |
|---|---|---|
| `NVIDIA_TF32_OVERRIDE=0` | PR ggml #27754: top-1 agreement su GLM-5.3 da 0,896 a 0,9995 | Sì, gratis — già nei driver |
| `--moe-cache-mib N` + `--moe-cache-cpu-share F` (PR #29887, #30966) | 1,6-2,2× su 4090/5090; ma su 12 GB il break-even è ~85% hit rate e probabilmente non si raggiunge | Misurare, non assumere |
| `-b 4096 -ub 4096` | su 3060 con offload dimezza il prefill (208 s → 80 s su 32K) | Sì |
| MTP nativo GLM (PR #29928, b11474) | 58,6 → 86,5 t/s su B200, lossless | No (file privo di testa; modulo separato 13,8 GB da convertire) |
| REAP saliency condizionata (ICLR 2026) | 50% pruning mantiene 97,6% coding, mentre frequency pruning collassa a 0,010 | Sì: migliorare il nostro potatore (oggi usa conteggi + sum2 come tie-break) |
| CISC / voto ponderato per confidenza (ACL 2025) | 8 campioni CISC battono 30 di self-consistency standard | Sì, da preregistrare con help/harm rate |
| Qwen3.5-9B Q4_K_M (~5,3 GB) | MMLU-Pro 82,5, GPQA-D 81,7, HumanEval+ 89,6 (vendor) | Sì: bake-off con duro-v1 n=15 |

Fonti principali: REAP arxiv 2510.13999; RCO repo `lapadatbogdan/GLM-5.3-Flash-Coder-RCO-152E`; ISTA GSQ arxiv 2604.18556; CAT-Q ICML 2026 e ScaleQ-1.58 arxiv 2608.01078; llama.cpp docs/moe-cache.md, docs/speculative.md, PR #27773/#29887/#29928/#30966; MMLU-Pro arxiv 2406.01574; CISC arxiv 2502.06233; "When Self-Consistency Hurts" arxiv 2611.11201; s1 arxiv 2501.19393; Dynamic Cheatsheet arxiv 2504.07952; "Revisit Self-Debugging" arxiv 2501.12793; RouteLLM/FrugalGPT; card JetBrains Mellum2.1 e card Qwen3.5-9B.

## 2b. Aggiornamento motori avanzati (secondo giro di ricerca, 17/11)

- **Strata 0.2.15**: verificato anche a mano (`strata launch --model-path ... --plan`): risponde *"this model type is not supported by the Strata engine"*. Supporta solo Qwen3.8-Flash-Next e Qwen3.6-35B-A3B; i fork GLM (lighttransport `glm53f`, OpenMOSE) vogliono 64-128 GB RAM e shape a 288/224/144 esperti, non 152. **Fuori**.
- **Gyro**: non è un motore ma un formato di quantizzazione trellis+Hadamard (AgentionAI) per Qwen3.8-Flash-Next. Nessun supporto `glm5-next`. **Fuori**.
- **Edge0 / edge0-strata**: il prerouter addestrato (arXiv 2609.18063) riduce la latenza, non i byte; è per-modello e serve training. **Fuori per il nostro GGUF**.
- **Project Maya**: l'unico motore dichiarato dalla scheda RCO-152E. Supporto Windows nativo, CUDA 13.4 e VS Build Tools verificati sul nostro PC; `--gguf-dir` permette di usare i nostri file invece dei pack da 89-108 GB. `--check` dice "This PC can run Maya". **In prova ora**.
- **llama.cpp + MoE cache/direct-I/O** resta il piano B: PR #25294 (SSD streaming) è ancora *open* e non mergiata; i flag in master sono `--moe-cache-mib`, `--moe-cache-cpu-share`, `-lm dio`. Con 47 GB su 12+16 GB le stime di letteratura per SSD-streaming sono 1,5-4 t/s: non è il risultato sperato, ma è misurabile.

## 3. Dreaming — tre traiettorie, in ordine di realizzabilità stanotte

### Traiettoria A — `GLM impossibile` (quello che vuoi, lo provo comunque)
1. Build **b12244 CUDA12.4**: unica che vede `CUDA0` *e* supporta `glm5-next` (la CUDA13.5 non vede GPU: tutti i vecchi numeri erano CPU-only).
2. `NVIDIA_TF32_OVERRIDE=0`, `-fa on` vs `off`, `-b/-ub 4096`, sweep di offload e ngram-speculative.
3. Imatrix → potatura aggressiva K=32/24/16/12/8 con misura di **t/s e PPL** per ogni variante. Le varianti si cancellano dopo la misura; resta solo la migliore.
4. Criterio di successo onesto: **≥15 t/s misurati**. Se la qualità crolla (PPL/duro), lo scrivo: è un esperimento di ingegneria, non la gem.

### Traiettoria B — `la gem che esiste davvero` (quella che può vincere)
1. **Mellum2.1 Q4_K_M**: 100 t/s (record della macchina). Primo duro-v1 completo a budget di default 1500: **50,0%** (sotto AREX 69,5). Ma test successivi mostrano che il collo è il **budget di pensiero**: su MMLU, 1500 → 4096 token alza l'accuratezza da **64,3% a 78,6%** e le risposte troncate da 28,6% a 7,1% (8192 non aggiunge nulla). Quindi il confronto corretto è a parità di budget 4096, non a 1500.
2. **MTP head ufficiale JetBrains** (`Mellum2.1-...-MTP-Q4_K_M.gguf`, 7,76 GB): `--spec-type draft-mtp -md <head>` → speedup atteso 1,4-2× senza cambiare le risposte.
3. **Bake-off**: stesso `duro.py`, stesse domande, n=15, per Mellum, **Qwen3.5-9B Q4_K_M** e (se avanza tempo) Gemma4-12B. Ipotesi preregistrata: vince Qwen3.5 se il composite pesa knowledge/math; vince Mellum se pesa codice/agentic.
4. **CISC gated**: N=8 con voto ponderato per confidenza, solo sui casi incerti. Misuro `greedy_ok`, `vote_ok`, `help`, `harm` per categoria; disattivo il voto dove l'harm supera l'help.

### Traiettoria C — `dopo l'upgrade` (non ora, come da decisione)
Con 64 GB RAM + 24 GB VRAM: Project Maya (tiering VRAM/RAM/NVMe + hot-expert scheduling, 18-21 t/s su 3090+128 GB) diventa sensato; Strix Halo 128 GB resta l'unica macchina che fa entrare entrambi i sogni. Niente acquisti stasera.

## 4. Regole di misura (preregistrazione)

- `duro-v1`, `n=15`, `jobs=1`, stessi set math/code/mmlu; differenze **<3 punti = rumore**, non si dichiara vincitore.
- Ogni nuovo risultato ha `engine`, `hardware`, `benchmark.suite`, `n`, `notes` con eventuali anomalie.
- I fallimenti si registrano (`status: "failed"`) con la causa, non si cancellano.
- Il sistema ibrido (Mellum+CISC+cascata) si chiama **`candidate-system-v2`**, non "punteggio di Mellum": mele con mele.
- Niente claim QLoRA <1000 esempi, niente n=8 spacciato per robusto, niente page file allargato.

## 5. Ordine di esecuzione della notte

1. Finish curva potatura Mellum (PPL corretta, duro completo) + commit.
2. Assalto GLM A (CUDA12 + potatura) — il desiderio esplicito.
3. Traiettoria B (MTP Mellum + bake-off Qwen3.5-9B + CISC gated) — la vera possibilità di una gem più intelligente.
4. A mattina: curva K/PPL, punteggi duro affiancati, decisione su cosa tenere.
