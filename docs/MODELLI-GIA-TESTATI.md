# Modelli già testati nel lab — CONTROLLARE PRIMA DI SCARICARE

Aggiornato 19 novembre 2026. Consultare questo file prima di lanciare qualsiasi nuovo download/test.

## Campioni attuali
- **Qwen3.8-35B-A3B-Distill IQ2_M** (empero, 11,7 GB) — VELOCITÀ: 72,3 t/s, duro 77,3%
- **Qwen3.8-27B UD-IQ2_XS** (unsloth, 7,1 GB) — QUALITÀ: 20,9 t/s, duro 84,1%

## Già testati (NON ripetere)
Agents-A1 (42+ varianti: Q4_K_M, ONYX, 35B-A3B, i1, 4B, REAP176...), BAAI_AREX-Turbo (14 test, Q4_K_M/IQ2_M/OMP), Mellum2.1 (9 test, base/RCO/pruning/MTP), Qwen3.8-27B (UD-IQ2_XS, GSQ-RCO, Dirk, bartowski, Flash-Next), Qwen3.8-35B-A3B empero (IQ2_M/IQ3_M), Qwen3.8-Flash-Next (8 test), GLM-5.3-Flash (10 test), NVIDIA-Nemotron-3.5-Bolt (9 test), Edge0-120B, Hy4-flash, Ornith-1.5-35B-A3B (APEX MTP/GGUF/9B), Arete-27B, Qwen3.5-9B-Kimi-k3, LFM2.5-24B-A2B, DeepSeek-V4-Pro-Qwen3.5-9B, Cyber-Tiel-Coder-35B, Laguna-XS, flint, Omni-Edu-27B, Mimo-V2.6-Distill, Sharp-Spark-X2.5-4B, OTel-2.0-Decision-26B, **Ternary-Bonsai-2-27B (10 nov + ritest 19 nov)**.

## Testati oggi 19 nov (nuovi)
- GSQ-RCO IQ2_XS su 27B: 75,0% (peggiore di UD 84,1%) — archiviato
- Sampler ufficiale Empero (temp 0.6) su 35B: 70,5% (peggiore di greedy 77,3%) — archiviato
- ThinkPRM best-of-4: OOM con due server — archiviato
- **SenseNova-Loop-9B 100K** (fork patched): 18,4 t/s, duro in corso — NUOVA ARCHITETTURA

## Testati oggi 19 nov (secondo round)
- **K2-Horizon-7B Q8_0** (IFM, 8,92 GB) — 34,9 t/s, duro in corso — NUOVO modello 7B dense settembre 2026
- Router q35/q27 con gate: fallito (confidenza non separa, troncamento non si fixa) — router mode llama.cpp pero verificato senza OOM

## Novità 17-19 nov verificate ma NON ancora testate
- DaisyRose-27B (18 nov, RL, SWE-Pro 94.7) — GGUF IQ2_XS 9 GB, 0 download, rischio 2-bit
- pi-adapted LoRA (16 nov, +25 pp) — serve base Q8_0 29GB, non entra in 12GB
- llamAmpere v0.7.0.1 (18 nov) — nessun dato 3060 12GB

## Lezioni
- MODEL CARD: Qwen3.8/Empero sconsigliano greedy su GENERAZIONI LUNGHE; il nostro benchmark chiede RISPOSTA: <numero> (domanda chiusa) dove EMNLP 2026 dice che greedy vince. Non confondere i due regimi.
- duro.py: BENCH_EXTRA sovrascrive temperature per tutti i voti; il parser prende l'ULTIMA RISPOSTA (m[-1]) — corretto.
- RAM 16GB: MAI due server LLM insieme → OOM. Un solo modello alla volta.
- Fork SenseNova: build con -G Ninja, CUDA_PATH, placeholder `workbox[hash].js` in build/tools/ui/dist.
