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
- **Basalt** (`jesdga95/basalt`, MIT, v0.4.1 16/11/2026, fork di Strata): reale e più veloce di Strata su prefill/TTFT/JSON (test indipendente: TTFT 100K 41,1 s vs 69,2), ma supporta solo Qwen3.8-Flash-Next / DeepSeek-V4.1 / Qwen3.8-27B, richiede GPU Blackwell sm_120 e 64-128 GB RAM. **Nessun supporto `glm5-next`** → valutato e scartato per il nostro caso.
- **llama.cpp + MoE cache/direct-I/O** resta il piano B: PR #25294 (SSD streaming) è ancora *open* e non mergiata; i flag in master sono `--moe-cache-mib`, `--moe-cache-cpu-share`, `-lm dio`. Con 47 GB su 12+16 GB le stime di letteratura per SSD-streaming sono 1,5-4 t/s: non è il risultato sperato, ma è misurabile.

## 2c. RISULTATO REALE (18/11, 00:52) — GLM RCO-152E gira su Maya

- `START-MAYA.bat --setup --gguf-dir <cartella RCO> --model glm-maya-s --context 4096 --no-vision` ha costruito il pack dal **nostro GGUF da 44,2 GiB** (1257 tensori), senza scaricare i pack da 89 GB.
- Tre ostacoli risolti senza toccare page file o app dell'utente: (1) `--vram-reserve-mib 300` + `MAYA_GLM_POOL_GB` per il pool, (2) `--repack` dopo aver letto i metadati, (3) **un llama-server mio rimasto vivo da 10,9 GB che esauriva il commit**: fermato, commit libero 9,9 → 20,8 GB.
- Top-7 (modifica sperimentale di `expert_used_count`) causava un access violation; **ripristinato top-8**, pack ricostruito, tutto stabile.
- Config finale: pool VRAM 4 GB (18 slot/layer), tier RAM, `MAYA_GLM_READ_CHUNKS=4`, `MAYA_GLM_PREFETCH_N=1`, `MAYA_GLM_RAM_HEADROOM_GB=3`.
- **Misura (protocollo abbreviato, 3 prompt × 48 token): 1,02 → 1,41 → 1,60 t/s** (mediana wallclock, engine ~2,0 t/s). Sanity corrette (391, palindromo, spiegazione).
- Collo dominante: **~180 letture SSD/token, ~470 ms/token**, hit VRAM ~45%. Il modello risponde ma è lontano dai 15 t/s: per avvicinarsi servono più RAM/VRAM, non un altro flag. Questo è registrato come `2026-11-18-glm-rco152e-maya-feasibility.json`.

## 2d. Seconda ondata di ricerca (18/11) — idee nuove e limiti duri

- **ik_llama.cpp** (`ikawrakow/ik_llama.cpp`): GLM-5.3-Flash (PR #2376), MTP (PR #2548), **DFlash2 (PR #2761: 16,35 → 20,61 t/s, +26% su code)**, MoE MMQ ottimizzato (#2831), dynamic expert cache (#2644). È il fork da provare prima di scrivere kernel nuovi. In prova ora.
- **Compilazione**: TMA/tcgen05/WGMMA **non esistono su sm_86** (RTX 3060 è Ampere): i megakernel 2026 vanno riscritti con `cp.async` + `mma.sync` int8. ggml fa già IQ2_XXS → int8 → tensor core; il guadagno concreto è estendere la fusione MMQ a IQ2_XXS (oggi solo Q4_K/Q8_0/IQ4_XS/…) e fondere gate/up. Stima onesta: 1,26 × 1,3 × 1,15 ≈ **2,3×**, non 10×.
- **Cascade Mellum→GLM**: con gate a confidenza sui logit (Mellum 60 t/s, GLM 1,6, escalate 5%) l'aritmetica dà ~19 t/s aggregati. Zero fork, 2-4 h: è l'esperimento ad alto ROI da fare dopo ik.
- **Verso il "quasi-lossless"**: **GSQ** (ISTA, ICML 2026) rifinisce GGUF in-formato (+6,25 pp su Q2_K) ma serve il teacher FP8/BF16 e 8×H200 per ~1-2 giorni; **EoRA/QER** aggiungono correzioni low-rank (+3…+11 pp task-specific, 128 sample). **Lossless matematico a 2 bit è impossibile**: mancano ~145 GB di informazione rispetto a FP8; 1-2 GB di adapter coprono <1,5%. "Task-lossless" dimostrato solo a 2,75-3,5 bpw.
- **Prefetch a impronta del router** (n-gram sugli ID esperti del prefill) e **per-expert bit-width dinamico** (DynaExq: +4,5 pp e 2,73× throughput su Qwen3-80B) sono le due idee originali da implementare nel fork Maya se il cascade non basta.

## 2e. Risultato self-consistency (18/11, 01:30) — il "candidate system v2"

- Mellum2.1 Q4_K_M, duro-v1 n=15, budget 4096, **8 voti con early-stop adattivo: 65,9%** (math 80,0 / code 60,0 / mmlu 57,1) contro **56,8%** con 1 voto. Un recheck vote=1 subito dopo ha riprodotto **esattamente 56,8%** con le stesse per-categoria → non era drift di sessione.
- Con **16 voti: 63,6%** (math/mmlu identici, code 53,3): oltre 8 non si guadagna, e il ramo codice (che non usa voting) oscilla di ±6,7 pp a n=15 → è rumore, non segnale.
- Significance: +9,1 pp è sopra il rumore di ±3, ma il confronto non è bootstrap appaiato per-domanda (non ho salvato le risposte singole del run storico): va trattato come **forte evidenza, non come prova statistica**. Replica di votes8 in corso.
- **Ik_llama.cpp**: compila, vede CUDA e carica glm5-next, ma richiede ~40 GB di RAM pinnabile; con `GGML_CUDA_NO_PINNED=1` e mmap serve ~5 min/prompt → più lento di Maya. Registrato come fallimento.
- Prossimo: **bake-off Qwen3.5-9B Q4_K_M** (5,29 GB, MMLU-Pro 82,5 vendor) con lo stesso metro e 8 voti. Se supera 69,5 (AREX) in modo riproducibile, è il nuovo campione del repo.

## 2f. Sperimentazione motoristica su GLM (18/11, 08:00-10:00)

- **DFlash2** (drafter 0,7 GB): 0,8 t/s contro 1,6 senza → **kill criterion**, scartato. Causa: costo di verify ~3,5-4,5× il decode su 47 GB con esperti offloadati; anche con accettazione perfetta non può vincere.
- **Profilo esperti / usage tracking**: 0,8 t/s nel test falsato da `--dflash`; ritest pulito con profilo STRP 46×152 e 5.834 coppie reali: 1,63 vs 1,60 → **nessun guadagno** (soglia 0,26). L'adaptive tier di Maya impara già a runtime.
- **Sweep 6 knob** (A/A due volte, baseline finale): noise 0,01, drift 0,29, soglia 0,58 → **nessun knob promosso**. `stream_fetch_off` peggiora (1,38). Termica stabile (45-52 °C, clock 1650): il drift non è termico.
- **Prefill**: 2.373 token richiedono 64 s (37 t/s). Con conversation/slot cache, il secondo turno riusa **2.372 token** e passa da **78,6 s a 14,6 s (5,4×)**. Il prefill va evitato, non ottimizzato.
- **Batching `--parallel 2` + `MAYA_GLM_GROUP_EXPERTS=1`**: 2 richieste parallele 63,1 s contro 61,1 s sequenziali (0,97×): la seconda KV sottrae VRAM al pool esperti più di quanto il grouping risparmi. Scartato.
- **Contesto con KV INT8**: 4K → 1,62 t/s, 32K → 1,62, 128K → 1,56, **256K → 1,36**. Il GLM RCO-152E regge 256K di contesto su 12+16 GB con Maya: il costo è solo −16% di decode, non un errore di memoria. Record nel repo: `2026-11-18-glm-maya-context-256k.json`.
- **Limite fisico**: 336 letture di esperti/token × 6,5 MB = 2,18 GB/token; a 2,2 GB/s di NVMe il limite senza cache è ~1 t/s. Per 16 t/s servirebbe 94% di hit, cioè ~39 GB residenti: impossibile con 28 GB totali. Il massimo onesto con 2-3× di hit improvement è 3-5 t/s; 16-17 t/s richiedono una macchina con 121 GB unificati (DGX Spark, misurato).

## 2g. SVOLTA: Edge0-120B su 12+16 GB (18/11, 15:00)

- Scaricato `Edge0/Edge0-120B-q3k` senza il PLE da 47,7 GB (41,7 GB di file reali), compilato `edge0-infer` (CPU/AVX2) e `edge0-strata` (CUDA 13.4, sm_86).
- Tre patch applicate al fork locale: (1) **`STRATA_ARENA_MMAP` portato a Windows** con `MapViewOfFile`: `experts.bin` da 26,7 GB mappato read-only invece di 26,7 GB di RAM/pinning; (2) **PLE ghost**: GGUF sparse NTFS da 47,68 GiB logici con header valido e righe a zero (occupazione ~0); (3) `experts.bin` esteso di 1.164.800 B di padding e `LatentA::open` allentato per accettare `want..want+blob`.
- **Risultato: il modello da 120B gira** a **2,29 t/s**, risposta corretta (391), **131072 token di contesto**, 105 esperti residenti, PCIe 13,1 GB/s, MTP attivo. `2026-11-18-edge0-120b-feasibility.json`.
- **4 round di tuning breve** (2-3 prompt da 128 token, nessun test da ore): contesto 131K→8K 4,32→7,51 t/s (1,74x); PCIe 0.36 ~ neutrale; KV q4_0 6,19-6,66 (peggiora a 8K, meglio solo sopra ~33K per il crossover calcolato); spec2 5,03; SM86_PERF=0 e spec0 non partono. Round 4: PCIe 0.20 7,18, async tier (`NOWAIT=0 LAG=1 ASYNC=1`) 7,08, baseline 6,84. **Combo finale PCIe 0.20 + async: 7,36 t/s** con warm-up esplicito (`2026-11-18-edge0-120b-combo-finale.json`).
- Gli agenti (GitHub + paper + meta) concordano: il PLE ghost costa <0,2 ms/token e `--no-ple` cambierebbe il modello (vietato); il collo è la **residenza degli esperti** da SSD/RAM. Il +2x osservato cold→warm (3,34→8,49 t/s) è paging, non intelligenza: da cui il warm-up nei benchmark. Duro corto n=15 budget 256 in corso per la qualità.
- È il "best brain" che gira su questa macchina: 120B contro i 12B/9B di prima. Duro-v1 con budget 512 in corso (con 4096 richiederebbe ore); il confronto con i run a 4096 non è diretto.
- Il PLE ghost degrada la memoria n-gram: se il duro delude, il passo successivo è generare righe PLE approssimate nel reader invece di zeri, senza scaricare 47,7 GB.

## 2h. Hy4-flash (18/11 pomeriggio) — il nuovo goal

- Modello uscito ieri: **220B totali, 11,3B attivi, 42 layer, 512 esperti/16 attivi, Engram da 63,4B, MLA ibrido + DFly2**. Arch GGUF `hy_v4_flash`.
- L'unico runtime esistente è **llama.cpp PR #32517 @ `0a054d0`** (draft), compilato con CUDA sm_86 aggirando un bug della web UI (placeholder `workbox[hash].js`). Build e comandi nel repo.
- GGUF ufficiali 125-150 GB: usata quant community **AtomicChat AD-IQ2_S/IQ2_XXS** (45,69+33,19 GB). Ghost Engram a zeri: modello carica ma **produce nonsense** (l'Engram è il 29% dei parametri) → scaricata tabella reale da 33,19 GB e sostituita; qualità restaurata (391 e codice corretti).
- **Router starvation** (patch in-place `expert_used_count` 16→K, offset 1050): K16 0,94 t/s ( corretto), K8 1,75 (corretto), **K6 3,41 (corretto, scelto)**, K5 2,21 (corretto), K4 4,89 (**sbagliato**: 391 errato). Curva non monotona: sotto 6 il danno supera il guadagno.
- Duro short K6: 57,1% MMLU ma 71,4% troncate a budget 256 → non confrontabile; duro a 512 in corso per il verdetto vero.
- Gli agenti (HF/GitHub/paper) dicono: 100 t/s su 12 GB non fisicamente credibile; target realistico **battere Edge0 7,36 t/s**. Per ora Hy4 è a 3,41 t/s ma con cervello molto più grande e qualità sanity corretta.

## 2i. Ricerca post-Hy4 (4 agenti, 18/11 sera) — cosa provare dopo

- **Campione attuale resta Mellum** (100 t/s, duro 56,8): miglior rapporto t/s÷qualità. Qwen3.8-7B/W7B **non esistono** come modelli ufficiali; i prune community 7B sono inutilizzabili (CE 6,26-6,34).
- **Pista calda mai provata**: `Qwen3.8-Flash-Next` **GSQ-RCO IQ2_XS** (39,2 GB backbone + 28,8 GB PLE separato) con engine **Betelgeuse** (fork llama.cpp): **16,8 t/s freddo / 22,03 caldo su RTX 3060 12GB + 16GB RAM**, bit-exact (0 diff su 576 token) e task-average 89,16 (AIME25 96,7, GPQA-D 87,4, LCB 83,4). Con Strata low-RAM solo 8,11 t/s, con llama.cpp stock 1,5. Serve ~68 GB di disco: liberabile cancellando Hy4 (78,9 GB) dopo il duro.
- **llamAmpere v0.7.0.1 preset 12dense + Qwen3.8-27B Swift 1.5 3bpw** (9 GB): 70,8 t/s @100K su 3080 Ti 12GB ma con ~90 GB RAM; con i nostri 16 GB partire da 64K e verificare. Binari solo Linux (da buildare o WSL2).
- **Router ibrido Mellum↔modello lento** (Gatekeeper/RouteLLM/speculative cascades): con deferral 5-10% si passa da 3,41 a 26-41 t/s blended; i tokenizer diversi vietano il draft token-level, va fatto a livello di routing.
- **Engram-PEFT / distillazione two-stage**: +2-5 punti innestando memoria n-gram in un 7B 4-bit; 10-30k tracce sintetiche via API (~$17) + QLoRA 1-3h danno +3-6 punti mirati, non +10 generali. Niente trapianto letterale di esperti tra architetture.
- **Glitch di sistema** (CUDA graphs + MTP + KV q4 + pinned I/O): 1,5-2,3× misurati su stack simili; da fare solo dopo aver scelto il modello.
- **Avvertenze**: PR Hy4 #32517 è draft con 2 finding bloccanti; DFly2 non è ancora nei GGUF (confermato Tencent 18/11 15:32); Halocline è Ada-only; Edge0-145B richiede 20 GB RAM.

## 3. Dreaming - tre traiettorie, in ordine di realizzabilità stanotte

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
