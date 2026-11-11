# Diario 11/11/2026 — giorno: IQ3 in RAM, router, FreeToken, distillazione

## 1. A1-35B IQ3_XXS: il 35B che entra in RAM

| Test | Risultato |
|---|---|
| duro | **76,3** (Q4: 89,5) |
| agente | **91,7** (Q4: 100) |
| omp (6 compiti) | **4/6 pieni** (92,2 di media parziale), 39,5-40,4 t/s |
| contesto 64K | **2/2 aghi**, prefill **816 t/s**, decode **26,6 t/s**, 79 s totali |
| contesto 128K | **fallito**: connessione interrotta (modello 12,7 GB + KV 128K non stanno in 16 GB) |

Dettaglio omp: `bug_prezzi` 1,0 · `config_loader` 1,0 · `rinomina` 1,0 · `paginazione` 0,778 · `dati_ordini` 0,0 · `rapporto` 1,0.

Il 128K non e' un problema di kernel ma di RAM: il modello IQ3 pesa 12,7 GB e la KV q4_0 a 128K
porta il working set oltre i 16 GB. A 64K il conto torna. Il Q4 (21 GB su disco, 34,5 t/s) resta
il riferimento di qualita'; l'IQ3 e' il modello da tenere in RAM quando serve velocita' e spazio.

## 2. Crash CUDA con la prompt cache

Due run omp sono morti con `CUDA error: an illegal memory access was encountered`
(`ggml_backend_cuda_synchronize`), sempre quando il server faceva spazio nella prompt cache
(`--cache-ram 512 --cache-reuse 256`) con offload parziale e contesto 64K. Con
`--flash-attn on`, `-c 32768`, niente cache-ram/reuse, il run e' arrivato in fondo (4/6).
Ipotesi operativa: l'eviction della prompt cache su un modello che non entra interamente in VRAM
lascia un buffer libero mentre la graph lo usa. Da non usare cache-ram/reuse su A1 IQ3.

## 3. Router per tipo: l'ipotesi non regge

- A1 IQ3 sui compiti di specifica (`config_loader`, `paginazione`, `rapporto`): **3/3, 100%**.
- AREX 4B sui compiti veloci/dati (`bug_prezzi`, `rinomina`, `dati_ordini`): **1/3** (bug 1,0;
  rinomina 0,333 in 2 s, omp si e' fermato subito; dati_ordini 0,0). 70 t/s.
- Somma dei due: `rinomina` peggiora rispetto ad A1 da solo (1,0 -> 0,333) e `dati_ordini` resta
  0 per entrambi.

**Conclusione:** il router non conviene. A1 IQ3 da solo fa meglio ed e' piu' semplice; AREX resta
solo un fallback di velocita' (70 vs 40 t/s) a costo di qualita'.

## 4. FreeToken su Windows: muore per RAM, non per VRAM

Installato dai wheel ufficiali (`FreeToken-Web`, 0.2.2+cu130, venv Python 3.12 con
`torch==2.14.1+cu130`, CUDA visibile). `ft serve` riconosce il GGUF e seleziona `offload`, poi
`HostOutOfMemoryError` (`WinError 1455`): i bank degli esperti (11,7 GB) non stanno nei 16 GB di
RAM e il build seriale chiede un buffer extra per shard. Page file gia' a 24,5 GB: allargarlo
sposterebbe il problema sul disco. Record `n27-freetoken-a1-iq3` (failed, nascosto dal sito).

FreeToken risolve il caso "GPU piccola, RAM grande"; noi abbiamo il vincolo opposto.

## 5. Distillazione verificata: primo ciclo completo, nessun guadagno misurabile

Pipeline pronta in `llm/fabbrica/training/` (`gen_tasks.py` -> `distill_run.py` -> `gen_traces.py` ->
`qlora_train.py` -> `eval_ab.py`). 196 tracce verificate dal teacher A1 IQ3, QLoRA 4-bit su AREX-Turbo
(21,2M parametri, 147 step, 22 min, loss 1,02 -> 0,15).

**A/B su 20 compiti di holdout separati per seed: base 6/20 (30%), LoRA 7/20 (35%).** Cinque compiti
migliorati e quattro peggiorati: +1 su 20 e' dentro il rumore, quindi **nessun guadagno dichiarato**.
Dettagli e next step in [`TRAINING-2026-11-11-DISTILLAZIONE-AREX.md`](TRAINING-2026-11-11-DISTILLAZIONE-AREX.md).

Pesi HF di AREX-Turbo (8,5 GB) su `L:\hf` per non riempire C:.

## 6. Spazio e igiene

- Cancellati build vecchi non referenziati (`llm\bin`, `llm\bin-b10955`, 3,4 GB) per far posto al
  test router e alla distillazione. C: libero ~12 GB.
- Non toccati i modelli con launcher attivi (Flash-Next/Strata su C: e L:).

## 7. Sera: MoE cache, ubatch e il re in RAM

Tre esperimenti rapidi per capire cosa si puo' spremere dal 35B su 12+16 GB.

| Test | Prefill | Decode | Esito |
|---|---:|---:|---|
| A1 IQ3 64K, KV q4, `-ub 1024` (baseline) | 816 t/s | **26,6 t/s** | riferimento |
| A1 IQ3 64K, `-ub 2048` | **945 t/s (+16%)** | 22,5 t/s (-15%) | piu' lettura, meno scrittura, RAM 0,1 GB: non adottato |
| A1 IQ3 64K, `-cmoe --moe-cache-mib 5000` | 309 t/s | 14,5 t/s | **peggio del baseline**: gli esperti su CPU costano piu' di quanto la cache GPU recuperi |
| A1 Q4 64K (da L:), `-ub 2048` | 129 t/s | 2,8 t/s | **thrash**: 19,7 GB di modello non stanno in 16 GB di RAM con il desktop aperto |

**Lezione:** il Q4 e' il re per intelligenza (duro 89,5) ma **solo su macchina scarica**: con browser,
Codex e app aperte la RAM libera va a 0,1 GB e il decode crolla da 26 a 2,8 t/s. L'IQ3 (12,7 GB) e'
l'unico 35B che convive con il desktop.

## 8. E4B: il miglior compromesso qualita/velocita/contesto nella fascia piccola

Rimesso su disco (`L:\modelli\e4b`, 3,45 GB) e ritestato:

| Test | Risultato |
|---|---|
| duro n=8 | **78,9** (30/38: code 100, mbpp 87,5, mmlu 71,4, math 62,5) |
| agente | 50 |
| omp 6 compiti | **1/6 pieni** (parziale 34,6%), 61,6 t/s |
| contesto 128K | 3/3 aghi, prefill 1426 t/s, decode 19,1 t/s (test precedente) |
| contesto 256K | **impossibile**: il server si autolimita a 128K per memoria |

E4B e' quindi il modello "tutti i giorni" per qualita' generale e contesto lungo a velocita' alta;
sull'agentica resta indietro (1/6 contro 4/6 di A1 IQ3).

## 9. Il campione conta: n=8 era ottimista (stesso protocollo del Mac, n=15 / 59 domande)

| Modello | duro n=8 (38 domande) | **duro n=15 (59 domande)** | agente | decode |
|---|---:|---:|---:|---:|
| AREX-Turbo 4B (2,9 GB) | 76,3 | **69,5** | 91,7 | 74 t/s |
| A1-35B IQ3 (12,7 GB) | 76,3 | **64,4** | 91,7 | 39 t/s |
| Nemotron E4B (3,5 GB) | 78,9 | **62,7** | 50 | 62 t/s |
| A1-35B Q4 (19,7 GB) | 89,5 | da rifare (RAM) | - | 34,5 t/s |

Con lo stesso campione del Mac (n=15) il quadro cambia: **AREX 4B e' il migliore dei tre piccoli**
(69,5, identico all'ONYX-mini del Mac), l'IQ3 perde 12 punti rispetto all'n=8, E4B scende a 62,7.
Le classifiche precedenti su n=8 erano ottimiste: da oggi il QI va letto a n=15.


