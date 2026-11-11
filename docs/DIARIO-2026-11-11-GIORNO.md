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

## 5. Distillazione verificata (in corso)

Pipeline pronta in `llm/fabbrica/training/`:
`gen_tasks.py` (compiti sintetici con test nascosti) -> `distill_run.py` (teacher A1 IQ3) ->
`gen_traces.py` (tiene solo le soluzioni che passano) -> `qlora_train.py` (4-bit nf4 + LoRA) ->
`eval_adapter.py` (A/B base vs adapter su holdout separato per seed).
Pesi HF di AREX-Turbo (8,5 GB) su `L:\hf` per non riempire C:.

Split pulito: train seed 7 (240 compiti), holdout seed 99 (40). Nessun compito viene da
MBPP/HumanEval, quindi `duro --sets=code,mbpp` resta uncontaminato.

## 6. Spazio e igiene

- Cancellati build vecchi non referenziati (`llm\bin`, `llm\bin-b10955`, 3,4 GB) per far posto al
  test router e alla distillazione. C: libero ~12 GB.
- Non toccati i modelli con launcher attivi (Flash-Next/Strata su C: e L:).
