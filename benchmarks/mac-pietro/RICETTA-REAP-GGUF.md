# Ricetta: Agents-A1 REAP-192 / REAP-176 IQ3_XXS per llama.cpp (Mac M4, 16 GB), 14 novembre 2026

Il nuovo `a1-locale veloce` sul Mac e **REAP-192 IQ3_XXS con solo ngram-mod** (senza DFlash, che con 10,9 GB di pesi non
entra in memoria): **duro n=30 75,4 (89/118), OMP agente-mini 6/6 in 292 s, esplorazione 3/3 in 96 s, ~52 t/s**.
Batte i riferimenti di Windows di Motti (AREX duro 69,5, OMP 5/6). REAP-176 (stessa ricetta, 10,1 GB) entra con DFlash ma
perde molto: duro 64,4, OMP 4/6.

La ricetta prende la classifica degli expert di REAP-144 per MLX (vedi `RICETTA-REAP-MLX.md`) e la porta su llama.cpp,
che ha IQ3_XXS e gli speculativi.

## Idea
- ONYX-224r ordinava gli expert con i conteggi dell'imatrix e li teneva quasi tutti a 2 bit.
- Qui: **meno expert (176 su 256) ordinati con REAP**, e quelli rimasti a **IQ3_XXS** invece di IQ2_XXS.
- Tutti gli altri tensori (attenzione, strati lineari, router, embedding) restano **identici** al Q4_K_M di partenza,
  grazie a un file dei tipi.

## Passi
`<A>` = cartella dei modelli A1, `<P>` = profilo REAP `mep_profile.json` (stesso di MLX, da `optiq prune-experts --profile-only`).
```bash
# 1. potatura del GGUF Q4_K_M (mradermacher) a K expert per strato (192 o 176), classifica REAP
python -I prune_experts.py <A>/Agents-A1.Q4_K_M.gguf <A>/imatrix-a1-v2.gguf 176 <A>/tmp-reap176-q4km.gguf --reap=<P>
# 2. stesso taglio sull'imatrix (servono gli stessi expert per riquantizzare)
python -I prune_imatrix.py <A>/imatrix-a1-v2.gguf 256 176 <A>/tmp-imatrix-reap176.gguf --reap=<P>
# 3. file dei tipi: expert IQ3_XXS, il resto com'e
python -I tipi_reap.py <A>/tmp-reap176-q4km.gguf iq3_xxs tipi-reap176.txt
# 4. riquantizzazione
llama-quantize --allow-requantize --imatrix <A>/tmp-imatrix-reap176.gguf --tensor-type-file tipi-reap176.txt \
  <A>/tmp-reap176-q4km.gguf <A>/Agents-A1-REAP176-iq3.gguf IQ3_XXS 4
```
Risultato: 10,92 GB (192) o 10,14 GB (176). I comandi sopra sono per 176: per 192 cambiare 176 in 192. Il file Q4_K_M potato intermedio (circa 15 GB) si puo cancellare.

## Server (profilo `a1-locale veloce`)
```bash
llama-server -m Agents-A1-REAP192-iq3.gguf -ngl 99 -fa on --jinja --parallel 1 --cache-ram 512 \
  -c 65536 --cache-type-k q4_0 --cache-type-v q4_0 -b 512 -ub 256 \
  --spec-type ngram-mod --dry-multiplier 0.5 --dry-allowed-length 8 --reasoning off
```
Per REAP-176 (`a1-locale veloce-176`) si aggiunge il drafter: `--spec-type draft-dflash,ngram-mod -md Qwen3.5-35B-A3B-DFlash-Q4_K_M.gguf -ngld 99`.

## Confronto (Mac M4 16 GB)
| Modello | Motore | GB | t/s | duro n=15 | agente | OMP agente-mini | OMP esplora |
|---|---|---|---|---|---|---|---|
| **REAP-192 IQ3_XXS** | llama.cpp ngram-mod | 10,92 | ~52 | **71,2** (n=30: **75,4**) | 11/12 | **6/6, 292 s** | **3/3, 96 s** |
| REAP-200 IQ3_XXS | llama.cpp ngram-mod | 11,31 | ~56 | n=30: 76,3 | - | 6/6, 624 s | 2/3, 664 s |
| REAP-176 IQ3_XXS | llama.cpp DFlash+ngram | 10,14 | ~42,5 (solo ngram: ~42) | 64,4 | 12/12 | 4/6, 698 s | 2/3, 114 s |
| ONYX-224r | llama.cpp DFlash+ngram | 10,21 | ~51 | 55,9 | 12/12 | 4/6, 687 s | 1/3, 896 s |
| REAP-144 MLX 3 bit | MLX | 10,33 | 39 | 67,8 (n=30: 67,8) | 12/12 | 4/6, 1723 s | 2/3, 908 s |
| REAP-128 MLX 3 bit (profilo ricco) | MLX | 9,44 | 36 | 69,5 (n=30: 66,1) | 11/12 | - | - |

OMP: `omp_agente.py --snello --profile=locale`, ragionamento spento. REAP-192 ripetuto: agente-mini 5/6 in 267 s, esplora 2/3 in 120 s.
REAP-200: duro entro il rumore di REAP-192, ma in OMP il doppio dei passi e dei secondi, con la memoria al limite: 192 resta il punto giusto.

DFlash non serve sempre: su REAP-176 con o senza drafter la velocita e quasi uguale (~42 t/s), e REAP-192 senza drafter
va piu veloce (~52 t/s), forse perche risponde in modo piu compatto e ngram-mod trova piu ripetizioni.

## Cosa non funziona
- **REAP-192 con DFlash a 64K**: 10,9 GB di pesi piu il drafter superano la memoria della GPU
  (`kIOGPUCommandBufferCallbackErrorOutOfMemory` al primo prompt). Sul M4 da 16 GB il limite pratico con DFlash e circa 10,2 GB.
