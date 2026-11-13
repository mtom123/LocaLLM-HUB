# Ricetta: Agents-A1 REAP-144 a 3 bit per MLX (Mac M4, 16 GB), 13 novembre 2026

Il miglior Agents-A1 sul Mac da 16 GB finora: **duro n=15 67,8 (40/59), agente 12/12, 39 t/s** (core-speed, a orologio).
Batte ONYX-224r su llama.cpp (55,9 / 12/12 / ~51 t/s con DFlash) di 12 punti su duro, e REAP-128 pubblico (64,4 / 9/12 / 38 t/s).

## Perche MLX
Sul M4 base mlx-lm 0.32.2 fa girare i MoE circa 2 volte piu veloci di llama.cpp senza speculativo
(Qwen3.5-35B-A3B: 55 t/s contro 28). Ma MLX non ha i formati IQ2/IQ3 di llama.cpp: sotto i 3 bit crolla.
- 2 bit affine standard: modello distrutto (duro 1,7, 88% risposte in ciclo).
- 2 bit con la ricerca del range di optiq (`quantize_best`): meglio, ma peggio della potatura (ONYX-224 per MLX: 54,2).
- Quindi: **3 bit per gli expert e meno expert** al posto di 2 bit e tutti gli expert.

## Quale classifica degli expert
- Conteggi dell'imatrix di llama.cpp (come ONYX): **crolla a 160** (duro 13,6). Spiega anche il vecchio crollo di ONYX-176.
- REAP (peso del router x norma dell'uscita dell'expert, `optiq prune-experts --profile-only`): regge fino a 128-144.

## Passi
1. Sorgente quasi senza perdite: `mlx-community/Agents-A1-8bit` (37,7 GB, resta sul disco e si legge a pezzi).
2. Calibrazione ricca (`calib-reap-a1.jsonl`): i 40 campioni del mix di optiq piu 209 brani di prosa da 5.000 caratteri
   (Wikipedia italiano e inglese, codice) presi da `calibrazione-a1-v2.txt`. Una riga JSON `{"text": ...}` per campione.
3. Profilo REAP (circa 13 minuti sul M4):
   ```bash
   optiq prune-experts <M>/Agents-A1-8bit -o <M>/reap-tmp/out --profile-only --n-samples 96 --calibration-mix calib-reap-a1.jsonl
   ```
   Scrive `<M>/reap-tmp/mep_profile.json` (punteggio REAP per expert e per strato). Con optiq attuale il comando finisce con un
   TypeError in `_measure_ranking` (manca `target_retention` con `--profile-only`), ma il profilo e gia scritto: si usa quello.
4. Potatura e ri-quantizzazione:
   ```bash
   python mlx_onyx.py <M>/Agents-A1-8bit <M>/reap-tmp/mep_profile.json <M>/A1-R144-r3 --keep=144 --piano=r3
   ```
   Expert 3 bit gruppo 64 (ricerca del range di optiq), router 8 bit, lm_head 6 bit, resto 4 bit: 10,33 GB.
5. Server: `python mlx_serve.py --model <M>/A1-R144-r3 --port 8080 --max-tokens 8192`
   (ragionamento spento: `--chat-template-args '{"enable_thinking":false}'`).
   Per OMP: profilo `omp-profilo-mlx/` in `~/.omp/profiles/mlx/agent/`, poi `omp --profile=mlx --model mlx/agents-a1`.

## Varianti provate (stesso profilo)
| Variante | GB | t/s | duro n=15 | agente |
|---|---|---|---|---|
| REAP-144, 3 bit g64 | 10,33 | 39,2 | **67,8** | **12/12** |
| REAP-160, 3 bit g128 | 10,58 | 35,1 | 62,7 | 11/12 |
| REAP-128 pubblico (19B), 3 bit g64 | 8,59 | 38 | 64,4 | 9/12 |
| Potato a 160 da conteggi imatrix, 3 bit g128 | ~10,6 | - | 13,6 | 8/12 |
| ONYX-224 per MLX (2/3 bit per strato) | 11,4 | - | 54,2 | fuori memoria |

Il gruppo 64 conta piu dei 16 expert in piu: REAP-144 g64 batte REAP-160 g128 su qualita e velocita.
Rumore: con 59 domande ±6 punti; con 15 per categoria ±13.

## Cosa non funziona (ancora)
- **Speculativo in mlx-lm 0.32.2**: con qualunque drafter (provato Qwen3.5-0.8B-4bit, stesso tokenizer) il server rifiuta:
  `Speculative decoding requires a trimmable prompt cache (got {'ArraysCache'})`. Gli strati a attenzione lineare di
  Qwen3.5/A1 hanno uno stato che non si puo riavvolgere quando una bozza viene scartata. Su llama.cpp invece DFlash funziona.
- **OMP con le impostazioni predefinite del server**: dopo 1-2 compiti la GPU finisce la memoria
  (`kIOGPUCommandBufferCallbackErrorOutOfMemory`): 10,3 GB di pesi piu passi di prefill da 2048 token e piu cache dei prompt
  superano i 12,7 GB che MLX puo usare. Con `--prefill-step-size 512 --prompt-cache-size 1 --prompt-cache-bytes 1000000000`
  (vedi risultati G12b/E12b/E13b).

## OMP vero su REAP-144 (profilo `omp-profilo-mlx`, server con i limiti di memoria sopra)
| Prova | REAP-144 MLX | ONYX-224r llama.cpp veloce (DFlash) |
|---|---|---|
| agente-mini, ragionamento spento | 4/6 pieni, 1723 s (G12b) | 4/6, 687 s (G11) |
| esplorazione, ragionamento spento | 2/3, 908 s (E12b) | 1/3, 896 s (E11) |
| esplorazione, ragionamento acceso | 2/3, 770 s (E13b) | 2/3, 766 s (E10) |

Conclusione: per domande e chiamate di funzione brevi REAP-144 su MLX e il migliore (piu intelligente, 39 t/s). Per OMP, con
contesti da 15-40K token riletti a ogni passo, resta piu veloce ONYX-224r su llama.cpp con DFlash + ngram-mod (`a1-locale veloce`).
