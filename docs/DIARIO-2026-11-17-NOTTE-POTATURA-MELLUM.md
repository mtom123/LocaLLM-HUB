# Diario notte 17/11/2026 — potatura esperti fatta in casa

## Cosa e successo
- Baseline **Mellum2.1-12B-A2.5B-Thinking Q4_K_M: ~100 t/s** in VRAM (core-speed): il piu veloce mai misurato qui.
- GLM RCO-152E (44 GB): gira ma ~0,6 t/s su 12+16 GB, registrato come feasibility failed. SCOPERTA: b12244 CUDA13.5 non vede la GPU; b12141 e il nuovo b12244 CUDA12.4 la vedono.
- Pipeline nostra (fork di prune_experts di Pietro): imatrix con conteggi di routing -> taglio a K esperti/layer -> eval.

## Variante | PPL heldout | GB | speedtest exit
| K=64 (base) | 4.3652 | 7,52 | - |
| K=56 | 4.5261 | 6.65 | 0 |
| K=48 | 4.6883 | 5.79 | 0 |
| K=40 | 4.982 | 4.93 | 0 |
| K=32 | 6.6614 | 4.06 | 0 |
| K=24 | 7.9064 | 3.2 | 0 |

## Intelligenza
duro-v1 (math/code/mmlu, n=15): in corso% — vedi results/windows/2026-11-17-mellum21-duro-intelligenza.json.

## Interpretazione
La velocita non cambia (8 esperti attivi per token): la potatura compra VRAM e dimensione, non t/s. La PPL dice quanto cervello si perde.

## Files
- risultati: results/windows/2026-11-17-mellum21-potatura-esperti-curva.json + speedtest per variante
- strumenti: benchmarks/windows-fabbrica/prune_experts_win.py
- modelli: solo la migliore tenuta; altre cancellate dopo misura (permesso Motti), rigenerabili
- imatrix: imatrix-mellum-base.gguf (78 MB)
