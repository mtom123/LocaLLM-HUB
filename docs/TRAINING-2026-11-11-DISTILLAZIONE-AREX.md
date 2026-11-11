# Distillazione A1 IQ3 -> AREX 4B: primo giro completo (11/11/2026)

Pipeline in `llm/fabbrica/training/` (descrizione in `training/README.md`). Primo ciclo end-to-end eseguito
oggi: compiti sintetici -> tracce verificate del teacher -> QLoRA 4-bit -> A/B su holdout separato.

## Cosa e' stato fatto
| Fase | Numero |
|---|---|
| Compiti generati (seed 7, train) | 240 |
| Tracce accettate dal teacher A1 IQ3 (solo soluzioni che passano i test) | **196** (74% dei tentativi) |
| Compiti di holdout (seed 99, mai visti) | 40 |
| Training | QLoRA 4-bit nf4, LoRA r=16 su q/k/v/o + gate/up/down, 21,2M parametri (0,5%), 196 esempi, 3 epoche, 147 step, loss 1,02 -> 0,15, **~22 min** su RTX 3060 |
| Valutazione A/B | 20 compiti di holdout, stesso motore 4-bit, greedy |

## Risultato
| | Pass |
|---|---|
| AREX-Turbo base | **6/20 (30%)** |
| AREX-Turbo + LoRA | **7/20 (35%)** |

Dettaglio: 5 compiti risolti in piu', 4 risolti in meno. **+1 compito su 20 e' dentro il rumore** (con n=20
l'intervallo e' di decine di punti): non si dichiara nessun guadagno.

## Cosa dice davvero
- La pipeline funziona: l'adapter cambia le risposte (5 migliorati, 4 peggiorati) e la loss scende; non e' un
  problema di masking o di adapter non applicato.
- Il 4B non impara a generalizzare da 196 esempi su 6 template sintetici: le tracce sono poche e molto
  omogenee, e i compiti di holdout sono fuori distribuzione rispetto al tuning agentico di AREX.
- La base e' debole su questi compiti (30%): c'e' margine, ma serve un dataset piu' vario (piu' template, piu'
  linguaggi, esempi con errori e correzioni) e piu' epoche, non un trucco.

## Prossimi passi (se lo si riprende)
1. Fare il **fit check**: valutare l'adapter su 10 compiti *di training*. Se li risolve, il problema e'
   generalizzazione (servono dati); se non li risolve, il training va rinforzato (epoche, lr, rank).
2. Generare il resto dei compiti (i 240 non finiti per il `bad allocation` del teacher) e aggiungere template
   nuovi (parsing CSV, classi, test) per almeno 600-800 esempi.
3. Solo se il holdout migliora in modo chiaro: convertire il LoRA in GGUF (`convert_lora_to_gguf.py`) e
   misurare `duro`/`omp` con `llama-server --lora`, per vedere se il guadagno sopravvive al cambio di motore.

## Nota di metodo
Il teacher ha smesso di generare a ~129 tracce per `bad allocation` (RAM: modello 12,7 GB + KV + cache prompt
su 16 GB); il secondo giro e' ripartito in resume e ha aggiunto 67 tracce. La pipeline riprende da sola, ma su
questa macchina conviene lanciarla in blocchi da ~100 compiti.
