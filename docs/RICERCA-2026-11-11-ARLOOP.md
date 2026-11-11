# Ricerca 11/11/2026 — ARLoop (looped transformer con attention residuals)

Paper: *ARLoop: Fully-Open Pre-trained Looped Transformers with Attention Residuals* — Yao, Du, Zhu, Sun (OSU NLP),
arXiv [2611.05234](https://arxiv.org/abs/2611.05234), 3 novembre 2026. Tutto aperto: codice
[OSU-NLP-Group/ARLoop](https://github.com/OSU-NLP-Group/ARLoop), pesi su HF (`osunlp/ARLoop-0.6B`, `ARLoop-1.2B`,
baseline `PreNormLoop-0.6B`), ricette di training e dati.

## Cosa propone
Nei **looped transformer** lo stesso blocco di layer viene applicato piu' volte (R ricorrenze): si aumenta la
"profondita' effettiva" senza aumentare i parametri. Il problema e' che il residual standard accumula tutto in un
unico stream e non sceglie *quali* rappresentazioni intermedie riusare. ARLoop sostituisce il residual con
**attention residuals**: l'input di ogni sublayer e' un'aggregazione via attenzione su tutta la profondita'.
Scoperta utile: non serve tutta la storia dei residual, basta una **finestra scorrevole sulla profondita'**
(embedding iniziale + stati recenti) — stessa qualita' o meglio, meno costo.

## Numeri dichiarati
- 140M e 370M, R = 1..5: ARLoop batte i residual precedenti per looped transformer.
- 0.6B e 1.2B con ricetta di pretraining realistica (0,5T token, 3 ricorrenze).
- **1.2B batte Huginn-3.5B** (l'altro looped LLM aperto) su tutti i task valutati ed e' **competitivo con Qwen3.5-2B**
  con molti meno token di training; guadagno piu' forte su math e coding.

## Perche' ci interessa (e quanto)
- E' la stessa idea che serve a noi: **piu' intelligenza per GB**. Un 1.2B con 3 ricorrenze usa ~2,4 GB in bf16 e
  dichiara qualita' da 2B. Se l'architettura regge, e' il tipo di modello che sta comodo in 12 GB con contesto lungo.
- Al momento **non e' utilizzabile nei nostri test**: non esiste GGUF, non c'e' supporto llama.cpp, e i modelli sono
  **base** (non instruct). Per confrontarli con duro servirebbe un instruct o un harness few-shot, e gireranno su
  transformers/vLLM (i kernel `flash-linear-attention` mancano su Windows, quindi versione reference lenta).
- Rischio: i guadagni dichiarati sono contro Huginn (famiglia diversa) e su benchmark accademici; il salto reale
  rispetto a un instruct 4B moderno non e' dimostrato.

## Verdetto
**Da leggere e tenere d'occhio, non da testare adesso.** Se esce un ARLoop instruct (o una conversione GGUF), il test
sensato e': ARLoop-1.2B instruct contro Qwen3.5-2B/AREX 4B su `duro` e `tiny`, a parita' di GB occupati. Fino ad
allora non produce numeri per la dashboard e non sposta la scelta del modello di produzione.

Riferimenti utili: `huskydoge/Awesome-Loop-Models` e `asimfish/awesome_loop_transformer` (liste aggiornate).
