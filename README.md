# Multimodal Emotion Recognition su IEMOCAP

Questo repository contiene il codice sorgente per gli esperimenti condotti nell'ambito della tesi di laurea in Ingegneria e Scienze Informatiche per la Cybersecurity, dal titolo 'Un approccio Zero-Shot per l'analisi e il confronto tra modelli di Emotion Recognition multimodale nel contesto della prevenzione di Insider Threats'. Il progetto valuta le prestazioni di architetture visivo-linguistiche (LLaVA-NeXT) integrate con modelli NLP (DistilRoBERTa) per il riconoscimento multimodale delle emozioni.

## Struttura della Repository
I file sono suddivisi in due macro-categorie:

*   **Script di Esecuzione (Esperimenti):**
    *   `esperimento1_base.py`: Pipeline Zero-Shot con elaborazione simultanea video/testo.
    *   `esperimento2_canali_separati.py`: Approccio sequenziale (analisi linguistica BERT passata come prompt a LLaVA).
    *   `esperimento2_v2_distribuzione.py`: Iniezione dell'embedding probabilistico di BERT nel VLM.
    *   `esperimento3_solo_BERT.py`: Isolamento e valutazione delle prestazioni della sola componente testuale.
*   **Analisi Dati e Metriche:**
    *   `matrice_confusione_*.py`: Script per la generazione grafica delle matrici di confusione a partire dai CSV di output.
    *   `analisi_disaccordi.py`: Script per l'analisi dei casi di disaccordo multimodale.

## Riproducibilità e Installazione
Tutti gli script sono stati ottimizzati per l'esecuzione su cluster HPC (High Performance Computing) purpleJeans. 
Per installare le dipendenze necessarie, eseguire:
pip install -r requirements.txt
