# Multimodal Emotion Recognition su IEMOCAP

Questo repository contiene il codice sorgente per gli esperimenti condotti nell'ambito della tesi di laurea in Ingegneria Informatica. Il progetto valuta le prestazioni di architetture visivo-linguistiche (LLaVA-NeXT) integrate con modelli NLP (DistilRoBERTa) per il riconoscimento multimodale delle emozioni.

## Struttura della Repository
I file sono suddivisi in due macro-categorie:

*   **Script di Esecuzione (Esperimenti):**
    *   `esperimento1_base.py`: Pipeline Zero-Shot con elaborazione simultanea video/testo.
    *   `esperimento2_canali_separati.py`: Approccio sequenziale (Analisi linguistica BERT passata come prompt a LLaVA).
    *   `esperimento2_v2_distribuzione.py`: Iniezione dell'embedding probabilistico di BERT nel VLM.
    *   `esperimento3_solo_BERT.py`: Isolamento e valutazione delle prestazioni della sola componente testuale.
*   **Analisi Dati e Metriche:**
    *   `matrice_confusione_*.py`: Script per la generazione grafica delle matrici di confusione a partire dai CSV di output.
    *   `analisi_disaccordi.py`: Strumento per l'isolamento dei casi di divergenza tra Ground Truth e predizione.

## Note sul Dataset (IEMOCAP)
A causa dei vincoli di licenza e degli accordi di non divulgazione (NDA), il dataset originale IEMOCAP e i video sorgenti non sono inclusi in questa repository. Il dataset può essere richiesto per scopi accademici agli autori presso la University of Southern California (USC).

## Riproducibilità e Installazione
Tutti gli script sono stati ottimizzati per l'esecuzione su cluster HPC (ambiente Linux/Slurm). 
Per installare le dipendenze necessarie, eseguire:
pip install -r requirements.txt
