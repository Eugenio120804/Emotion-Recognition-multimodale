import pandas as pd

# ==========================================
# CONFIGURAZIONE PERCORSO
# ==========================================

csv_path = "risultati_EXP2_Canali_Separati.csv"

print("\n" + "="*50)
print(" ANALISI DISACCORDI: BERT vs LLaVA (EXP 2)")
print("="*50 + "\n")

try:
    # Caricamento del dataset (usiamo il separatore ';' come da script originale)
    df = pd.read_csv(csv_path, sep=';')
    
    # BERT usa etichette intere (es: 'anger'), VLM usa 3 lettere (es: 'ang').
    # Creiamo una colonna mappata per poterli confrontare direttamente.
    bert_map = {
        'anger': 'ang', 'neutral': 'neu', 'surprise': 'sur',
        'sadness': 'sad', 'disgust': 'dis', 'joy': 'hap', 'fear': 'fea'
    }
    df['BERT_Mapped'] = df['BERT_Prediction'].map(bert_map)

    # ==========================================
    # 1. STATISTICHE GENERALI
    # ==========================================
    totale_battute = len(df)
    
    # Filtriamo solo le righe dove BERT e VLM hanno predetto cose diverse
    disaccordi = df[df['BERT_Mapped'] != df['VLM_Prediction']]
    num_disaccordi = len(disaccordi)

    # Sotto-categorie di disaccordo
    vlm_ragione_bert_torto = len(disaccordi[(disaccordi['VLM_Prediction'] == disaccordi['Ground_Truth']) & (disaccordi['BERT_Mapped'] != disaccordi['Ground_Truth'])])
    bert_ragione_vlm_torto = len(disaccordi[(disaccordi['BERT_Mapped'] == disaccordi['Ground_Truth']) & (disaccordi['VLM_Prediction'] != disaccordi['Ground_Truth'])])
    entrambi_torto = len(disaccordi[(disaccordi['VLM_Prediction'] != disaccordi['Ground_Truth']) & (disaccordi['BERT_Mapped'] != disaccordi['Ground_Truth'])])

    print(f"Totale battute analizzate: {totale_battute}")
    print(f"Totale disaccordi:         {num_disaccordi} ({(num_disaccordi/totale_battute)*100:.2f}%)")
    print(f" -> VLM ha corretto BERT:  {vlm_ragione_bert_torto}")
    print(f" -> VLM ha ignorato BERT (sbagliando): {bert_ragione_vlm_torto}")
    print(f" -> Entrambi hanno torto:  {entrambi_torto}")

    # ==========================================
    # 2. PATTERN DI DISACCORDO PIÙ FREQUENTI
    # ==========================================
    print("\n" + "="*50)
    print(" TOP 10 ERRORI SISTEMATICI (Cortocircuiti)")
    print("="*50)
    
    # Raggruppiamo per Vera Emozione, Scelta di BERT e Scelta finale di VLM
    pattern = disaccordi.groupby(['Ground_Truth', 'BERT_Mapped', 'VLM_Prediction']).size().reset_index(name='Frequenza')
    pattern = pattern.sort_values(by='Frequenza', ascending=False).head(10)
    
    # Rinominiamo le colonne per chiarezza in output
    pattern.columns = ['Vera Emozione', 'Suggerimento BERT', 'Decisione VLM', 'Frequenza']
    print(pattern.to_string(index=False))

    # ==========================================
    # 3. ESTRAZIONE CASI STUDIO (Esempi)
    # ==========================================
    print("\n" + "="*50)
    print(" ESTRAZIONE CASI STUDIO (Chain of Thought)")
    print("="*50)
    
    # Caso 1: LLaVA corregge BERT
    casi_vlm_corretti = disaccordi[(disaccordi['VLM_Prediction'] == disaccordi['Ground_Truth']) & (disaccordi['BERT_Mapped'] != disaccordi['Ground_Truth'])]
    if not casi_vlm_corretti.empty:
        esempio_1 = casi_vlm_corretti.iloc[0]
        print("\n[ESEMPIO 1] LLaVA IGNORA BERT E INDOVINA L'EMOZIONE:")
        print(f"File: {esempio_1['file_name']}")
        print(f"Ground Truth: {esempio_1['Ground_Truth'].upper()} | Suggerimento testuale (BERT): {esempio_1['BERT_Mapped'].upper()} | Scelta VLM: {esempio_1['VLM_Prediction'].upper()}")
        print(f"Ragionamento LLaVA: {esempio_1['CoT_Reasoning'][:400]}...\n")

    # Caso 2: BERT aveva ragione, ma LLaVA sbaglia
    casi_bert_corretti = disaccordi[(disaccordi['BERT_Mapped'] == disaccordi['Ground_Truth']) & (disaccordi['VLM_Prediction'] != disaccordi['Ground_Truth'])]
    if not casi_bert_corretti.empty:
        esempio_2 = casi_bert_corretti.iloc[0]
        print("[ESEMPIO 2] LLaVA SI FA INGANNARE DAL VIDEO E SBAGLIA (Ignorando BERT):")
        print(f"File: {esempio_2['file_name']}")
        print(f"Ground Truth: {esempio_2['Ground_Truth'].upper()} | Suggerimento testuale (BERT): {esempio_2['BERT_Mapped'].upper()} | Scelta VLM: {esempio_2['VLM_Prediction'].upper()}")
        print(f"Ragionamento LLaVA: {esempio_2['CoT_Reasoning'][:400]}...")

except FileNotFoundError:
    print(f"\n[ERRORE] Il file '{csv_path}' non è stato trovato. Assicurati di essere nella stessa cartella.")
except Exception as e:
    print(f"\n[ERRORE DURANTE L'ANALISI]: {e}")
