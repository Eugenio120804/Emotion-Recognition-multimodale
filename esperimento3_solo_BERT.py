import os
import glob
import re
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

# ==========================================
# CONFIGURAZIONE PERCORSI 
# ==========================================
base_iemocap = "/home/e.alligrande/data/IEMOCAP_full_release/"
output_csv = "./risultati_EXP3_BERT_Only.csv"

# ==========================================
# 1. ESTRAZIONE GROUND TRUTH 
# ==========================================
print(" Estrazione Ground Truth da tutte le sessioni in corso...")
ground_truth_dict = {}
eval_files = glob.glob(os.path.join(base_iemocap, "Session*", "dialog", "EmoEvaluation", "*.txt"))

for file in eval_files:
    if os.path.basename(file).startswith('.'): continue
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('['):
                parti = line.split('\t')
                if len(parti) >= 3:
                    ground_truth_dict[parti[1].strip()] = parti[2].strip().lower()

print(f" Etichette reali caricate: {len(ground_truth_dict)}")

# ==========================================
# INIZIALIZZAZIONE MODELLO (SOLO BERT)
# ==========================================
print(" Inizializzazione modello testuale DistilRoBERTa (BERT) su GPU...")
# Usiamo device=0 per usare la GPU del cluster e renderlo fulmineo
text_emotion_analyzer = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base", 
    use_safetensors=True,
    device=0 
)
print(" Modello BERT caricato e pronto!")

# Mappatura dalle emozioni in uscita da BERT alle sigle di IEMOCAP
mappa_hartmann_iemocap = {
    'anger': 'ang',
    'joy': 'hap',
    'sadness': 'sad',
    'neutral': 'neu',
    'surprise': 'sur',
    'disgust': 'dis',
    'fear': 'fea'
}

# ==========================================
# AVVIO PIPELINE SUL DATASET COMPLETO
# ==========================================
trans_files = sorted(glob.glob(os.path.join(base_iemocap, "Session*", "dialog", "transcriptions", "*.txt")))
report_esperimento = []
contatore = 0

print(f"\n AVVIO ESPERIMENTO 3 (SOLO TESTO - BERT) SU TUTTE E 5 LE SESSIONI...\n")

for trans_file in trans_files:
    with open(trans_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for line in lines:
        match = re.search(r'([a-zA-Z0-9_]+)\s*\[(\d+\.\d+)-(\d+\.\d+)\]:\s*(.*)', line)
        if match:
            utt_id = match.group(1)
            transcript = match.group(4).strip()

            # Salta le battute senza Ground Truth valida
            if utt_id not in ground_truth_dict or ground_truth_dict[utt_id] == 'xxx':
                continue

            vera_emozione = ground_truth_dict[utt_id]

            # SALTA LE FASI VUOTE
            if not transcript:
                continue

            # PREDIZIONE BERT
            try:
                # Tronchiamo a 500 caratteri per evitare errori di limite token su BERT
                testo_bert = transcript[:500] 
                bert_res = text_emotion_analyzer(testo_bert)
                bert_label_raw = bert_res[0]['label']
                
                # Traduciamo l'etichetta nel formato IEMOCAP
                bert_prediction_short = mappa_hartmann_iemocap.get(bert_label_raw, 'oth')
            except Exception as e:
                print(f"Errore BERT su {utt_id}: {e}")
                bert_prediction_short = "oth"

            print(f"[{contatore+1}] {utt_id} | Testo: \"{transcript[:30]}...\" | Vera Em: {vera_emozione.upper()} -> BERT: {bert_prediction_short.upper()}")

            report_esperimento.append({
                'file_name': utt_id,
                'Ground_Truth': vera_emozione,
                'Transcript': transcript,
                'BERT_Prediction': bert_prediction_short
            })
            contatore += 1

# ==========================================
# CALCOLO METRICHE FINALI E SALVATAGGIO
# ==========================================
if report_esperimento:
    df = pd.DataFrame(report_esperimento)
    df.to_csv(output_csv, index=False, sep=';')
    
    print("\n" + "="*50 + "\n CALCOLO METRICHE ESPERIMENTO 3 (BERT)\n" + "="*50)
    
    accuracy = accuracy_score(df['Ground_Truth'], df['BERT_Prediction'])
    precision = precision_score(df['Ground_Truth'], df['BERT_Prediction'], average='weighted', zero_division=0)
    f1 = f1_score(df['Ground_Truth'], df['BERT_Prediction'], average='weighted', zero_division=0)

    print(f"Accuratezza (Accuracy): {accuracy:.4f}")
    print(f"Precisione (Precision): {precision:.4f}")
    print(f"F1-Score:               {f1:.4f}\n")
    print(classification_report(df['Ground_Truth'], df['BERT_Prediction'], zero_division=0))
    
    with open("metriche_EXP3_BERT_Only.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nF1-Score: {f1:.4f}\n\n")
        f.write(classification_report(df['Ground_Truth'], df['BERT_Prediction'], zero_division=0))
        
    print(f" Dati salvati in: {output_csv} e metriche_EXP3_BERT_Only.txt")
