import os
import glob
import json
import re
import subprocess
import pandas as pd
import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import (
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration, 
    BitsAndBytesConfig,
    pipeline
)
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

# ==========================================
# BLOCCO DI PROTEZIONE GPU
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# ==========================================
# CONFIGURAZIONE PERCORSI 
# ==========================================
base_iemocap = "/home/e.alligrande/data/IEMOCAP_full_release/"
output_csv = "./risultati_EXP2_Canali_Separati.csv"

# ==========================================
# PARAMETRI ESPERIMENTO
# ==========================================
NUM_FRAMES = 8
MAX_UTTERANCES = None

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

# ==========================================
# FUNZIONE DI PRE-TRATTAMENTO VIDEO
# ==========================================
def crop_video_split(input_video_path, base_name):
    left_output = os.path.join(".", f"{base_name}_left.mp4")
    if os.path.exists(left_output): return left_output
    subprocess.run(["./ffmpeg_static", "-y", "-i", input_video_path, "-filter:v", "crop=iw/2:ih:0:0", "-an", left_output], capture_output=True)
    return left_output

# ==========================================
# INIZIALIZZAZIONE MODELLI (BERT + LLaVA)
# ==========================================
print(" Inizializzazione modello testuale DistilRoBERTa (BERT) su CPU...")
text_emotion_analyzer = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base", 
    use_safetensors=True,
    device="cpu" # Lasciamo BERT su CPU per non rubare memoria preziosa a LLaVA
)

print(" Inizializzazione LLaVA-NeXT sulla GPU...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id, quantization_config=quantization_config, device_map="auto"
)
print(" Modelli pronti!")

# ==========================================
# AVVIO PIPELINE ESPERIMENTO 2
# ==========================================
trans_files = sorted(glob.glob(os.path.join(base_iemocap, "Session*", "dialog", "transcriptions", "*.txt")))
report_esperimento = []
contatore = 0

print(f"\n AVVIO ESPERIMENTO 2 (CANALI SEPARATI: BERT -> VLM) SU TUTTE E 5 LE SESSIONI...\n")

for trans_file in trans_files:
    if MAX_UTTERANCES and contatore >= MAX_UTTERANCES: break

    nome_base = os.path.basename(trans_file).replace(".txt", "")
    session_dir = os.path.dirname(os.path.dirname(trans_file))
    video_originale = os.path.join(session_dir, "avi", "DivX", f"{nome_base}.avi")

    if not os.path.exists(video_originale): continue

    video_left_path = crop_video_split(video_originale, nome_base)

    try:
        vr = VideoReader(video_left_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
    except Exception: 
        continue

    with open(trans_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for line in lines:
        if MAX_UTTERANCES and contatore >= MAX_UTTERANCES: break

        match = re.search(r'([a-zA-Z0-9_]+)\s*\[(\d+\.\d+)-(\d+\.\d+)\]:\s*(.*)', line)
        if match:
            utt_id = match.group(1)
            start_t = float(match.group(2))
            end_t = float(match.group(3))
            transcript = match.group(4)

            if utt_id not in ground_truth_dict or ground_truth_dict[utt_id] == 'xxx':
                continue

            vera_emozione = ground_truth_dict[utt_id]
            print(f"[{contatore+1}] {utt_id} | Vera Em: {vera_emozione.upper()}")

            # --- ANALISI CANALE SEPARATO (BERT) ---
            try:
                testo_bert = transcript[:500]
                bert_res = text_emotion_analyzer(testo_bert)
                bert_emotion = bert_res[0]['label']
            except Exception:
                bert_emotion = "neutral"

            start_f, end_f = int(start_t * fps), min(int(end_t * fps), len(vr) - 1)
            if end_f <= start_f: continue

            indices = np.linspace(start_f, end_f, NUM_FRAMES).astype(int)
            video_frames = vr.get_batch(indices).asnumpy()
            
            # --- PROMPT VLM CON VERDETTO BERT ---
            prompt = f"""USER: <video>
You are an expert academic researcher evaluating the IEMOCAP dataset.
Analyze these video frames capturing the actor's facial expressions.

DIALOGUE PRONOUNCED: "{transcript[:1000]}"
LINGUISTIC ANALYSIS: A specialized NLP model evaluated this dialogue and predicted the emotion is "{bert_emotion.upper()}".

Task:
1. REASONING: Explain your thought process combining the visual facial expressions and the NLP linguistic prediction. Do they match?
2. JSON: Provide the final emotion probabilities based on your multimodal assessment.

Respond EXACTLY in this format:
REASONING: [Your explanation]
JSON: {{"angry": <score>, "happy": <score>, "sad": <score>, "neutral": <score>, "frustrated": <score>, "excited": <score>, "fearful": <score>, "disgusted": <score>, "other": <score>}}
ASSISTANT:"""
            
            subset_frames = video_frames[::12]
            image_tags = "<image>\n" * len(subset_frames)
            final_prompt = f"{image_tags}\n{prompt}"
           
            torch.cuda.empty_cache()
            inputs = processor(text=final_prompt, images=subset_frames, return_tensors="pt").to("cuda")
            
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=350)

            generated_text = processor.batch_decode(out, skip_special_tokens=True)[0]
            response = generated_text.split("ASSISTANT:")[-1].strip()

            match_json = re.search(r'JSON:\s*(\{.*?\})', response, re.DOTALL | re.IGNORECASE)
            if not match_json: match_json = re.search(r'\{.*?\}', response, re.DOTALL)

            if match_json:
                try:
                    dati_grezzi = json.loads(match_json.group(1).replace("'", '"'))
                    emozioni_valide = {k: float(v) for k, v in dati_grezzi.items() if k in ['angry', 'happy', 'sad', 'neutral', 'frustrated', 'excited', 'fearful', 'disgusted', 'other']}
                    
                    if emozioni_valide:
                        mappa_emo = {'angry':'ang', 'happy':'hap', 'sad':'sad', 'neutral':'neu', 'frustrated':'fru', 'excited':'exc', 'fearful':'fea', 'disgusted':'dis', 'other':'oth'}
                        
                        vlm_prediction_full = max(emozioni_valide, key=emozioni_valide.get).lower()
                        vlm_prediction_short = mappa_emo.get(vlm_prediction_full, 'oth')

                        emozioni_valide['file_name'] = utt_id
                        emozioni_valide['Ground_Truth'] = vera_emozione
                        emozioni_valide['BERT_Prediction'] = bert_emotion # Salviamo la scelta di BERT
                        emozioni_valide['VLM_Prediction'] = vlm_prediction_short
                        emozioni_valide['CoT_Reasoning'] = response.split("JSON:")[0].replace("REASONING:", "").strip()
                        report_esperimento.append(emozioni_valide)
                        
                        print(f"     -> BERT predice: {bert_emotion.upper()} | VLM decide: {vlm_prediction_short.upper()}")
                except Exception as e:
                    print(f"     [ERRORE PARSING JSON] Dettaglio: {e}")
            contatore += 1

# ==========================================
# CALCOLO METRICHE FINALI E SALVATAGGIO
# ==========================================
if report_esperimento:
    df = pd.DataFrame(report_esperimento)
    
    colonne_base = ['file_name', 'Ground_Truth', 'BERT_Prediction', 'VLM_Prediction']
    colonne_emozioni = [c for c in df.columns if c not in colonne_base + ['CoT_Reasoning']]
    df = df[colonne_base + colonne_emozioni + ['CoT_Reasoning']]
    
    df.to_csv(output_csv, index=False, sep=';')
    
    print("\n" + "="*50 + "\n CALCOLO METRICHE ESPERIMENTO 2\n" + "="*50)
    
    accuracy = accuracy_score(df['Ground_Truth'], df['VLM_Prediction'])
    precision = precision_score(df['Ground_Truth'], df['VLM_Prediction'], average='weighted', zero_division=0)
    f1 = f1_score(df['Ground_Truth'], df['VLM_Prediction'], average='weighted', zero_division=0)

    print(f"Accuratezza (Accuracy): {accuracy:.4f}")
    print(f"Precisione (Precision): {precision:.4f}")
    print(f"F1-Score:               {f1:.4f}\n")
    print(classification_report(df['Ground_Truth'], df['VLM_Prediction'], zero_division=0))
    
    with open("metriche_EXP2_Canali_Separati.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nF1-Score: {f1:.4f}\n\n")
        f.write(classification_report(df['Ground_Truth'], df['VLM_Prediction'], zero_division=0))
        
    print(f" Dati salvati in: {output_csv} e metriche_EXP2_Canali_Separati.txt")
