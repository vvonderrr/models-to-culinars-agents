import os
import torch
import time
import open_clip
from PIL import Image
import numpy as np
import pickle
from deep_translator import GoogleTranslator
from huggingface_hub import hf_hub_download
import sys
import logging
import warnings

warnings.filterwarnings("ignore")
# 🔇 Отключаем информационные сообщения от Hugging Face
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# 👇 Добавь эту строку, чтобы включить поддержку эмодзи
sys.stdout.reconfigure(encoding='utf-8')
def translate_to_english(text):
    """Переводит русский текст на английский"""
    try:
        translator = GoogleTranslator(source='auto', target='en')
        translated = translator.translate(text)
        print(f"🌐 Перевод: {text} → {translated}")
        return translated
    except Exception as e:
        print(f"⚠️ Ошибка перевода: {e}")
        return text  # Если ошибка, используем оригинал


# 1. Настройка
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Устройство: {device}")

INDEX_FILE = "food_clip_index.pkl"
IMAGE_FOLDER = "images"

# Имена для open_clip
MODEL_NAME = "ViT-B-32"
PRETRAINED_OPENAI = "openai" # Базовые веса для текста и структуры

print("📥 Инициализация модели...")

try:
    # 1. Создаем модель и процессор на базе стандартного OpenAI CLIP ViT-B/32
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_OPENAI)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    
    # 2. Загружаем кастомные веса для ВИЗУАЛЬНОЙ части (Food101)
    print("   Загрузка улучшенных весов для еды (Vision Encoder)...")
    
    # Скачиваем файл safetensors локально
    model_path = hf_hub_download(
        repo_id="tanganke/clip-vit-base-patch32_food101", 
        filename="model.safetensors"
    )
    
    # Загружаем веса из safetensors
    import safetensors.torch
    state_dict = safetensors.torch.load_file(model_path, device=device)
    
    # В модели tanganke ключи могут называться иначе или содержать лишнее.
    # Нам нужно обновить только vision_model.
    # open_clip хранит веса в model.visual
    
    # Фильтруем ключи, которые относятся к визуальной части
    # Обычно в таких чекпоинтах ключи начинаются с 'vision_model.' или просто совпадают с структурой visual
    # Попробуем загрузить напрямую в visual часть
    
    # Создаем новый state_dict только для visual части, сопоставляя ключи
    visual_state_dict = {}
    for k, v in state_dict.items():
        # Если ключ начинается с vision_model, убираем префикс, так как open_clip ожидает ключи внутри visual
        if k.startswith("vision_model."):
            new_key = k.replace("vision_model.", "")
            visual_state_dict[new_key] = v
        elif k.startswith("visual."): # Иногда бывает такой префикс
             new_key = k.replace("visual.", "")
             visual_state_dict[new_key] = v
        # Если ключи без префикса, значит они уже готовы для visual части (зависит от того, как сохраняли)
        # Для tanganke обычно это full model dump, поэтому проверяем наличие ключей visual
    
    # Попытка загрузить веса в визуальную часть модели
    # strict=False позволяет игнорировать несовпадения, если какие-то ключи не найдены
    missing, unexpected = model.visual.load_state_dict(visual_state_dict, strict=False)
    
    if missing:
        print(f"   ⚠️ Недостающие ключи при загрузке весов еды: {len(missing)} шт.")
    if unexpected:
        print(f"   ⚠️ Лишние ключи: {len(unexpected)} шт.")
        
    model.to(device)
    model.eval()
    print("✅ Модель готова (Text: OpenAI, Vision: Food101 Fine-tuned)")

except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- ФУНКЦИИ ИНДЕКСАЦИИ И ПОИСКА ---

def get_all_images_recursively(folder_path):
    image_files = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    if not os.path.exists(folder_path):
        return []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def build_index():
    print("\n📁 Сканирование папки...")
    all_images = get_all_images_recursively(IMAGE_FOLDER)
    if not all_images:
        print(f"❌ Папка '{IMAGE_FOLDER}' пуста или не найдена.")
        return None, None
    
    print(f"Найдено {len(all_images)} изображений. Кодирование...\n")
    image_paths = []
    image_features_list = []
    
    for i, path in enumerate(all_images):
        if (i + 1) % 500 == 0:
            print(f"  Обработано: {i+1}/{len(all_images)}")
        try:
            image = Image.open(path).convert("RGB")
            # Препроцессинг open_clip
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Кодирование изображения
                image_feat = model.encode_image(image_input)
                # Нормализация
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                
            image_paths.append(path)
            image_features_list.append(image_feat.cpu().numpy().flatten())
            
        except Exception as e:
            print(f"    ⚠️ Пропуск {path}: {e}")

    if not image_paths: return None, None
    
    features_matrix = np.array(image_features_list).astype('float32')
    
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump({'paths': image_paths, 'features': features_matrix}, f)
        
    print(f"\n✅ Индекс сохранен: {len(image_paths)} картинок")
    return image_paths, features_matrix

def load_index():
    if not os.path.exists(INDEX_FILE): return None, None
    with open(INDEX_FILE, 'rb') as f:
        data = pickle.load(f)
    print(f"💾 Индекс загружен: {len(data['paths'])} картинок")
    return data['paths'], data['features']

def search(query, top_k=5, paths=None, feats=None):
    if paths is None: return []
    
    #  1. Переводим запрос на английский перед обработкой
    try:
        translator = GoogleTranslator(source='auto', target='en')
        translated_query = translator.translate(query)
        print(f"🔍 Перевод: '{query}' -> '{translated_query}'")
    except Exception as e:
        print(f"⚠️ Ошибка перевода: {e}. Использую оригинал.")
        translated_query = query

    # 2. Токенизация и кодирование текста (теперь передаём translated_query)
    text_tokens = tokenizer([translated_query]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_tokens)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        
    text_feat_np = text_feat.cpu().numpy().flatten()
    
    # 3. Поиск сходства (этот блок не меняем)
    sims = np.dot(feats, text_feat_np)
    top_idx = sims.argsort()[-top_k:][::-1]
    
    # 👇 Дальше код функции остаётся точно таким же, как у тебя был
    results = []
    for idx in top_idx:
        results.append({
            'path': paths[idx],
            'score': float(sims[idx]),
            'rel_path': os.path.relpath(paths[idx], IMAGE_FOLDER)
        })
    return results

# --- ЗАПУСК ---

paths, feats = load_index()
if paths is None:
    paths, feats = build_index()
if paths is None: exit()

print("\n" + "="*50)
print("🍔 Food Search (OpenCLIP + Food101 Weights)")
print("="*50)

while True:
    q = input("\n Запрос: ").strip()
    if q.lower() in ['exit', 'quit']: break
    if not q: continue
    
    # ⏱️ Замер чистого времени поиска
    t0 = time.perf_counter()
    res = search(q, top_k=3, paths=paths, feats=feats)
    elapsed_time = time.perf_counter() - t0
    print(f"⏱️ Поиск занял: {elapsed_time:.3f} сек.")
    
    # Вывод результатов
    for i, r in enumerate(res):
        print(f"  {i+1}. [{r['score']:.3f}] {r['rel_path']}")
    
    # Показ картинки
    if res:
        try:
            Image.open(res[0]['path']).show()
        except: pass