# ВНУТРИ recipe_generator.py
import json
from openai import OpenAI

# 1. Настраиваем подключение к твоему локальному серверу LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# 2. Твой идеальный Системный Промпт (без лишней проверки вежливости)
SYSTEM_PROMPT = """You are a professional chef. Your task is to respond to the user's request by generating a recipe strictly in valid JSON format.
CRITICAL: All text values inside the JSON (messages, names, ingredients, steps, tips) MUST BE WRITTEN IN RUSSIAN.

### JSON RESPONSE FORMAT:
{
  "status": "success",
  "message": "дружелюбное приветствие на русском языке (например: 'Конечно, вот рецепт...')",
  "recipes": [
    {
      "id": 1,
      "name": "название блюда на русском",
      "cooking_time": "время приготовления на русском",
      "difficulty": "Легко / Средне / Сложно",
      "description": "краткое описание вкуса на русском",
      "ingredients": [
        "ингредиент 1 - количество на русском"
      ],
      "steps": [
        "Шаг 1: ... (на русском)"
      ],
      "tips": "полезный совет на русском"
    }
  ]
}

### RULES:
1. Output ONLY raw valid JSON. No markdown formatting, no ```json wrapper.
2. Fill all fields completely in Russian language.
3. Field "steps" must contain 5-10 detailed steps.
"""


def get_recipe_json(dish_name):
    try:
        response = client.chat.completions.create(
            model="Smoffyy/Qwen3.5-4B-Instruct-Revised-GGUF", 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Приготовь рецепт для блюда: {dish_name}"}
            ],
            temperature=0.3  # Убрали строчку response_format, теперь ошибки 400 не будет!
        )
        return response.choices[0].message.content
    except Exception as e:
        error_json = {"status": "error", "message": f"Ошибка сервера: {str(e)}"}
        return json.dumps(error_json, ensure_ascii=False, indent=2)



# --- БЛОК ЗАПУСКА СКРИПТА ---
if __name__ == "__main__":
    print("--- Локальный генератор JSON-рецептов запущен ---")
    user_input = input("Введите название блюда (например: Пицца): ")
    
    print("\nЗапрос отправлен в LM Studio... Ожидайте генерации...\n")
    result_json = get_recipe_json(user_input)
    
    # ВОТ ЭТИ СТРОЧКИ ДОЛЖНЫ БЫТЬ НА КОНЦЕ, ЧТОБЫ ВЫВЕСТИ ОТВЕТ В КОНСОЛЬ:
    print("--- ОТВЕТ ОТ МОДЕЛИ В ФОРМАТЕ JSON ---")
    print(result_json)
