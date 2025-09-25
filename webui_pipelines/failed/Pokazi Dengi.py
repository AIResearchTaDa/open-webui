"""
title: Auto Token Logger and Display on Click
author: Gurasis Osahan (modified for auto logging and click display)
author_url: https://github.com/asisosahan
funding_url: https://buymeacoffee.com/gosahan
version: 0.2.2
description:
    Automatically logs token counts and costs to `chat_cost_log` after model response.
    Displays metrics in UI only on button click, avoiding duplicate DB entries.
"""

import logging
import time
import os
from typing import Any, Dict, List, Optional

import psycopg2
import tiktoken
from fastapi.requests import Request
from pydantic import BaseModel, Field

# Налаштування логера
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.set_name("auto_token_logger")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

MODEL_PRICING = {
    # ------------------------------
    # GPT-4 and GPT-3.5 placeholders
    # ------------------------------
    "gpt-4": {"input": 30.00, "output": 60.00},  # 8k context
    "gpt-4-32k": {"input": 60.00, "output": 120.00},  # 32k context
    "gpt-3.5-turbo": {"input": 1.50, "output": 2.00},
    "gpt-3.5-turbo-16k": {"input": 3.00, "output": 4.00},
    "gpt-3.5-turbo-0613": {"input": 1.50, "output": 2.00},
    "gpt-3.5-turbo-16k-0613": {"input": 3.00, "output": 4.00},
    # Legacy references
    "davinci-002": {"input": 2.00, "output": 2.00},
    "babbage-002": {"input": 0.40, "output": 0.40},
    # -------------------------------------------------------------------------
    # Anthropic Claude placeholders - edit to match your actual costs
    # -------------------------------------------------------------------------
    "claude-3.5-sonnet-2024-10-22": {"input": 20.00, "output": 40.00},
    "claude-3.5-sonnet-2024-06-20": {"input": 20.00, "output": 40.00},
    "claude-3.5-haiku": {"input": 15.00, "output": 30.00},
    "claude-3-opus": {"input": 25.00, "output": 50.00},
    "claude-3-sonnet": {"input": 20.00, "output": 40.00},
    "claude-3-haiku": {"input": 15.00, "output": 30.00},
    "claude-2.1": {"input": 12.00, "output": 24.00},
    "claude-2.0": {"input": 12.00, "output": 24.00},
    # -------------------------------------------------------------------------
    # GPT-4o and other placeholders from earlier versions
    # -------------------------------------------------------------------------
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-4o-audio-preview": {"input": 2.50, "output": 10.00},
    "gpt-4o-audio-preview-2024-12-17": {"input": 2.50, "output": 10.00},
    "gpt-4o-audio-preview-2024-10-01": {"input": 2.50, "output": 10.00},
    "gpt-4o-realtime-preview": {"input": 5.00, "output": 20.00},
    "gpt-4o-realtime-preview-2024-12-17": {"input": 5.00, "output": 20.00},
    "gpt-4o-realtime-preview-2024-10-01": {"input": 5.00, "output": 20.00},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-2024-12-17": {"input": 15.00, "output": 60.00},
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o1-preview-2024-09-12": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o1-mini-2024-09-12": {"input": 3.00, "output": 12.00},
    # --------------------------------------------------------------
    # Fallback categories: "small", "medium", "large" for unknown models
    # --------------------------------------------------------------
    "small": {"input": 0.50, "output": 1.50},
    "medium": {"input": 5.00, "output": 15.00},
    "large": {"input": 15.00, "output": 60.00},
}


class Action:
    class Valves(BaseModel):
        show_on_click: bool = Field(
            default=True, description="Show metrics in UI only on button click."
        )
        log_to_db: bool = Field(
            default=True, description="Automatically log to DB after model response."
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            logger.error("DATABASE_URL environment variable not set. Cannot log costs.")

    def get_token_count_estimate(self, text: str, model: str) -> int:
        """Оцінює кількість токенів за допомогою tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def get_model_prices(self, model_id: str) -> Dict[str, float]:
        """Отримує ціни для моделі, повертаючись до категорій, якщо модель не знайдена."""
        if model_id in MODEL_PRICING:
            return MODEL_PRICING[model_id]

        model_lower = model_id.lower()
        if any(term in model_lower for term in ["opus", "gpt-4", "large"]):
            return MODEL_PRICING["large"]
        if any(term in model_lower for term in ["sonnet", "gpt-3.5", "medium"]):
            return MODEL_PRICING["medium"]
        return MODEL_PRICING["small"]

    def save_cost_to_db(self, data: Dict[str, Any]):
        """Зберігає дані про токени та витрати в базу даних."""
        if not self.db_url or not self.valves.log_to_db:
            logger.error("DB logging disabled or no DATABASE_URL. Skipping DB save.")
            return

        sql = """
            INSERT INTO chat_cost_log (chat_id, user_id, model_id, prompt_tokens, completion_tokens, total_tokens, cost)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        conn = None
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            cur.execute(
                sql,
                (
                    data["chat_id"],
                    data["user_id"],
                    data["model_id"],
                    data.get("prompt_tokens"),
                    data.get("completion_tokens"),
                    data["total_tokens"],
                    data["cost"],
                ),
            )
            conn.commit()
            cur.close()
            logger.info(f"Logged cost for chat_id: {data['chat_id']}")
        except Exception as e:
            logger.error(f"Failed to write cost to DB: {e}")
        finally:
            if conn:
                conn.close()

    async def action(
        self,
        body: Dict[str, Any],
        __request__: Request,
        __user__: Optional[Any] = None,
        __event_emitter__: Optional[Any] = None,
        __model__: Optional[Dict[str, str]] = None,
    ) -> None:
        start_time = time.time()

        try:
            # Перевіряємо, чи є відповідь моделі
            messages = body.get("messages", [])
            if not messages or messages[-1].get("role") != "assistant":
                logger.info("No model response detected. Skipping DB logging.")
                return

            # Основна інформація
            model_id = __model__.get("id") if __model__ else "unknown"
            chat_id = body.get("chat_id")
            user_id = __user__.get("id") if __user__ else "anonymous"

            if not chat_id:
                logger.warning("No chat_id found in body. Cannot log cost.")
                return

            prompt_tokens = 0
            completion_tokens = 0
            cost = 0.0

            # Отримуємо дані usage, якщо доступні
            usage_data = body.get("usage")
            if usage_data and isinstance(usage_data, dict):
                logger.info(f"Received usage data from API: {usage_data}")
                prompt_tokens = usage_data.get("prompt_tokens", 0)
                completion_tokens = usage_data.get("completion_tokens", 0)

                prices = self.get_model_prices(model_id)
                prompt_cost = (prompt_tokens / 1_000_000) * prices.get("input", 0)
                completion_cost = (completion_tokens / 1_000_000) * prices.get(
                    "output", 0
                )
                cost = prompt_cost + completion_cost
            else:
                logger.warning(
                    f"No 'usage' data for model {model_id}. Falling back to tiktoken estimation."
                )
                # Оцінка токенів для всіх повідомлень (промпт) і останнього (відповідь)
                prompt_text = " ".join([m.get("content", "") for m in messages[:-1]])
                prompt_tokens = self.get_token_count_estimate(prompt_text, model_id)
                completion_text = messages[-1].get("content", "")
                completion_tokens = self.get_token_count_estimate(
                    completion_text, model_id
                )

                prices = self.get_model_prices(model_id)
                prompt_cost = (prompt_tokens / 1_000_000) * prices.get("input", 0)
                completion_cost = (completion_tokens / 1_000_000) * prices.get(
                    "output", 0
                )
                cost = prompt_cost + completion_cost

            total_tokens = prompt_tokens + completion_tokens

            # Автоматичне збереження в БД після відповіді
            if self.valves.log_to_db:
                log_data = {
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "model_id": model_id,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost,
                }
                self.save_cost_to_db(log_data)

            # Відображення в UI лише при натисканні (за умовчанням)
            if self.valves.show_on_click and __event_emitter__:
                elapsed_time = time.time() - start_time
                status_message = f"Витрачено токенів: {total_tokens} | Вартість: ${cost:.5f} | Час: {elapsed_time:.2f}с"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": status_message, "done": True},
                    }
                )

        except Exception as e:
            logger.error(f"Помилка в обчисленні вартості: {e}", exc_info=True)
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Помилка при обчисленні вартості",
                            "done": True,
                        },
                    }
                )
