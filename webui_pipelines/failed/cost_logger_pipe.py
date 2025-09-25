"""
title: Automatic Token Cost Logger (Pipe)
author: Gurasis Osahan (adapted for Pipelines by AI)
author_url: https://github.com/asisosahan
funding_url: https://buymeacoffee.com/gosahan
version: 1.0.0
description:
    Automatically logs token counts and costs to the 'chat_cost_log' table
    after every model response. Designed to work as an Open WebUI Pipeline.
"""

import logging
import time
import os
from typing import Any, Dict, List, Optional

# Спроба імпортувати psycopg2. Якщо не вийде, конвеєр все одно не впаде.
try:
    import psycopg2
except ImportError:
    psycopg2 = None

# Спроба імпортувати tiktoken.
try:
    import tiktoken
except ImportError:
    tiktoken = None

from pydantic import BaseModel, Field

# Налаштування логера
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.set_name("auto_token_logger_pipe")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

MODEL_PRICING = {
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-32k": {"input": 60.00, "output": 120.00},
    "gpt-3.5-turbo": {"input": 1.50, "output": 2.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-3-opus": {"input": 25.00, "output": 50.00},
    "claude-3-sonnet": {"input": 20.00, "output": 40.00},
    # --- Резервні категорії ---
    "small": {"input": 0.50, "output": 1.50},
    "medium": {"input": 5.00, "output": 15.00},
    "large": {"input": 15.00, "output": 60.00},
}

class Pipe:
    """
    Цей конвеєр автоматично розраховує та зберігає вартість кожного запиту до моделі.
    """
    class Valves(BaseModel):
        log_to_db: bool = Field(
            default=True, description="Enable or disable logging to the database."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.db_url = os.getenv("DATABASE_URL")

        if psycopg2 is None:
            logger.error("The 'psycopg2-binary' library is not installed. DB logging is disabled.")
            self.valves.log_to_db = False
        
        if tiktoken is None:
            logger.error("The 'tiktoken' library is not installed. Token estimation will not work.")

        if not self.db_url and self.valves.log_to_db:
            logger.error("DATABASE_URL environment variable is not set for the pipeline service. DB logging disabled.")
            self.valves.log_to_db = False


    def get_token_count_estimate(self, text: str, model: str) -> int:
        if tiktoken is None:
            return 0
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def get_model_prices(self, model_id: str) -> Dict[str, float]:
        if model_id in MODEL_PRICING:
            return MODEL_PRICING[model_id]
        model_lower = model_id.lower()
        if any(term in model_lower for term in ["opus", "gpt-4", "large"]):
            return MODEL_PRICING["large"]
        if any(term in model_lower for term in ["sonnet", "gpt-3.5", "medium"]):
            return MODEL_PRICING["medium"]
        return MODEL_PRICING["small"]

    def save_cost_to_db(self, data: Dict[str, Any]):
        if not self.valves.log_to_db:
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
                    data["chat_id"], data["user_id"], data["model_id"],
                    data.get("prompt_tokens"), data.get("completion_tokens"),
                    data["total_tokens"], data["cost"],
                ),
            )
            conn.commit()
            cur.close()
            logger.info(f"Successfully logged cost for chat_id: {data['chat_id']}")
        except Exception as e:
            logger.error(f"Failed to write cost to DB: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()

    async def pipe(self, body: dict, __user__: Optional[dict] = None) -> dict:
        try:
            messages = body.get("messages", [])
            if not messages or messages[-1].get("role") != "assistant":
                return body

            model_id = body.get("model", "unknown")
            chat_id = body.get("chat_id")
            user_id = __user__.get("id") if __user__ else "anonymous"

            if not chat_id:
                logger.warning("No chat_id in body. Cannot log cost.")
                return body

            usage_data = body.get("usage")
            if usage_data and isinstance(usage_data, dict):
                prompt_tokens = usage_data.get("prompt_tokens", 0)
                completion_tokens = usage_data.get("completion_tokens", 0)
            else:
                prompt_text = " ".join([m.get("content", "") for m in messages[:-1]])
                prompt_tokens = self.get_token_count_estimate(prompt_text, model_id)
                completion_text = messages[-1].get("content", "")
                completion_tokens = self.get_token_count_estimate(completion_text, model_id)
            
            prices = self.get_model_prices(model_id)
            prompt_cost = (prompt_tokens / 1_000_000) * prices.get("input", 0)
            completion_cost = (completion_tokens / 1_000_000) * prices.get("output", 0)

            log_data = {
                "chat_id": chat_id, "user_id": user_id, "model_id": model_id,
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": prompt_cost + completion_cost,
            }
            self.save_cost_to_db(log_data)

        except Exception as e:
            logger.error(f"Error in cost calculation pipe: {e}", exc_info=True)
        
        return body