from aiogram import Bot
from open_webui.env import TG_BOT_TOKEN
import logging

try:
    if not TG_BOT_TOKEN:
        bot = None
        logging.warning("TG_BOT_TOKEN is not set. Bot will not be initialized.")
    bot = Bot(token=TG_BOT_TOKEN)
    
except ValueError as e:
    logging.warning(f"Error initi: {e}")
except Exception as e:
    logging.warning(f"Произошла ошибка при инициализации бота: {e}")