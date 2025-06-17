import logging
from datetime import date
from hashlib import sha256
from typing import Any, Dict, Optional
from open_webui.env import (
    MY_ODOO_API_URL, 
    MY_ODOO_SECRET_KEY,
)
import requests


logger = logging.getLogger("odoo")


def _generate_auth_token(secret_key: str) -> str:
    logger.info("Generate token for odoo")
    today_str = date.today().strftime("%d/%m/%Y")
    string_to_hash = today_str + secret_key
    return sha256(string_to_hash.encode("utf-8")).hexdigest()


def check_email_exist(email) -> Optional[Dict[str, Any]]:
    logger.info(f"--- Send request in: {MY_ODOO_API_URL} ---")
    json = {"email": email}
    token = _generate_auth_token(MY_ODOO_SECRET_KEY)
    headers = {"token": token}
    try:
        response = requests.get(
            f"{MY_ODOO_API_URL}/tada/user_with_email_exists",
            headers=headers,
            json=json,
            timeout=15,
        )
        response.raise_for_status()

        logger.info("\n✅ Запит успішно виконано!")
        if response.json() == True:
            logger.info("✅ Користувач з таким email існує.")
            return True
        else:
            logger.info("❌ Користувач з таким email не знайдений.")
            return False

    except requests.exceptions.HTTPError as http_err:
        logger.info(f"❌ Помилка HTTP: {http_err}")
        logger.info(f"Текст відповіді сервера: {http_err.response.text}")
    except requests.exceptions.RequestException as err:
        logger.info(f"❌ Не вдалося виконати запит: {err}")

    return False