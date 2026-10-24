import asyncio
import datetime as dt
import html
import logging
import os

import aiosqlite
import httpx
from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from openai import AsyncOpenAI
import feedparser

load_dotenv()
#
logging.basicConfig(
    filename='sci-bot.log',
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("sci-news-bot")

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Укажите BOT_TOKEN в .env")

DB_PATH = os.getenv("DB_PATH", "data/bot.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY","")


DIGEST_HOUR = int(os.getenv("DIGEST_HOUR", "9"))
DIGEST_MINUTE = int(os.getenv("DIGEST_MINUTE", "0"))
ARTICLES_PER_USER = int(os.getenv("ARTICLES_PER_USER", "5"))
SEARCH_DAYS = int(os.getenv("SEARCH_DAYS", "2"))

llm = AsyncOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

router = Router()


async def get_db():
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    return conn


async def init_db():
    conn = await get_db()
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                chat_id INTEGER PRIMARY KEY,
                keywords TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS seen (
                chat_id INTEGER NOT NULL,
                work_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (chat_id, work_id)
            )
        """)

        await conn.commit()
    finally:
        await conn.close()


async def ensure_user(chat_id: int):
    conn = await get_db()
    try:
        await conn.execute(
            "INSERT OR IGNORE INTO users(chat_id) VALUES (?)",
            (chat_id,)
        )
        await conn.commit()
    finally:
        await conn.close()


async def save_keywords(chat_id: int, keywords: str):
    conn = await get_db()
    try:
        await conn.execute(
            """
            INSERT INTO users(chat_id, keywords)
            VALUES (?, ?)
            ON CONFLICT(chat_id)
            DO UPDATE SET keywords = excluded.keywords
            """,
            (chat_id, keywords)
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_keywords(chat_id: int) -> str:
    conn = await get_db()
    try:
        cur = await conn.execute(
            "SELECT keywords FROM users WHERE chat_id = ?",
            (chat_id,)
        )
        row = await cur.fetchone()
        return row["keywords"] if row else ""
    finally:
        await conn.close()


async def mark_seen(chat_id: int, work_id: str) -> bool:
    """
    Возвращает True, если статья новая и была отмечена как отправленная.
    Возвращает False, если она уже была отправлена ранее.
    """
    conn = await get_db()
    try:
        cur = await conn.execute(
            "INSERT OR IGNORE INTO seen(chat_id, work_id) VALUES (?, ?)",
            (chat_id, work_id)
        )
        await conn.commit()
        return cur.rowcount > 0
    finally:
        await conn.close()


def reconstruct_abstract(inverted_index):
    """
    OpenAlex часто отдает аннотацию в inverted index формате:
    {
        "Word1": [0, 10],
        "Word2": [1]
    }
    Эта функция собирает текст обратно.
    """
    if not inverted_index:
        return ""

    max_pos = -1
    for positions in inverted_index.values():
        if positions:
            max_pos = max(max_pos, max(positions))

    if max_pos < 0:
        return ""

    words = [""] * (max_pos + 1)

    for word, positions in inverted_index.items():
        for pos in positions:
            if 0 <= pos <= max_pos:
                words[pos] = word

    return " ".join(words).strip()


async def fetch_openalex(keywords: list[str], limit: int = 5, days: int = 2):
    url = "https://api.openalex.org/works"
    since = dt.date.today() - dt.timedelta(days=days)

    params = {
        "filter": f"from_created_date:{since.isoformat()}",
        "sort": "publication_date",
        "order": "desc",
        "per-page": min(max(limit, 1), 25),
        "api_key": OPENALEX_API_KEY
    }

    # if OPENALEX_MAILTO:
    #     params["mailto"] = OPENALEX_MAILTO

    if keywords:
        params["search"] = " ".join(keywords)

    headers = {
        "User-Agent": "sci-news-bot/0.1"
    }

    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()

    items = []

    for work in payload.get("results", []):
        primary_location = work.get("primary_location") or {}
        article_url = primary_location.get("landing_page_url") or work.get("doi_url") or ""

        items.append({
            "id": work.get("id") or article_url,
            "title": work.get("display_name") or "Без названия",
            "date": work.get("publication_date") or "",
            "url": article_url,
            "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        })

    return items

async def fetch_arxiv(query: str, max_results: int = 5):
    url = "https://export.arxiv.org/api/query"

    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()

    feed = feedparser.parse(response.text)

    items = []
    for entry in feed.entries:
        items.append({
            "id": entry.id,
            "title": entry.title,
            "date": entry.published,
            "url": entry.link,
            "abstract": entry.summary,
        })

    return items

async def fetch_semantic_scholar(query: str, limit: int = 5):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": query,
        "fields": "title,abstract,url,publicationDate,externalIds",
        "limit": min(limit, 20),
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    items = []

    for paper in data.get("data", []):
        items.append({
            "id": paper.get("paperId"),
            "title": paper.get("title"),
            "date": paper.get("publicationDate"),
            "url": paper.get("url"),
            "abstract": paper.get("abstract") or "",
        })

    return items

async def summarize(title: str, abstract: str) -> str:
    if not abstract.strip():
        return "Аннотация недоступна, поэтому саммари по названию может быть поверхностным."

    system_prompt = (
        "Ты — научный редактор. Сделай краткое саммари статьи на русском языке, "
        "опираясь только на название и аннотацию. Не выдумывай факты. "
        "Формат: 3-6 предложений или маркированный список: суть, метод, результат, почему важно. "
        "Если информации мало, так и скажи."
    )

    user_prompt = f"""
Название:
{title}

Аннотация:
{abstract[:8000]}

Сделай саммари:
""".strip()

    try:
        response = await llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("Ошибка LLM")
        return f"Не удалось получить саммари: {e}"


def format_item(item: dict, summary: str) -> str:
    title = html.escape(str(item.get("title", "")))
    date = html.escape(str(item.get("date", "")))
    summary = html.escape(summary)

    url = item.get("url")
    if url:
        link = f'<a href="{html.escape(url, quote=True)}">Открыть статью</a>'
    else:
        link = "Ссылка недоступна"

    return (
        f"📄 <b>{title}</b>\n"
        f"Дата: {date}\n\n"
        f"<b>Саммари:</b>\n"
        f"{summary}\n\n"
        f"{link}"
    )


@router.message(CommandStart())
async def cmd_start(message: types.Message):
    await ensure_user(message.chat.id)
    await message.answer(
        "Привет! Я бот для дайджеста научных статей.\n\n"
        "Задайте ключевые слова командой:\n"
        "/set machine learning, genomics, LLM agents\n\n"
        "Полезные команды:\n"
        "/keywords — показать текущие ключевые слова\n"
        "/test — проверить поиск и саммари прямо сейчас"
    )


@router.message(Command("set"))
async def cmd_set(message: types.Message):
    parts = (message.text or "").split(maxsplit=1)

    if len(parts) < 2 or not parts[1].strip():
        await message.answer(
            "Использование:\n"
            "/set machine learning, genomics, LLM agents"
        )
        return

    keywords = parts[1].strip()
    await save_keywords(message.chat.id, keywords)

    await message.answer(
        f"Ключевые слова сохранены:\n"
        f"{keywords}\n\n"
        "Проверить работу можно командой /test"
    )


@router.message(Command("keywords"))
async def cmd_keywords(message: types.Message):
    keywords = await get_keywords(message.chat.id)

    if not keywords:
        await message.answer(
            "Ключевые слова пока не заданы.\n"
            "Используйте /set тема1, тема2"
        )
        return

    await message.answer(f"Ваши ключевые слова:\n{keywords}")


@router.message(Command("test"))
async def cmd_test(message: types.Message):
    keywords_raw = await get_keywords(message.chat.id)
    keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

    full_list = []

    if not keywords:
        keywords = ["large language models"]

    await message.answer(f"Проверяю по темам: {', '.join(keywords)}")

    try:
        items_oa = await fetch_openalex(
            keywords=keywords,
            limit=1,
            days=2
        )
        full_list.extend(items_oa)
    except Exception as e:
        logger.exception("[openalex] Ошибка получения статей")
        await message.answer(f"[openalex] Ошибка источника статей: {e}")

    await asyncio.sleep(5)

    try:
        items_arx = await fetch_arxiv(
            query=keywords,
        )
        full_list.extend(items_arx)
    except Exception as e:
        logger.exception("[arxiv] Ошибка получения статей")
        await message.answer(f"[arxiv] Ошибка источника статей: {e}")
        return

    try:
        items_ss = await fetch_semantic_scholar(
            query=keywords,
            limit=1,
        )
        full_list.extend(items_ss)
    except Exception as e:
        logger.exception("[semantic scholar] Ошибка получения статей")
        await message.answer(f"[semanticscholar] Ошибка источника статей: {e}")


    if not full_list:
        await message.answer(
            "За последние дни ничего не нашлось. "
            "Попробуйте другие ключевые слова или расширить период."
        )
        return

    for item in full_list[:2]:
        summary = await summarize(item["title"], item["abstract"])
        await message.answer(
            format_item(item, summary),
            parse_mode="HTML",
            disable_web_page_preview=True
        )
        await asyncio.sleep(1.1)


async def send_digest_to_user(bot: Bot, chat_id: int, keywords_raw: str):
    keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

    if not keywords:
        return
    
    # try:
    #     items = await fetch_openalex(
    #         keywords=keywords,
    #         limit=ARTICLES_PER_USER * 3,
    #         days=SEARCH_DAYS
    #     )
    # except Exception as e:
    #     logger.warning("Ошибка получения статей для %s: %s", chat_id, e)
    #     return
    for kw in keywords:
        try:
            items = await fetch_arxiv(
                query=kw,
            )
        except Exception as e:
            logger.exception("[arxiv] Ошибка получения статей")
            return
        
        await asyncio.sleep(5) 

    sent = 0

    for item in items:
        if sent >= ARTICLES_PER_USER:
            break

        work_id = item.get("id")
        if not work_id:
            continue

        is_new = await mark_seen(chat_id, work_id)
        if not is_new:
            continue

        summary = await summarize(item["title"], item["abstract"])

        try:
            await bot.send_message(
                chat_id,
                format_item(item, summary),
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            sent += 1
            await asyncio.sleep(1.2)
        except Exception as e:
            logger.warning("Ошибка отправки сообщения пользователю %s: %s", chat_id, e)
            break

    if sent > 0:
        logger.info("Отправлено %s статей пользователю %s", sent, chat_id)


async def digest_job(bot: Bot):
    conn = await get_db()
    try:
        cur = await conn.execute(
            """
            SELECT chat_id, keywords
            FROM users
            WHERE keywords IS NOT NULL AND keywords != ''
            """
        )
        users = await cur.fetchall()
    finally:
        await conn.close()

    logger.info("Запуск дайджеста для %s пользователей", len(users))

    for user in users:
        try:
            await send_digest_to_user(
                bot=bot,
                chat_id=user["chat_id"],
                keywords_raw=user["keywords"]
            )
        except Exception:
            logger.exception("Ошибка дайджеста для пользователя %s", user["chat_id"])

        await asyncio.sleep(0.5)

    logger.info("Дайджест завершен")


async def main():
    await init_db()

    from aiogram.client.session.aiohttp import AiohttpSession
    #from aiohttp_socks import ProxyConnector
    # connector = ProxyConnector.from_url("socks5://sciproxyuser:6F:A4oY5Ah8v}WA@195.135.253.9:1080")
    # bot = Bot(token=BOT_TOKEN, session=AiohttpSession(connector=connector))

    session = AiohttpSession("socks5://sciproxyuser:6F:A4oY5Ah8v}WA@195.135.253.9:1080")
    bot = Bot(token=BOT_TOKEN, session=session)


    #bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    await bot.set_my_commands([
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="set", description="Задать ключевые слова"),
        BotCommand(command="keywords", description="Показать ключевые слова"),
        BotCommand(command="test", description="Проверить дайджест"),
    ])

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        digest_job,
        trigger="cron",
        hour=DIGEST_HOUR,
        minute=DIGEST_MINUTE,
        args=[bot],
    )
    scheduler.start()

    logger.info(
        "Дайджест будет запускаться в %02d:%02d UTC",
        DIGEST_HOUR,
        DIGEST_MINUTE
    )

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())