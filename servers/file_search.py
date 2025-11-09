import os, re, mimetypes, asyncio, logging
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from google import genai
from google.genai import types

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GeminiFileSearchMCP")

client = genai.Client()
mcp = FastMCP("Google Gemini File Search")

# --- Helpers -------------------------------------------------------------
def _get_name(obj):
    """Return the name field regardless of SDK object type."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return obj.get("name") or obj.get("id") or str(obj)
    return getattr(obj, "name", str(obj))

def _sanitize_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0].lower()
    safe = re.sub(r"[^a-z0-9-]+", "-", base).strip("-")
    return safe or "file"

def _safe_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime:
        return mime
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".log", ".md"): return "text/plain"
    if ext == ".json": return "application/json"
    if ext == ".csv": return "text/csv"
    if ext == ".pdf": return "application/pdf"
    return "application/octet-stream"

# --- 1. Upload and index -------------------------------------------------
@mcp.tool()
async def upload_and_index(file_path: str, display_name: str = None) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    mime_type = _safe_mime(file_path)
    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        raise ValueError("❌ DOCX is not yet supported by File Search. Please convert to PDF first.")

    display_name = display_name or os.path.basename(file_path)
    logger.info(f"📂 Uploading {display_name} ({mime_type})")

    store = client.file_search_stores.create(config={"display_name": f"store_{int(asyncio.get_event_loop().time())}"})
    store_name = _get_name(store)
    logger.info(f"🪣 Created FileSearchStore: {store_name}")

    op = client.file_search_stores.upload_to_file_search_store(
        file_search_store_name=store_name,
        file=file_path,
        config={
            "display_name": display_name,
            "mime_type": mime_type,
            "chunking_config": {
                "white_space_config": {"max_tokens_per_chunk": 500, "max_overlap_tokens": 100}
            },
        },
    )

    op_name = _get_name(op)
    for _ in range(60):
        current = client.operations.get(op_name)
        if getattr(current, "done", False):
            logger.info("✅ Upload and indexing complete.")
            break
        await asyncio.sleep(3)

    return store_name

# --- 2. Import via Files API --------------------------------------------
@mcp.tool()
async def import_file(file_path: str, display_name: str = None) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    mime_type = _safe_mime(file_path)
    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        raise ValueError("❌ DOCX unsupported — convert to PDF first.")

    display_name = display_name or os.path.basename(file_path)
    safe_name = _sanitize_name(file_path)
    logger.info(f"⬆️  Uploading via Files API as {safe_name}")

    sample_file = client.files.upload(
        file=file_path,
        config={"name": safe_name, "display_name": display_name, "mime_type": mime_type},
    )
    file_name = _get_name(sample_file)

    store = client.file_search_stores.create(config={"display_name": display_name})
    store_name = _get_name(store)
    logger.info(f"🪣 Created FileSearchStore: {store_name}")

    op = client.file_search_stores.import_file(
        file_search_store_name=store_name,
        file_name=file_name,
    )

    op_name = _get_name(op)
    for _ in range(60):
        current = client.operations.get(op_name)
        if getattr(current, "done", False):
            logger.info("✅ File imported and indexed.")
            break
        await asyncio.sleep(3)

    return store_name

# --- 3. Query ------------------------------------------------------------
@mcp.tool()
async def query_file_search(store_name: str, question: str) -> dict:
    if not store_name or not question:
        raise ValueError("Missing store_name or question.")
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question,
            config=types.GenerateContentConfig(
                tools=[types.Tool(file_search=types.FileSearch(file_search_store_names=[store_name]))]
            ),
        ),
    )
    grounding = getattr(resp.candidates[0], "grounding_metadata", None)
    sources = [c.retrieved_context.title for c in getattr(grounding, "grounding_chunks", [])]
    return {"answer": resp.text, "sources": sources, "store": store_name}

# --- 4–6. List / Get / Delete ------------------------------------------
@mcp.tool()
async def list_stores() -> list:
    stores = client.file_search_stores.list()
    return [getattr(s, "to_dict", lambda: s)() for s in stores]

@mcp.tool()
async def get_store(store_name: str) -> dict:
    store = client.file_search_stores.get(name=store_name)
    return getattr(store, "to_dict", lambda: store)()

@mcp.tool()
async def delete_store(store_name: str, force: bool = False) -> str:
    client.file_search_stores.delete(name=store_name, config={"force": force})
    return f"🗑️ Deleted File Search Store: {store_name} (force={force})"

if __name__ == "__main__":
    logger.info("🚀 Starting Gemini File Search MCP Server...")
    mcp.run()
