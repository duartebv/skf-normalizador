import asyncio
import io
import os
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
import chardet
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

load_dotenv(Path(__file__).parent / ".env", override=True)

import time

from normalizer.cache import CacheNormalizer
from normalizer.catalog import CatalogValidator
from normalizer.claude_client import ClaudeNormalizer
from normalizer.rules import clean_description, normalize_ref_candidate
from normalizer.db import init_db, Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DB_HOST = os.getenv("DB_HOST", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "")
DB_PORT = int(os.getenv("DB_PORT", "3306"))

# In-memory stores
_results_store: dict[str, dict] = {}
_progress_store: dict[str, dict] = {}
_claude_session_cache: dict[str, str] = {}  # clean_desc -> ref

# Global components
cache: CacheNormalizer | None = None
catalog: CatalogValidator | None = None
claude: ClaudeNormalizer | None = None
db: Database | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cache, catalog, claude, db, _claude_session_cache

    model_path = DATA_DIR / "Modelo de aprendizaje.xlsx"
    catalog_path = DATA_DIR / "Data_ref.csv"

    try:
        cache = CacheNormalizer(model_path)
    except Exception as e:
        logger.error(f"No se pudo cargar el modelo de aprendizaje: {e}")

    try:
        catalog = CatalogValidator(catalog_path)
    except Exception as e:
        logger.error(f"No se pudo cargar el catálogo: {e}")

    if API_KEY and cache:
        examples = cache.get_examples(40)
        claude = ClaudeNormalizer(API_KEY, examples)
        logger.info("Claude API configurado correctamente")
    else:
        logger.warning("ANTHROPIC_API_KEY no configurada — sólo se usará la caché")

    # Inicializar base de datos (falla silenciosamente si no está configurada)
    if DB_HOST and DB_USER and DB_PASS and DB_NAME:
        try:
            db = init_db(DB_HOST, DB_USER, DB_PASS, DB_NAME, DB_PORT)
            if db.available():
                # Cargar caché persistente de Claude al arranque
                db_cache = db.get_all_claude_cache()
                _claude_session_cache.update(db_cache)
                logger.info(f"BD conectada · {len(db_cache)} entradas de caché Claude cargadas")
            else:
                logger.warning("BD configurada pero no disponible")
        except Exception as e:
            logger.warning(f"BD no disponible: {e}")
    else:
        logger.info("BD no configurada — funcionando sin persistencia MySQL")

    logger.info("Arranque completado")
    yield

    if db:
        db.close()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(title="SKF Normalizador de Referencias", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------------

def _process_one(description: str, use_claude: bool = True, log_to_db: bool = False) -> dict:
    """Normaliza una descripción. Devuelve dict con ref, status, confidence, notes, source."""
    t0 = time.monotonic()

    if not description or not description.strip():
        return {"ref": None, "status": "NOT_FOUND", "confidence": "LOW", "notes": "Descripción vacía", "source": "none"}

    desc = description.strip()
    desc_clean = clean_description(desc)
    desc_norm = normalize_ref_candidate(desc_clean)

    result = None

    # --- 0. Lookup directo en catálogo ---
    if catalog:
        for candidate in ([desc_norm, desc_clean] if desc_norm != desc_clean else [desc_clean]):
            if not candidate:
                continue
            found_direct, matched_direct, score_direct = catalog.validate(candidate)
            if found_direct and score_direct == 100:
                result = {"ref": matched_direct, "status": "FOUND", "confidence": "HIGH",
                          "notes": "Referencia directa en catálogo", "source": "catalog"}
                break

    if result is None:
        # --- 1. Cache lookup ---
        if cache:
            ref_cache, score_cache, method_cache = cache.lookup(desc, cleaned=desc_clean)
        else:
            ref_cache, score_cache, method_cache = None, 0.0, "none"

        if method_cache == "exact" or (method_cache == "fuzzy" and score_cache >= 90):
            if catalog:
                found, matched, _cat_score = catalog.validate(ref_cache)
            else:
                found, matched = True, ref_cache

            confidence = "HIGH" if score_cache == 100 else "MEDIUM"
            status = "FOUND" if found else "REVIEW"
            notes = "Coincidencia exacta en modelo" if method_cache == "exact" else f"Coincidencia fuzzy {score_cache:.0f}% en modelo"
            if not found:
                notes += " (no validado en catálogo)"
            result = {"ref": matched or ref_cache, "status": status, "confidence": confidence, "notes": notes, "source": "cache"}

        # --- 2. Claude API ---
        elif use_claude and claude:
            ref_claude = claude.normalize_single(desc)

            if ref_claude and ref_claude.upper() not in ("UNKNOWN", "NAN", ""):
                if catalog:
                    found, matched_ref, cat_score = catalog.validate(ref_claude)
                else:
                    found, matched_ref, cat_score = True, ref_claude, 100.0

                if found:
                    confidence = "HIGH" if cat_score == 100 else "MEDIUM"
                    notes = "API Claude" if cat_score == 100 else f"API Claude + catálogo fuzzy {cat_score:.0f}%"
                    result = {"ref": matched_ref or ref_claude, "status": "FOUND", "confidence": confidence, "notes": notes, "source": "claude"}
                else:
                    if ref_cache and score_cache >= 75:
                        result = {
                            "ref": ref_cache,
                            "status": "REVIEW",
                            "confidence": "LOW",
                            "notes": f"Claude sugirió {ref_claude} (no en catálogo). Cache sugiere {ref_cache} ({score_cache:.0f}%)",
                            "source": "claude",
                        }
                    else:
                        result = {
                            "ref": ref_claude,
                            "status": "NOT_FOUND",
                            "confidence": "LOW",
                            "notes": f"Claude sugirió {ref_claude} pero no existe en catálogo",
                            "source": "claude",
                        }

        # --- 3. Fallback: weak cache suggestion ---
        if result is None:
            if ref_cache and score_cache >= 75:
                result = {"ref": ref_cache, "status": "REVIEW", "confidence": "LOW", "notes": f"Sólo coincidencia fuzzy débil {score_cache:.0f}%", "source": "cache"}
            else:
                result = {"ref": None, "status": "NOT_FOUND", "confidence": "LOW", "notes": "No identificado", "source": "none"}

    if log_to_db and db and db.available():
        ms = int((time.monotonic() - t0) * 1000)
        db.log_query(desc, desc_clean, result["ref"], result["status"],
                     result["confidence"], result["source"], result.get("notes", ""), ms)

    return result


def _read_file(content: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    detected = chardet.detect(content)
    encoding = detected.get("encoding") or "utf-8"
    try:
        return pd.read_csv(io.BytesIO(content), encoding=encoding, on_bad_lines="skip")
    except Exception:
        return pd.read_csv(io.BytesIO(content), encoding="latin-1", on_bad_lines="skip")


def _detect_desc_column(df: pd.DataFrame) -> str | None:
    keywords = ["descripcion", "descripción", "description", "referencia", "ref",
                "articulo", "artículo", "producto", "texto", "material", "item", "denominacion"]
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    for col in df.columns:
        if df[col].dtype == object:
            return col
    return None


def _build_output_excel(df_original: pd.DataFrame, results: list[dict]) -> bytes:
    df_out = df_original.copy()
    df_out["REF_NORMALIZADA"] = [r["ref"] if r["ref"] else "" for r in results]
    df_out["ESTADO"] = [r["status"] for r in results]
    df_out["CONFIANZA"] = [r["confidence"] for r in results]
    df_out["NOTAS"] = [r["notes"] for r in results]

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Resultado")
        ws = writer.sheets["Resultado"]

        header_fill = PatternFill("solid", fgColor="0040A0")
        header_font = Font(color="FFFFFF", bold=True)
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")

        ws.freeze_panes = "A2"

        status_col_idx = df_out.columns.get_loc("ESTADO") + 1
        fills = {
            "FOUND": PatternFill("solid", fgColor="C6EFCE"),
            "NOT_FOUND": PatternFill("solid", fgColor="FFC7CE"),
            "REVIEW": PatternFill("solid", fgColor="FFEB9C"),
        }

        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            status_val = row[status_col_idx - 1].value
            fill = fills.get(status_val)
            if fill:
                for cell in row:
                    cell.fill = fill

        for col_idx, col in enumerate(df_out.columns, 1):
            max_len = max(
                len(str(col)),
                df_out.iloc[:, col_idx - 1].astype(str).str.len().max() if len(df_out) > 0 else 0,
            )
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 50)

    output.seek(0)
    return output.read()


def _build_output_csv(df_original: pd.DataFrame, results: list[dict]) -> bytes:
    df_out = df_original.copy()
    df_out["REF_NORMALIZADA"] = [r["ref"] if r["ref"] else "" for r in results]
    df_out["ESTADO"] = [r["status"] for r in results]
    df_out["CONFIANZA"] = [r["confidence"] for r in results]
    df_out["NOTAS"] = [r["notes"] for r in results]
    return df_out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def _count_statuses(results: list[dict]) -> dict:
    counts = {"FOUND": 0, "NOT_FOUND": 0, "REVIEW": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# PRO background processing
# ---------------------------------------------------------------------------

def _run_pro_sync(token: str):
    """Runs in thread pool. Updates _progress_store[token] as it goes."""
    global _claude_session_cache

    entry = _results_store.get(token)
    if not entry or not claude:
        _progress_store[token] = {"status": "error", "message": "Datos no disponibles"}
        return

    descriptions = entry["descriptions"]
    results = [dict(r) for r in entry["results"]]

    indices = [i for i, r in enumerate(results) if r["status"] in ("NOT_FOUND", "REVIEW")]
    total_pending = len(indices)

    _progress_store[token] = {"status": "processing", "current": 0, "total": total_pending, "improved": 0}

    if total_pending == 0:
        entry["results_pro"] = results
        entry["excel_pro"] = entry["excel_basic"]
        entry["csv_pro"] = entry["csv_basic"]
        counts = _count_statuses(results)
        _progress_store[token] = {
            "status": "done",
            "found": counts["FOUND"], "not_found": counts["NOT_FOUND"], "review": counts["REVIEW"],
            "improved": 0, "cost_eur": 0.0, "total": len(results),
        }
        return

    # Deduplicate: only send unique descriptions to Claude
    unique_descs = list(dict.fromkeys(descriptions[i] for i in indices))

    # Filter out session-cached results
    to_request = [d for d in unique_descs if clean_description(d) not in _claude_session_cache]

    def _progress_cb(current, total_chunks):
        pct_done = current / max(total_chunks, 1)
        _progress_store[token]["current"] = int(pct_done * total_pending)

    if to_request:
        claude_refs = claude.normalize_batch(to_request, progress_callback=_progress_cb)
        for desc, ref in zip(to_request, claude_refs):
            clean_key = clean_description(desc)
            _claude_session_cache[clean_key] = ref
            # Persist to DB cache
            if db and db.available():
                status_for_cache = "FOUND" if ref and ref.upper() not in ("UNKNOWN", "NAN", "") else "NOT_FOUND"
                db.save_claude_cache(clean_key, ref if status_for_cache == "FOUND" else None, status_for_cache)

    improved = 0
    for original_idx in indices:
        desc = descriptions[original_idx]
        ref_claude = _claude_session_cache.get(clean_description(desc), "UNKNOWN")

        if not ref_claude or ref_claude.upper() in ("UNKNOWN", "NAN", ""):
            continue
        if catalog:
            found, matched_ref, cat_score = catalog.validate(ref_claude)
        else:
            found, matched_ref, cat_score = True, ref_claude, 100.0

        if found:
            confidence = "HIGH" if cat_score == 100 else "MEDIUM"
            notes = "Claude API (Versión PRO)" if cat_score == 100 else f"Claude API + catálogo fuzzy {cat_score:.0f}%"
            results[original_idx] = {"ref": matched_ref or ref_claude, "status": "FOUND",
                                     "confidence": confidence, "notes": notes, "source": "claude"}
            improved += 1
        else:
            results[original_idx]["notes"] += f" | Claude sugirió: {ref_claude} (no en catálogo)"

    df = _read_file(entry["file_content"], entry["original_filename"])
    entry["results_pro"] = results
    entry["excel_pro"] = _build_output_excel(df, results)
    entry["csv_pro"] = _build_output_csv(df, results)

    counts = _count_statuses(results)
    cost_eur = claude.real_cost_eur() if claude else 0.0

    if db and db.available():
        est = _results_store.get(token, {}).get("_cost_estimated", 0.0)
        db.log_batch_pro(token, improved, est, cost_eur,
                         counts["FOUND"], counts["REVIEW"], counts["NOT_FOUND"])

    _progress_store[token] = {
        "status": "done",
        "found": counts["FOUND"], "not_found": counts["NOT_FOUND"], "review": counts["REVIEW"],
        "improved": improved, "cost_eur": cost_eur, "total": len(results),
    }
    logger.info(f"PRO batch done: token={token}, improved={improved}, cost=€{cost_eur}")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def status() -> JSONResponse:
    return JSONResponse({
        "cache_loaded": cache is not None,
        "catalog_loaded": catalog is not None,
        "claude_available": claude is not None,
        "cache_entries": len(cache.df) if cache else 0,
        "catalog_entries": len(catalog.refs) if catalog else 0,
    })


@app.post("/api/columns")
async def get_columns(file: UploadFile = File(...)) -> JSONResponse:
    content = await file.read()
    try:
        df = _read_file(content, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Error leyendo archivo: {e}")

    suggested = _detect_desc_column(df)

    # Send first 5 rows of suggested column for preview
    preview_rows = []
    if suggested and suggested in df.columns:
        preview_rows = df[suggested].astype(str).head(5).tolist()

    return JSONResponse({
        "columns": list(df.columns),
        "suggested": suggested,
        "rows": len(df),
        "preview_rows": preview_rows,
    })


@app.post("/api/normalize/single")
async def normalize_single(description: str = Form(...)) -> JSONResponse:
    if not description.strip():
        raise HTTPException(400, "La descripción no puede estar vacía")
    result = _process_one(description.strip(), log_to_db=True)
    return JSONResponse(result)


@app.post("/api/normalize/batch")
async def normalize_batch(
    file: UploadFile = File(...),
    col: Optional[str] = Form(None),
) -> JSONResponse:
    if not file.filename.lower().endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(400, "El archivo debe ser Excel (.xlsx/.xls) o CSV (.csv)")

    content = await file.read()

    try:
        df = _read_file(content, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Error leyendo archivo: {e}")

    if df.empty:
        raise HTTPException(400, "El archivo está vacío")

    if col and col in df.columns:
        desc_col = col
    else:
        desc_col = _detect_desc_column(df)
        if desc_col is None:
            raise HTTPException(400, "No se pudo detectar la columna de descripciones")

    descriptions = df[desc_col].astype(str).tolist()

    # Basic processing with deduplication (cache + catalog only — no Claude)
    dedup_cache: dict[str, dict] = {}
    results = []
    for desc in descriptions:
        key = desc.strip().lower() if desc.strip() not in ("nan", "") else ""
        raw = desc.strip() if desc.strip() not in ("nan", "") else ""
        if key not in dedup_cache:
            dedup_cache[key] = _process_one(raw, use_claude=False)
        results.append(dedup_cache[key])

    excel_basic = _build_output_excel(df, results)
    csv_basic = _build_output_csv(df, results)

    token = str(uuid.uuid4())
    orig_name = file.filename.rsplit(".", 1)[0]
    _results_store[token] = {
        "descriptions": descriptions,
        "results": results,
        "results_pro": None,
        "excel_basic": excel_basic,
        "csv_basic": csv_basic,
        "excel_pro": None,
        "csv_pro": None,
        "file_content": content,
        "original_filename": file.filename,
        "desc_col": desc_col,
        "orig_name": orig_name,
    }

    counts = _count_statuses(results)

    if db and db.available():
        db.log_batch(token, file.filename, len(results),
                     counts["FOUND"], counts["REVIEW"], counts["NOT_FOUND"])

    return JSONResponse({
        "token": token,
        "total": len(results),
        "found": counts["FOUND"],
        "not_found": counts["NOT_FOUND"],
        "review": counts["REVIEW"],
        "desc_col": desc_col,
        "claude_available": claude is not None,
    })


@app.get("/api/estimate/{token}")
async def estimate_cost(token: str) -> JSONResponse:
    entry = _results_store.get(token)
    if not entry:
        raise HTTPException(404, "Token no encontrado")

    pending = [r for r in entry["results"] if r["status"] in ("NOT_FOUND", "REVIEW")]

    # Deduplicate estimate (unique descriptions only)
    unique_pending = set(
        entry["descriptions"][i].strip().lower()
        for i, r in enumerate(entry["results"])
        if r["status"] in ("NOT_FOUND", "REVIEW")
    )
    # Subtract already session-cached
    uncached = {d for d in unique_pending if clean_description(d) not in _claude_session_cache}
    n_api_calls = len(uncached)

    batches = max(1, -(-n_api_calls // 10))
    input_tokens = batches * 2700
    output_tokens = batches * 150
    cost_usd = (input_tokens * 3 + output_tokens * 15) / 1_000_000
    cost_usd_max = cost_usd * 1.35
    cost_eur = cost_usd * 0.92
    cost_eur_max = cost_usd_max * 0.92

    return JSONResponse({
        "pending": len(pending),
        "unique_api_calls": n_api_calls,
        "session_cached": len(unique_pending) - n_api_calls,
        "claude_available": claude is not None,
        "cost_usd": round(cost_usd, 4),
        "cost_eur": round(cost_eur, 4),
        "cost_usd_max": round(cost_usd_max, 4),
        "cost_eur_max": round(cost_eur_max, 4),
    })


@app.post("/api/normalize/batch/pro/{token}")
async def normalize_batch_pro_start(token: str) -> JSONResponse:
    """Inicia el procesamiento PRO en background. Responde inmediatamente."""
    entry = _results_store.get(token)
    if not entry:
        raise HTTPException(404, "Token no encontrado")
    if not claude:
        raise HTTPException(400, "Claude API no configurada. Añade ANTHROPIC_API_KEY al archivo .env")

    # Avoid double-start
    existing = _progress_store.get(token, {})
    if existing.get("status") == "processing":
        return JSONResponse({"status": "already_processing"})

    _progress_store[token] = {"status": "processing", "current": 0, "total": 0, "improved": 0}
    asyncio.create_task(asyncio.to_thread(_run_pro_sync, token))
    return JSONResponse({"status": "started"})


@app.get("/api/progress/{token}")
async def get_progress(token: str) -> JSONResponse:
    progress = _progress_store.get(token)
    if not progress:
        raise HTTPException(404, "No hay progreso registrado")
    return JSONResponse(progress)


@app.get("/api/download/{token}")
async def download_result(
    token: str,
    version: str = "basic",
    fmt: str = "xlsx",
    status_filter: str = "all",
) -> StreamingResponse:
    entry = _results_store.get(token)
    if not entry:
        raise HTTPException(404, "Resultado no encontrado o expirado")

    if version == "pro":
        results = entry.get("results_pro")
        if not results:
            raise HTTPException(400, "Versión PRO no disponible todavía")
        orig_data = entry["excel_pro"]
        orig_csv = entry.get("csv_pro")
        suffix = "PRO"
    else:
        results = entry["results"]
        orig_data = entry["excel_basic"]
        orig_csv = entry.get("csv_basic")
        suffix = "basico"

    orig_name = entry["orig_name"]

    # Apply status filter if requested
    if status_filter != "all" and results:
        df = _read_file(entry["file_content"], entry["original_filename"])
        filtered_results = []
        filtered_rows = []
        for i, r in enumerate(results):
            if r["status"] == status_filter.upper():
                filtered_results.append(r)
                filtered_rows.append(i)
        df_filtered = df.iloc[filtered_rows].reset_index(drop=True)
        if fmt == "csv":
            data = _build_output_csv(df_filtered, filtered_results)
        else:
            data = _build_output_excel(df_filtered, filtered_results)
        suffix = f"{suffix}_{status_filter}"
    else:
        data = orig_csv if fmt == "csv" else orig_data

    if fmt == "csv":
        filename = f"resultado_{orig_name}_{suffix}.csv"
        media_type = "text/csv"
    else:
        filename = f"resultado_{orig_name}_{suffix}.xlsx"
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Serve frontend (must be last)
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
