"""
Módulo de base de datos MySQL para SKF Normalizador.
Gestiona el log de consultas, caché persistente de Claude y log de batches.
"""
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pymysql
    import pymysql.cursors
    _PYMYSQL_AVAILABLE = True
except ImportError:
    _PYMYSQL_AVAILABLE = False
    logger.warning("PyMySQL no instalado — base de datos desactivada")


class Database:
    """Conexión a MySQL con reconexión automática y degradación elegante."""

    def __init__(self, host: str, user: str, password: str, database: str, port: int = 3306):
        if not _PYMYSQL_AVAILABLE:
            self._conn = None
            return
        self._config = dict(host=host, user=user, password=password, db=database,
                            port=port, charset="utf8mb4", connect_timeout=10,
                            cursorclass=pymysql.cursors.DictCursor,
                            autocommit=True)
        self._conn = None
        self._connect()

    def _connect(self):
        try:
            import pymysql
            self._conn = pymysql.connect(**self._config)
            logger.info("MySQL conectado correctamente")
        except Exception as e:
            logger.warning(f"MySQL no disponible: {e}")
            self._conn = None

    def _cursor(self):
        if not self._conn:
            return None
        try:
            self._conn.ping(reconnect=True)
            return self._conn.cursor()
        except Exception:
            self._connect()
            if self._conn:
                return self._conn.cursor()
            return None

    def available(self) -> bool:
        return self._conn is not None

    # ── Query log ────────────────────────────────────────────────────────

    def log_query(self, description_original: str, description_clean: str,
                  ref_found: Optional[str], status: str, confidence: str,
                  source: str, notes: str, response_ms: int):
        """Registra una consulta individual en query_log."""
        cur = self._cursor()
        if not cur:
            return
        try:
            cur.execute("""
                INSERT INTO query_log
                    (description_original, description_clean, ref_found, status, confidence, source, notes, response_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (description_original[:2000], description_clean[:500] if description_clean else None,
                  ref_found[:100] if ref_found else None, status, confidence, source,
                  notes[:2000] if notes else None, response_ms))
        except Exception as e:
            logger.debug(f"log_query error: {e}")

    # ── Claude persistent cache ──────────────────────────────────────────

    def get_claude_cache(self, description_clean: str) -> Optional[str]:
        """Busca una descripción en la caché persistente de Claude. Devuelve ref o None."""
        cur = self._cursor()
        if not cur:
            return None
        try:
            cur.execute("""
                SELECT ref_result FROM claude_cache WHERE description_clean = %s
            """, (description_clean[:500],))
            row = cur.fetchone()
            if row:
                # Actualizar contador de uso
                cur.execute("""
                    UPDATE claude_cache SET used_count = used_count + 1, last_used_at = NOW()
                    WHERE description_clean = %s
                """, (description_clean[:500],))
                return row["ref_result"]
        except Exception as e:
            logger.debug(f"get_claude_cache error: {e}")
        return None

    def save_claude_cache(self, description_clean: str, ref_result: Optional[str], status: str):
        """Guarda el resultado de Claude en la caché persistente."""
        cur = self._cursor()
        if not cur:
            return
        try:
            cur.execute("""
                INSERT INTO claude_cache (description_clean, ref_result, status)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    ref_result = VALUES(ref_result),
                    status = VALUES(status),
                    used_count = used_count + 1,
                    last_used_at = NOW()
            """, (description_clean[:500],
                  ref_result[:100] if ref_result else None,
                  status))
        except Exception as e:
            logger.debug(f"save_claude_cache error: {e}")

    def get_all_claude_cache(self) -> dict[str, str]:
        """Carga toda la caché de Claude al inicio (para el _claude_session_cache)."""
        cur = self._cursor()
        if not cur:
            return {}
        try:
            cur.execute("SELECT description_clean, ref_result FROM claude_cache WHERE status = 'FOUND'")
            rows = cur.fetchall()
            return {r["description_clean"]: r["ref_result"] for r in rows if r["ref_result"]}
        except Exception as e:
            logger.debug(f"get_all_claude_cache error: {e}")
            return {}

    # ── Batch log ────────────────────────────────────────────────────────

    def log_batch(self, token: str, filename: str, total_rows: int,
                  found: int, review: int, not_found: int):
        """Registra un trabajo de análisis masivo básico."""
        cur = self._cursor()
        if not cur:
            return
        try:
            cur.execute("""
                INSERT INTO batch_log (token, filename, total_rows, found_count, review_count, not_found_count)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    found_count = VALUES(found_count),
                    review_count = VALUES(review_count),
                    not_found_count = VALUES(not_found_count)
            """, (token, filename[:255] if filename else None, total_rows, found, review, not_found))
        except Exception as e:
            logger.debug(f"log_batch error: {e}")

    def log_batch_pro(self, token: str, improved: int, cost_eur_estimated: float,
                      cost_eur_real: float, found: int, review: int, not_found: int):
        """Actualiza el log de batch con los resultados PRO."""
        cur = self._cursor()
        if not cur:
            return
        try:
            cur.execute("""
                UPDATE batch_log SET
                    used_pro = 1,
                    cost_eur_estimated = %s,
                    cost_eur_real = %s,
                    found_count = %s,
                    review_count = %s,
                    not_found_count = %s
                WHERE token = %s
            """, (cost_eur_estimated, cost_eur_real, found, review, not_found, token))
        except Exception as e:
            logger.debug(f"log_batch_pro error: {e}")

    # ── Stats ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Devuelve estadísticas generales para mostrar en el status."""
        cur = self._cursor()
        if not cur:
            return {}
        try:
            cur.execute("""
                SELECT
                    COUNT(*) as total_queries,
                    SUM(status = 'FOUND') as found,
                    SUM(status = 'NOT_FOUND') as not_found,
                    SUM(status = 'REVIEW') as review
                FROM query_log
            """)
            row = cur.fetchone()
            cur.execute("SELECT COUNT(*) as cache_entries FROM claude_cache")
            cache = cur.fetchone()
            cur.execute("SELECT COALESCE(SUM(cost_eur_real), 0) as total_cost FROM batch_log WHERE used_pro = 1")
            cost = cur.fetchone()
            return {
                "total_queries": row["total_queries"],
                "found": row["found"],
                "not_found": row["not_found"],
                "review": row["review"],
                "claude_cache_entries": cache["cache_entries"],
                "total_cost_eur": float(cost["total_cost"]) if cost["total_cost"] else 0.0,
            }
        except Exception as e:
            logger.debug(f"get_stats error: {e}")
            return {}

    def close(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass


# Singleton — se inicializa desde app.py lifespan
db: Optional[Database] = None


def init_db(host: str, user: str, password: str, database: str, port: int = 3306) -> Database:
    global db
    db = Database(host=host, user=user, password=password, database=database, port=port)
    return db
