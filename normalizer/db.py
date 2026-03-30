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
            self._migrate()
        except Exception as e:
            logger.warning(f"MySQL no disponible: {e}")
            self._conn = None

    def _migrate(self):
        """Añade columnas nuevas de forma no destructiva."""
        cur = self._cursor()
        if not cur:
            return
        migrations = [
            "ALTER TABLE query_log ADD COLUMN IF NOT EXISTS username VARCHAR(50) DEFAULT NULL",
        ]
        for sql in migrations:
            try:
                cur.execute(sql)
            except Exception as e:
                logger.debug(f"Migration skipped: {e}")

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
                  source: str, notes: str, response_ms: int, username: str = ""):
        """Registra una consulta individual en query_log."""
        cur = self._cursor()
        if not cur:
            return
        try:
            cur.execute("""
                INSERT INTO query_log
                    (description_original, description_clean, ref_found, status, confidence, source, notes, response_ms, username)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (description_original[:2000], description_clean[:500] if description_clean else None,
                  ref_found[:100] if ref_found else None, status, confidence, source,
                  notes[:2000] if notes else None, response_ms, username[:50] if username else ""))
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

    # ── Feedback ─────────────────────────────────────────────────────────────

    def log_feedback(self, description: str, ref: Optional[str], thumbs_up: bool, source: str = ""):
        """Registra el feedback del usuario (👍/👎) sobre un resultado."""
        cur = self._cursor()
        if not cur:
            return
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    description VARCHAR(2000),
                    ref_found VARCHAR(100),
                    thumbs_up TINYINT(1),
                    source VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                INSERT INTO feedback_log (description, ref_found, thumbs_up, source)
                VALUES (%s, %s, %s, %s)
            """, (description[:2000], ref[:100] if ref else None, 1 if thumbs_up else 0, source or ""))
        except Exception as e:
            logger.debug(f"log_feedback error: {e}")

    # ── Stats detalladas ─────────────────────────────────────────────────────

    def get_detailed_stats(self) -> dict:
        """Estadísticas detalladas para el dashboard."""
        cur = self._cursor()
        if not cur:
            return {}
        try:
            result = {}
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(status = 'FOUND') as found,
                    SUM(status = 'NOT_FOUND') as not_found,
                    SUM(status = 'REVIEW') as review,
                    SUM(source = 'cache') as from_cache,
                    SUM(source = 'claude') as from_claude,
                    SUM(source = 'catalog') as from_catalog,
                    AVG(response_ms) as avg_ms
                FROM query_log
            """)
            row = cur.fetchone()
            result.update({
                "total_queries": int(row["total"] or 0),
                "found": int(row["found"] or 0),
                "not_found": int(row["not_found"] or 0),
                "review": int(row["review"] or 0),
                "from_cache": int(row["from_cache"] or 0),
                "from_claude": int(row["from_claude"] or 0),
                "from_catalog": int(row["from_catalog"] or 0),
                "avg_response_ms": round(float(row["avg_ms"] or 0)),
            })
            cur.execute("""
                SELECT ref_found, COUNT(*) as cnt FROM query_log
                WHERE status = 'FOUND' AND ref_found IS NOT NULL
                GROUP BY ref_found ORDER BY cnt DESC LIMIT 10
            """)
            result["top_refs"] = [{"ref": r["ref_found"], "count": r["cnt"]} for r in cur.fetchall()]
            cur.execute("""
                SELECT DATE(created_at) as day, COUNT(*) as cnt FROM query_log
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL 14 DAY)
                GROUP BY DATE(created_at) ORDER BY day ASC
            """)
            result["queries_by_day"] = [{"day": str(r["day"]), "count": r["cnt"]} for r in cur.fetchall()]
            try:
                cur.execute("""
                    SELECT username, COUNT(*) as cnt FROM query_log
                    WHERE username IS NOT NULL AND username != ''
                    GROUP BY username ORDER BY cnt DESC LIMIT 10
                """)
                result["top_users"] = [{"user": r["username"], "count": r["cnt"]} for r in cur.fetchall()]
            except Exception:
                result["top_users"] = []
            cur.execute("SELECT COUNT(*) as cnt FROM claude_cache WHERE status = 'FOUND'")
            result["claude_cache_entries"] = int(cur.fetchone()["cnt"] or 0)
            cur.execute("""
                SELECT COUNT(*) as total_batches, COALESCE(SUM(total_rows),0) as total_rows,
                       SUM(used_pro=1) as pro_batches, COALESCE(SUM(cost_eur_real),0) as total_cost
                FROM batch_log
            """)
            br = cur.fetchone()
            result["total_batches"] = int(br["total_batches"] or 0)
            result["total_batch_rows"] = int(br["total_rows"] or 0)
            result["pro_batches"] = int(br["pro_batches"] or 0)
            result["total_cost_eur"] = round(float(br["total_cost"] or 0), 4)
            try:
                cur.execute("""
                    SELECT COUNT(*) as total, SUM(thumbs_up=1) as pos, SUM(thumbs_up=0) as neg
                    FROM feedback_log
                """)
                fb = cur.fetchone()
                result["feedback_total"] = int(fb["total"] or 0)
                result["feedback_positive"] = int(fb["pos"] or 0)
                result["feedback_negative"] = int(fb["neg"] or 0)
            except Exception:
                result["feedback_total"] = 0
                result["feedback_positive"] = 0
                result["feedback_negative"] = 0
            return result
        except Exception as e:
            logger.debug(f"get_detailed_stats error: {e}")
            return {}

    # ── Stats básicas ────────────────────────────────────────────────────────

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
