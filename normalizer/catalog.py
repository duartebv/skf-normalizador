import re
import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz, process
import logging

logger = logging.getLogger(__name__)

CATALOG_COL = "Product Designation / Charge Description"


def _norm_ws(s: str) -> str:
    """Normaliza espacios en una referencia para comparación."""
    return re.sub(r'\s+', ' ', s).strip().upper()


class CatalogValidator:
    def __init__(self, catalog_path: Path):
        self.refs: list[str] = []
        self._load(catalog_path)

    def _load(self, path: Path) -> None:
        try:
            df = pd.read_csv(path, encoding='utf-8')
            if CATALOG_COL not in df.columns:
                df.columns = [CATALOG_COL] + list(df.columns[1:])
            self.refs = df[CATALOG_COL].astype(str).str.strip().dropna().tolist()
            # Versiones upper con espacios normalizados para comparación rápida
            self._refs_upper = [_norm_ws(r) for r in self.refs]
            logger.info(f"Catálogo cargado: {len(self.refs)} referencias desde {path.name}")
        except Exception as e:
            logger.error(f"Error cargando catálogo: {e}")
            raise

    def validate(self, reference: str) -> tuple[bool, str | None, float]:
        """
        Valida que una referencia exista en el catálogo.
        Intenta también la versión normalizada para cubrir variantes de formato.
        Returns: (found, matched_ref, confidence)
        """
        if not reference or reference.upper() in ('UNKNOWN', 'NAN', ''):
            return False, None, 0.0

        from normalizer.rules import normalize_ref_candidate

        ref_upper = _norm_ws(reference)

        # Candidatos a buscar: original y versión re-normalizada
        candidates = [ref_upper]
        ref_renorm = _norm_ws(normalize_ref_candidate(reference))
        if ref_renorm and ref_renorm != ref_upper:
            candidates.append(ref_renorm)

        for candidate in candidates:
            # 1. Búsqueda exacta (case-insensitive, espacios normalizados)
            for i, r in enumerate(self._refs_upper):
                if r == candidate:
                    return True, self.refs[i], 100.0

        # 2. Búsqueda fuzzy sobre todos los candidatos
        best_score = 0.0
        best_idx = -1
        for candidate in candidates:
            result = process.extractOne(
                candidate,
                self._refs_upper,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=85,
            )
            if result and float(result[1]) > best_score:
                best_score = float(result[1])
                best_idx = result[2]

        if best_idx >= 0:
            return True, self.refs[best_idx], best_score

        return False, None, 0.0
