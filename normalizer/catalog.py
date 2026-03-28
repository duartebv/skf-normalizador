import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz, process
import logging

logger = logging.getLogger(__name__)

CATALOG_COL = "Product Designation / Charge Description"


class CatalogValidator:
    def __init__(self, catalog_path: Path):
        self.refs: list[str] = []
        self._load(catalog_path)

    def _load(self, path: Path) -> None:
        try:
            df = pd.read_csv(path, encoding='utf-8')
            if CATALOG_COL not in df.columns:
                # Try first column
                df.columns = [CATALOG_COL] + list(df.columns[1:])
            self.refs = df[CATALOG_COL].astype(str).str.strip().dropna().tolist()
            self._refs_upper = [r.upper() for r in self.refs]
            logger.info(f"Catálogo cargado: {len(self.refs)} referencias desde {path.name}")
        except Exception as e:
            logger.error(f"Error cargando catálogo: {e}")
            raise

    def validate(self, reference: str) -> tuple[bool, str | None, float]:
        """
        Valida que una referencia exista en el catálogo.
        Returns: (found, matched_ref, confidence)
        """
        if not reference or reference.upper() in ('UNKNOWN', 'NAN', ''):
            return False, None, 0.0

        ref_upper = reference.strip().upper()

        # Búsqueda exacta (case-insensitive)
        for i, r in enumerate(self._refs_upper):
            if r == ref_upper:
                return True, self.refs[i], 100.0

        # Búsqueda fuzzy
        result = process.extractOne(
            ref_upper,
            self._refs_upper,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=85,
        )

        if result:
            _match, score, idx = result
            return True, self.refs[idx], float(score)

        return False, None, 0.0
