import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz, process
import logging

logger = logging.getLogger(__name__)

DESC_COL = "Description Cliente"
REF_COL = "REF. SKF"
DESC_CLEAN_COL = "_DESC_CLEAN"


class CacheNormalizer:
    def __init__(self, model_path: Path):
        self.df: pd.DataFrame = pd.DataFrame()
        self._load(model_path)

    def _load(self, path: Path) -> None:
        # Import here to avoid circular import at module level
        from normalizer.rules import clean_description
        try:
            df = pd.read_excel(path)
            df.columns = [c.strip() for c in df.columns]
            df[DESC_COL] = df[DESC_COL].astype(str).str.upper().str.strip()
            df[REF_COL] = df[REF_COL].astype(str).str.strip()
            df = df[df[REF_COL].notna() & (df[REF_COL] != '') & (df[REF_COL] != 'nan')]
            # Pre-compute cleaned versions for smarter matching
            df[DESC_CLEAN_COL] = df[DESC_COL].apply(clean_description)
            self.df = df.reset_index(drop=True)
            logger.info(f"Cache cargado: {len(self.df)} entradas desde {path.name}")
        except Exception as e:
            logger.error(f"Error cargando cache: {e}")
            raise

    def lookup(self, description: str, cleaned: str | None = None) -> tuple[str | None, float, str]:
        """
        Busca en el modelo de aprendizaje sobre descripción original Y limpiada.
        Returns: (referencia, score, method)  method: 'exact' | 'fuzzy' | 'none'
        """
        if self.df.empty:
            return None, 0.0, 'none'

        from normalizer.rules import clean_description
        desc_upper = description.upper().strip()
        desc_clean = cleaned if cleaned is not None else clean_description(desc_upper)

        # 1. Exact match on original
        exact = self.df[self.df[DESC_COL] == desc_upper]
        if not exact.empty:
            return exact.iloc[0][REF_COL], 100.0, 'exact'

        # 2. Exact match on cleaned (catches "BEARING 6205" == "RODAMIENTO 6205" after cleaning)
        if desc_clean:
            exact_clean = self.df[self.df[DESC_CLEAN_COL] == desc_clean]
            if not exact_clean.empty:
                return exact_clean.iloc[0][REF_COL], 100.0, 'exact'

        # 3. Fuzzy on cleaned descriptions (most robust)
        best_ref, best_score, best_method = None, 0.0, 'none'

        if desc_clean:
            choices_clean = self.df[DESC_CLEAN_COL].tolist()
            result = process.extractOne(
                desc_clean,
                choices_clean,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=75,
            )
            if result:
                _match, score, idx = result
                best_ref = self.df.iloc[idx][REF_COL]
                best_score = float(score)
                best_method = 'fuzzy'

        # 4. Fuzzy on original (fallback, may catch cases where cleaning removed too much)
        choices_orig = self.df[DESC_COL].tolist()
        result_orig = process.extractOne(
            desc_upper,
            choices_orig,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=75,
        )
        if result_orig and float(result_orig[1]) > best_score:
            _match, score, idx = result_orig
            best_ref = self.df.iloc[idx][REF_COL]
            best_score = float(score)
            best_method = 'fuzzy'

        return best_ref, best_score, best_method

    def get_examples(self, n: int = 40) -> list[dict]:
        """Devuelve n ejemplos del modelo de aprendizaje para el prompt de Claude."""
        sample = self.df.head(n)
        return [
            {"desc": row[DESC_COL], "ref": row[REF_COL]}
            for _, row in sample.iterrows()
        ]
