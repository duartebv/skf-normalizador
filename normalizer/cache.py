import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz, process
import logging

logger = logging.getLogger(__name__)

DESC_COL = "Description Cliente"
REF_COL = "REF. SKF"
DESC_CLEAN_COL = "_DESC_CLEAN"
DESC_NORM_COL = "_DESC_NORM"   # clean + normalize_ref_candidate aplicado


class CacheNormalizer:
    def __init__(self, model_path: Path):
        self.df: pd.DataFrame = pd.DataFrame()
        self._load(model_path)

    def _load(self, path: Path) -> None:
        from normalizer.rules import clean_description, normalize_ref_candidate
        try:
            df = pd.read_excel(path)
            df.columns = [c.strip() for c in df.columns]
            df[DESC_COL] = df[DESC_COL].astype(str).str.upper().str.strip()
            df[REF_COL] = df[REF_COL].astype(str).str.strip()
            df = df[df[REF_COL].notna() & (df[REF_COL] != '') & (df[REF_COL] != 'nan')]
            # Pre-computar versiones limpias y normalizadas para matching más preciso
            df[DESC_CLEAN_COL] = df[DESC_COL].apply(clean_description)
            df[DESC_NORM_COL] = df[DESC_CLEAN_COL].apply(normalize_ref_candidate)
            self.df = df.reset_index(drop=True)
            logger.info(f"Cache cargado: {len(self.df)} entradas desde {path.name}")
        except Exception as e:
            logger.error(f"Error cargando cache: {e}")
            raise

    def lookup(self, description: str, cleaned: str | None = None) -> tuple[str | None, float, str]:
        """
        Busca en el modelo de aprendizaje sobre descripción original, limpiada y normalizada.
        Returns: (referencia, score, method)  method: 'exact' | 'fuzzy' | 'none'
        """
        if self.df.empty:
            return None, 0.0, 'none'

        from normalizer.rules import clean_description, normalize_ref_candidate
        desc_upper = description.upper().strip()
        desc_clean = cleaned if cleaned is not None else clean_description(desc_upper)
        desc_norm = normalize_ref_candidate(desc_clean) if desc_clean else ""

        # 1. Exact match on original description
        exact = self.df[self.df[DESC_COL] == desc_upper]
        if not exact.empty:
            return exact.iloc[0][REF_COL], 100.0, 'exact'

        # 2. Exact match on cleaned description
        if desc_clean:
            exact_clean = self.df[self.df[DESC_CLEAN_COL] == desc_clean]
            if not exact_clean.empty:
                return exact_clean.iloc[0][REF_COL], 100.0, 'exact'

        # 3. Exact match on normalized form — capta "22322 E C3" == "22322 E/C3"
        if desc_norm:
            exact_norm = self.df[self.df[DESC_NORM_COL] == desc_norm]
            if not exact_norm.empty:
                return exact_norm.iloc[0][REF_COL], 100.0, 'exact'

            # También comparar normalizado contra REF_COL directamente
            exact_ref = self.df[self.df[REF_COL].str.upper() == desc_norm.upper()]
            if not exact_ref.empty:
                return exact_ref.iloc[0][REF_COL], 100.0, 'exact'

        best_ref, best_score, best_method = None, 0.0, 'none'

        # 4. Fuzzy en columna normalizada (más preciso: "22322 E/C3" vs "22322 E/C3")
        if desc_norm:
            choices_norm = self.df[DESC_NORM_COL].tolist()
            result_norm = process.extractOne(
                desc_norm,
                choices_norm,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=78,
            )
            if result_norm:
                _match, score, idx = result_norm
                best_ref = self.df.iloc[idx][REF_COL]
                best_score = float(score)
                best_method = 'fuzzy'

        # 5. Fuzzy en columna limpiada
        if desc_clean:
            choices_clean = self.df[DESC_CLEAN_COL].tolist()
            result_clean = process.extractOne(
                desc_clean,
                choices_clean,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=75,
            )
            if result_clean and float(result_clean[1]) > best_score:
                _match, score, idx = result_clean
                best_ref = self.df.iloc[idx][REF_COL]
                best_score = float(score)
                best_method = 'fuzzy'

        # 6. Fuzzy en descripción original (fallback; capta casos donde cleaning eliminó demasiado)
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
        """Devuelve n ejemplos del modelo de aprendizaje para el prompt de Gemini."""
        sample = self.df.head(n)
        return [
            {"desc": row[DESC_COL], "ref": row[REF_COL]}
            for _, row in sample.iterrows()
        ]
