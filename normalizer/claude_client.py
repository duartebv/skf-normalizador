import re
import time
import logging
from typing import Optional, List
import google.generativeai as genai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts — diseñados para ser CORTOS y minimizar consumo de tokens gratuitos
# ---------------------------------------------------------------------------

# Modo verificación: cuando ya tenemos un candidato extraído por reglas.
# Solo pedimos a Gemini que verifique/corrija el formato. ~120 tokens.
_PROMPT_VERIFY = """Eres un normalizador de referencias SKF. Verifica el candidato y devuelve la referencia SKF correcta en formato de catálogo.

Formato SKF: guión para protecciones (6205-2RSH, 6205-2Z, 6205-Z), barra para holgura/ranura (6205-2RSH/C3, 22322 E/C3, 24026 CC/W33).
Conversiones: E1→E | EM1→ECM | ZZ→-2Z | 2RS/2RS1→-2RSH | RS/RS1→-RSH | C3 suelto→/C3 | W33 suelto→/W33

Responde SOLO con la referencia o UNKNOWN."""

# Modo extracción: cuando no hay candidato claro.
# Incluye reglas completas + pocos ejemplos clave. ~250 tokens base + ejemplos.
_PROMPT_EXTRACT = """Eres un normalizador SKF. Extrae y normaliza la referencia SKF de catálogo.

Ignora: tipo (RODAMIENTO, BEARING, COJINETE...), marca (FAG, INA, NSK, TIMKEN...), contexto (PARA REDUCTOR/MOTOR...).
Convierte: E1→E | EM1→ECM | ZZ→-2Z | 2RS→-2RSH | RS→-RSH | C3 suelto→/C3 | W33 suelto→/W33
Formato: 6205-2RSH/C3 | 22322 E/C3 | 24026 CC/W33 | AH 3240 | NU 206 ECP | 7318 BECBM

Ejemplos:
{examples}

Responde SOLO con la referencia o UNKNOWN."""

# Ejemplos fijos y representativos (series más comunes) — ~150 tokens
_FIXED_EXAMPLES = [
    ("RODAMIENTO FAG 6205 2RS C3", "6205-2RSH/C3"),
    ("BEARING 22322 E1/C3", "22322 E/C3"),
    ("RODILLO ESFÉRICO 24026 CC W33", "24026 CC/W33"),
    ("MANGUITO AH 3240", "AH 3240"),
    ("CYLINDRICAL ROLLER NU 206 ECP", "NU 206 ECP"),
    ("NSK 32314", "32314"),
    ("6310 ZZ C3", "6310-2Z/C3"),
    ("7318 BECBM", "7318 BECBM"),
]


def _fmt_examples(extras: List[dict]) -> str:
    """Combina ejemplos fijos con algunos del modelo de aprendizaje."""
    lines = [f'  "{d}" → "{r}"' for d, r in _FIXED_EXAMPLES]
    # Añadir hasta 7 ejemplos del modelo (los más cortos = más eficientes en tokens)
    model_sorted = sorted(extras, key=lambda x: len(x["desc"]))[:7]
    for e in model_sorted:
        lines.append(f'  "{e["desc"]}" → "{e["ref"]}"')
    return "\n".join(lines)


def _clean_response(text: str) -> str:
    """Extrae solo la referencia de la respuesta de Gemini, ignorando texto explicativo."""
    text = text.strip()
    if not text:
        return "UNKNOWN"
    # Quitar markdown (negritas, código)
    text = re.sub(r'[`*_]+', '', text)
    # Primera línea solo
    first_line = text.split('\n')[0].strip()
    # Quitar prefijos comunes que Gemini añade
    first_line = re.sub(
        r'^(?:referencia\s+(?:skf\s*)?:?\s*|la\s+referencia\s+(?:es\s*|skf\s*)?:?\s*'
        r'|ref\.?\s*:?\s*|skf\s*:?\s*|resultado\s*:?\s*)',
        '', first_line, flags=re.IGNORECASE,
    )
    # Eliminar puntuación final
    first_line = first_line.rstrip('.,;:')
    return first_line.strip() or "UNKNOWN"


class ClaudeNormalizer:
    """Normalizador SKF usando Google Gemini (free tier)."""

    def __init__(self, api_key: str, examples: List[dict]):
        genai.configure(api_key=api_key)
        self._gen_config = genai.types.GenerationConfig(
            temperature=0.0,        # determinista
            max_output_tokens=32,   # referencias son cortas; ahorra quota
        )
        self.model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=self._gen_config,
        )
        self.examples = examples
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _call_with_retry(self, prompt: str, retries: int = 2) -> str:
        """Llama a Gemini con reintentos ante errores de quota (429)."""
        for attempt in range(retries + 1):
            try:
                response = self.model.generate_content(prompt)
                if response.usage_metadata:
                    self.total_input_tokens += response.usage_metadata.prompt_token_count or 0
                    self.total_output_tokens += response.usage_metadata.candidates_token_count or 0
                return response.text.strip()
            except Exception as e:
                msg = str(e)
                is_quota = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()
                if is_quota and attempt < retries:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"Gemini quota hit, reintentando en {wait}s (intento {attempt+1}/{retries})")
                    time.sleep(wait)
                else:
                    logger.error(f"Gemini API error: {e}")
                    return "UNKNOWN"
        return "UNKNOWN"

    def normalize_single(
        self,
        description: str,
        cleaned: str = "",
        candidate: str = "",
    ) -> str:
        """
        Normaliza una descripción individual.
        Si hay candidato → usa prompt corto de verificación.
        Si no hay candidato → usa prompt de extracción con ejemplos.
        """
        has_candidate = bool(candidate and candidate not in ("", "UNKNOWN"))

        if has_candidate:
            # Modo verificación: prompt mínimo (~120 tokens entrada)
            prompt = (
                f"{_PROMPT_VERIFY}\n\n"
                f"Descripción: {description}\n"
                f"Candidato: {candidate}\n\n"
                f"Referencia SKF:"
            )
        else:
            # Modo extracción: prompt con ejemplos (~400 tokens entrada)
            examples_text = _fmt_examples(self.examples)
            prompt = (
                f"{_PROMPT_EXTRACT.format(examples=examples_text)}\n\n"
                f"Descripción: {description}"
            )
            if cleaned and cleaned.upper() != description.upper():
                prompt += f"\nTexto limpiado: {cleaned}"
            prompt += "\n\nReferencia SKF:"

        raw = self._call_with_retry(prompt)
        return _clean_response(raw)

    def normalize_batch(
        self,
        descriptions: List[str],
        cleaned_list: Optional[List[str]] = None,
        candidate_list: Optional[List[str]] = None,
        progress_callback=None,
    ) -> List[str]:
        """
        Normaliza una lista de descripciones en batches de 10.
        Agrupa ítems con/sin candidato para optimizar tokens usados.
        """
        results: List[str] = ["UNKNOWN"] * len(descriptions)
        total = len(descriptions)

        # Separar ítems con candidato (verify) vs sin candidato (extract)
        verify_indices = []
        extract_indices = []
        for i, desc in enumerate(descriptions):
            cand = (candidate_list[i] if candidate_list and i < len(candidate_list) else "") or ""
            if cand and cand not in ("", "UNKNOWN"):
                verify_indices.append(i)
            else:
                extract_indices.append(i)

        # Procesar batch de verificación (prompts muy cortos)
        for start in range(0, len(verify_indices), 10):
            chunk_idx = verify_indices[start: start + 10]
            chunk_descs = [descriptions[i] for i in chunk_idx]
            chunk_cands = [(candidate_list[i] if candidate_list else "") or "" for i in chunk_idx]
            chunk_results = self._batch_verify(chunk_descs, chunk_cands)
            for i, ref in zip(chunk_idx, chunk_results):
                results[i] = ref
            if progress_callback:
                progress_callback(min(start + 10, total), total)

        # Procesar batch de extracción (prompts con ejemplos)
        examples_text = _fmt_examples(self.examples)
        for start in range(0, len(extract_indices), 10):
            chunk_idx = extract_indices[start: start + 10]
            chunk_descs = [descriptions[i] for i in chunk_idx]
            chunk_clean = [(cleaned_list[i] if cleaned_list else "") or "" for i in chunk_idx]
            chunk_results = self._batch_extract(chunk_descs, chunk_clean, examples_text)
            for i, ref in zip(chunk_idx, chunk_results):
                results[i] = ref
            if progress_callback:
                progress_callback(min(start + 10, total), total)

        return results

    def _batch_verify(self, descriptions: List[str], candidates: List[str]) -> List[str]:
        """Batch para ítems con candidato ya extraído — prompt ultra-corto."""
        lines = []
        for j, (desc, cand) in enumerate(zip(descriptions, candidates)):
            lines.append(f"{j+1}. {desc}  [candidato: {cand}]")

        prompt = (
            f"{_PROMPT_VERIFY}\n\n"
            f"Para cada ítem devuelve exactamente: 'N. REFERENCIA' o 'N. UNKNOWN'\n\n"
            + "\n".join(lines)
        )
        raw = self._call_with_retry(prompt)
        return self._parse_batch_response(raw, len(descriptions))

    def _batch_extract(
        self,
        descriptions: List[str],
        cleaned_list: List[str],
        examples_text: str,
    ) -> List[str]:
        """Batch para ítems sin candidato — incluye ejemplos."""
        lines = []
        for j, (desc, clean) in enumerate(zip(descriptions, cleaned_list)):
            line = f"{j+1}. {desc}"
            if clean and clean.upper() != desc.upper():
                line += f"  [limpiado: {clean}]"
            lines.append(line)

        prompt = (
            f"{_PROMPT_EXTRACT.format(examples=examples_text)}\n\n"
            f"Para cada ítem devuelve exactamente: 'N. REFERENCIA' o 'N. UNKNOWN'\n\n"
            + "\n".join(lines)
        )
        raw = self._call_with_retry(prompt)
        return self._parse_batch_response(raw, len(descriptions))

    def _parse_batch_response(self, text: str, n: int) -> List[str]:
        """Parsea respuesta batch de Gemini. Acepta '1.' '1)' y variantes."""
        parsed = ["UNKNOWN"] * n
        for line in text.splitlines():
            line = line.strip()
            m = re.match(r'^(\d+)[.)]\s*(.+)$', line)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < n:
                    parsed[idx] = _clean_response(m.group(2))
        return parsed

    def real_cost_eur(self) -> float:
        """Gemini free tier: sin coste."""
        return 0.0
