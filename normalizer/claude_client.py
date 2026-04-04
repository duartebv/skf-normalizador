import re
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt del sistema
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Eres un experto en referencias SKF. Tu tarea: extraer y normalizar la referencia SKF de catálogo a partir de una descripción libre.

## FORMATO DEL CATÁLOGO SKF
- Guión para protecciones: `6205-2RSH`, `6205-2Z`, `6205-RSH`, `6205-Z`
- Barra para holgura/ranura: `6205-2RSH/C3`, `22322 E/C3`, `24026 CC/W33`, `22322 EK/W33/C3`
- Espacio entre base y sufijo de serie: `22322 E`, `22322 EK`, `NU 206 ECP`
- Manguitos y casquillos: `AH 3240`, `H 3140`, `AN 20` (espacio entre prefijo y número)
- Plummer blocks: `SY 40 TF`, `SNL 510`

## CONVERSIONES OBLIGATORIAS
| Entrada | Salida |
|---------|--------|
| E1 (notación FAG) | E |
| ZZ, 2ZR | -2Z |
| 2RS, 2RS1, 2RSH, 2RSL | -2RSH |
| RS, RS1, RSL | -RSH |
| RZ | -RZ |
| C3 / C4 / C5 separado | /C3 / /C4 / /C5 |
| W33 separado | /W33 |

## PALABRAS A IGNORAR (en cualquier idioma)
- Tipo: RODAMIENTO, BEARING, ROULEMENT, LAGER, ROLAMENTO, CUSCINETTO, COJINETE, MANGUITO
- Marcas: FAG, INA, NSK, TIMKEN, NTN, KOYO, NACHI, SNR, TORRINGTON
- Contexto: PARA REDUCTOR/MOTOR/BOMBA, REF., No., P/N, DIN, ISO, marca, modelo
- Calidades: VITON, NBR, similar, equivalente

## REGLAS
1. Devuelve ÚNICAMENTE la referencia SKF normalizada, sin texto adicional.
2. Si hay un "Candidato extraído" proporcionado, verifícalo y corrígelo si es necesario.
3. Si no puedes determinar la referencia con seguridad → devuelve exactamente: UNKNOWN
4. No inventes referencias que no sigan el estándar SKF."""


class ClaudeNormalizer:
    """Normalizador SKF usando Google Gemini (free tier)."""

    def __init__(self, api_key: str, examples: list[dict]):
        genai.configure(api_key=api_key)
        self._gen_config = genai.types.GenerationConfig(
            temperature=0.0,          # determinista: misma entrada → misma salida
            max_output_tokens=64,     # referencias son cortas; ahorra quota
        )
        self.model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=self._gen_config,
        )
        self.examples = examples[:40]
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _examples_text(self) -> str:
        return "\n".join(f'  "{e["desc"]}" → "{e["ref"]}"' for e in self.examples)

    def normalize_single(
        self,
        description: str,
        cleaned: str = "",
        candidate: str = "",
    ) -> str:
        """
        Normaliza una descripción.
        - cleaned: versión ya limpiada por clean_description()
        - candidate: mejor candidato extraído por normalize_ref_candidate()
        Devuelve la referencia o 'UNKNOWN'.
        """
        context = f"Descripción original: {description}"
        if cleaned and cleaned.upper() != description.upper():
            context += f"\nTexto limpiado (sin ruido): {cleaned}"
        if candidate and candidate not in ("", "UNKNOWN"):
            context += f"\nCandidato extraído (verificar/corregir formato): {candidate}"

        prompt = f"""{SYSTEM_PROMPT}

EJEMPLOS DEL MODELO:
{self._examples_text()}

---
{context}

Referencia SKF:"""
        try:
            response = self.model.generate_content(prompt)
            if response.usage_metadata:
                self.total_input_tokens += response.usage_metadata.prompt_token_count or 0
                self.total_output_tokens += response.usage_metadata.candidates_token_count or 0
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error (single): {e}")
            return "UNKNOWN"

    def normalize_batch(
        self,
        descriptions: list[str],
        cleaned_list: list[str] | None = None,
        candidate_list: list[str] | None = None,
        progress_callback=None,
    ) -> list[str]:
        """
        Normaliza una lista de descripciones en batches de 10.
        - cleaned_list: versiones limpias correspondientes (misma longitud)
        - candidate_list: candidatos extraídos (misma longitud)
        """
        results: list[str] = []
        total = len(descriptions)
        for i in range(0, total, 10):
            chunk_descs = descriptions[i: i + 10]
            chunk_clean = (cleaned_list[i: i + 10] if cleaned_list else None)
            chunk_cands = (candidate_list[i: i + 10] if candidate_list else None)
            results.extend(self._normalize_chunk(chunk_descs, chunk_clean, chunk_cands))
            if progress_callback:
                progress_callback(min(i + 10, total), total)
        return results

    def _normalize_chunk(
        self,
        descriptions: list[str],
        cleaned_list: list[str] | None = None,
        candidate_list: list[str] | None = None,
    ) -> list[str]:
        """Procesa hasta 10 descripciones en una sola llamada."""
        lines = []
        for j, desc in enumerate(descriptions):
            line = f"{j + 1}. {desc}"
            clean = (cleaned_list[j] if cleaned_list and j < len(cleaned_list) else "") or ""
            cand = (candidate_list[j] if candidate_list and j < len(candidate_list) else "") or ""
            if clean and clean.upper() != desc.upper():
                line += f"  [limpiado: {clean}]"
            if cand and cand not in ("", "UNKNOWN"):
                line += f"  [candidato: {cand}]"
            lines.append(line)

        numbered = "\n".join(lines)
        prompt = f"""{SYSTEM_PROMPT}

EJEMPLOS DEL MODELO:
{self._examples_text()}

INSTRUCCIÓN BATCH: Para cada línea numerada devuelve exactamente una línea con el número, punto y referencia SKF normalizada.
Formato esperado: "1. 6205-2RSH/C3"
Si no puedes determinar una referencia → "N. UNKNOWN"

DESCRIPCIONES:
{numbered}"""
        try:
            response = self.model.generate_content(prompt)
            if response.usage_metadata:
                self.total_input_tokens += response.usage_metadata.prompt_token_count or 0
                self.total_output_tokens += response.usage_metadata.candidates_token_count or 0
            text = response.text.strip()
            parsed = ["UNKNOWN"] * len(descriptions)
            for line in text.splitlines():
                line = line.strip()
                # Acepta: "1. REF", "1) REF", "1 REF"
                m = re.match(r'^(\d+)[.)]\s*(.+)$', line)
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(descriptions):
                        parsed[idx] = m.group(2).strip()
            return parsed
        except Exception as e:
            logger.error(f"Gemini API error (batch): {e}")
            return ["UNKNOWN"] * len(descriptions)

    def real_cost_eur(self) -> float:
        """Gemini free tier: sin coste."""
        return 0.0
