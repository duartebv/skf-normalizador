import re
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un experto en rodamientos SKF. Tu tarea es normalizar descripciones libres de productos a referencias de catálogo SKF.

Las descripciones pueden venir en cualquier idioma: español, inglés, francés, alemán, portugués, italiano u otros.
Palabras equivalentes a ignorar según idioma:
- ES: RODAMIENTO, RETEN, ANILLO, GUARDAPOLVOS
- EN: BEARING, SEAL, RING, OIL SEAL, BALL BEARING, ROLLER BEARING
- FR: ROULEMENT, JOINT, BAGUE
- DE: LAGER, WELLENDICHTRING, DICHTRING
- PT: ROLAMENTO, RETENTOR
- IT: CUSCINETTO, PARAOLIO

REGLAS GENERALES:
1. Devuelve ÚNICAMENTE la referencia SKF normalizada, sin texto adicional
2. Si no puedes determinar la referencia con seguridad, devuelve "UNKNOWN"
3. Nunca inventes referencias que no sean estándar SKF
4. Si la descripción ya ES una referencia SKF válida (sin prefijos de tipo), devuélvela normalizada directamente

REGLAS DE NORMALIZACIÓN:

RODAMIENTOS:
- Elimina palabras clave en cualquier idioma (ver lista arriba)
- Elimina marcas de otros fabricantes: FAG, SNR, INA, NSK, TIMKEN, NTN, KOYO, NACHI
- Une dígitos separados por espacios que forman una referencia: "22 207" → "22207"
- Sufijos de tipo: CC/CCK/C/CK/K → E/EK según corresponda a la serie
- C3, C4 → /C3, /C4 (holgura radial)
- W33 → ranura de lubricación, conservar si aparece en catálogo
- 2RS, 2RSH, ZZ, 2Z → protecciones, conservar normalizadas
- ETN9, EKTN9 → jaula de poliamida, conservar
- J2, J2/Q → rodillos cónicos series 300/320/330, conservar

RETENES (oil seals):
- Formato de salida: DxdxA TIPO MATERIAL
  Ejemplos: 20X40X7 HMSA10 RG | 120X150X12 HMSA10 V
- Separadores dimensiones: "-", " x ", "x", " X " → "X" mayúscula
- Material NBR (por defecto): HMSA10 RG
- Material Viton (si menciona VITON/V/FPM/FKM): HMSA10 V
- Labio simple: HMS5 RG cuando la descripción lo indique

V-RINGS:
- Formato: DDD VA/VS R
  VA = tipo A (guardapolvo), VS = tipo S
  Ejemplo: 100 VS R"""


class ClaudeNormalizer:
    """Normalizador SKF usando Google Gemini (misma interfaz que el cliente anterior)."""

    def __init__(self, api_key: str, examples: list[dict]):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.examples = examples[:40]
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _examples_text(self) -> str:
        return "\n".join(f'  "{e["desc"]}" → "{e["ref"]}"' for e in self.examples)

    def normalize_single(self, description: str) -> str:
        """Normaliza una sola descripción. Devuelve la referencia o 'UNKNOWN'."""
        prompt = f"""{SYSTEM_PROMPT}

EJEMPLOS DE ENTRENAMIENTO:
{self._examples_text()}

DESCRIPCIÓN A NORMALIZAR:
{description}"""
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
        progress_callback=None,
    ) -> list[str]:
        """Normaliza una lista de descripciones en batches de 10."""
        results: list[str] = []
        total = len(descriptions)
        for i in range(0, total, 10):
            chunk = descriptions[i : i + 10]
            results.extend(self._normalize_chunk(chunk))
            if progress_callback:
                progress_callback(min(i + 10, total), total)
        return results

    def _normalize_chunk(self, descriptions: list[str]) -> list[str]:
        """Procesa hasta 10 descripciones en una sola llamada."""
        numbered = "\n".join(f"{j + 1}. {d}" for j, d in enumerate(descriptions))
        prompt = f"""{SYSTEM_PROMPT}

EJEMPLOS DE ENTRENAMIENTO:
{self._examples_text()}

INSTRUCCIÓN BATCH: Para cada descripción numerada devuelve en una línea separada el número, un punto y la referencia normalizada. Ejemplo: "1. 6205 2RS1"

DESCRIPCIONES A NORMALIZAR:
{numbered}"""
        try:
            response = self.model.generate_content(prompt)
            if response.usage_metadata:
                self.total_input_tokens += response.usage_metadata.prompt_token_count or 0
                self.total_output_tokens += response.usage_metadata.candidates_token_count or 0
            text = response.text.strip()
            parsed = ["UNKNOWN"] * len(descriptions)
            for line in text.splitlines():
                m = re.match(r"^(\d+)\.\s*(.+)$", line.strip())
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(descriptions):
                        parsed[idx] = m.group(2).strip()
            return parsed
        except Exception as e:
            logger.error(f"Gemini API error (batch): {e}")
            return ["UNKNOWN"] * len(descriptions)

    def real_cost_eur(self) -> float:
        """Gemini es gratuito en el free tier."""
        return 0.0
