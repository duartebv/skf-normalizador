import re

# Palabras a eliminar completamente (multilingüe)
STOP_WORDS = {
    # Español — tipos de componentes
    'RODAMIENTO', 'RODAMINTO', 'RODAMIENTOS', 'RETEN', 'RETÉN', 'RETENTOR',
    'ANILLO', 'GUARDAPOLVOS', 'CORREA', 'DENTADA', 'JUNTA', 'SELLO',
    'OSCILANTE', 'BOLAS', 'RODILLOS', 'CILÍNDRICO', 'CILINDRICO',
    'CILÍNDRICOS', 'CILINDRICOS',
    'ESFÉRICO', 'ESFERICO', 'ESFÉRICOS', 'ESFERICOS', 'ESFÉRICAS', 'ESFERICAS',
    'AGUJA', 'AXIAL', 'RADIAL', 'CÓNICO', 'CONICO', 'CÓNICOS', 'CONICOS',
    'COJINETE', 'COJINETES',           # faltaba (español estándar)
    'MANGUITO', 'MANGUITOS',           # adaptador de montaje (AH, H series)
    'CASQUILLO', 'CASQUILLOS',         # bushing
    # Español — ruido contextual
    'TIPO', 'REF', 'REFª', 'RLT', 'DE', 'A', 'EN', 'CON', 'Y', 'O',
    'MARCA', 'MODELO', 'CALIDAD',      # etiquetas de ruido
    'REDUCTOR', 'REDUCTORES',          # contexto de maquinaria
    'MOTOR', 'MOTORES',
    'BOMBA', 'BOMBAS',
    'COMPRESOR', 'COMPRESORES',
    'VENTILADOR', 'VENTILADORES',
    'CAJA', 'TRANSMISION', 'TRANSMISIÓN',
    # Inglés
    'BEARING', 'BEARINGS', 'BALL', 'ROLLER', 'TAPERED', 'SPHERICAL',
    'NEEDLE', 'THRUST', 'SEAL', 'SEALS', 'OIL', 'SHAFT', 'RING', 'RINGS',
    'TYPE', 'REF', 'NO', 'NUMBER',
    # Francés
    'ROULEMENT', 'ROULEMENTS', 'JOINT', 'BAGUE', 'ANNEAU',
    # Alemán
    'LAGER', 'WELLENDICHTRING', 'DICHTRING', 'RILLENKUGELLAGER',
    'PENDELROLLENLAGER', 'KEGELROLLENLAGER', 'ZYLINDERROLLENLAGER',
    # Portugués
    'ROLAMENTO', 'ROLAMENTOS', 'RETENTOR', 'ANEL',
    # Italiano
    'CUSCINETTO', 'CUSCINETTI', 'PARAOLIO',
    # Marcas a ignorar
    'SKF', 'FAG', 'SNR', 'INA', 'NSK', 'TIMKEN', 'NTN', 'TORRINGTON',
    'UCSJ', 'SCHAEFFLER', 'KOYO', 'NACHI', 'ZKL', 'RHP', 'URB',
    'GENERIC', 'STANDARD',
}

# Patrones a eliminar
PATTERNS_TO_REMOVE = [
    r'\([^)]*\)',                                          # texto entre paréntesis
    r'"[^"]*"',                                            # texto entre comillas dobles
    r"'[^']*'",                                            # texto entre comillas simples
    r'(?i)\b(ou?\s+fag|ou?\s+similar|ou?\s+equivalente|or\s+similar|or\s+equivalent)\b',
    r'(?i)\b(din[-\s]?\d+|iso[-\s]?\d+|uni[-\s]?\d+)\b',
    r'(?i)(?:\s*[-,;/]\s*|\s+)\b(para|segun|según|norma|ref[ae]?\.|pour|für|per)\b.*$',
    r'(?i)\b(fabricaci[oó]n\s+obligatoria|fabri\.?\s*oblig\.?)\b',
    r'(?i)\b(calidad\s+viton|calidad\s+nbr|calidad\s+similar|quality\s+\w+)\b',
    r'(?i)\b(material:\s*\w+)\b',
    r'(?i)\bN[º¢°\.]\s*',                                 # Nº, N¢, N°, N.
    r'(?i)\bREF[ª\.]?\s*',                                # REF. REFª
    r'(?i)\s*[-–]\s*DIN.*$',                              # trailing DIN
    r'(?i)\b(p/?n|part\s*n[o°]?\.?|part\s*number)\s*:?\s*', # P/N, Part No.
    r'(?i)\b(ref\.?\s*:?\s*)',                             # REF:
    r'(?i)\s*\b(oblig\.?|obligatorio|mandatory)\b',
]


def clean_description(text: str) -> str:
    """Limpia y pre-procesa una descripción de producto para normalización."""
    if not isinstance(text, str):
        return ""

    text = text.upper().strip()

    # Aplicar patrones de limpieza
    for pattern in PATTERNS_TO_REMOVE:
        text = re.sub(pattern, ' ', text)

    # Eliminar stop words por token
    words = text.split()
    cleaned = [w for w in words if w not in STOP_WORDS and len(w) > 0 and w not in {'.', ',', ';', ':', '-', '–'}]
    text = ' '.join(cleaned)

    # Colapsar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()

    # Eliminar puntuación suelta al final (residuos tras eliminar paréntesis/patrones)
    text = re.sub(r'[,;:\.]+$', '', text).strip()

    # Juntar dígitos separados por espacios que forman una referencia
    # Ej: "22 205" → "22205", "32 30 9" → "32309", "12 09" → "1209"
    text = join_split_digits(text)

    return text


def join_split_digits(text: str) -> str:
    """
    Une dígitos separados por espacio que forman códigos de referencia SKF.
    Se aplica iterativamente hasta que no haya más cambios.
    Patrones cubiertos:
      "22 205"  → "22205"   (serie 222xx: 2+3 dígitos)
      "32 30 9" → "32309"   (serie 323xx: 2+2+1)
      "12 09"   → "1209"    (serie 12xx: 2+2)
      "160 02"  → "16002"   (serie 160xx: 3+2)
    """
    # Patrón: número de 1-3 dígitos seguido de espacio y otro número de 1-3 dígitos,
    # donde la concatenación da 4-6 dígitos (longitud típica de referencia SKF)
    pattern = re.compile(r'\b(\d{1,3})\s+(\d{1,3})\b')

    prev = None
    while prev != text:
        prev = text
        def _join(m):
            combined = m.group(1) + m.group(2)
            # Solo unir si el resultado tiene entre 4 y 6 dígitos (referencia plausible)
            if 4 <= len(combined) <= 6:
                return combined
            return m.group(0)  # dejar sin cambios
        text = pattern.sub(_join, text)

    return text


def normalize_ref_candidate(text: str) -> str:
    """
    Normaliza un candidato a referencia SKF a la convención del catálogo.
    Se aplica DESPUÉS de clean_description, cuando ya solo queda la referencia en bruto.

    Ejemplos:
      "6000 C3"       → "6000/C3"
      "6205 2RS C3"   → "6205-2RSH/C3"
      "6205 ZZ"       → "6205-2Z"
      "6205 2Z C3"    → "6205-2Z/C3"
      "22207 E C3"    → "22207 E/C3"
    """
    if not text:
        return text

    t = text.upper().strip()

    # 0. Normalizar notación FAG → SKF: E1 → E (jaula reforzada)
    t = re.sub(r'\bE1\b', 'E', t)

    # 1. Normalizar variantes de 2RS → -2RSH (antes de C3 para no confundir)
    t = re.sub(r'[\s\-]+(2RS1?H?|2RSH|2RSL)\b', r'-2RSH', t)
    t = re.sub(r'[\s\-]+(2RS)\b', r'-2RSH', t)

    # 2. Normalizar ZZ / 2ZR → -2Z
    t = re.sub(r'[\s\-]+(ZZ|2ZR?)\b', r'-2Z', t)

    # 3. Normalizar C3/C4/C5 suelto → /C3
    #    Soporta: "6000 C3", "6000-C3", "6000C3" (pegado al número)
    t = re.sub(r'[\s\-/]+(C[3-5])\b', r'/\1', t)
    t = re.sub(r'(\d)(C[3-5])\b', r'\1/\2', t)  # "6000C3" pegado

    # 4. Eliminar espacios sobrantes alrededor de /
    t = re.sub(r'\s*/\s*', '/', t)

    return t.strip()
