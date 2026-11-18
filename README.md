# 📖 ETER — Relatos que cobran vida

Aplicación web interactiva construida con **Streamlit** y estética **pixel-art retro** que transforma una idea breve en un **libro ilustrado, narrado y musicalizado en español** utilizando modelos locales de **IA generativa** (texto, imagen, voz y música de fondo).

---

## 📁 Estructura del proyecto

```text
ProyectoIA/
├── .python-version              # Versión de Python fijada (3.10)
├── pyproject.toml               # Configuración del proyecto, índice CUDA y dependencias (uv)
├── requirements.txt             # Dependencias exportadas para instalación con pip
├── uv.lock                      # Lockfile reproducible de dependencias
├── README.md                    # Documentación completa del proyecto
├── Documento_Tecnico_ETER.pdf   # Documento técnico de arquitectura, pruebas y ética
└── src/
    ├── proyectointegradoria/    # Paquete base para build de uv
    │   └── __init__.py
    ├── main.py                  # Aplicación Streamlit, pipelines IA, mezcla audio y PDF/ZIP
    └── salida/                  # Artefactos generados del último libro (JSON, PNG, WAV, PDF, ZIP)
```

---

## 💻 Requisitos de hardware

| Componente | Mínimo (solo CPU) | Recomendado (híbrido CPU + GPU) |
| :--- | :--- | :--- |
| **Procesador (CPU)** | 4 núcleos / 8 hilos | 6+ núcleos / 12+ hilos (p. ej. AMD64 x86_64) |
| **Memoria RAM** | 16 GB | 16 GB o más |
| **Tarjeta gráfica (GPU)** | No requerida (más lento) | NVIDIA con **4 GB+ VRAM** y soporte CUDA (probado en **RTX 2050 4 GB**) |
| **Almacenamiento** | 15 GB libres | 20 GB libres (para entorno virtual + caché de modelos) |
| **Red** | Internet en la primera ejecución | Internet en la primera ejecución (descarga de modelos) |

---

## 🛠️ Instalación paso a paso

### Opción A: Usando `uv` (recomendado)

1. Verifica que tengas **Python 3.10+** y **`uv`** instalados.
2. Clona el repositorio y entra en la carpeta del proyecto:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd ProyectoIA
   ```
3. Sincroniza el entorno virtual `.venv` y todas las dependencias (incluyendo PyTorch con soporte CUDA `cu130`):
   ```bash
   uv sync
   ```

### Opción B: Usando `pip` y `requirements.txt`

1. Crea y activa un entorno virtual con Python 3.10:
   ```bash
   python -m venv .venv
   # En Windows:
   .venv\Scripts\activate
   # En Linux/macOS:
   source .venv/bin/activate
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
   > **Nota sobre GPU NVIDIA:** si `pip` instala PyTorch solo para CPU, reinstala PyTorch con CUDA ejecutando:
   > `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130`

### 💻 ¿Qué hacer si NO cuentas con una GPU NVIDIA CUDA?

Si tu equipo usa solo CPU, GPU integrada o tarjeta AMD/Apple Silicon:

1. **No necesitas cambiar [src/main.py](file:///c:/Users/sebastian/Desktop/ProyectoIA/src/main.py):** el proyecto detecta automáticamente `torch.cuda.is_available() == False` y cambia a modo **CPU (`float32`)**.
2. **Instalación más liviana (opcional, para no descargar ~2 GB de librerías CUDA):**
   - Con **`pip`**: instala primero PyTorch para CPU y luego el resto de dependencias:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     pip install -r requirements.txt
     ```
   - Con **`uv`**: en [pyproject.toml](file:///c:/Users/sebastian/Desktop/ProyectoIA/pyproject.toml), cambia la URL `https://download.pytorch.org/whl/cu130` por `https://download.pytorch.org/whl/cpu` y ejecuta `uv sync`.
3. **Recomendaciones en modo solo CPU:**
   - Se recomiendan **16 GB de RAM** o más.
   - El tiempo por libro completo será mayor (**~15 a 25 minutos**). Puedes desactivar el toggle **`🎵 Incluir música de fondo (IA)`** en el estudio si deseas reducir el tiempo de espera.

---

## 📦 Descarga y ubicación de modelos

No necesitas descargar pesos manualmente ni configurar claves de API. En la **primera generación**, `transformers`, `diffusers` y `kokoro` descargan automáticamente los modelos desde **Hugging Face Hub**:

| Función | Modelo en Hugging Face | Uso aproximado |
| :--- | :--- | :--- |
| **Historia (LLM)** | `Qwen/Qwen3-1.7B` | ~3.4 GB (fp16) |
| **Ilustraciones (Base)** | `Lykon/dreamshaper-8` | ~2.0–4.0 GB |
| **Aceleración visual** | `latent-consistency/lcm-lora-sdv1-5` | ~130 MB |
| **Narración (TTS)** | `hexgrad/Kokoro-82M` (vía `kokoro`) | ~300 MB |
| **Música de fondo (IA)** | `facebook/musicgen-small` | ~1.2 GB |

### Ubicación local en disco
Todos los modelos se almacenan en la caché estándar de Hugging Face:
- **Windows:** `C:\Users\<usuario>\.cache\huggingface\hub\`
- **Linux / macOS:** `~/.cache/huggingface/hub/`

*(Si deseas cambiar la ruta, define la variable de entorno `HF_HOME` antes de iniciar).*

---

## ⚙️ Configuración

Los parámetros principales están centralizados al inicio de [src/main.py](file:///c:/Users/sebastian/Desktop/ProyectoIA/src/main.py):

- **Dispositivo e hilos:** `USE_CUDA = torch.cuda.is_available()`, `DEVICE` (`"cuda"` o `"cpu"`) y `TORCH_THREADS` (hasta 10 hilos de CPU).
- **Historia:** `N_SCENES = 5`, `SCENE_WORDS_MIN = 150`, `SCENE_WORDS_MAX = 190`.
- **Imágenes:** `IMG_SIZE = 512`, `IMG_REFINE_SIZE = 768`, `IMG_UPSCALE = 2` (`1536×1536` final), `IMG_STEPS = 8`, `IMG_GUIDANCE = 1.6`, `IMG_REFINE_STEPS = 8`, `IMG_REFINE_STRENGTH = 0.5`.
- **Voz y música:** `KOKORO_LANG = "e"`, `GENRE_VOICE` (mezcla de voces `em_santa`, `em_alex`, `ef_dora`), `MUSIC_MAX_TOKENS = 750` (~15 s), `MUSIC_BG_GAIN = 0.14` (~-17 dB respecto al RMS de la voz) y `GENRE_MUSIC`.
- **Estilos y géneros:** `GENRES` (6 géneros) y `STYLE_PROMPT` (5 estilos artísticos).

---

## ▶️ Ejecución local

Con `uv`:
```bash
uv run streamlit run src/main.py
```

O con el entorno virtual activado:
```bash
streamlit run src/main.py
```

Abre en tu navegador la dirección local que indique la terminal (habitualmente `http://localhost:8501`).

---

## 🧪 Ejemplo de uso

1. En la pantalla de bienvenida haz clic en **`▶ START`**.
2. En el **Santuario Literario**, configura tu libro:
   - **Idea inicial:** *"Una científica descubre que el sistema digital de su laboratorio replica los recuerdos de su hermano."*
   - **Género:** `Ciencia ficción`
   - **Estilo de ilustración:** `arte conceptual`
   - **Música de fondo (IA):** activado (`🎵 Incluir música de fondo (IA)`).
3. Pulsa **`⚡ GENERAR LIBRO`**. Verás el progreso en vivo con el cronómetro (`⏱️ MM:SS`), las 5 escenas escribiéndose en streaming, los bocetos e ilustraciones pintándose en GPU mientras Kokoro sintetiza las voces en CPU, y la composición de la música de fondo.
4. Al finalizar (aprox. **3–4.5 minutos** con modelos cacheados en RTX 2050 + CPU), abre el libro para leerlo y escucharlo página por página, o descárgalo en **PDF** o **ZIP** (`libro.txt`, `historia.json`, `escena_1..5.png`, `voz_1..5.wav`, `musica_fondo.wav` y `.pdf`).

---

## 📜 Licencia, privacidad y redistribución

- **Código fuente:** distribuido bajo licencia **MIT** (uso, modificación y redistribución libre manteniendo el aviso de copyright).
- **Modelos de IA:** cada modelo conserva su licencia original en Hugging Face (`Qwen3-1.7B`: Apache 2.0; `Kokoro-82M`: Apache 2.0; `lcm-lora-sdv1-5`: CreativeML Open RAIL-M; `dreamshaper-8`: CreativeML Open RAIL-M; `facebook/musicgen-small`: pesos CC-BY-NC 4.0 para uso no comercial).
- **Privacidad y buenas prácticas:** este repositorio **no incluye** claves de API, tokens, datos personales ni pesos redistribuidos de modelos (los pesos se descargan directamente desde Hugging Face Hub al ejecutarse localmente).
