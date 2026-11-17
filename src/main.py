import base64
import gc
import html
import io
import time
import os
import json
import re
import shutil
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import soundfile as sf

# ==================================
# CONFIGURACION
# ==================================
st.set_page_config(
    page_title="ETER | Relatos que cobran vida",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "salida"
DEVICE = "cpu"

LLM_ID = "Qwen/Qwen3-1.7B"
IMG_ID = "Lykon/dreamshaper-8"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

KOKORO_LANG = "e"
KOKORO_VOICE = "ef_dora"
KOKORO_SPEED = 0.88
SENTENCE_PAUSE = 0.35
SCENE_END_PAUSE = 0.60

TORCH_THREADS = max(1, min(8, os.cpu_count() or 2))
torch.set_num_threads(TORCH_THREADS)

N_SCENES = 5
IMG_SIZE = 512          
IMG_REFINE_SIZE = 768   
IMG_UPSCALE = 2         

# <-- CAMBIO A: con LCMScheduler bastan 4-8 pasos y guidance ~1.0.
#     Con guidance <= 1.0 no se usa CFG (la mitad de cómputo por paso).
IMG_STEPS = 6
IMG_GUIDANCE = 1.0
# <-- CAMBIO B: pase 2 reparado (img2img). Se ejecutan steps * strength pasos (8 * 0.5 = 4)
IMG_REFINE_STEPS = 8
IMG_REFINE_STRENGTH = 0.5
# Si el LoRA LCM no se pudiera cargar, se usa el scheduler normal con estos valores
IMG_STEPS_FALLBACK = 20
IMG_GUIDANCE_FALLBACK = 7.0
IMG_SEED = 1234
IMG_PROMPT_MAX_TOKENS = 77   # límite del codificador de texto CLIP

# <-- CAMBIO: escenas más largas (~10-13 líneas en el libro)
SCENE_WORDS_MIN = 150
SCENE_WORDS_MAX = 190
LLM_MAX_TOKENS = 5000   # ya no se usa de golpe: la prosa se genera escena por escena

NEGATIVE_PROMPT = (
    "ugly, deformed, noisy, blurry, distorted, low quality, "
    "bad anatomy, bad proportions, duplicate, watermark, "
    "signature, text, cropped, extra limbs, missing limbs, 3d render, plastic"
)

# <-- CAMBIO C: la interfaz sigue en ESPAÑOL (claves), pero el motor de imágenes
#     (CLIP) entiende mucho mejor el inglés, así que cada estilo se traduce internamente.
STYLE_EN = {
    "pintura digital": "digital painting",
    "acuarela": "watercolor painting",
    "arte conceptual": "concept art",
    "ilustración detallada": "detailed illustration",
    "estilo óleo clásico": "classic oil painting",
}

GENRES = {
    "Fantasía": {
        "visual": "epic fantasy storybook illustration, rich oil painting, warm golden light, masterpiece",
        "cover": "fantasy",
        "symbol": "⚔️",
    },
    "Terror": {
        "visual": "gothic horror illustration, dark moody painting, deep shadows, eerie fog, cinematic",
        "cover": "horror",
        "symbol": "💀",
    },
    "Misterio": {
        "visual": "noir mystery illustration, foggy night, dramatic cinematic lighting, detailed painting",
        "cover": "mystery",
        "symbol": "🔍",
    },
    "Aventura": {
        "visual": "adventure storybook illustration, vivid rich colors, dynamic composition, painted style",
        "cover": "adventure",
        "symbol": "🗺️",
    },
    "Ciencia ficción": {
        "visual": "retro science fiction illustration, steampunk aesthetic, classic pulp sci-fi art",
        "cover": "scifi",
        "symbol": "👾",
    },
    "Infantil": {
        "visual": "children's vintage picture book illustration, soft watercolor, cute, bright pastel colors",
        "cover": "children",
        "symbol": "⭐",
    },
}


# ============================================================
# ESTILOS RETRO / PIXEL ART 8-BIT & 16-BIT
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

/* GLOBAL PIXEL STYLE */
* {
    image-rendering: pixelated;
}

.stApp {
    background-color: #0d0e15;
    background-image: 
        radial-gradient(#1a1c2b 15%, transparent 16%),
        radial-gradient(#1a1c2b 15%, transparent 16%);
    background-size: 16px 16px;
    background-position: 0 0, 8px 8px;
    color: #e2e8f0;
    font-family: 'VT323', monospace !important;
    font-size: 20px;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* ============================================================
   BARRA SUPERIOR PIXEL ART
============================================================ */
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 24px;
    margin-bottom: 24px;
    background: #181425;
    border: 4px solid #262b44;
    box-shadow: 
        -4px 0 0 0 #000,
         4px 0 0 0 #000,
         0 -4px 0 0 #000,
         0  4px 0 0 #000,
         6px 6px 0px 0px #000;
}

.brand {
    font-family: 'Press Start 2P', monospace;
    font-size: 20px;
    color: #ffcd75;
    text-shadow: 3px 3px 0px #8b4049;
    letter-spacing: 2px;
}

.tag {
    font-family: 'Press Start 2P', monospace;
    font-size: 9px;
    color: #a5e06b;
    text-transform: uppercase;
}

/* ===== ANIMACIÓN PIXEL EN LA TOPBAR ===== */
.pixel-sky {
    flex: 1;
    display: flex;
    justify-content: center;
    margin: -10px 16px;
}
.pixel-sky svg { width: 320px; height: 64px; } /* escala x2 = pixeles enteros */
@media (max-width: 760px) { .pixel-sky { display: none; } }

.p-bird   { animation: p-fly 10s linear infinite; }
.p-up     { animation: p-flap-a .5s steps(1) infinite; }
.p-down   { animation: p-flap-b .5s steps(1) infinite; }
.p-bolt   { opacity: 0; animation: p-bolt 10s steps(1) infinite; }
.p-flash  { opacity: 0; animation: p-flash 10s steps(1) infinite; }
.p-smoke  { opacity: 0; animation: p-smoke 10s ease-out infinite; }
.p-talis  { opacity: 0; animation: p-talis 10s ease-in-out infinite; }
.p-sword  { opacity: 0; animation: p-sword 10s ease-out infinite; }
.p-spark  { animation: p-blink .6s steps(1) infinite; }

@keyframes p-fly {
    0%   { transform: translateX(-14px); opacity: 1; }
    40%  { transform: translateX(66px);  opacity: 1; }
    45%  { transform: translateX(66px);  opacity: 1; }
    46%  { transform: translateX(66px);  opacity: 0; }
    100% { transform: translateX(66px);  opacity: 0; }
}
@keyframes p-flap-a { 0% { opacity: 1; } 50% { opacity: 0; } }
@keyframes p-flap-b { 0% { opacity: 0; } 50% { opacity: 1; } }
@keyframes p-bolt {
    0%   { opacity: 0; }
    42%  { opacity: 1; }
    43.5%{ opacity: 0; }
    44.5%{ opacity: 1; }
    46.5%{ opacity: 0; }
}
@keyframes p-flash {
    0%   { opacity: 0; }
    42%  { opacity: .35; }
    43.5%{ opacity: 0; }
    44.5%{ opacity: .35; }
    46.5%{ opacity: 0; }
}
@keyframes p-smoke {
    0%, 45% { opacity: 0; transform: translateY(4px); }
    47%     { opacity: .9; transform: translateY(2px); }
    64%     { opacity: 0;  transform: translateY(-12px); }
    100%    { opacity: 0; }
}
@keyframes p-talis {
    0%, 47% { opacity: 0; transform: translateY(6px); }
    52%     { opacity: 1; transform: translateY(0); }
    58%     { transform: translateY(-2px); }
    64%     { transform: translateY(0); }
    70%     { opacity: 1; transform: translateY(-2px); }
    72%, 100% { opacity: 0; }
}
@keyframes p-sword {
    0%, 71% { opacity: 0; transform: translateY(8px); }
    76%     { opacity: 1; transform: translateY(0); }
    92%     { opacity: 1; }
    97%, 100% { opacity: 0; }
}
@keyframes p-blink { 0% { opacity: 1; } 50% { opacity: 0; } }

/* ============================================================
   LIBRO FORMANDOSE EN VIVO
   Es el mismo libro retro del resultado final: las paginas se
   van llenando escena por escena (texto -> boceto -> ilustracion
   -> voz) y al terminar la app pasa a la portada del libro.
============================================================ */
.book-live .paper { overflow: hidden; min-height: 500px; }
.book-live.turn .paper { animation: page-turn .55s steps(6) both; }
@keyframes page-turn {
    0%   { opacity: 0; transform: translateX(16px); }
    100% { opacity: 1; transform: none; }
}

/* hueco vacio de una pagina que aun no se llena */
.slot-empty {
    height: 380px;
    border: 4px dashed #c0cbdc;
    background: #ececdf;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-family: 'Press Start 2P', monospace;
    font-size: 11px;
    line-height: 1.9;
    color: #8b9bb4;
}
.slot-empty span { animation: p-blink 1.2s steps(2) infinite; }

/* ilustracion: boceto borroso que se pinta por encima de arriba a abajo */
.img-stack { position: relative; }
.paper .img-stack img.under {
    position: absolute; left: 0; top: 0;
    width: 100%; height: 100%; max-height: none;
    filter: blur(5px) saturate(.6);
}
.paper img.img-draft { filter: blur(5px) saturate(.6); }
.paper img.img-reveal { animation: img-paint 1.8s steps(14) both; position: relative; }
@keyframes img-paint {
    0%   { clip-path: inset(0 0 100% 0); filter: blur(6px) saturate(.4); }
    100% { clip-path: inset(0 0 0 0);    filter: none; }
}

/* texto: las palabras aparecen una a una, con cursor parpadeando */
.paper .w { opacity: 0; animation: w-in .01s steps(1) forwards; }
@keyframes w-in { to { opacity: 1; } }
.paper .cursor {
    display: inline-block; width: 10px; height: 20px;
    background: #181425; vertical-align: -3px;
    animation: p-blink .8s steps(2) infinite;
}

/* tira de progreso por escena: T = texto, I = ilustracion, V = voz */
.wip-steps {
    display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;
    max-width: 1100px; margin: -4px auto 4px;
}
.wip-step {
    font-family: 'Press Start 2P', monospace; font-size: 9px;
    background: #262b44; color: #8b9bb4;
    padding: 8px 10px; box-shadow: 4px 4px 0 #000;
    display: flex; gap: 6px; align-items: center;
}
.wip-step.active { outline: 3px solid #ffcd75; color: #ffcd75; }
.wip-step i { font-style: normal; padding: 3px 5px; background: #3a4466; color: #8b9bb4; }
.wip-step i.on { background: #a5e06b; color: #181425; }
.wip-legend {
    text-align: center; font-family: 'VT323', monospace;
    font-size: 18px; color: #8b9bb4; margin-bottom: 6px;
}

/* ============================================================
   BOTONES RETRO 3D CON RELEVE Y PRESION LATERAL (PIXEL STYLE)
============================================================ */
.stButton > button {
    font-family: 'Press Start 2P', monospace !important;
    font-size: 12px !important;
    background-color: #597dce !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 0px !important;
    padding: 14px 20px !important;
    position: relative !important;
    cursor: pointer !important;

    /* Sombra 3D lateral en Pixel Art */
    box-shadow: 
        inset -4px -4px 0px 0px #243467,
        inset  4px  4px 0px 0px #8cb0ff,
        -4px 0 0 0 #000,
         4px 0 0 0 #000,
         0 -4px 0 0 #000,
         0  4px 0 0 #000,
         5px 5px 0px 0px #000 !important;
    transition: all 0.05s linear !important;
}

.stButton > button:hover {
    background-color: #6990e7 !important;
    color: #ffcd75 !important;
    box-shadow: 
        inset -4px -4px 0px 0px #243467,
        inset  4px  4px 0px 0px #a2c2ff,
        -4px 0 0 0 #000,
         4px 0 0 0 #000,
         0 -4px 0 0 #000,
         0  4px 0 0 #000,
         7px 7px 0px 0px #000 !important;
    transform: translate(-2px, -2px) !important;
}

/* EFECTO 3D PRESIONADO HACIA ABAJO Y DERECHA */
.stButton > button:active {
    transform: translate(4px, 4px) !important;
    box-shadow: 
        inset 4px 4px 0px 0px #243467,
        -4px 0 0 0 #000,
         4px 0 0 0 #000,
         0 -4px 0 0 #000,
         0  4px 0 0 #000,
         1px 1px 0px 0px #000 !important;
}

/* INPUTS Y SELECTBOX PIXELATED */
.stTextArea textarea,
.stTextInput input,
div[data-baseweb="select"] > div {
    background-color: #181425 !important;
    color: #ffcd75 !important;
    border: none !important;
    border-radius: 0px !important;
    font-family: 'VT323', monospace !important;
    font-size: 22px !important;
    box-shadow: 
        inset 3px 3px 0px 0px #000000,
        -3px 0 0 0 #3b5dc9,
         3px 0 0 0 #3b5dc9,
         0 -3px 0 0 #3b5dc9,
         0  3px 0 0 #3b5dc9 !important;
}

label {
    color: #a5e06b !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 10px !important;
    text-transform: uppercase;
    margin-bottom: 8px !important;
}

/* ============================================================
   BIENVENIDA Y BOTÓN DE START CENTRADO
============================================================ */
.eter-title {
    width: 100%;
    text-align: center;
    font-family: 'Press Start 2P', monospace;
    font-size: clamp(36px, 6vw, 64px);
    color: #ffcd75;
    text-shadow: 6px 6px 0px #8b4049, 10px 10px 0px #000;
    margin: 0;
    line-height: 1.2;
}

.eter-sub {
    width: 100%;
    text-align: center;
    font-family: 'VT323', monospace;
    font-size: 28px;
    color: #a5e06b;
    margin: 20px 0 40px;
    letter-spacing: 1px;
}

/* CENTRADO PERFECTO DEL BOTÓN START */
div[data-testid="stColumn"]:has(div.st-key-welcome_play) {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

.st-key-welcome_play {
    display: flex !important;
    justify-content: center !important; /* Centra el botón en su contenedor */
    align-items: center !important;
    width: 100% !important;
    transform: translateX(-85px); /* <--- AJUSTA ESTE VALOR (-10px, -20px, etc.) PARA DESPLAZARLO A LA IZQUIERDA */
}

.st-key-welcome_play button {
    width: 160px !important;
    height: 60px !important;
    font-size: 16px !important;
    background-color: #e43b44 !important;
    box-shadow: 
        inset -6px -6px 0px 0px #8b4049,
        inset  6px  6px 0px 0px #feae34,
        -4px 0 0 0 #000,
         4px 0 0 0 #000,
         0 -4px 0 0 #000,
         0  4px 0 0 #000,
         8px 8px 0px 0px #000 !important;
}

.st-key-welcome_play button:hover {
    background-color: #f75157 !important;
}

/* ============================================================
   ESTRUCTURA DE LIBRO RETRO ABIERTO
============================================================ */
.book {
    display: grid;
    grid-template-columns: 1fr 1fr;
    max-width: 1100px;
    min-height: 540px;
    margin: 12px auto 20px;
    background: #262b44;
    position: relative;
    box-shadow: 
        -6px 0 0 0 #000,
         6px 0 0 0 #000,
         0 -6px 0 0 #000,
         0  6px 0 0 #000,
         10px 10px 0px 0px #000;
    padding: 12px;
    gap: 12px;
}

.paper {
    position: relative;
    padding: 32px 28px;
    color: #181425;
    background: #f4f4e8;
    box-shadow: inset 0 0 0 4px #c0cbdc;
    font-family: 'VT323', monospace;
}

.paper h2 {
    text-align: center;
    font-family: 'Press Start 2P', monospace;
    font-size: 14px;
    color: #181425;
    border-bottom: 4px solid #c0cbdc;
    padding-bottom: 12px;
    margin: 0 0 16px;
}

.paper p {
    font-size: 22px;
    line-height: 1.35;
    text-align: left;
    margin: 0;
}

.paper img {
    width: 100%;
    max-height: 380px;
    object-fit: cover;
    border: 4px solid #181425;
    box-shadow: 4px 4px 0px 0px #8b4049;
    image-rendering: auto !important; /* MANTIENE LA IMAGEN SUAVE DE ALTA CALIDAD */
}

.pnum {
    position: absolute;
    bottom: 10px;
    right: 14px;
    color: #8b9bb4;
    font-family: 'Press Start 2P', monospace;
    font-size: 10px;
}

/* ============================================================
   PORTADAS PIXEL ART PERSONALIZADAS POR GÉNERO
============================================================ */
.cover {
    position: relative;
    max-width: 550px;
    min-height: 500px;
    margin: 20px auto;
    padding: 40px 30px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    box-shadow: 
        -6px 0 0 0 #000,
         6px 0 0 0 #000,
         0 -6px 0 0 #000,
         0  6px 0 0 #000,
         10px 10px 0px 0px #000;
}

.cover small {
    font-family: 'Press Start 2P', monospace;
    font-size: 9px;
    letter-spacing: 1px;
    margin-bottom: 16px;
}

.cover h1 {
    font-family: 'Press Start 2P', monospace;
    font-size: clamp(20px, 3.5vw, 30px);
    line-height: 1.4;
    margin: 20px 0;
    text-shadow: 4px 4px 0px #000;
}

.cover .sym {
    font-size: 64px;
    margin-bottom: 10px;
}

.cover p {
    font-family: 'VT323', monospace;
    font-size: 26px;
}

/* FANTASÍA PIXEL */
.cover.fantasy {
    background: #3b2137;
    color: #ffcd75;
    box-shadow: inset 0 0 0 8px #b45252, -6px 0 0 0 #000, 6px 0 0 0 #000, 0 -6px 0 0 #000, 0 6px 0 0 #000, 10px 10px 0px 0px #000;
}

/* TERROR PIXEL */
.cover.horror {
    background: #181425;
    color: #e43b44;
    box-shadow: inset 0 0 0 8px #8b4049, -6px 0 0 0 #000, 6px 0 0 0 #000, 0 -6px 0 0 #000, 0 6px 0 0 #000, 10px 10px 0px 0px #000;
}

/* MISTERIO PIXEL */
.cover.mystery {
    background: #1a1c2b;
    color: #a5e06b;
    box-shadow: inset 0 0 0 8px #3b5dc9, -6px 0 0 0 #000, 6px 0 0 0 #000, 0 -6px 0 0 #000, 0 6px 0 0 #000, 10px 10px 0px 0px #000;
}

/* AVENTURA PIXEL */
.cover.adventure {
    background: #8b4049;
    color: #ffffff;
    box-shadow: inset 0 0 0 8px #feae34, -6px 0 0 0 #000, 6px 0 0 0 #000, 0 -6px 0 0 #000, 0 6px 0 0 #000, 10px 10px 0px 0px #000;
}

/* CIENCIA FICCIÓN PIXEL */
.cover.scifi {
    background: #08233b;
    color: #63c74d;
    box-shadow: inset 0 0 0 8px #00e43a, -6px 0 0 0 #000, 6px 0 0 0 #000, 0 -6px 0 0 #000, 0 6px 0 0 #000, 10px 10px 0px 0px #000;
}

/* INFANTIL PIXEL */
.cover.children {
    background: #ff7ba3;
    color: #ffffff;
    box-shadow: inset 0 0 0 8px #ffcd75, -6px 0 0 0 #000, 6px 0 0 0 #000, 0 -6px 0 0 #000, 0 6px 0 0 #000, 10px 10px 0px 0px #000;
}

.audio-label {
    font-family: 'Press Start 2P', monospace;
    font-size: 10px;
    color: #a5e06b;
    margin: 12px 0 6px;
}

@media(max-width:850px) {
    .book {
        grid-template-columns: 1fr;
    }
}
</style>
""", unsafe_allow_html=True)



# ============================================================
# FONDOS TEMATICOS (CSS puro) SEGUN EL GENERO
# - Pantalla de bienvenida: "biblioteca nocturna" (fondo general)
# - Estudio y libro: el fondo cambia con el genero elegido
# ============================================================
def _stars(c1: str, c2: str, c3: str) -> str:
    """Capa de estrellitas pixel (puntos pequenos en 5 mosaicos de distinto tamano)."""
    img = (
        f"radial-gradient(circle at 20px 30px,{c1} 0 1.6px,transparent 2.2px),"
        f"radial-gradient(circle at 90px 110px,{c2} 0 1.2px,transparent 1.8px),"
        f"radial-gradient(circle at 150px 60px,{c3} 0 1px,transparent 1.6px),"
        f"radial-gradient(circle at 60px 200px,{c1} 0 1px,transparent 1.6px),"
        f"radial-gradient(circle at 220px 140px,{c2} 0 1.4px,transparent 2px)"
    )
    return (
        f"background-image:{img};"
        "background-size:137px 137px,173px 173px,211px 211px,251px 251px,293px 293px;"
        "background-position:0 0,40px 70px,120px 30px,10px 90px,80px 20px;"
        "animation:bg-twinkle 3s steps(2) infinite alternate;"
    )


def _teeth(n: int, low: int, tips: tuple) -> str:
    """Silueta dentada (cercas / tumbas / arboles secos) para el fondo de Terror."""
    pts = ["0 100%", f"0 {low}%"]
    w = 100 / n
    for i in range(n):
        x0 = i * w
        pts += [f"{x0:.2f}% {low}%", f"{x0 + w / 2:.2f}% {tips[i % len(tips)]}%"]
    pts += [f"100% {low}%", "100% 100%"]
    return "polygon(" + ",".join(pts) + ")"


_BG_BASE = """
.stApp{isolation:isolate;background-attachment:fixed!important;}
.stApp::before,.stApp::after{content:'';position:fixed;pointer-events:none;z-index:-1;}
.stApp::before{inset:0;}
.stApp::after{left:0;right:0;bottom:0;}
@keyframes bg-twinkle{0%{opacity:.45}100%{opacity:1}}
@keyframes bg-drift{0%{transform:translateX(-5%)}100%{transform:translateX(5%)}}
@keyframes bg-rain{from{background-position:0 0}to{background-position:0 120px}}
@keyframes bg-grid{from{background-position:0 0}to{background-position:0 48px}}
@media (prefers-reduced-motion:reduce){.stApp::before,.stApp::after{animation:none!important;}}
"""

_BG = {
    # --- General: biblioteca nocturna (bienvenida) ---
    "_general": (
        ".stApp{background:linear-gradient(180deg,#0b0a1a 0%,#1a1033 60%,#2b1638 100%)!important;}"
        ".stApp::before{" + _stars("#ffcd75", "#ffffff", "#9fb4e2") + "}"
        ".stApp::after{height:96px;border-top:6px solid #5a3a2a;opacity:.55;"
        "background:repeating-linear-gradient(90deg,#3b2137 0 18px,#262b44 18px 30px,"
        "#8b4049 30px 42px,#1a1c2b 42px 60px,#b45252 60px 72px,#181425 72px 92px);}"
    ),
    # --- Fantasia: cielo violeta, luna dorada y castillo ---
    "Fantasía": (
        ".stApp{background:radial-gradient(circle at 82% 16%,#ffe9a8 0 28px,#ffe9a833 29px 64px,transparent 65px),"
        "linear-gradient(180deg,#120a22 0%,#2a1633 45%,#5b2b45 100%)!important;}"
        ".stApp::before{" + _stars("#ffcd75", "#fff4c2", "#f7a8d8") + "}"
        ".stApp::after{height:30vh;background:#0f0818;clip-path:polygon("
        "0 100%,0 70%,6% 70%,6% 55%,9% 55%,9% 45%,12% 45%,12% 55%,15% 55%,15% 70%,"
        "30% 70%,30% 62%,36% 62%,36% 50%,38% 50%,38% 38%,40% 38%,40% 32%,42% 32%,"
        "42% 38%,44% 38%,44% 50%,46% 50%,46% 62%,52% 62%,52% 70%,70% 70%,70% 58%,"
        "74% 58%,74% 48%,77% 48%,77% 58%,80% 58%,80% 70%,100% 70%,100% 100%);}"
    ),
    # --- Terror: noche negra, luna roja, niebla y cercas dentadas ---
    "Terror": (
        ".stApp{background:radial-gradient(circle at 50% 120%,#3a0a12 0%,transparent 60%),"
        "radial-gradient(circle at 80% 15%,#c9303a 0 22px,#c9303a33 23px 55px,transparent 56px),"
        "linear-gradient(180deg,#050308 0%,#120712 60%,#1e0a14 100%)!important;}"
        ".stApp::before{background:"
        "radial-gradient(ellipse 40% 12% at 20% 80%,#8b8fa244,transparent),"
        "radial-gradient(ellipse 45% 10% at 70% 70%,#8b8fa233,transparent),"
        "radial-gradient(ellipse 50% 14% at 45% 90%,#8b8fa255,transparent);"
        "animation:bg-drift 14s ease-in-out infinite alternate;}"
        ".stApp::after{height:22vh;background:#030205;clip-path:"
        + _teeth(26, 55, (8, 28, 14, 34)) + ";}"
    ),
    # --- Misterio: noir azul, luna palida, lluvia y ciudad ---
    "Misterio": (
        ".stApp{background:radial-gradient(circle at 78% 14%,#dfe8ff 0 20px,#dfe8ff22 21px 70px,transparent 71px),"
        "linear-gradient(180deg,#05070f 0%,#0c1630 60%,#16254a 100%)!important;}"
        ".stApp::before{opacity:.55;background:linear-gradient(100deg,transparent 48%,#9fb4e2 50%,transparent 52%);"
        "background-size:40px 120px;animation:bg-rain 1.2s linear infinite;}"
        ".stApp::after{height:34vh;clip-path:polygon("
        "0 100%,0 55%,5% 55%,5% 30%,10% 30%,10% 48%,14% 48%,14% 20%,19% 20%,19% 40%,"
        "24% 40%,24% 60%,30% 60%,30% 35%,36% 35%,36% 50%,41% 50%,41% 15%,46% 15%,"
        "46% 45%,52% 45%,52% 58%,58% 58%,58% 28%,64% 28%,64% 42%,70% 42%,70% 22%,"
        "75% 22%,75% 52%,81% 52%,81% 36%,88% 36%,88% 56%,94% 56%,94% 40%,100% 40%,100% 100%);"
        "background:radial-gradient(circle,#ffd96677 0 2px,transparent 2.5px) 0 0/22px 26px,#02030a;}"
    ),
    # --- Aventura: atardecer, nubes pixel y montanas ---
    "Aventura": (
        ".stApp{background:radial-gradient(circle at 50% 78%,#ffe08a 0 60px,#ffb04a55 61px 120px,transparent 121px),"
        "linear-gradient(180deg,#2b1232 0%,#8b3a4a 35%,#e0763a 70%,#f7b455 100%)!important;}"
        ".stApp::before{background:"
        "linear-gradient(#ffffff2e,#ffffff2e) 12% 16%/150px 14px no-repeat,"
        "linear-gradient(#ffffff2e,#ffffff2e) calc(12% + 24px) calc(16% - 14px)/90px 14px no-repeat,"
        "linear-gradient(#ffffff2e,#ffffff2e) 68% 28%/180px 14px no-repeat,"
        "linear-gradient(#ffffff2e,#ffffff2e) calc(68% + 30px) calc(28% - 14px)/100px 14px no-repeat;"
        "animation:bg-drift 22s ease-in-out infinite alternate;}"
        ".stApp::after{height:26vh;background:#3a1a24;clip-path:polygon("
        "0 100%,0 60%,8% 60%,8% 45%,14% 45%,14% 30%,20% 30%,20% 45%,26% 45%,26% 62%,"
        "38% 62%,38% 50%,46% 50%,46% 62%,60% 62%,60% 40%,66% 40%,66% 24%,72% 24%,"
        "72% 40%,78% 40%,78% 55%,90% 55%,90% 68%,100% 68%,100% 100%);}"
    ),
    # --- Ciencia ficcion: espacio profundo y piso de rejilla neon ---
    "Ciencia ficción": (
        ".stApp{background:radial-gradient(ellipse at 50% 100%,#00e43a22,transparent 60%),"
        "linear-gradient(180deg,#02060d 0%,#06182b 55%,#0a2f3a 100%)!important;}"
        ".stApp::before{" + _stars("#7df9ff", "#ffffff", "#63c74d") + "}"
        ".stApp::after{height:80vh;transform:perspective(400px) rotateX(64deg);transform-origin:50% 100%;"
        "background:linear-gradient(0deg,#00e43a66 2px,transparent 2px),"
        "linear-gradient(90deg,#00e43a66 2px,transparent 2px);background-size:48px 48px;"
        "-webkit-mask-image:linear-gradient(to top,#000,transparent 85%);"
        "mask-image:linear-gradient(to top,#000,transparent 85%);"
        "animation:bg-grid 2.4s linear infinite;}"
    ),
    # --- Infantil: cielo de cuento, estrellas pastel y colinas ---
    "Infantil": (
        ".stApp{background:radial-gradient(circle at 85% 14%,#fff3b0 0 30px,#fff3b055 31px 64px,transparent 65px),"
        "linear-gradient(180deg,#2a2a78 0%,#5a49b8 50%,#b06ad0 100%)!important;}"
        ".stApp::before{" + _stars("#fff3b0", "#ffb3d9", "#b3f0ff") + "}"
        ".stApp::after{height:20vh;background:"
        "radial-gradient(ellipse 45% 100% at 15% 100%,#6fd69a 0 99%,transparent 100%),"
        "radial-gradient(ellipse 55% 85% at 55% 100%,#4fc088 0 99%,transparent 100%),"
        "radial-gradient(ellipse 40% 95% at 95% 100%,#86e0a8 0 99%,transparent 100%);}"
    ),
}


def apply_background(genre=None):
    """Inyecta el fondo CSS: general si genre es None, tematico si es un genero."""
    css = _BG.get(genre, _BG["_general"]) + _BG_BASE
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ============================================================
# UTILIDADES
# ============================================================

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def img_to_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(
        path.read_bytes()
    ).decode()


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", text))


def is_spanish(text: str) -> bool:
    if has_cjk(text):
        return False
    words = re.findall(r"[a-záéíóúñü]+", text.lower())
    if not words:
        return False
    common = {"el", "la", "los", "las", "de", "que", "y", "en", "un", "una", "con", "por", "su", "se", "del"}
    return sum(w in common for w in words) / len(words) > 0.15


# ============================================================
# CARGA DE MODELOS EN CACHÉ (RENDIMIENTO)
# Los modelos se cargan UNA sola vez por sesión: la primera
# generación tarda lo mismo, pero las siguientes saltan por
# completo la carga (la mayor pérdida de tiempo).
# ============================================================

@st.cache_resource(show_spinner=False)
def load_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_ID,
        torch_dtype=torch.float32
    ).to(DEVICE)
    model.eval()
    return tok, model


@st.cache_resource(show_spinner=False)
def load_image_pipe():
    from diffusers import AutoPipelineForText2Image, LCMScheduler
    pipe = AutoPipelineForText2Image.from_pretrained(
        IMG_ID,
        torch_dtype=torch.float32,
        use_safetensors=True
    ).to(DEVICE)

    # LCM-LoRA acelera la generación en CPU (pocos pasos).
    # Requiere la librería "peft": uv pip install peft
    try:
        pipe.load_lora_weights(LCM_LORA_ID)
        pipe.fuse_lora()
        # <-- CAMBIO A: el LoRA LCM SOLO funciona bien con LCMScheduler.
        #     Sin esto salían imágenes ruidosas/con artefactos.
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    except Exception:
        # Recarga limpia (sin LoRA a medias) y scheduler original
        pipe = AutoPipelineForText2Image.from_pretrained(
            IMG_ID,
            torch_dtype=torch.float32,
            use_safetensors=True
        ).to(DEVICE)

    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


@st.cache_resource(show_spinner=False)
def load_img2img_pipe():
    """<-- CAMBIO B: pipeline img2img que COMPARTE los pesos del pipeline base
    (from_pipe no duplica memoria). El pipeline text2img no acepta image=/strength=,
    por eso el pase 2 fallaba en silencio."""
    from diffusers import AutoPipelineForImage2Image
    img2img = AutoPipelineForImage2Image.from_pipe(load_image_pipe())
    img2img.set_progress_bar_config(disable=True)
    return img2img


@st.cache_resource(show_spinner=False)
def load_kokoro():
    from kokoro import KPipeline
    return KPipeline(lang_code=KOKORO_LANG, device=DEVICE)


# ====================
# 1. HISTORIA
# ====================
def fallback_story(idea: str, genre: str) -> dict:
    idea = idea.strip().rstrip(".")
    return {
        "title": idea[:60].capitalize(),
        "character": "a focused human protagonist with distinctive features, practical clothing, sturdy gear",
        "setting": "a mysterious and dark ancient location",
        "scenes": [
            {
                "title": "El umbral de las sombras",
                "text": f"El viento helado soplaba con fuerza sobre los riscos cuando {idea}. La figura principal ajustó las correas de su pesado morral de cuero con manos firmes, sintiendo cómo el silencio sepulcral del entorno parecía respirar a su alrededor. Ante sus ojos se abría un paso oculto cuyos maderos carcomidos crujían como huesos viejos bajo la presión de la intemperie. Cada paso adelante despertaba ecos antiguos que se arrastraban entre las piedras, como si la montaña misma reconociera su presencia y contuviera la respiración.",
                "action": "a focused human protagonist holding an oil lantern entering a dark mysterious entrance"
            },
            {
                "title": "Los guardianes del abismo",
                "text": "A medida que avanzaba con extrema cautela por los senderos laberínticos de la gruta, el aire se tornó pesado, cargado por completo con un olor metálico y sumamente antiguo. De súbito, el haz titilante de la luz chocó contra un par de destellos intensos y amenazantes en la oscuridad. Eran ojos. Enormes, pacientes, clavados en la penumbra como brasas enterradas en ceniza fría. Las criaturas no rugieron ni atacaron: simplemente observaban, con la serena autoridad de quien lleva siglos custodiando aquel umbral prohibido.",
                "action": "massive dark mysterious creatures blocking a stone corridor with glowing eyes"
            },
            {
                "title": "El lenguaje del silencio",
                "text": "Manteniéndose completamente firme a pesar del evidente temblor en sus manos, levantó la palma abierta en una clara señal universal de paz, intentando descifrar el propósito real de aquellas presencias imponentes que la observaban sin parpadear. Un rumor profundo recorrió la caverna, semejante a una lengua olvidada pronunciada bajo el agua. Poco a poco comprendió que no le exigían palabras, sino intención: la honestidad silenciosa de quien llega a pedir, no a tomar.",
                "action": "the protagonist cautiously communicating with mysterious entities"
            },
            {
                "title": "El secreto oculto en la roca",
                "text": "Guiada por una intuición febril y persistente que desafiaba el sentido común, se deslizó con sumo cuidado más allá de los vigilantes, adentrándose en el corazón mismo del recinto. En el centro exacto reposaba un artefacto geométrico de cristal que flotaba a un palmo del suelo, girando lentísimo sobre su propio eje. Al acercarse, la piel le hormigueó con una electricidad antigua, y en el reflejo del cristal no vio su rostro, sino una versión de sí misma más valiente, más antigua, más despierta.",
                "action": "discovering a glowing geometric crystal artifact inside a secret chamber"
            },
            {
                "title": "El horizonte desconocido",
                "text": "Sosteniendo firmemente el valioso artefacto entre sus manos, sintió una repentina y cálida oleada de conocimiento antiguo que recorrió todo su cuerpo, disipando los temores iniciales. Las sombras de la gruta se apartaron como cortinas ante un actor esperado, y una senda luminosa se abrió hacia un valle que ningún mapa había registrado jamás. Ya no era una simple persona ordinaria: era la primera voz de una historia nueva, y el mundo entero esperaba ser escrito por sus pasos.",
                "action": "a protagonist standing victoriously looking into a vast new mysterious world"
            }
        ]
    }

def build_outline_messages(idea: str, genre: str):
    """FASE 1: escaleta detallada (rápida de generar, JSON pequeño)."""
    system = (
        "Eres un escritor galardonado de literatura de ficción en español. "
        "REGLA ABSOLUTA: el título del libro, los títulos de las escenas y los "
        "resúmenes se redactan 100% EN ESPAÑOL. "
        "EXCEPCIÓN ÚNICA: los campos \"character\", \"setting\" y \"action\" van "
        "EN INGLÉS para el motor de imágenes. "
        "Respondes SIEMPRE y únicamente con un objeto JSON válido, sin texto adicional."
    )

    user = f"""Diseña la ESCALETA de una obra de género {genre} basada en esta idea: "{idea}"

Reglas:
- Exactamente {N_SCENES} escenas conectadas con fluidez, fieles a la idea del usuario.
- "title": título del libro, evocativo, 100% EN ESPAÑOL, máx. 6 palabras.
- "character": descripción física fija del protagonista EN INGLÉS, máx. 20 palabras.
- "setting": escenario principal EN INGLÉS, máx. 12 palabras.
- Cada escena: "title" EN ESPAÑOL (máx. 6 palabras), "action" EN INGLÉS (máx. 20 palabras,
  visual, para el motor de imágenes) y "summary" EN ESPAÑOL: 2-3 frases que resuman qué
  sucede, con un arco claro (presentación, desarrollo, giro, clímax, cierre).
- Las escenas deben sentirse distintas entre sí: alterna acción, diálogo, atmósfera y giro.

Responde con este formato JSON exacto (solo JSON):
{{
  "title": "...",
  "character": "...",
  "setting": "...",
  "scenes": [
    {{"title": "...", "action": "...", "summary": "..."}}
  ]
}}

Idea: "{idea}". Genera la escaleta de {N_SCENES} escenas."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]


def build_scene_prose_messages(idea: str, genre: str, outline: dict, idx: int, prev_ending: str):
    """FASE 2: prosa de UNA sola escena (salida corta = menos truncamientos)."""
    sc = outline["scenes"][idx]

    system = (
        "Eres un escritor galardonado de literatura de ficción en español. "
        "Escribes 100% EN ESPAÑOL con una prosa rica, melódica y atmosférica, "
        "digna de un narrador profesional de audiolibros. "
        "Respondes ÚNICAMENTE con el párrafo narrativo solicitado, sin título, "
        "sin comillas que lo envuelvan, sin explicaciones y sin texto adicional."
    )

    continuation = (
        f"\nLa escena anterior terminó así: \"...{prev_ending}\". "
        "Continúa la historia de forma fluida y coherente con ese final."
        if prev_ending else ""
    )

    user = f"""Escribe ÚNICAMENTE la escena {idx + 1} de {N_SCENES} de esta obra de género {genre}.

Idea original del autor: "{idea}"
Título de la escena: "{sc['title']}"
Resumen de lo que debe suceder: "{sc['summary']}"{continuation}

Reglas obligatorias:
- Prosa de MÍNIMO {SCENE_WORDS_MIN} palabras y máximo {SCENE_WORDS_MAX} palabras,
  100% EN ESPAÑOL, en UN solo párrafo (sin saltos de línea).
- Extensión equivalente a unas 10-13 líneas de libro: desarrolla la escena con varias
  frases, diálogos o pensamientos breves, descripciones sensoriales y ritmo pausado.
- NO repitas el nombre ni el sustantivo del protagonista una y otra vez: menciónalo
  como máximo UNA vez y usa después pronombres ("él", "ella", "le", "su").
- Alterna el inicio de las oraciones: nunca comiences varias frases seguidas
  con la misma palabra.
- Desarrolla fielmente el resumen indicado y deja la escena lista para que la
  siguiente continúe con naturalidad.

Responde solo con el párrafo."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]


JSON_PREFILL = '{\n  "title": "'


def parse_outline(raw: str):
    """JSON tolerante para la escaleta (bloques ```json, comas finales, etc.)."""
    text = re.sub(r"```(?:json)?", "", raw)
    first = text.find("{")
    if first == -1:
        return None
    last = text.rfind("}")
    text = text[first:last + 1] if last > first else text[first:]
    candidate = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        data = json.loads(candidate, strict=False)
        scenes = data.get("scenes")
        if not isinstance(scenes, list) or len(scenes) < N_SCENES:
            return None
        scenes = scenes[:N_SCENES]
        for s in scenes:
            for k in ("title", "action", "summary"):
                if not str(s.get(k, "")).strip():
                    return None
        data["scenes"] = scenes
        data.setdefault("title", str(scenes[0].get("title", "Relato")))
        data.setdefault("character", "a focused human protagonist with distinctive features")
        data.setdefault("setting", "a mysterious atmospheric location")
        return data
    except Exception:
        return None


def _clean_prose(text: str) -> str:
    """Limpia la salida de prosa: sin comillas envolventes ni etiquetas."""
    text = text.strip().strip('"').strip()
    text = re.sub(r"```(?:\w+)?", "", text)
    return " ".join(text.split())


def generate_story(idea: str, genre: str, on_event=None) -> tuple[dict, bool]:
    """on_event(evento, i, dato) permite mostrar el libro formandose:
    'outline' (dato = escaleta), 'scene_start' (dato = titulo), 'scene_done' (dato = escena)."""
    import traceback

    tok, model = load_llm()   # en caché: no se recarga en cada generación

    # ============ FASE 1: ESCALETA (una sola llamada, JSON pequeño) ============
    outline = None
    debug_lines = []

    for temp in (0.5, 0.7):
        try:
            prompt = tok.apply_chat_template(
                build_outline_messages(idea, genre),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ) + JSON_PREFILL

            inputs = tok([prompt], return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=1200,     # la escaleta es corta
                    do_sample=True,
                    temperature=temp,
                    top_p=0.9,
                    repetition_penalty=1.05
                )

            raw = JSON_PREFILL + tok.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            debug_lines.append(f"===== ESCALETA (temp={temp}) =====\n{raw}\n")
            outline = parse_outline(raw)
            if outline:
                break
        except Exception:
            debug_lines.append(f"===== ESCALETA ERROR =====\n{traceback.format_exc()}\n")
            continue

    if not outline:
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "debug_llm_raw.txt").write_text("\n".join(debug_lines), encoding="utf-8")
        except Exception:
            pass
        return fallback_story(idea, genre), True

    if on_event:
        on_event("outline", None, outline)

    # ============ FASE 2: PROSA ESCENA POR ESCENA (chunking) ============
    scenes = []
    prev_ending = ""

    for i in range(N_SCENES):
        text = None
        if on_event:
            on_event("scene_start", i, outline["scenes"][i]["title"])

        for temp in (0.6, 0.8):
            try:
                prompt = tok.apply_chat_template(
                    build_scene_prose_messages(idea, genre, outline, i, prev_ending),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                inputs = tok([prompt], return_tensors="pt").to(DEVICE)
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=700,   # una sola escena: salida corta
                        do_sample=True,
                        temperature=temp,
                        top_p=0.9,
                        repetition_penalty=1.05
                    )

                candidate = _clean_prose(tok.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ))

                # Aceptamos si se acerca al mínimo pedido (80%)
                if len(candidate.split()) >= int(SCENE_WORDS_MIN * 0.8):
                    text = candidate
                    break
                debug_lines.append(f"===== ESCENA {i+1} (temp={temp}) CORTA ({len(candidate.split())} palabras) =====\n{candidate}\n")
            except Exception:
                debug_lines.append(f"===== ESCENA {i+1} ERROR =====\n{traceback.format_exc()}\n")
                continue

        if text is None:
            # Escena irrecuperable: usamos el relato de contingencia completo
            try:
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                (OUT_DIR / "debug_llm_raw.txt").write_text("\n".join(debug_lines), encoding="utf-8")
            except Exception:
                pass
            return fallback_story(idea, genre), True

        scenes.append({
            "title": outline["scenes"][i]["title"],
            "text": text,
            "action": outline["scenes"][i]["action"],
        })
        if on_event:
            on_event("scene_done", i, scenes[-1])
        # Contexto para la siguiente escena: final de la actual
        prev_ending = text[-220:]

    return {
        "title": outline["title"],
        "character": outline["character"],
        "setting": outline["setting"],
        "scenes": scenes,
    }, False


# ============================================================
# 2. IMÁGENES ESTÁTICAS DE ALTA CALIDAD (ILUSTRACIONES ORIGINALES)
#    Pipeline de 3 fases por escena:
#      1) Generación base 512x512 con LCM (rápida)
#      2) Refinado img2img a 768x768 con ruido bajo (detalle)
#      3) Escalado x2 + denoise suave + nitidez (PIL, sin GPU)
# ============================================================

def _polish_image(img, target_size: int):
    """Escalado de alta calidad + reducción de ruido + nitidez (PIL)."""
    from PIL import Image, ImageFilter

    # 1) Escalado x2 con Lanczos (mucho más nítido que el escalado del navegador)
    img = img.resize((target_size, target_size), resample=Image.LANCZOS)
    # 2) Denoise suave: elimina el grano/ruido del diffusion sin borrar detalle
    img = img.filter(ImageFilter.MedianFilter(size=3))
    # 3) Nitidez final para recuperar el detalle tras el denoise
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=2))
    return img


def build_image_prompt(pipe, parts: list) -> str:
    """<-- CAMBIO C: une las partes POR PRIORIDAD sin pasar el límite de CLIP
    (77 tokens). Lo que no cabe se descarta, en vez de cortarse al azar al final."""
    tok = pipe.tokenizer
    chosen = []
    for part in parts:
        part = str(part).strip().strip(",").strip()
        if not part:
            continue
        candidate = ", ".join(chosen + [part])
        if not chosen or len(tok(candidate).input_ids) <= IMG_PROMPT_MAX_TOKENS:
            chosen.append(part)
    return ", ".join(chosen)


def generate_scene_images(story: dict, genre: str, style: str, progress_cb, preview_cb=None) -> list[Path]:
    import traceback
    from PIL import Image
    from diffusers import LCMScheduler

    pipe = load_image_pipe()          # en caché: no se recarga en cada generación
    img2img = load_img2img_pipe()     # en caché y comparte pesos con pipe

    # <-- CAMBIO A: parámetros según el scheduler realmente activo
    lcm_ok = isinstance(pipe.scheduler, LCMScheduler)
    steps = IMG_STEPS if lcm_ok else IMG_STEPS_FALLBACK
    refine_steps = IMG_REFINE_STEPS if lcm_ok else IMG_STEPS_FALLBACK
    guidance = IMG_GUIDANCE if lcm_ok else IMG_GUIDANCE_FALLBACK
    negative = NEGATIVE_PROMPT if guidance > 1.0 else None   # sin CFG no se usa

    visual = GENRES[genre]["visual"]
    style_en = STYLE_EN.get(style, "digital painting")
    paths = []
    debug_lines = []

    for i, scene in enumerate(story["scenes"], 1):
        # <-- CAMBIO C: orden por prioridad (acción > personaje > estilo > género > escenario)
        prompt = build_image_prompt(pipe, [
            scene["action"],
            story["character"],
            style_en,
            visual,
            story["setting"],
            "masterpiece, highly detailed",
        ])

        generator = torch.Generator(device=DEVICE).manual_seed(IMG_SEED + i * 150)

        # --- FASE 1: generación base con LCM ---
        with torch.inference_mode():
            base = pipe(
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=IMG_SIZE,
                width=IMG_SIZE,
                generator=generator
            ).images[0]

        if preview_cb:
            preview_cb(i, base)   # boceto base 512 para mostrarlo en el libro

        # --- FASE 2: refinado img2img a mayor resolución (hires fix) ---
        # <-- CAMBIO B: el img2img toma el tamaño de la imagen de entrada,
        #     por eso se escala la base a 768 antes de refinar.
        try:
            base_up = base.resize((IMG_REFINE_SIZE, IMG_REFINE_SIZE), resample=Image.LANCZOS)
            ref_generator = torch.Generator(device=DEVICE).manual_seed(IMG_SEED + i * 150)
            with torch.inference_mode():
                refined = img2img(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=base_up,
                    strength=IMG_REFINE_STRENGTH,
                    num_inference_steps=refine_steps,
                    guidance_scale=guidance,
                    generator=ref_generator
                ).images[0]
        except Exception:
            # Ya no falla en silencio: queda registrado en salida/debug_img.txt
            debug_lines.append(f"===== ESCENA {i}: ERROR EN REFINADO =====\n{traceback.format_exc()}\n")
            refined = base

        # --- FASE 3: escalado + denoise + nitidez ---
        final = _polish_image(refined, IMG_REFINE_SIZE * IMG_UPSCALE)

        path = OUT_DIR / f"escena_{i}.png"
        final.save(path, optimize=True)
        paths.append(path)
        progress_cb(i, path)

    if debug_lines:
        try:
            (OUT_DIR / "debug_img.txt").write_text("\n".join(debug_lines), encoding="utf-8")
        except Exception:
            pass

    return paths


# ============================================================
# 3. VOZ (KOKORO)
# ============================================================

def generate_narrations(story: dict, progress_cb) -> list[Path]:
    pipeline = load_kokoro()   # <-- en caché: no se recarga en cada generación
    paths = []

    for i, scene in enumerate(story["scenes"], 1):
        chunks = []
        with torch.inference_mode():
            for _, _, audio in pipeline(
                scene["text"],
                voice=KOKORO_VOICE,
                speed=KOKORO_SPEED,
                split_pattern=r"(?<=[.!?…])\s+"
            ):
                if audio is not None:
                    chunks.append(audio.detach().cpu().numpy().astype(np.float32).reshape(-1))
                    # Pausa entre frase y frase (SENTENCE_PAUSE segundos)
                    chunks.append(np.zeros(int(24000 * SENTENCE_PAUSE), dtype=np.float32))

        if not chunks:
            raise RuntimeError(f"Kokoro no produjo audio para la escena {i}.")

        # Pausa de respiración al final de la escena
        chunks.append(np.zeros(int(24000 * SCENE_END_PAUSE), dtype=np.float32))

        waveform = np.concatenate(chunks)
        path = OUT_DIR / f"voz_{i}.wav"
        sf.write(str(path), waveform, 24000, subtype="PCM_16")
        paths.append(path)
        progress_cb(i)

    return paths


# ============================================================
# LIBRO FORMANDOSE EN VIVO (ESCENA POR ESCENA)
# Usa las mismas paginas del libro real. Mientras se genera:
#   1) la pagina derecha se escribe palabra por palabra
#   2) la izquierda muestra un boceto que se pinta de arriba a abajo
#   3) al final se pasa de pagina escena por escena (voz lista)
# ============================================================

def pil_to_uri(img, quality: int = 70) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _words_html(text: str, animate: bool, cursor: bool) -> str:
    words = text.split()
    if not animate:
        return html.escape(" ".join(words))
    step = min(0.04, 4.0 / max(len(words), 1))   # el texto completo se "escribe" en <= 4 s
    out = " ".join(
        f'<span class="w" style="animation-delay:{k * step:.2f}s">{html.escape(w)}</span>'
        for k, w in enumerate(words)
    )
    if cursor:
        out += ' <span class="cursor"></span>'
    return out


def render_book_live(slot, idx, scene, steps, image_uri=None, img_mode="none",
                     draft_uri=None, text_mode="none", turn=False):
    """Dibuja el libro abierto en la escena `idx`.
    scene: {"title","text"} (text puede ir vacio mientras se escribe) o None
    img_mode: none | draft | reveal | done      text_mode: none | write | done
    """
    title = html.escape(scene["title"]) if scene and scene.get("title") else "..."
    text = (scene or {}).get("text", "")

    # --- pagina izquierda: ilustracion ---
    if image_uri and img_mode == "reveal" and draft_uri:
        left = (f'<div class="img-stack"><img class="under" src="{draft_uri}" />'
                f'<img class="img-reveal" src="{image_uri}" /></div>')
    elif image_uri:
        cls = {"draft": "img-draft", "reveal": "img-reveal"}.get(img_mode, "")
        left = f'<img class="{cls}" src="{image_uri}" />'
    else:
        left = '<div class="slot-empty"><span>LIENZO<br>EN BLANCO</span></div>'

    # --- pagina derecha: narracion ---
    if text:
        right = "<p>" + _words_html(text, text_mode == "write", text_mode == "write") + "</p>"
    else:
        right = '<div class="slot-empty"><span>ESCRIBIENDO...</span></div>'

    chips = "".join(
        f'<div class="wip-step{" active" if k == idx else ""}"><b>{k + 1}</b>'
        f'<i class="{"on" if s["t"] else ""}">T</i>'
        f'<i class="{"on" if s["i"] else ""}">I</i>'
        f'<i class="{"on" if s["v"] else ""}">V</i></div>'
        for k, s in enumerate(steps)
    )

    # Todo en una sola linea: una linea en blanco rompería el bloque HTML de markdown.
    slot.markdown(
        f'<div class="book book-live{" turn" if turn else ""}">'
        f'<div class="paper"><h2>{title}</h2>{left}<div class="pnum">PAG {idx * 2 + 1}</div></div>'
        f'<div class="paper"><h2>Escena {idx + 1} / {N_SCENES}</h2>{right}<div class="pnum">PAG {idx * 2 + 2}</div></div>'
        f'</div>'
        f'<div class="wip-steps">{chips}</div>'
        f'<div class="wip-legend">T = texto &nbsp; I = ilustración &nbsp; V = voz</div>',
        unsafe_allow_html=True
    )


# ============================================================
# ESTADO DE LA APLICACION
# ============================================================

for k, v in {
    "screen": "welcome",
    "book": None,
    "page": -1,
    "narration_auto": True
}.items():
    st.session_state.setdefault(k, v)


# ============================================================
# BARRA SUPERIOR RETRO
# ============================================================

st.markdown("""
<div class="topbar">
<div class="brand">ETER</div>
<div class="pixel-sky">
<svg viewBox="0 0 160 32" shape-rendering="crispEdges" xmlns="http://www.w3.org/2000/svg">
<rect class="p-flash" width="160" height="32" fill="#ffffff"/>
<g class="p-bird"><g transform="translate(0,14)">
<rect x="0" y="3" width="2" height="1" fill="#41a6f6"/>
<rect x="2" y="3" width="5" height="2" fill="#41a6f6"/>
<rect x="7" y="2" width="2" height="2" fill="#73eff7"/>
<rect x="8" y="2" width="1" height="1" fill="#0d0e15"/>
<rect x="9" y="3" width="1" height="1" fill="#ffcd75"/>
<g class="p-up" fill="#41a6f6"><rect x="4" y="0" width="1" height="1"/><rect x="3" y="1" width="3" height="1"/><rect x="3" y="2" width="3" height="1"/></g>
<g class="p-down" fill="#41a6f6"><rect x="3" y="5" width="3" height="1"/><rect x="4" y="6" width="1" height="1"/></g>
</g></g>
<g class="p-bolt" fill="#ffe66d">
<rect x="73" y="0" width="3" height="4"/><rect x="71" y="4" width="3" height="4"/>
<rect x="73" y="8" width="3" height="4"/><rect x="70" y="12" width="3" height="3"/>
<rect x="72" y="15" width="2" height="2"/>
</g>
<g class="p-smoke">
<rect x="66" y="14" width="4" height="4" fill="#8b9bb4"/>
<rect x="71" y="12" width="5" height="5" fill="#c0cbdc"/>
<rect x="76" y="15" width="3" height="3" fill="#566c86"/>
</g>
<g class="p-talis">
<rect x="68" y="6" width="6" height="10" fill="#3b1459"/>
<rect x="69" y="7" width="4" height="8" fill="#b45cff"/>
<rect x="70" y="9" width="2" height="1" fill="#ffcd75"/>
<rect x="70" y="11" width="2" height="1" fill="#ffcd75"/>
<rect x="70" y="13" width="2" height="1" fill="#ffcd75"/>
</g>
<g class="p-sword">
<rect x="70" y="1" width="2" height="17" fill="#e2e8f0"/>
<rect x="70" y="1" width="1" height="17" fill="#ffffff"/>
<rect x="66" y="18" width="10" height="2" fill="#ffcd75"/>
<rect x="70" y="20" width="2" height="5" fill="#8b4049"/>
<rect x="69" y="25" width="4" height="2" fill="#ffcd75"/>
<g class="p-spark"><rect x="63" y="6" width="1" height="1" fill="#ffffff"/><rect x="78" y="10" width="1" height="1" fill="#ffffff"/><rect x="65" y="22" width="1" height="1" fill="#b45cff"/><rect x="77" y="24" width="1" height="1" fill="#b45cff"/></g>
</g>
</svg>
</div>
<div class="tag">ESTUDIO DE HISTORIAS RETRO</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PANTALLA 1: BIENVENIDA
# ============================================================
if st.session_state.screen == "welcome":
    apply_background(None)
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([2, 2, 2])

    with center_col:
        st.markdown(
            '<div class="eter-title">ETER</div>'
            '<div class="eter-sub">TUS RELATOS ILUSTRADOS EN FORMATO RETRO</div>',
            unsafe_allow_html=True
        )

        if st.button("▶ START", key="welcome_play", help="Iniciar"):
            st.session_state.screen = "studio"
            st.rerun()

# ============================================================
# PANTALLA 2: ESTUDIO DE CREACION
# ============================================================
elif st.session_state.screen == "studio":
    st.markdown("""
    <div style="max-width: 900px; margin: 0 auto 2rem auto; text-align: center;">
        <div style="font-family: 'Press Start 2P', monospace; font-size: 22px; color: #ffcd75; margin-bottom: 12px;">
            SANTUARIO LITERARIO
        </div>
        <div style="color: #a5e06b; font-size: 22px; font-family: 'VT323', monospace;">
            Configura los parámetros para forjar tu aventura.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1.3, 1], gap="large")

    with col_a:
        st.markdown("**Idea inicial de la historia**")
        idea = st.text_area(
            "Idea inicial de la historia",
            label_visibility="collapsed",
            height=165,
            placeholder="Ej: Una cartógrafa descubre un reino oculto bajo las montañas…"
        )

    with col_b:
        st.markdown("**Tipo de historia (Género)**")
        genre = st.selectbox(
            "Tipo de historia (Género)",
            label_visibility="collapsed",
            options=list(GENRES.keys())
        )
        apply_background(genre)

        st.markdown("**Estilo de ilustración**")
        style = st.selectbox(
            "Estilo de ilustración",
            label_visibility="collapsed",
            options=list(STYLE_EN.keys())
        )

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])

    with btn_col:
        crear_libro = st.button("⚡ GENERAR LIBRO", use_container_width=True)

    if crear_libro:
        if not idea.strip():
            st.error("Escribe una idea para la historia.")
        else:
            if OUT_DIR.exists():
                shutil.rmtree(OUT_DIR)
            OUT_DIR.mkdir(parents=True, exist_ok=True)

            with st.status("Creando tu obra literaria…", expanded=True) as status:
                # El libro real formándose en vivo, escena por escena
                book_slot = st.empty()
                note_slot = st.empty()
                steps = [{"t": False, "i": False, "v": False} for _ in range(N_SCENES)]
                drafts = {}

                def show(idx, scene, **kw):
                    render_book_live(book_slot, idx, scene, steps, **kw)

                def note(msg):
                    note_slot.markdown(f'<div class="audio-label">{html.escape(msg)}</div>',
                                       unsafe_allow_html=True)

                # --- Fase 1: texto, escena por escena ---
                show(0, None)
                note("Trazando la escaleta de la historia...")

                def on_story_event(ev, i, data):
                    if ev == "scene_start":
                        show(i, {"title": data, "text": ""}, turn=True)
                        note(f"Escribiendo la escena {i + 1}/{N_SCENES}...")
                    elif ev == "scene_done":
                        steps[i]["t"] = True
                        show(i, data, text_mode="write")

                story, used_fallback = generate_story(idea.strip(), genre, on_event=on_story_event)

                if used_fallback:
                    st.warning("Se usó un relato base alternativo por contingencia.")
                    for s in steps:
                        s["t"] = True

                scenes = story["scenes"]

                # --- Fase 2: ilustraciones (boceto -> pintado) ---
                bar_img = st.progress(0, text="Ilustrando escenas (base 512 + refinado 768)...")
                show(0, scenes[0], text_mode="done", turn=True)
                note("Ilustrando la escena 1...")

                def on_img_preview(i, pil_img):
                    drafts[i] = pil_to_uri(pil_img)
                    show(i - 1, scenes[i - 1], image_uri=drafts[i], img_mode="draft",
                         text_mode="done", turn=True)
                    note(f"Escena {i}: boceto listo, refinando detalles...")

                def on_img_done(i, path):
                    steps[i - 1]["i"] = True
                    show(i - 1, scenes[i - 1], image_uri=img_to_uri(path), img_mode="reveal",
                         draft_uri=drafts.get(i), text_mode="done")
                    note(f"Escena {i}: ilustración pintada.")
                    bar_img.progress(i / N_SCENES, text=f"Ilustración {i}/{N_SCENES}")

                images = generate_scene_images(story, genre, style, on_img_done, preview_cb=on_img_preview)

                # --- Fase 3: voz (el libro pasa de página escena por escena) ---
                bar_voice = st.progress(0, text="Sintetizando voz narradora...")
                show(0, scenes[0], image_uri=img_to_uri(images[0]), img_mode="done",
                     text_mode="done", turn=True)
                note("Narrando la escena 1...")

                def on_voice_done(i):
                    steps[i - 1]["v"] = True
                    nxt = min(i, N_SCENES - 1)
                    show(nxt, scenes[nxt], image_uri=img_to_uri(images[nxt]), img_mode="done",
                         text_mode="done", turn=(i < N_SCENES))
                    note(f"Escena {i}: voz lista." if i == N_SCENES else f"Narrando la escena {i + 1}...")
                    bar_voice.progress(i / N_SCENES, text=f"Voz {i}/{N_SCENES}")

                voices = generate_narrations(story, on_voice_done)

                for i, sc in enumerate(story["scenes"]):
                    sc["image"] = str(images[i])
                    sc["voice"] = str(voices[i])

                (OUT_DIR / "historia.json").write_text(
                    json.dumps(story, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

                note("¡Libro encuadernado!")
                status.update(label="¡Aventura lista!", state="complete")
                time.sleep(1.2)

            st.session_state.book = {
                "story": story,
                "genre": genre,
                "idea": idea.strip()
            }
            st.session_state.page = -1
            st.session_state.screen = "book"
            st.rerun()

# ============================================================
# PANTALLA 3: LIBRO CON VISUALIZACIÓN RETRO
# ============================================================
elif st.session_state.screen == "book":
    book = st.session_state.book
    story = book["story"]
    genre = book["genre"]
    apply_background(genre)
    scenes = story["scenes"]
    page = st.session_state.page

    if page == -1:
        g = GENRES.get(genre, GENRES["Fantasía"])
        st.markdown(f"""
<div class="cover {g['cover']}">
    <small>EDICION LITERARIA ILUSTRADA</small>
    <div class="sym">{g['symbol']}</div>
    <h1>{html.escape(story['title'])}</h1>
    <p>Genero: {html.escape(genre)}</p>
</div>
""", unsafe_allow_html=True)

        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            if st.button("📖 ABRIR LIBRO", use_container_width=True):
                st.session_state.page = 0
                st.rerun()
    else:
        page = max(0, min(page, N_SCENES - 1))
        sc = scenes[page]

        st.markdown(f"""
<div class="book">
    <div class="paper">
        <h2>{html.escape(sc['title'])}</h2>
        <img src="{img_to_uri(Path(sc['image']))}" />
        <div class="pnum">PAG {page * 2 + 1}</div>
    </div>
    <div class="paper">
        <h2>Escena {page + 1} / {N_SCENES}</h2>
        <p>{html.escape(sc['text'])}</p>
        <div class="pnum">PAG {page * 2 + 2}</div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="audio-label">NARRACIÓN DE VOZ</div>', unsafe_allow_html=True)
        st.audio(sc["voice"], autoplay=st.session_state.narration_auto)

    n1, n2, n3, n4 = st.columns(4)
    with n1:
        if st.button("🏠 PORTADA", use_container_width=True):
            st.session_state.page = -1
            st.rerun()
    with n2:
        if st.button("◀ ANTERIOR", disabled=page <= 0, use_container_width=True):
            st.session_state.page = page - 1
            st.rerun()
    with n3:
        if st.button("SIGUIENTE ▶", disabled=page >= N_SCENES - 1, use_container_width=True):
            st.session_state.page = page + 1
            st.rerun()
    with n4:
        if st.button("➕ NUEVO", use_container_width=True):
            st.session_state.screen = "studio"
            st.session_state.page = -1
            st.rerun()

    st.session_state.narration_auto = st.toggle(
        "Reproducción automática de voz",
        value=st.session_state.narration_auto
    )