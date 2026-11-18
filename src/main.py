import base64
import gc
import html
import io
import logging
import time
import os
import json
import re
import shutil
import threading
import traceback
import zipfile
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
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

LLM_ID = "Qwen/Qwen3-1.7B"
IMG_ID = "Lykon/dreamshaper-8"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"

KOKORO_LANG = "e"
KOKORO_VOICE = "ef_dora"      
KOKORO_SPEED = 0.88
SENTENCE_PAUSE = 0.35
SCENE_END_PAUSE = 0.60

GENRE_VOICE = {
    "Fantasía":        {"mix": {"em_santa": 0.55, "em_alex": 0.45}, "speed": 0.92, "pause": 0.40},
    "Terror":          {"mix": {"em_santa": 0.80, "ef_dora": 0.20}, "speed": 0.80, "pause": 0.55},
    "Misterio":        {"mix": {"em_alex": 0.75, "ef_dora": 0.25},  "speed": 0.88, "pause": 0.50},
    "Aventura":        {"mix": {"em_alex": 1.00},                   "speed": 1.02, "pause": 0.25},
    "Ciencia ficción": {"mix": {"ef_dora": 0.50, "em_alex": 0.50},  "speed": 0.97, "pause": 0.30},
    "Infantil":        {"mix": {"ef_dora": 1.00},                   "speed": 0.95, "pause": 0.40},
}

MUSIC_ID = "facebook/musicgen-small"
MUSIC_MAX_TOKENS = 750    # ~15s de audio a 50 Hz
MUSIC_BG_GAIN = 0.14      # volumen de fondo relativo al RMS de la voz (~-17 dB)

GENRE_MUSIC = {
    "Fantasía":        "cinematic fantasy orchestral soundtrack, soft harp, warm strings, gentle flute, ambient background music, calm, no vocals",
    "Terror":          "dark gothic horror ambient soundtrack, eerie low cello drone, subtle haunting piano, suspenseful atmosphere, no vocals",
    "Misterio":        "noir mystery ambient soundtrack, soft upright bass, muted piano chords, atmospheric rain mood, subtle, no vocals",
    "Aventura":        "epic adventure orchestral soundtrack, warm brass, light acoustic guitar, uplifting strings, cinematic background music, no vocals",
    "Ciencia ficción": "retro sci-fi ambient synth soundtrack, warm analog pads, gentle futuristic arpeggio, atmospheric space mood, no vocals",
    "Infantil":        "gentle whimsical music box and acoustic strings soundtrack, warm playful lullaby melody, soft background music, no vocals",
}

TORCH_THREADS = max(1, min(10, os.cpu_count() or 2))
torch.set_num_threads(TORCH_THREADS)

N_SCENES = 5
IMG_SIZE = 512          
IMG_REFINE_SIZE = 768   
IMG_UPSCALE = 2         


IMG_STEPS = 8
IMG_GUIDANCE = 1.6

IMG_REFINE_STEPS = 8
IMG_REFINE_STRENGTH = 0.5

IMG_STEPS_FALLBACK = 20
IMG_GUIDANCE_FALLBACK = 7.0
IMG_SEED = 1234
IMG_PROMPT_MAX_TOKENS = 77   


SCENE_WORDS_MIN = 150
SCENE_WORDS_MAX = 190
LLM_MAX_TOKENS = 5000   

NEGATIVE_PROMPT = (
    "ugly, deformed, noisy, blurry, distorted, low quality, "
    "bad anatomy, bad proportions, duplicate, watermark, "
    "signature, text, cropped, extra limbs, missing limbs, 3d render, plastic, picture frame, border"
)


STYLE_EN = {
    "pintura digital": "digital painting",
    "acuarela": "watercolor painting",
    "arte conceptual": "concept art",
    "ilustración detallada": "detailed illustration",
    "estilo óleo clásico": "classic oil painting",
}


STYLE_PROMPT = {
    "pintura digital": "digital painting, smooth shading, painterly brush strokes, vibrant colors, artstation",
    "acuarela": "watercolor painting, soft wet washes, paper texture, pigment bleeding, pastel tones, loose brushwork",
    "arte conceptual": "concept art, cinematic matte painting, epic scale, dramatic lighting, wide establishing shot",
    "ilustración detallada": "detailed pen and ink illustration, intricate linework, fine hatching, crisp outlines, detailed book art",
    "estilo óleo clásico": "classic oil painting, thick impasto brush strokes, canvas texture, rich oil brushwork, chiaroscuro lighting",
}

STYLE_NEGATIVE = {
    "pintura digital": "photo, canvas texture",
    "acuarela": "oil painting, photo, thick outlines, heavy shadows",
    "arte conceptual": "cartoon, childish, flat colors",
    "ilustración detallada": "painterly, blurry, photo",
    "estilo óleo clásico": "watercolor, digital art, cartoon, flat colors",
}

GENRES = {
    "Fantasía": {
        "visual": "epic fantasy storybook illustration, rich oil painting, warm golden light, masterpiece",
        "mood": "epic fantasy scene, warm golden light, magical atmosphere",
        "cover": "fantasy",
        "symbol": "⚔️",
    },
    "Terror": {
        "visual": "dark horror illustration, moody painting, deep shadows, eerie fog, cinematic",
        "mood": "dark horror scene, deep shadows, eerie fog, suspenseful atmosphere",
        "cover": "horror",
        "symbol": "💀",
    },
    "Misterio": {
        "visual": "mystery illustration, dramatic shadows, atmospheric fog, cinematic lighting, detailed painting",
        "mood": "mystery scene, dramatic shadows, atmospheric mood, cinematic lighting",
        "cover": "mystery",
        "symbol": "🔍",
    },
    "Aventura": {
        "visual": "adventure storybook illustration, vivid rich colors, dynamic composition, painted style",
        "mood": "adventure scene, vivid rich colors, dynamic composition",
        "cover": "adventure",
        "symbol": "🗺️",
    },
    "Ciencia ficción": {
        "visual": "futuristic science fiction illustration, high-tech sci-fi art, holographic lighting",
        "mood": "futuristic science fiction scene, high-tech sci-fi atmosphere, dramatic lighting",
        "cover": "scifi",
        "symbol": "👾",
    },
    "Infantil": {
        "visual": "children's picture book illustration, soft watercolor, cute, bright pastel colors",
        "mood": "children's picture book scene, cute, bright pastel colors",
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
.paper img.img-step { filter: none; image-rendering: pixelated !important; }
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

/* tira de progreso por escena: botones reales (T = texto, I = ilustracion, V = voz) */
.st-key-strip .stButton > button {
    font-size: 10px !important;
    padding: 12px 6px !important;
    background-color: #1d2b53 !important;      /* azul marino: se leen las letras y los cuadros */
    color: #ffcd75 !important;
    transform: none !important;
    box-shadow:
        inset -3px -3px 0px 0px #0f1838,
        inset  3px  3px 0px 0px #3b5dc9,
        -4px 0 0 0 #000, 4px 0 0 0 #000, 0 -4px 0 0 #000, 0 4px 0 0 #000 !important;
}
/* el texto del boton vive dentro de un <p>: se fuerza para que no salga blanco */
.st-key-strip .stButton > button *,
.st-key-strip .stButton > button p {
    color: inherit !important;
    background: transparent !important;
    font-family: 'Press Start 2P', monospace !important;
    font-size: 10px !important;
}
.st-key-strip .stButton > button:hover:not(:disabled) {
    background-color: #3b5dc9 !important;
    color: #ffffff !important;
    transform: translate(-2px, -2px) !important;
}
/* escena elegida / EN VIVO: ambar con letras oscuras */
.st-key-strip .stButton > button[kind="primary"],
.st-key-strip .stButton > button[data-testid="stBaseButton-primary"] {
    background-color: #e6a23c !important;
    color: #181425 !important;
    box-shadow:
        inset -3px -3px 0px 0px #8b5a1a,
        inset  3px  3px 0px 0px #ffd98a,
        -4px 0 0 0 #000, 4px 0 0 0 #000, 0 -4px 0 0 #000, 0 4px 0 0 #000 !important;
}
.st-key-strip .stButton > button[kind="primary"]:hover:not(:disabled),
.st-key-strip .stButton > button[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
    background-color: #ffcd75 !important;
    color: #181425 !important;
}
.st-key-strip .stButton > button:disabled {
    background-color: #151a33 !important;
    color: #6b7794 !important;
}
/* escena activa / boton principal */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background-color: #e6a23c !important;
    color: #181425 !important;
    box-shadow:
        inset -4px -4px 0px 0px #8b5a1a,
        inset  4px  4px 0px 0px #ffd98a,
        -4px 0 0 0 #000, 4px 0 0 0 #000, 0 -4px 0 0 #000, 0 4px 0 0 #000,
        5px 5px 0px 0px #000 !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background-color: #ffcd75 !important;
    color: #181425 !important;
}
.stButton > button:disabled,
.stDownloadButton > button:disabled {
    opacity: .45 !important;
    cursor: not-allowed !important;
    transform: none !important;
}
.wip-legend {
    text-align: center; font-family: 'VT323', monospace;
    font-size: 18px; color: #8b9bb4; margin-bottom: 6px;
}

/* ============================================================
   BOTONES RETRO 3D CON RELEVE Y PRESION LATERAL (PIXEL STYLE)
============================================================ */
.stButton > button,
.stDownloadButton > button {
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

.stButton > button,
.stDownloadButton > button:hover {
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
.stButton > button,
.stDownloadButton > button:active {
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

/* SELECTBOX: caja y lista desplegable en azul marino retro (sin blanco).
   Se pinta TODO lo que hay dentro de la caja y de la lista (Streamlit anida varios
   contenedores y la lista se dibuja en un portal fuera de la app). */
:root { color-scheme: dark; }
div[data-testid="stSelectbox"] *,
div[data-baseweb="select"] *,
div[data-baseweb="popover"],
div[data-baseweb="popover"] *,
div[data-baseweb="menu"],
div[data-baseweb="menu"] *,
[data-testid="stSelectboxVirtualDropdown"],
[data-testid="stSelectboxVirtualDropdown"] *,
ul[role="listbox"],
ul[role="listbox"] * {
    background-color: #1d2b53 !important;
    color: #ffcd75 !important;
    border-radius: 0px !important;
    font-family: 'VT323', monospace !important;
    font-size: 22px !important;
}
div[data-baseweb="select"] svg,
div[data-baseweb="select"] svg * {
    fill: #ffcd75 !important;
    background-color: transparent !important;
}
div[data-baseweb="popover"] > div {
    border: 3px solid #3b5dc9 !important;
    box-shadow: 4px 4px 0px 0px #000 !important;
}
li[role="option"]:hover, li[role="option"]:hover *,
div[role="option"]:hover, div[role="option"]:hover *,
li[role="option"][aria-selected="true"], li[role="option"][aria-selected="true"] *,
div[role="option"][aria-selected="true"], div[role="option"][aria-selected="true"] * {
    background-color: #3b5dc9 !important;
    color: #ffffff !important;
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
    # --- Misterio: noir azul, luna palida, niebla y ciudad ---
    "Misterio": (
        ".stApp{background:radial-gradient(circle at 78% 14%,#dfe8ff 0 20px,#dfe8ff22 21px 70px,transparent 71px),"
        "linear-gradient(180deg,#05070f 0%,#0c1630 60%,#16254a 100%)!important;}"
        ".stApp::before{background:radial-gradient(ellipse 40% 10% at 25% 78%,#9fb4e22e,transparent),"
        "radial-gradient(ellipse 45% 9% at 72% 66%,#9fb4e226,transparent),"
        "radial-gradient(ellipse 50% 12% at 45% 88%,#9fb4e23a,transparent);"
        "animation:bg-drift 18s ease-in-out infinite alternate;}"
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


# ============================================================
# ANIMACIONES PIXEL DEL FONDO (una escena distinta por genero)
# Sprites ASCII -> SVG <rect>; se mueven solo con CSS (sin JS).
# ============================================================
def _rects(rows, pal):
    out = []
    for y, row in enumerate(rows):
        x = 0
        while x < len(row):
            c = row[x]
            if c in ". ":
                x += 1
                continue
            x0 = x
            while x < len(row) and row[x] == c:
                x += 1
            out.append(f'<rect x="{x0}" y="{y}" width="{x - x0}" height="1" fill="{pal[c]}"/>')
    return "".join(out)


def _svg(frames, pal, px=4, anim=0.5):
    """frames: lista de bitmaps. Con 2 frames se alternan (aleteo) sin interpolar."""
    w = max(len(r) for r in frames[0])
    h = len(frames[0])
    inner = ""
    for k, rows in enumerate(frames):
        cls = f' class="px-f{k}" style="animation-duration:{anim}s"' if len(frames) > 1 else ""
        inner += f"<g{cls}>{_rects(rows, pal)}</g>"
    return (f'<svg width="{w * px}" height="{h * px}" viewBox="0 0 {w} {h}" '
            f'shape-rendering="crispEdges" xmlns="http://www.w3.org/2000/svg">{inner}</svg>')


SPR = {}

SPR["dragon"] = ([
    [   # alas arriba
        "..........#...........",
        ".........###..........",
        "........#####.....#.#..",
        ".......#######...#####.",
        "......##########.######",
        "...#...###########.####",
        "..###.############.####",
        ".#################GGGGG",
        "###################...",
        ".####.###########.....",
        "..#....##...##........",
    ],
    [   # alas abajo
        "......................",
        "......................",
        "......................",
        "..................#.#.",
        "...#.............#####",
        "..###...##########.###",
        ".#####.###########.###",
        "###################GGG",
        ".#################....",
        "..###.####..######....",
        "...#...##.....##......",
    ],
], {"#": "#6fd69a", "G": "#ffcd75"})

SPR["star4"] = ([[
    "..#..",
    "..#..",
    "#####",
    "..#..",
    "..#..",
]], {"#": "#fff4c2"})

SPR["bat"] = ([
    [
        "#.............#",
        "##...#...#...##",
        "###..##.##..###",
        "####.#####.####",
        "#####.###.#####",
        ".###..###..###.",
        "..#....#....#..",
    ],
    [
        "......#...#....",
        "....#######....",
        "..###########..",
        ".#####.#.#####.",
        "###...#####...#",
        "##.....###.....",
        "#.......#......",
    ],
], {"#": "#7a4a8a"})

SPR["ghost"] = ([[
    "..######..",
    ".########.",
    "##########",
    "##kk##kk##",
    "##kk##kk##",
    "##########",
    "##########",
    "##########",
    "##########",
    "##.####.##",
    "#...##...#",
]], {"#": "#c9d1e8", "k": "#181425"})

SPR["crow"] = ([
    [
        "......#.....",
        ".....##.....",
        "....###.....",
        "..#######...",
        "###########y",
        ".#########..",
        "...##.##....",
        "............",
    ],
    [
        "............",
        "............",
        "..#.........",
        ".###.#####..",
        "###########y",
        "..#########.",
        "....#####...",
        "....##.##...",
    ],
], {"#": "#3a4a7a", "y": "#ffcd75"})

SPR["qmark"] = ([[
    ".###.",
    "#...#",
    "....#",
    "...#.",
    "..#..",
    ".....",
    "..#..",
]], {"#": "#9fb4e2"})

SPR["lupa"] = ([[
    "..#####....",
    ".#.....#...",
    "#..ww...#..",
    "#..w....#..",
    "#.......#..",
    ".#.....#...",
    "..#####.b..",
    "........bb.",
    ".........bb",
]], {"#": "#dfe8ff", "w": "#ffffff", "b": "#a56a3a"})

SPR["gull"] = ([
    [
        "##.......##",
        ".##.....##.",
        "..##...##..",
        "...#####...",
    ],
    [
        "...#####...",
        "..##...##..",
        ".##.....##.",
        "##.......##",
    ],
], {"#": "#fff1e0"})

SPR["balloon_hot"] = ([[
    "...rrrr...",
    ".rrrrrrrr.",
    "rrryyyyrrr",
    "rrryyyyrrr",
    "rrryyyyrrr",
    "rrryyyyrrr",
    ".rrryyrrr.",
    "..rryyrr..",
    "...rrrr...",
    "...b..b...",
    "...b..b...",
    "...bbbb...",
    "...bbbb...",
]], {"r": "#d6483a", "y": "#ffcd75", "b": "#5a3a2a"})

SPR["ufo"] = ([
    [
        "......cccccc......",
        ".....cccccccc.....",
        "..gggggggggggggg..",
        ".gggggggggggggggg.",
        "..gyggyggyggyggg..",
        "....gg......gg....",
    ],
    [
        "......cccccc......",
        ".....cccccccc.....",
        "..gggggggggggggg..",
        ".gggggggggggggggg.",
        "..grggrggrggrggg..",
        "....gg......gg....",
    ],
], {"c": "#7df9ff", "g": "#8b9bb4", "y": "#ffe66d", "r": "#ff5d8f"})

SPR["invader"] = ([
    [
        "..#.....#..",
        "...#...#...",
        "..#######..",
        ".##.###.##.",
        "###########",
        "#.#######.#",
        "#.#.....#.#",
        "...##.##...",
    ],
    [
        "..#.....#..",
        "#..#...#..#",
        "#.#######.#",
        "###.###.###",
        "###########",
        ".#########.",
        "..#.....#..",
        ".#.......#.",
    ],
], {"#": "#63c74d"})

SPR["balloon"] = ([[
    "..rrr..",
    ".rrrrr.",
    "rrwrrrr",
    "rrwrrrr",
    "rrrrrrr",
    ".rrrrr.",
    "..rrr..",
    "...r...",
    "...b...",
    "..b....",
    "...b...",
]], {"r": "#ff7ba3", "w": "#ffffff", "b": "#c0cbdc"})

SPR["butterfly"] = ([
    [
        "##.....##",
        "###...###",
        "####b####",
        ".###b###.",
        "####b####",
        "###...###",
        "##.....##",
    ],
    [
        "....b....",
        "...###...",
        "..##b##..",
        "..##b##..",
        "..##b##..",
        "...###...",
        "....b....",
    ],
], {"#": "#ffb3d9", "b": "#181425"})

SPR["book"] = ([[
    "..########..",
    ".#bbbb#bbbb#.",
    ".#bbbb#bbbb#.",
    ".#bbbb#bbbb#.",
    ".#wwww#wwww#.",
    "..########..",
]], {"#": "#5a3a2a", "b": "#f4f4e8", "w": "#c0cbdc"})


def _sprite(name, px, **pal_over):
    frames, pal = SPR[name]
    return _svg(frames, {**pal, **pal_over}, px)


def _fly(name, px, y, dur, delay, rtl=False, bob=None, opacity=None, **pal_over):
    """Sprite que cruza la pantalla (izq->der, o der->izq si rtl) en bucle."""
    svg = _sprite(name, px, **pal_over)
    if bob:
        svg = f'<div style="animation:px-bob {bob}s steps(2) infinite">{svg}</div>'
    op = f"opacity:{opacity};" if opacity else ""
    return (f'<div class="sp{" flip" if rtl else ""}" style="--y:{y}%;{op}'
            f'animation:{"px-rtl" if rtl else "px-ltr"} {dur}s linear infinite;'
            f'animation-delay:-{delay}s">{svg}</div>')


def _still(name, px, x, y, bob=None, delay=0.0, twinkle=False, **pal_over):
    """Sprite fijo que se mece (bob) o titila (twinkle)."""
    svg = _sprite(name, px, **pal_over)
    if bob:
        svg = f'<div style="animation:px-bob {bob}s steps(2) infinite">{svg}</div>'
    return (f'<div class="sp{" tw" if twinkle else ""}" style="--x:{x}%;--y:{y}%;'
            f'animation-delay:-{delay}s">{svg}</div>')


def _rise_sprite(name, px, x, dur, delay, **pal_over):
    """Sprite que sube desde abajo (globos)."""
    return (f'<div class="sp rise" style="--x:{x}%;--d:{dur}s;--dl:-{delay}s">'
            f'{_sprite(name, px, **pal_over)}</div>')


def _particles(n, colors, seed, fall=False, size=(5, 8), dur=(12, 24), sway=(-30, 30)):
    """Cuadraditos pixel que suben (brasas, orbes) o caen (confeti)."""
    import random
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(
            f'<i class="pt{" fall" if fall else ""}" style="--x:{rnd.randint(2, 97)}%;'
            f'--s:{rnd.choice(range(size[0], size[1] + 1, 1))}px;--c:{rnd.choice(colors)};'
            f'--d:{rnd.uniform(*dur):.1f}s;--dl:-{rnd.uniform(0, dur[1]):.1f}s;'
            f'--sw:{rnd.randint(*sway)}px"></i>'
        )
    return "".join(out)


def _eyes(x, y, delay):
    return f'<i class="eyes" style="--x:{x}%;--y:{y}%;animation-delay:-{delay}s"></i>'


def _shoot(x, y, dur, delay):
    return f'<i class="shoot" style="--x:{x}%;--y:{y}%;--d:{dur}s;--dl:-{delay}s"></i>'


def _march(y):
    row = "".join(_sprite("invader", 3) for _ in range(5))
    return f'<div class="sp march" style="--y:{y}%">{row}</div>'


def _build_px_layers() -> dict:
    L = {}
    # --- General: biblioteca nocturna (libros flotando, motas doradas, estrella fugaz) ---
    L["_general"] = (
        _still("book", 4, 7, 24, bob=2.4) + _still("book", 3, 88, 34, bob=3.0, delay=1)
        + _still("book", 3, 50, 12, bob=2.8, delay=2)
        + _particles(14, ["#ffcd75", "#fff4c2", "#f7a8d8"], seed=1)
        + _shoot(10, 8, 11, 3)
    )
    # --- Fantasia: dragones, destellos y orbes magicos ---
    L["Fantasía"] = (
        _fly("dragon", 5, 12, 48, 14) + _fly("dragon", 3, 28, 75, 30, rtl=True, opacity=0.7)
        + "".join(_still("star4", 3, x, y, twinkle=True, delay=d)
                  for x, y, d in ((10, 12, 0), (24, 32, 1.1), (60, 8, 0.6), (92, 40, 1.7)))
        + _particles(16, ["#ffcd75", "#f7a8d8", "#a5e06b", "#fff4c2"], seed=2)
    )
    # --- Terror: murcielagos, fantasma, ojos en la oscuridad y brasas ---
    L["Terror"] = (
        _fly("bat", 4, 10, 30, 6, bob=0.8) + _fly("bat", 3, 24, 40, 22, rtl=True, bob=1.0)
        + _fly("bat", 3, 38, 52, 35, bob=0.7)
        + _fly("ghost", 5, 48, 70, 20, rtl=True, opacity=0.5, bob=2.4)
        + _eyes(5, 62, 0) + _eyes(91, 54, 2.3) + _eyes(14, 74, 4.1) + _eyes(80, 70, 1.2)
        + _particles(8, ["#c9303a", "#ff6b4a"], seed=3, size=(4, 6), dur=(16, 28))
    )
    # --- Misterio: cuervos, lupa que busca, signos de pregunta y polvo (SIN lluvia) ---
    L["Misterio"] = (
        _fly("crow", 4, 16, 36, 9) + _fly("crow", 3, 28, 54, 30, rtl=True)
        + '<div class="sp search" style="--x:6%;--y:56%">' + _sprite("lupa", 5) + "</div>"
        + "".join(_still("qmark", 4, x, y, twinkle=True, delay=d)
                 for x, y, d in ((12, 30, 0), (30, 20, 0.8), (70, 26, 1.5), (88, 44, 2.1), (48, 40, 1.1)))
        + _particles(8, ["#9fb4e2", "#dfe8ff"], seed=4, size=(4, 5), dur=(20, 34))
    )
    # --- Aventura: gaviotas, globo aerostatico y chispas de atardecer ---
    L["Aventura"] = (
        _fly("gull", 4, 14, 26, 4, bob=0.6) + _fly("gull", 3, 22, 34, 18, bob=0.7)
        + _fly("gull", 3, 30, 42, 28, rtl=True, bob=0.6)
        + _fly("balloon_hot", 5, 20, 95, 40, rtl=True, bob=3.0)
        + _particles(10, ["#ffe08a", "#ffb04a", "#fff1e0"], seed=5, size=(4, 6))
    )
    # --- Ciencia ficcion: OVNIs, invasores marchando y estrellas fugaces ---
    L["Ciencia ficción"] = (
        _fly("ufo", 4, 16, 28, 5) + _fly("ufo", 3, 32, 44, 20, rtl=True)
        + _march(6)
        + _shoot(15, 10, 9, 1) + _shoot(55, 6, 13, 6) + _shoot(80, 22, 11, 9)
    )
    # --- Infantil: globos que suben, mariposas y confeti ---
    L["Infantil"] = (
        "".join(_rise_sprite("balloon", 4, x, d, dl, r=c)
                for x, d, dl, c in ((8, 26, 3, "#ff7ba3"), (24, 32, 14, "#ffcd75"),
                                    (46, 28, 20, "#73eff7"), (66, 34, 8, "#a5e06b"),
                                    (84, 30, 25, "#b45cff"), (94, 27, 11, "#ff7ba3")))
        + _fly("butterfly", 4, 20, 34, 6, bob=0.5) + _fly("butterfly", 3, 36, 46, 22, rtl=True, bob=0.6)
        + _fly("butterfly", 3, 50, 58, 40, bob=0.5)
        + "".join(_still("star4", 3, x, y, twinkle=True, delay=d, **{"#": c})
                  for x, y, d, c in ((12, 10, 0, "#fff3b0"), (38, 24, 0.9, "#ffb3d9"),
                                     (62, 8, 1.6, "#b3f0ff"), (90, 30, 0.4, "#fff3b0")))
        + _particles(18, ["#ffb3d9", "#b3f0ff", "#fff3b0", "#ffffff"], seed=6, fall=True, size=(4, 6), dur=(14, 26))
    )
    return L


_PX_HTML = _build_px_layers()

_PX_CSS = """
.px-bg{position:fixed;inset:0;overflow:hidden;pointer-events:none;z-index:-1;}
.px-bg .sp{position:absolute;left:var(--x,0);top:var(--y,20%);will-change:transform;}
.px-bg .sp svg{display:block;image-rendering:pixelated;}
.px-bg .sp.flip svg{transform:scaleX(-1);}
@keyframes px-fa{0%{opacity:1}50%{opacity:0}}
@keyframes px-fb{0%{opacity:0}50%{opacity:1}}
.px-bg .px-f0{animation:px-fa .5s steps(1) infinite;}
.px-bg .px-f1{animation:px-fb .5s steps(1) infinite;}
@keyframes px-ltr{from{transform:translateX(-20vw)}to{transform:translateX(112vw)}}
@keyframes px-rtl{from{transform:translateX(112vw)}to{transform:translateX(-20vw)}}
@keyframes px-bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
@keyframes px-tw{0%,100%{opacity:.2}50%{opacity:1}}
.px-bg .sp.tw{animation:px-tw 2.6s steps(2) infinite;}
@keyframes px-riseS{from{transform:translate(0,0)}to{transform:translate(26px,-125vh)}}
.px-bg .sp.rise{top:auto;bottom:-110px;opacity:.9;animation:px-riseS var(--d,28s) linear infinite;animation-delay:var(--dl,0s);}
@keyframes px-search{from{transform:translate(0,0) rotate(-8deg)}50%{transform:translate(22vw,-6vh) rotate(6deg)}to{transform:translate(44vw,2vh) rotate(-4deg)}}
.px-bg .sp.search{opacity:.85;animation:px-search 26s ease-in-out infinite alternate;}
@keyframes px-march{from{transform:translateX(2vw)}to{transform:translateX(52vw)}}
.px-bg .sp.march{display:flex;gap:18px;opacity:.8;animation:px-march 9s steps(14) infinite alternate;}
.px-bg .sp.march .px-f0,.px-bg .sp.march .px-f1{animation-duration:1s !important;}
.px-bg i.pt{position:absolute;left:var(--x);bottom:-12px;width:var(--s,6px);height:var(--s,6px);background:var(--c,#fff);opacity:0;animation:px-rise var(--d,16s) linear infinite;animation-delay:var(--dl,0s);}
.px-bg i.pt.fall{bottom:auto;top:-12px;animation-name:px-fall;}
@keyframes px-rise{0%{transform:translate(0,0);opacity:0}10%{opacity:.9}85%{opacity:.9}100%{transform:translate(var(--sw,20px),-105vh);opacity:0}}
@keyframes px-fall{0%{transform:translate(0,0);opacity:0}10%{opacity:.9}90%{opacity:.9}100%{transform:translate(var(--sw,20px),105vh);opacity:0}}
.px-bg i.eyes{position:absolute;left:var(--x);top:var(--y);width:6px;height:6px;background:#ff3b3b;box-shadow:18px 0 #ff3b3b;animation:px-blink 7s steps(1) infinite;}
@keyframes px-blink{0%,88%,100%{opacity:1}92%{opacity:0}}
.px-bg i.shoot{position:absolute;left:var(--x);top:var(--y);width:4px;height:4px;background:#fff;box-shadow:-4px 0 #ffffffcc,-8px 0 #ffffff99,-12px 0 #ffffff66,-16px 0 #ffffff33;opacity:0;animation:px-shoot var(--d,10s) linear infinite;animation-delay:var(--dl,0s);}
@keyframes px-shoot{0%{opacity:0;transform:rotate(28deg) translateX(0)}2%{opacity:1}10%{opacity:1;transform:rotate(28deg) translateX(46vw)}12%,100%{opacity:0;transform:rotate(28deg) translateX(46vw)}}
@media (prefers-reduced-motion:reduce){.px-bg{display:none;}}
"""


def apply_background(genre=None):
    """Inyecta el fondo CSS + la capa de animaciones pixel del genero (general si genre es None)."""
    key = genre if genre in _BG else "_general"
    css = _BG[key] + _BG_BASE + _PX_CSS
    st.markdown(f"<style>{css}</style><div class='px-bg' aria-hidden='true'>{_PX_HTML[key]}</div>",
                unsafe_allow_html=True)


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
    if USE_CUDA:
        # Hibrido GPU + CPU: prioriza VRAM disponible (~2.9 GB para RTX 2050 4 GB)
        # y descarga el resto de capas a RAM/CPU automaticamente via accelerate.
        model = AutoModelForCausalLM.from_pretrained(
            LLM_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "2900MiB", "cpu": "16GiB"},
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            LLM_ID,
            torch_dtype=torch.float32,
        ).to("cpu")
    model.eval()
    return tok, model


@st.cache_resource(show_spinner=False)
def load_image_pipe():
    from diffusers import AutoPipelineForText2Image, LCMScheduler
    dtype = torch.float16 if USE_CUDA else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        IMG_ID,
        torch_dtype=dtype,
        use_safetensors=True
    )

    # LCM-LoRA acelera la generación (pocos pasos).
    # Requiere la librería "peft".
    try:
        pipe.load_lora_weights(LCM_LORA_ID)
        pipe.fuse_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    except Exception:
        # Recarga limpia (sin LoRA a medias) y scheduler original
        pipe = AutoPipelineForText2Image.from_pretrained(
            IMG_ID,
            torch_dtype=dtype,
            use_safetensors=True
        )

    if USE_CUDA:
        # Offload CPU<->GPU: mantiene submodelos inactivos en RAM y mueve a VRAM solo el activo.
        pipe.enable_model_cpu_offload()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
    else:
        pipe.to("cpu")

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
    # Kokoro corre en CPU para ejecutarse en paralelo mientras la GPU genera las imagenes.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        pipe = KPipeline(lang_code=KOKORO_LANG, device="cpu")
        if getattr(pipe, "model", None) is not None:
            pipe.model.float()
        return pipe
    finally:
        torch.set_default_dtype(prev_dtype)


@st.cache_resource(show_spinner=False)
def load_musicgen():
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    dtype = torch.float16 if USE_CUDA else torch.float32
    processor = AutoProcessor.from_pretrained(MUSIC_ID)
    model = MusicgenForConditionalGeneration.from_pretrained(
        MUSIC_ID,
        torch_dtype=dtype,
    ).to(DEVICE)
    model.eval()
    return processor, model


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


def _repair_truncated_json(text: str) -> str:
    """Cierra comillas, corchetes o llaves abiertas si el LLM se truncó."""
    in_str = False
    escape = False
    stack = []
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack and stack[-1] == ch:
                stack.pop()
    repaired = text.rstrip().rstrip(",")
    if in_str:
        repaired += '"'
    repaired += "".join(reversed(stack))
    return re.sub(r",\s*([}\]])", r"\1", repaired)


def _regex_extract_outline(text: str):
    """Plan B si json.loads falla (p.ej. comillas sin escapar dentro de un texto)."""
    def _field(name: str, src: str, default: str = "") -> str:
        m = re.search(rf'"{name}"\s*:\s*"(.*?)"\s*(?:,|\n|\}})', src, re.DOTALL)
        return m.group(1).replace('\\"', '"').strip() if m else default

    scenes_pos = text.find('"scenes"')
    head = text[:scenes_pos] if scenes_pos != -1 else text
    body = text[scenes_pos:] if scenes_pos != -1 else text

    pattern = re.compile(
        r'\{\s*"title"\s*:\s*"(.*?)"\s*,\s*"action"\s*:\s*"(.*?)"\s*,\s*"summary"\s*:\s*"(.*?)"\s*\}',
        re.DOTALL,
    )
    scenes = []
    for t, a, s in pattern.findall(body):
        t, a, s = t.strip(), a.strip(), s.strip()
        if t and a and s:
            scenes.append({"title": t, "action": a, "summary": s})
    if not scenes:
        return None
    return {
        "title": _field("title", head, scenes[0]["title"]),
        "character": _field("character", head, "a focused human protagonist with distinctive features"),
        "setting": _field("setting", head, "a mysterious atmospheric location"),
        "scenes": scenes,
    }


def parse_outline(raw: str, allow_partial: bool = False):
    """JSON tolerante + reparación de truncamientos + extracción por regex."""
    text = re.sub(r"```(?:json)?", "", raw)
    first = text.find("{")
    if first == -1:
        return None
    last = text.rfind("}")
    sliced = text[first:last + 1] if last > first else text[first:]
    candidate = re.sub(r",\s*([}\]])", r"\1", sliced)

    data = None
    for attempt in (candidate, _repair_truncated_json(text[first:])):
        try:
            data = json.loads(attempt, strict=False)
            break
        except Exception:
            continue

    if not isinstance(data, dict):
        data = _regex_extract_outline(text[first:])
    if not isinstance(data, dict):
        return None

    raw_scenes = data.get("scenes")
    if not isinstance(raw_scenes, list):
        return None

    scenes = []
    for s in raw_scenes:
        if not isinstance(s, dict):
            continue
        t = str(s.get("title", "")).strip()
        a = str(s.get("action", "")).strip()
        sm = str(s.get("summary", "")).strip()
        if t and a and sm:
            scenes.append({"title": t, "action": a, "summary": sm})

    if len(scenes) < N_SCENES:
        if not allow_partial or len(scenes) < 3:
            return None
        while len(scenes) < N_SCENES:
            idx = len(scenes) + 1
            scenes.append({
                "title": f"Escena {idx}",
                "action": scenes[-1]["action"],
                "summary": f"Continuación directa tras {scenes[-1]['title'].lower()}, avanzando hacia el desenlace.",
            })

    data["scenes"] = scenes[:N_SCENES]
    data.setdefault("title", str(scenes[0].get("title", "Relato")))
    data.setdefault("character", "a focused human protagonist with distinctive features")
    data.setdefault("setting", "a mysterious atmospheric location")
    return data


def _clean_prose(text: str) -> str:
    """Limpia la salida de prosa: sin comillas envolventes ni etiquetas."""
    text = text.strip().strip('"').strip()
    text = re.sub(r"```(?:\w+)?", "", text)
    return " ".join(text.split())


def _make_streamer(tok, cb, every: int = 3):
    """Streamer de transformers: avisa con el texto parcial cada `every` tokens."""
    from transformers.generation.streamers import BaseStreamer

    class _ProseStreamer(BaseStreamer):
        def __init__(self):
            self.ids, self.n, self.started = [], 0, False

        def put(self, value):
            if not self.started:      # la primera llamada trae el prompt, no se muestra
                self.started = True
                return
            self.ids.extend(value.reshape(-1).tolist())
            self.n += 1
            if self.n % every == 0:
                cb(tok.decode(self.ids, skip_special_tokens=True))

        def end(self):
            if self.ids:
                cb(tok.decode(self.ids, skip_special_tokens=True))

    return _ProseStreamer()


def _needs_english(text: str) -> bool:
    """Detecta si un campo visual (que debe ir en inglés para CLIP) salió en español."""
    if re.search(r"[áéíóúñü¿¡]", text.lower()):
        return True
    words = set(re.findall(r"[a-z]+", text.lower()))
    es_markers = {"el", "la", "los", "las", "un", "una", "con", "para", "por", "que", "su", "del", "en", "y"}
    return len(words & es_markers) >= 2


def _ensure_english_visual(tok, model, outline: dict) -> None:
    """Si el LLM dejó character/setting/action en español, los traduce al inglés en 1 llamada."""
    fields = [outline.get("character", ""), outline.get("setting", "")] + [
        s.get("action", "") for s in outline["scenes"]
    ]
    if not any(_needs_english(f) for f in fields):
        return

    numbered = "\n".join(f"{idx + 1}. {val}" for idx, val in enumerate(fields))
    msgs = [
        {"role": "system", "content": "Translate each numbered line into a concise visual English phrase (max 14 words each) for Stable Diffusion. Keep the exact numbering 1..7 and output only the translated lines."},
        {"role": "user", "content": numbered},
    ]
    try:
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tok([prompt], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=280, do_sample=False)
        raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = {}
        for ln in raw.splitlines():
            m = re.match(r"^\s*(\d+)[.)]\s*(.+)$", ln.strip())
            if m:
                parsed[int(m.group(1))] = m.group(2).strip().strip('"')
        if 1 in parsed:
            outline["character"] = parsed[1]
        if 2 in parsed:
            outline["setting"] = parsed[2]
        for idx, sc in enumerate(outline["scenes"]):
            if (idx + 3) in parsed:
                sc["action"] = parsed[idx + 3]
    except Exception:
        pass


def generate_story(idea: str, genre: str, on_event=None) -> tuple[dict, bool]:
    """on_event(evento, i, dato) permite mostrar el libro formandose:
    'outline' (dato = escaleta), 'scene_start' (dato = titulo), 'scene_text' (dato = texto parcial
    que se va escribiendo), 'scene_done' (dato = escena)."""
    import traceback

    tok, model = load_llm()   # en caché: no se recarga en cada generación

    # ============ FASE 1: ESCALETA (una sola llamada, JSON pequeño) ============
    outline = None
    debug_lines = []
    raw_attempts = []

    for temp in (0.5, 0.7, 0.3):
        try:
            prompt = tok.apply_chat_template(
                build_outline_messages(idea, genre),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ) + JSON_PREFILL

            inputs = tok([prompt], return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=1600,     # margen extra para evitar JSON truncado
                    do_sample=True,
                    temperature=temp,
                    top_p=0.9,
                    repetition_penalty=1.05
                )

            raw = JSON_PREFILL + tok.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            raw_attempts.append(raw)
            debug_lines.append(f"===== ESCALETA (temp={temp}) =====\n{raw}\n")
            outline = parse_outline(raw)
            if outline:
                break
        except Exception:
            debug_lines.append(f"===== ESCALETA ERROR =====\n{traceback.format_exc()}\n")
            continue

    if not outline:
        for raw in raw_attempts:
            outline = parse_outline(raw, allow_partial=True)
            if outline:
                break

    if not outline:
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "debug_llm_raw.txt").write_text("\n".join(debug_lines), encoding="utf-8")
        except Exception:
            pass
        return fallback_story(idea, genre), True

    _ensure_english_visual(tok, model, outline)

    if on_event:
        on_event("outline", None, outline)

    # ============ FASE 2: PROSA ESCENA POR ESCENA (chunking) ============
    scenes = []
    prev_ending = ""

    for i in range(N_SCENES):
        text = None
        best_candidate = ""
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

                inputs = tok([prompt], return_tensors="pt").to(model.device)
                streamer = None
                if on_event:
                    streamer = _make_streamer(
                        tok, lambda partial, i=i: on_event("scene_text", i, _clean_prose(partial)))
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=700,   # una sola escena: salida corta
                        do_sample=True,
                        temperature=temp,
                        top_p=0.9,
                        repetition_penalty=1.05,
                        streamer=streamer
                    )

                candidate = _clean_prose(tok.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ))
                if len(candidate.split()) > len(best_candidate.split()):
                    best_candidate = candidate

                # Aceptamos si se acerca al mínimo pedido (80%)
                if len(candidate.split()) >= int(SCENE_WORDS_MIN * 0.8):
                    text = candidate
                    break
                debug_lines.append(f"===== ESCENA {i+1} (temp={temp}) CORTA ({len(candidate.split())} palabras) =====\n{candidate}\n")
            except Exception:
                debug_lines.append(f"===== ESCENA {i+1} ERROR =====\n{traceback.format_exc()}\n")
                continue

        if text is None and len(best_candidate.split()) >= 50:
            # En lugar de descartar toda la historia, ampliamos o aceptamos el mejor intento.
            try:
                ext_msgs = [
                    {"role": "system", "content": "Eres un escritor literario en español. Continúa el párrafo de forma fluida con 60 a 80 palabras más, sin repetir lo anterior."},
                    {"role": "user", "content": f"Texto actual:\n{best_candidate}\n\nContinúa directamente desde la última frase:"},
                ]
                ext_prompt = tok.apply_chat_template(
                    ext_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                ext_inputs = tok([ext_prompt], return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    ext_out = model.generate(
                        **ext_inputs, max_new_tokens=220, do_sample=True, temperature=0.7, top_p=0.9
                    )
                extra = _clean_prose(tok.decode(
                    ext_out[0][ext_inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ))
                text = f"{best_candidate} {extra}".strip() if extra else best_candidate
            except Exception:
                text = best_candidate

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

    if USE_CUDA:
        # Libera la VRAM ocupada por el LLM antes de cargar Stable Diffusion (4 GB VRAM).
        del model
        load_llm.clear()
        free_memory()

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


def latents_to_preview(latents, size: int = 512):
    """Vista previa casi gratis de los latents (SD 1.5, 4 canales) -> RGB.
    Es una aproximación lineal (sin pasar por el VAE): colores aproximados, y como
    el latent es 64x64 se ve literalmente pixel por pixel."""
    from PIL import Image
    w = torch.tensor([[60, -60, 25, -70],
                      [60,  -5, 15, -50],
                      [60,  10, -5, -35]], dtype=torch.float32).T      # (4, 3)
    b = torch.tensor([150, 140, 130], dtype=torch.float32)
    lat = latents[0].detach().float().cpu()                            # (4, h, w)
    rgb = torch.einsum("lxy,lr->rxy", lat, w) + b[:, None, None]
    arr = rgb.clamp(0, 255).byte().numpy().transpose(1, 2, 0)
    return Image.fromarray(arr).resize((size, size), resample=Image.NEAREST)


def generate_scene_images(story: dict, genre: str, style: str, progress_cb,
                          preview_cb=None, step_cb=None) -> list[Path]:
    import traceback
    from PIL import Image
    from diffusers import LCMScheduler

    pipe = load_image_pipe()          
    img2img = load_img2img_pipe()     

    # <-- CAMBIO A: parámetros según el scheduler realmente activo
    lcm_ok = isinstance(pipe.scheduler, LCMScheduler)
    steps = IMG_STEPS if lcm_ok else IMG_STEPS_FALLBACK
    refine_steps = IMG_REFINE_STEPS if lcm_ok else IMG_STEPS_FALLBACK
    guidance = IMG_GUIDANCE if lcm_ok else IMG_GUIDANCE_FALLBACK
    negative = (NEGATIVE_PROMPT + ", " + STYLE_NEGATIVE.get(style, "")) if guidance > 1.0 else None

    mood = GENRES[genre].get("mood", GENRES[genre]["visual"])
    style_prompt = STYLE_PROMPT.get(style, STYLE_EN.get(style, "digital painting"))
    paths = []
    debug_lines = []

    for i, scene in enumerate(story["scenes"], 1):
        
        style_short = STYLE_EN.get(style, "digital painting")
        prompt = build_image_prompt(pipe, [
            style_short,             # etiqueta breve de estilo para no gastar los 77 tokens
            scene["action"],
            mood,
            story["setting"],
            story["character"],
            style_prompt,
            "highly detailed",
        ])

        generator = torch.Generator(device="cpu").manual_seed(IMG_SEED + i * 150)

        # --- FASE 1: generación base con LCM (con vista previa en cada paso) ---
        base_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=IMG_SIZE,
            width=IMG_SIZE,
            generator=generator,
        )

        if step_cb:
            step_cb(i, 0, steps, None)       # lienzo en blanco de esta escena

            def _on_step(_pipe, step, _t, kw, _i=i):
                step_cb(_i, step + 1, steps, latents_to_preview(kw["latents"], IMG_SIZE))
                return kw

            base_kwargs.update(callback_on_step_end=_on_step,
                               callback_on_step_end_tensor_inputs=["latents"])

        with torch.inference_mode():
            try:
                base = pipe(**base_kwargs).images[0]
            except TypeError as e:
                # diffusers antiguo sin callback_on_step_end: se genera sin vista previa
                if "callback_on_step_end" not in str(e):
                    raise
                base_kwargs.pop("callback_on_step_end", None)
                base_kwargs.pop("callback_on_step_end_tensor_inputs", None)
                generator = torch.Generator(device="cpu").manual_seed(IMG_SEED + i * 150)
                base_kwargs["generator"] = generator
                base = pipe(**base_kwargs).images[0]

        if preview_cb:
            preview_cb(i, base)   # boceto base 512 para mostrarlo en el libro

        # --- FASE 2: refinado img2img a mayor resolución.
        try:
            base_up = base.resize((IMG_REFINE_SIZE, IMG_REFINE_SIZE), resample=Image.LANCZOS)
            ref_generator = torch.Generator(device="cpu").manual_seed(IMG_SEED + i * 150)
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

    free_memory()
    return paths

# ============================================================
# 3. VOZ (KOKORO)
# ============================================================

def _resolve_voice(pipeline, mix: dict):
    """Mezcla voces de Kokoro con pesos (p. ej. 55% em_santa + 45% em_alex).
    Si algo falla devuelve la voz base para que la narración nunca se caiga."""
    try:
        total = sum(mix.values())
        voice = sum(pipeline.load_voice(name) * w for name, w in mix.items()) / total
        return voice.float()
    except Exception:
        return KOKORO_VOICE


def generate_narrations(story: dict, progress_cb, genre: str = "Fantasía") -> list[Path]:
    pipeline = load_kokoro()   # <-- en caché: no se recarga en cada generación
    paths = []

    cfg = GENRE_VOICE.get(genre, GENRE_VOICE["Fantasía"])
    voice = _resolve_voice(pipeline, cfg["mix"])
    speed = cfg["speed"]
    pause = cfg["pause"]
    rng = np.random.default_rng(7)   # variación pequeña y repetible en las pausas

    for i, scene in enumerate(story["scenes"], 1):
        chunks = []
        with torch.inference_mode():
            for gs, _, audio in pipeline(
                scene["text"],
                voice=voice,
                speed=speed,
                split_pattern=r"(?<=[.!?…])\s+"
            ):
                if audio is not None:
                    chunks.append(audio.detach().cpu().numpy().astype(np.float32).reshape(-1))
                    # Pausa entre frases con variación natural (una pausa idéntica suena a robot);
                    # más larga tras puntos suspensivos, interrogaciones y exclamaciones.
                    end = str(gs).rstrip()[-1:]
                    mult = 1.5 if end in "…?!" else 1.0
                    gap = pause * mult * float(rng.uniform(0.8, 1.3))
                    chunks.append(np.zeros(int(24000 * gap), dtype=np.float32))

        if not chunks:
            raise RuntimeError(f"Kokoro no produjo audio para la escena {i}.")

        # Pausa de respiración al final de la escena
        chunks.append(np.zeros(int(24000 * (SCENE_END_PAUSE + pause * 0.5)), dtype=np.float32))

        waveform = np.concatenate(chunks)
        path = OUT_DIR / f"voz_{i}.wav"
        sf.write(str(path), waveform, 24000, subtype="PCM_16")
        paths.append(path)
        progress_cb(i, path)

    return paths


# ============================================================
# 4. MÚSICA DE FONDO (MUSICGEN) + MEZCLA CON NARRACIÓN
# ============================================================

def generate_background_music(story: dict, genre: str):
    """Genera una pista instrumental de fondo acorde al género y escenario."""
    processor, model = load_musicgen()
    base_prompt = GENRE_MUSIC.get(genre, GENRE_MUSIC["Fantasía"])
    setting = str(story.get("setting", "")).strip()
    prompt = f"{base_prompt}, {setting}" if setting else base_prompt

    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        audio_values = model.generate(**inputs, max_new_tokens=MUSIC_MAX_TOKENS)

    sr = int(getattr(model.config.audio_encoder, "sampling_rate", 32000))
    audio = audio_values[0, 0].detach().float().cpu().numpy().reshape(-1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-6:
        audio = (audio / peak) * 0.9

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "musica_fondo.wav"
    sf.write(str(path), audio.astype(np.float32), sr, subtype="PCM_16")

    if USE_CUDA:
        # Libera la VRAM de MusicGen para mantener limpia la GPU de 4 GB.
        del model
        load_musicgen.clear()
        free_memory()

    return path


def _loop_with_crossfade(track: np.ndarray, target_len: int, sr: int, fade_sec: float = 1.0) -> np.ndarray:
    """Extiende una pista musical al largo objetivo usando crossfade suave."""
    if len(track) >= target_len:
        return track[:target_len].copy()

    fade = min(int(sr * fade_sec), len(track) // 4)
    if fade <= 0:
        reps = int(np.ceil(target_len / max(1, len(track))))
        return np.tile(track, reps)[:target_len]

    step = len(track) - fade
    out = np.zeros(target_len + len(track), dtype=np.float32)
    win_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    win_out = 1.0 - win_in

    pos = 0
    first = True
    while pos < target_len:
        chunk = track.copy()
        if not first:
            chunk[:fade] *= win_in
            out[pos:pos + fade] *= win_out
        out[pos:pos + len(chunk)] += chunk
        pos += step
        first = False
    return out[:target_len]


def mix_narration_with_music(voice_paths: list[Path], music_path: Path, bg_gain: float = MUSIC_BG_GAIN) -> None:
    """Mezcla la música de fondo (a volumen bajo) detrás de cada narración."""
    from scipy.signal import resample_poly
    from math import gcd

    music_raw, m_sr = sf.read(str(music_path), dtype="float32")
    if music_raw.ndim > 1:
        music_raw = music_raw.mean(axis=1)

    target_sr = 24000
    if m_sr != target_sr:
        g = gcd(target_sr, int(m_sr))
        music_24k = resample_poly(music_raw, target_sr // g, int(m_sr) // g).astype(np.float32)
    else:
        music_24k = music_raw.astype(np.float32)

    music_rms = float(np.sqrt(np.mean(music_24k ** 2))) or 1e-4

    for vpath in voice_paths:
        voice, v_sr = sf.read(str(vpath), dtype="float32")
        if voice.ndim > 1:
            voice = voice.mean(axis=1)
        if not voice.size:
            continue

        bg = _loop_with_crossfade(music_24k, len(voice), int(v_sr))

        # Fade-in y fade-out suaves al inicio y final de la escena
        fi = min(int(v_sr * 0.8), len(bg) // 4)
        fo = min(int(v_sr * 1.2), len(bg) // 4)
        if fi > 0:
            bg[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
        if fo > 0:
            bg[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)

        # Escala la música en función del RMS de la voz para que quede siempre de fondo
        voice_rms = float(np.sqrt(np.mean(voice ** 2))) or 0.05
        scaled_bg = bg * ((voice_rms * bg_gain) / music_rms)

        mixed = voice + scaled_bg
        peak = float(np.max(np.abs(mixed)))
        if peak > 0.98:
            mixed = mixed * (0.98 / peak)

        sf.write(str(vpath), mixed.astype(np.float32), int(v_sr), subtype="PCM_16")


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


def render_book_live(slot, idx, scene, image_uri=None, img_mode="none",
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
        cls = {"draft": "img-draft", "reveal": "img-reveal", "step": "img-step"}.get(img_mode, "")
        left = f'<img class="{cls}" src="{image_uri}" />'
    else:
        left = '<div class="slot-empty"><span>LIENZO<br>EN BLANCO</span></div>'

    # --- pagina derecha: narracion ---
    if text:
        if text_mode == "stream":
            body = html.escape(text) + ' <span class="cursor"></span>'
        else:
            body = _words_html(text, text_mode == "write", text_mode == "write")
        right = "<p>" + body + "</p>"
    else:
        right = '<div class="slot-empty"><span>ESCRIBIENDO...</span></div>'

    # Todo en una sola linea: una linea en blanco rompería el bloque HTML de markdown.
    slot.markdown(
        f'<div class="book book-live{" turn" if turn else ""}">'
        f'<div class="paper"><h2>{title}</h2>{left}<div class="pnum">PAG {idx * 2 + 1}</div></div>'
        f'<div class="paper"><h2>Escena {idx + 1} / {N_SCENES}</h2>{right}<div class="pnum">PAG {idx * 2 + 2}</div></div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ============================================================
# GENERACION EN SEGUNDO PLANO + VISTA EN VIVO CON TIRA CLICABLE
# El libro se crea en un hilo aparte (BookJob). La interfaz lo
# consulta cada ~1.5 s dentro de un fragmento: asi los botones de
# la tira de escenas responden SIN interrumpir la generacion.
# (requiere streamlit >= 1.37 por st.fragment)
# ============================================================
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment")

# El hilo no tiene ScriptRunContext: es normal, silenciamos el aviso.
for _n in ("streamlit.runtime.scriptrunner_utils.script_run_context",
           "streamlit.runtime.scriptrunner.script_run_context"):
    logging.getLogger(_n).setLevel(logging.ERROR)


class JobCancelled(BaseException):
    """El usuario cancelo (BaseException para que ningun `except Exception` la trague)."""


class BookJob:
    """Estado compartido entre el hilo generador y la interfaz (nunca llama a st.*)."""

    def __init__(self, idea: str, genre: str, style: str, include_music: bool = True):
        self.idea, self.genre, self.style = idea, genre, style
        self.include_music = include_music
        self.lock = threading.RLock()
        self.thread = None
        self.cancel = False
        self.status = "running"          # running | done | error | cancelled
        self.error = None
        self.used_fallback = False
        self.note = "Trazando la escaleta de la historia..."
        self.steps = [{"t": False, "i": False, "v": False} for _ in range(N_SCENES)]
        self.scenes = [None] * N_SCENES  # {"title","text"} cuando el texto esta listo
        self.drafts = {}                 # k -> boceto (uri)
        self.thumbs = {}                 # k -> ilustracion final liviana (uri)
        self.voices = {}                 # k -> Path del audio
        self.frame = {"idx": 0, "scene": None, "kw": {}}   # lo que muestra el modo "en vivo"
        self.story = None
        self.pdf_path = None
        self.zip_path = None
        self.pdf_note = ""
        self.started_at = time.monotonic()
        self.finished_at = None

    def set_frame(self, idx, scene, **kw):
        with self.lock:
            self.frame = {"idx": idx, "scene": scene, "kw": kw}

    def set_note(self, msg):
        with self.lock:
            self.note = msg

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = max(0.0, (self.finished_at or time.monotonic()) - self.started_at)
            return {
                "status": self.status, "note": self.note, "cancel": self.cancel,
                "steps": [dict(s) for s in self.steps], "scenes": list(self.scenes),
                "drafts": dict(self.drafts), "thumbs": dict(self.thumbs),
                "voices": dict(self.voices), "frame": dict(self.frame),
                "used_fallback": self.used_fallback, "elapsed": elapsed,
            }


@st.cache_resource(show_spinner=False)
def _registry() -> dict:
    """Guarda el trabajo activo (sobrevive a recargar la pagina y evita dos generaciones a la vez)."""
    return {"job": None}


def thumb_uri(path: Path, size: int = 768, quality: int = 82) -> str:
    """Version liviana de la ilustracion para la vista en vivo (la final sigue en PNG)."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        return pil_to_uri(im, quality)


# ---------- Exportacion: PDF ilustrado + paquete ZIP ----------
def _safe_name(title: str) -> str:
    """Nombre de archivo en ASCII (sin tildes ni simbolos) para descargas y ZIP."""
    import unicodedata
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s-]", "", s).strip()
    return re.sub(r"\s+", "_", s)[:50] or "libro"


def _pdf_txt(s: str) -> str:
    """Las fuentes base del PDF son WinAnsi (cubre espanol); lo demas se reemplaza."""
    return s.encode("cp1252", "replace").decode("cp1252")


_PDF_ACCENT = {"Fantasía": "#ffcd75", "Terror": "#c9303a", "Misterio": "#9fb4e2",
               "Aventura": "#f7b455", "Ciencia ficción": "#63c74d", "Infantil": "#ff7ba3"}


def build_pdf(story: dict, genre: str, out_path: Path) -> None:
    """Libro en PDF apaisado: portada, una doble pagina por escena y pagina final."""
    from xml.sax.saxutils import escape
    from PIL import Image
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader, simpleSplit
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph

    W, H = landscape(A4)
    ink, navy, cream, edge = HexColor("#181425"), HexColor("#262b44"), HexColor("#f4f4e8"), HexColor("#c0cbdc")
    accent = HexColor(_PDF_ACCENT.get(genre, "#ffcd75"))
    c = canvas.Canvas(str(out_path), pagesize=(W, H))
    c.setTitle(_pdf_txt(story["title"]))
    c.setAuthor("ETER | Relatos que cobran vida")

    # --- portada ---
    c.setFillColor(HexColor("#181425")); c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setStrokeColor(accent); c.setLineWidth(6); c.rect(30, 30, W - 60, H - 60, stroke=1, fill=0)
    c.setFillColor(HexColor("#8b9bb4")); c.setFont("Courier-Bold", 13)
    c.drawCentredString(W / 2, H - 110, "EDICION LITERARIA ILUSTRADA")
    c.setFillColor(accent); c.setFont("Courier-Bold", 34)
    lines = simpleSplit(_pdf_txt(story["title"]).upper(), "Courier-Bold", 34, W - 160)
    y = H / 2 + 20 * len(lines)
    for ln in lines:
        c.drawCentredString(W / 2, y, ln)
        y -= 44
    c.setFillColor(HexColor("#a5e06b")); c.setFont("Courier-Bold", 15)
    c.drawCentredString(W / 2, 90, _pdf_txt(f"Genero: {genre}"))
    c.setFillColor(HexColor("#8b9bb4")); c.setFont("Courier", 11)
    c.drawCentredString(W / 2, 62, "ETER | Relatos que cobran vida")
    c.showPage()

    # --- escenas: doble pagina (ilustracion | texto) ---
    mx, my, gap = 28, 28, 14
    pw, ph = (W - 2 * mx - gap) / 2, H - 2 * my
    pad = 22
    n = len(story["scenes"])
    for i, sc in enumerate(story["scenes"]):
        c.setFillColor(navy); c.rect(0, 0, W, H, stroke=0, fill=1)
        for px0 in (mx, mx + pw + gap):
            c.setFillColor(cream); c.rect(px0, my, pw, ph, stroke=0, fill=1)
            c.setStrokeColor(edge); c.setLineWidth(3); c.rect(px0 + 3, my + 3, pw - 6, ph - 6, stroke=1, fill=0)

        # izquierda: titulo + ilustracion
        c.setFillColor(ink); c.setFont("Courier-Bold", 14)
        tl = simpleSplit(_pdf_txt(sc["title"]).upper(), "Courier-Bold", 14, pw - 2 * pad)
        ty = my + ph - 34
        for ln in tl:
            c.drawCentredString(mx + pw / 2, ty, ln)
            ty -= 18
        c.setStrokeColor(edge); c.setLineWidth(2); c.line(mx + pad, ty + 8, mx + pw - pad, ty + 8)
        with Image.open(sc["image"]) as im:
            im = im.convert("RGB")
            im.thumbnail((1000, 1000))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=88)
        buf.seek(0)
        side = min(pw - 2 * pad, ty - my - 50)
        ix, iy = mx + (pw - side) / 2, ty - 18 - side
        c.setFillColor(HexColor("#8b4049")); c.rect(ix + 4, iy - 4, side, side, stroke=0, fill=1)
        c.drawImage(ImageReader(buf), ix, iy, side, side)
        c.setStrokeColor(ink); c.setLineWidth(3); c.rect(ix, iy, side, side, stroke=1, fill=0)
        c.setFillColor(HexColor("#8b9bb4")); c.setFont("Courier-Bold", 9)
        c.drawRightString(mx + pw - 16, my + 12, f"PAG {i * 2 + 1}")

        # derecha: cabecera + texto (la letra se ajusta para que siempre quepa)
        rx = mx + pw + gap
        c.setFillColor(ink); c.setFont("Courier-Bold", 14)
        c.drawCentredString(rx + pw / 2, my + ph - 34, f"ESCENA {i + 1} / {n}")
        c.setStrokeColor(edge); c.setLineWidth(2); c.line(rx + pad, my + ph - 42, rx + pw - pad, my + ph - 42)
        tw, th = pw - 2 * pad, ph - 100
        body = escape(_pdf_txt(sc["text"]))
        for fs in (15, 14, 13, 12, 11, 10, 9):
            style = ParagraphStyle("t", fontName="Times-Roman", fontSize=fs, leading=fs * 1.4,
                                   alignment=TA_JUSTIFY, textColor=ink)
            para = Paragraph(body, style)
            _, h = para.wrap(tw, th)
            if h <= th:
                break
        para.drawOn(c, rx + pad, my + ph - 58 - h)
        c.setFillColor(HexColor("#8b9bb4")); c.setFont("Courier-Bold", 9)
        c.drawRightString(rx + pw - 16, my + 12, f"PAG {i * 2 + 2}")
        c.showPage()

    # --- final ---
    c.setFillColor(HexColor("#181425")); c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(accent); c.setFont("Courier-Bold", 40)
    c.drawCentredString(W / 2, H / 2, "FIN")
    c.setFillColor(HexColor("#8b9bb4")); c.setFont("Courier", 11)
    c.drawCentredString(W / 2, H / 2 - 36, "ETER | Relatos que cobran vida")
    c.showPage()
    c.save()


def build_zip(story: dict, out_path: Path, pdf_path=None) -> None:
    """Paquete completo: texto, ilustraciones, narracion en audio y (si existe) el PDF."""
    base = _safe_name(story["title"])
    txt = [story["title"], "=" * len(story["title"]), ""]
    for i, sc in enumerate(story["scenes"], 1):
        txt += [f"Escena {i}: {sc['title']}", "", sc["text"], ""]
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{base}/libro.txt", "\n".join(txt))
        z.writestr(f"{base}/historia.json", json.dumps(story, ensure_ascii=False, indent=2))
        for i, sc in enumerate(story["scenes"], 1):
            z.write(sc["image"], f"{base}/escena_{i}.png", compress_type=zipfile.ZIP_STORED)
            z.write(sc["voice"], f"{base}/voz_{i}.wav")
        music_file = story.get("music")
        if music_file and Path(music_file).exists():
            z.write(music_file, f"{base}/musica_fondo.wav")
        if pdf_path and Path(pdf_path).exists():
            z.write(pdf_path, f"{base}/{base}.pdf", compress_type=zipfile.ZIP_STORED)


def build_downloads(job: "BookJob", story: dict) -> None:
    base = _safe_name(story["title"])
    try:
        pdf = OUT_DIR / f"{base}.pdf"
        build_pdf(story, job.genre, pdf)
        job.pdf_path = pdf
    except ImportError:
        job.pdf_note = "Para exportar a PDF instala reportlab:  pip install reportlab"
    except Exception:
        job.pdf_note = "No se pudo crear el PDF (el paquete .zip sí está disponible)."
        try:
            (OUT_DIR / "debug_pdf.txt").write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
    zp = OUT_DIR / f"{base}.zip"
    build_zip(story, zp, job.pdf_path)
    job.zip_path = zp


# ---------- Hilo generador ----------
def run_book_job(job: BookJob) -> None:
    def check():
        if job.cancel:
            raise JobCancelled()

    try:
        torch.set_num_threads(TORCH_THREADS)
        job.set_frame(0, None)

        # --- Fase 1: texto, escena por escena ---
        titles = {}

        def on_story_event(ev, i, data):
            check()
            if ev == "scene_start":
                titles[i] = data
                job.set_frame(i, {"title": data, "text": ""}, turn=True)
                job.set_note(f"Escribiendo la escena {i + 1}/{N_SCENES}...")
            elif ev == "scene_text":
                # texto real que el modelo va escribiendo, con cursor parpadeando
                job.set_frame(i, {"title": titles.get(i, "..."), "text": data}, text_mode="stream")
            elif ev == "scene_done":
                with job.lock:
                    job.steps[i]["t"] = True
                    job.scenes[i] = {"title": data["title"], "text": data["text"]}
                job.set_frame(i, data, text_mode="done")

        story, used_fallback = generate_story(job.idea, job.genre, on_event=on_story_event)
        check()
        job.used_fallback = used_fallback
        scenes = story["scenes"]
        with job.lock:
            for k, sc in enumerate(scenes):
                job.steps[k]["t"] = True
                job.scenes[k] = {"title": sc["title"], "text": sc["text"]}

        # --- Fase 2 y 3 en paralelo: ilustraciones (GPU) + voz (CPU) ---
        # Precarga los pipelines de imagen antes de lanzar el hilo de voz:
        # diffusers.from_pretrained(..., torch_dtype=float16) cambia temporalmente
        # torch.set_default_dtype(float16) de forma global, lo que romperia Kokoro en CPU.
        load_image_pipe()
        load_img2img_pipe()
        torch.set_default_dtype(torch.float32)

        voice_state = {"paths": None, "error": None, "done": False}
        images_done = threading.Event()

        def on_voice_done(i, path):
            check()
            with job.lock:
                job.voices[i - 1] = path
                job.steps[i - 1]["v"] = True
            if images_done.is_set():
                nxt = min(i, N_SCENES - 1)
                job.set_frame(nxt, scenes[nxt], image_uri=job.thumbs.get(nxt), img_mode="done",
                              text_mode="done", turn=(i < N_SCENES))
                job.set_note(f"Escena {i}: voz lista." if i == N_SCENES else f"Narrando la escena {i + 1}...")

        def _voice_worker():
            try:
                voice_state["paths"] = generate_narrations(story, on_voice_done, genre=job.genre)
            except BaseException as exc:
                voice_state["error"] = exc
            finally:
                voice_state["done"] = True

        voice_thread = threading.Thread(target=_voice_worker, daemon=True, name="eter-voice-cpu")
        voice_thread.start()

        job.set_frame(0, scenes[0], text_mode="done", turn=True)
        job.set_note("Ilustrando (GPU) y narrando (CPU) en paralelo...")

        def on_img_preview(i, pil_img):
            check()
            uri = pil_to_uri(pil_img)
            with job.lock:
                job.drafts[i - 1] = uri
            job.set_frame(i - 1, scenes[i - 1], image_uri=uri, img_mode="draft",
                          text_mode="done", turn=True)
            job.set_note(f"Escena {i}: boceto listo, refinando detalles...")

        def on_img_done(i, path):
            check()
            thumb = thumb_uri(path)
            with job.lock:
                job.thumbs[i - 1] = thumb
                job.steps[i - 1]["i"] = True
            job.set_frame(i - 1, scenes[i - 1], image_uri=thumb, img_mode="reveal",
                          draft_uri=job.drafts.get(i - 1), text_mode="done")
            job.set_note(f"Escena {i}: ilustración pintada.")

        def on_img_step(i, step, total, pil_img):
            check()
            if pil_img is None:   # empieza la imagen de la escena i: lienzo en blanco
                job.set_frame(i - 1, scenes[i - 1], text_mode="done", turn=(i > 1))
                job.set_note(f"Escena {i}: preparando el lienzo...")
                return
            job.set_frame(i - 1, scenes[i - 1], image_uri=pil_to_uri(pil_img, 80),
                          img_mode="step", text_mode="done")
            job.set_note(f"Escena {i}: dibujando la imagen, paso {step}/{total}...")

        images = generate_scene_images(story, job.genre, job.style, on_img_done,
                                       preview_cb=on_img_preview, step_cb=on_img_step)
        images_done.set()
        check()

        if not voice_state["done"]:
            job.set_note("Finalizando narración de voz...")
        voice_thread.join()
        check()
        if voice_state["error"] is not None:
            raise voice_state["error"]
        voices = voice_state["paths"]

        # --- Fase 4: música de fondo (MusicGen en GPU) + mezcla con la voz ---
        if job.include_music:
            try:
                job.set_note("Componiendo música de fondo (IA)...")
                music_path = generate_background_music(story, job.genre)
                check()
                job.set_note("Mezclando música suave detrás de la narración...")
                mix_narration_with_music(voices, music_path)
                story["music"] = str(music_path)
            except JobCancelled:
                raise
            except Exception:
                try:
                    (OUT_DIR / "debug_music.txt").write_text(traceback.format_exc(), encoding="utf-8")
                except Exception:
                    pass

        # --- Cierre: historia.json + archivos descargables ---
        for i, sc in enumerate(story["scenes"]):
            sc["image"] = str(images[i])
            sc["voice"] = str(voices[i])
        (OUT_DIR / "historia.json").write_text(
            json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8")
        job.set_note("Encuadernando el libro (PDF y paquete descargable)...")
        build_downloads(job, story)
        job.story = story
        job.set_note("¡Libro encuadernado!")
        job.finished_at = time.monotonic()
        job.status = "done"
    except JobCancelled:
        job.finished_at = time.monotonic()
        job.status = "cancelled"
    except Exception:
        job.finished_at = time.monotonic()
        job.error = traceback.format_exc()
        job.status = "error"


# ---------- Interfaz de la generacion ----------
def _pin_scene(k: int):
    st.session_state.pinned = k


def _follow_live():
    st.session_state.pinned = None


def _cancel_job():
    job = st.session_state.get("job")
    if job:
        job.cancel = True


def _open_finished_book(job: BookJob):
    st.session_state.book = {
        "story": job.story, "genre": job.genre, "idea": job.idea,
        "pdf": str(job.pdf_path) if job.pdf_path else None,
        "zip": str(job.zip_path) if job.zip_path else None,
        "pdf_note": job.pdf_note,
    }
    st.session_state.page = -1
    st.session_state.screen = "book"
    st.session_state.job = None
    st.session_state.pinned = None


@st.cache_data(show_spinner=False, max_entries=4)
def _file_bytes(path: str, mtime: float) -> bytes:
    return Path(path).read_bytes()


def render_downloads(pdf, zip_, pdf_note="", key="dl"):
    """Botones de descarga (solo cuando TODO el libro ya existe)."""
    pdf_ok = bool(pdf) and Path(pdf).exists()
    zip_ok = bool(zip_) and Path(zip_).exists()
    if not (pdf_ok or zip_ok):
        return
    st.markdown('<div class="audio-label">DESCARGAR LIBRO</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if pdf_ok:
        p = Path(pdf)
        c1.download_button("📄 DESCARGAR PDF", data=_file_bytes(str(p), p.stat().st_mtime),
                           file_name=p.name, mime="application/pdf",
                           use_container_width=True, key=f"{key}_pdf",
                           help="Libro ilustrado listo para leer o imprimir")
    elif pdf_note:
        c1.caption(pdf_note)
    if zip_ok:
        z = Path(zip_)
        c2.download_button("🎧 PAQUETE COMPLETO (.zip)", data=_file_bytes(str(z), z.stat().st_mtime),
                           file_name=z.name, mime="application/zip",
                           use_container_width=True, key=f"{key}_zip",
                           help="Texto, ilustraciones, narración en audio y PDF")


def _pinned_frame(snap: dict, k: int) -> dict:
    """Cuadro del libro para una escena elegida con la tira (muestra lo que ya existe)."""
    uri, mode = snap["thumbs"].get(k), "done"
    if not uri and snap["drafts"].get(k):
        uri, mode = snap["drafts"][k], "draft"
    kw = {"text_mode": "done"}
    if uri:
        kw.update(image_uri=uri, img_mode=mode)
    return {"idx": k, "scene": snap["scenes"][k], "kw": kw}


def _scene_strip(snap: dict, pinned):
    """Tira de progreso: un boton por escena (T texto, I ilustracion, V voz)."""
    mark = lambda ok: "🟩" if ok else "⬜"
    try:
        box = st.container(key="strip")
    except TypeError:            # streamlit < 1.39 no admite key= en containers
        box = st.container()
    with box:
        cols = st.columns(N_SCENES + 1, gap="small")
        for k in range(N_SCENES):
            s = snap["steps"][k]
            cols[k].button(
                f"{k + 1}  T{mark(s['t'])} I{mark(s['i'])} V{mark(s['v'])}",
                key=f"chip_{k}", on_click=_pin_scene, args=(k,),
                disabled=not s["t"], use_container_width=True,
                type="primary" if pinned == k else "secondary",
                help="Volver a esta escena" if s["t"] else "Esta escena aún no está lista",
            )
        cols[N_SCENES].button(
            "📡 EN VIVO", key="chip_live", on_click=_follow_live, use_container_width=True,
            type="primary" if pinned is None else "secondary",
            help="Seguir la escena que se está creando ahora",
        )
    st.markdown(
        '<div class="wip-legend">T = texto &nbsp; I = ilustración &nbsp; V = voz &nbsp;·&nbsp; '
        '🟩 listo &nbsp; ⬜ en espera &nbsp;·&nbsp; toca una escena para volver a verla</div>',
        unsafe_allow_html=True,
    )


def _generation_ui(polling: bool):
    job = st.session_state.get("job")
    if job is None:
        return
    snap = job.snapshot()
    status = snap["status"]
    if polling and status != "running":
        st.rerun()    # el trabajo termino: repinta la pagina completa (sin sondeo)

    pinned = st.session_state.get("pinned")
    if pinned is not None and not snap["scenes"][pinned]:
        pinned = None

    total = N_SCENES * 3
    done = sum(int(s[k]) for s in snap["steps"] for k in ("t", "i", "v"))
    mins, secs = divmod(int(snap.get("elapsed", 0)), 60)
    timer_txt = f"{mins:02d}:{secs:02d}"
    if status == "running" and snap["cancel"]:
        label = f"Cancelando... terminando el paso en curso · ⏱️ {timer_txt}"
    elif status == "done":
        label = f"¡Libro terminado! · ⏱️ Tiempo total: {timer_txt}"
    else:
        label = f"Creando tu obra literaria... {done}/{total} pasos · ⏱️ {timer_txt}"
    st.progress(done / total, text=label)
    st.markdown(f'<div class="audio-label">{html.escape(snap["note"])} · ⏱️ {timer_txt}</div>', unsafe_allow_html=True)
    if snap["used_fallback"]:
        st.warning("Se usó un relato base alternativo por contingencia.")

    fr = _pinned_frame(snap, pinned) if pinned is not None else snap["frame"]
    render_book_live(st, fr["idx"], fr["scene"], **fr["kw"])

    if pinned is not None and pinned in snap["voices"]:
        st.markdown('<div class="audio-label">NARRACIÓN DE VOZ</div>', unsafe_allow_html=True)
        st.audio(str(snap["voices"][pinned]))

    _scene_strip(snap, pinned)

    if status == "running":
        _, mid, _ = st.columns([1, 1, 1])
        mid.button("⛔ CANCELAR", key="gen_cancel", on_click=_cancel_job,
                   disabled=snap["cancel"], use_container_width=True)
    elif status == "done":
        st.markdown('<div class="audio-label">¡LIBRO ENCUADERNADO! ÁBRELO O DESCÁRGALO</div>',
                    unsafe_allow_html=True)
        if st.button("📖 ABRIR LIBRO", key="gen_open", use_container_width=True):
            _open_finished_book(job)
            st.rerun()
        render_downloads(job.pdf_path, job.zip_path, job.pdf_note, key="dl_gen")


@_fragment(run_every=0.8)
def generation_view_live():
    _generation_ui(polling=True)


@_fragment
def generation_view_static():
    _generation_ui(polling=False)


# ============================================================
# ESTADO DE LA APLICACION
# ============================================================

for k, v in {
    "screen": "welcome",
    "book": None,
    "page": -1,
    "narration_auto": True,
    "job": None,
    "pinned": None
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

        include_music = st.toggle(
            "🎵 Incluir música de fondo (IA)",
            value=True,
            help="Compone una pista instrumental acorde al género y la mezcla suavemente detrás de la narración."
        )

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])

    active_job = _registry()["job"]
    busy = active_job is not None and active_job.status == "running"

    with btn_col:
        crear_libro = st.button("⚡ GENERAR LIBRO", use_container_width=True, disabled=busy)
        if busy:
            st.markdown('<div class="audio-label">YA HAY UN LIBRO GENERÁNDOSE</div>', unsafe_allow_html=True)
            if st.button("↩ VOLVER A LA GENERACIÓN", key="resume_job", use_container_width=True):
                st.session_state.job = active_job
                st.session_state.pinned = None
                st.session_state.screen = "generating"
                st.rerun()

    if crear_libro:
        if not idea.strip():
            st.error("Escribe una idea para la historia.")
        else:
            if OUT_DIR.exists():
                shutil.rmtree(OUT_DIR)
            OUT_DIR.mkdir(parents=True, exist_ok=True)

            job = BookJob(idea.strip(), genre, style, include_music=include_music)
            job.thread = threading.Thread(target=run_book_job, args=(job,), daemon=True, name="eter-book-job")
            _registry()["job"] = job
            st.session_state.job = job
            st.session_state.pinned = None
            st.session_state.screen = "generating"
            job.thread.start()
            st.rerun()

# ============================================================
# PANTALLA 2B: GENERANDO (el libro se forma en vivo; la tira es clicable)
# ============================================================
elif st.session_state.screen == "generating":
    job = st.session_state.job
    if job is None:
        st.session_state.screen = "studio"
        st.rerun()
    apply_background(job.genre)
    st.markdown("""
    <div style="max-width: 900px; margin: 0 auto 1rem auto; text-align: center;">
        <div style="font-family: 'Press Start 2P', monospace; font-size: 18px; color: #ffcd75;">
            FORJANDO TU LIBRO
        </div>
    </div>
    """, unsafe_allow_html=True)

    if job.status == "cancelled":
        st.session_state.job = None
        st.session_state.screen = "studio"
        st.rerun()

    if job.status == "running":
        generation_view_live()
    else:
        generation_view_static()

    if job.status == "error":
        st.error("Algo falló durante la generación.")
        with st.expander("Detalles técnicos"):
            st.code(job.error or "sin detalles")
        if st.button("⬅ VOLVER AL ESTUDIO", key="gen_back"):
            st.session_state.job = None
            st.session_state.screen = "studio"
            st.rerun()
    elif job.status == "done" and st.session_state.pinned is None:
        time.sleep(1.2)               # deja ver el libro terminado un instante
        _open_finished_book(job)
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

    render_downloads(book.get("pdf"), book.get("zip"), book.get("pdf_note", ""), key="dl_book")

    st.session_state.narration_auto = st.toggle(
        "Reproducción automática de voz",
        value=st.session_state.narration_auto
    )