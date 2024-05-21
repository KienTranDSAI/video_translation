import os
import json
import torch
import sys
current_dir = os.getcwd()
sys.path.append(f"{current_dir}/tortoise-tts")
sys.path.append(f'{current_dir}/diff2lip/guided-diffusion/guided_diffusion')
sys.path.append(f'{current_dir}/diff2lip/guided-diffusion')
sys.path.append(f'{current_dir}/diff2lip')

import torch
import torchaudio
from tortoise.api import TextToSpeech,MODELS_DIR
import os
from tortoise.utils.audio import load_voices

from transformers import AutoConfig, AutoModelForSpeechSeq2Seq,WhisperTokenizer,WhisperFeatureExtractor,pipeline
from transformers import AutoModelForSeq2SeqLM,MBartTokenizer

import dist_util
import generate
import argparse
import cv2
import os
from os.path import join, basename, dirname, splitext
import shutil
import argparse
import numpy as np
import random
import torch, torchvision
import subprocess
from audio import audio
import face_detection
from tqdm import tqdm
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    tfg_model_and_diffusion_defaults,
    tfg_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from guided_diffusion.tfg_data_util import (
    tfg_process_batch,
)
from generate import *
def create_whisper():
    current_directory = os.getcwd()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load config
    config = AutoConfig.from_pretrained(f"{current_directory}/PhoWhisper-medium/config.json")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(f"{current_directory}/PhoWhisper-medium/pytorch_model.bin", config = config)
    with open(f"{current_directory}/PhoWhisper-medium/special_tokens_map.json", "r") as f:
        special_tokens_data = json.load(f)
        additional_special_tokens = special_tokens_data["additional_special_tokens"]
    wTokenizer = WhisperTokenizer(vocab_file=f"{current_directory}/PhoWhisper-medium/vocab.json",
                                           merges_file=f"{current_directory}/PhoWhisper-medium/merges.txt",
                                           normalizer_file=f"{current_directory}/PhoWhisper-medium/normalizer.json",
                                           additional_special_tokens = additional_special_tokens
                                           )
    wExtractor = WhisperFeatureExtractor.from_pretrained(f"{current_directory}/PhoWhisper-medium/preprocessor_config.json")
    speech_recog_pipe = pipeline("automatic-speech-recognition",
                    model = model,
                    tokenizer = wTokenizer,
                    feature_extractor = wExtractor,
                    device = device)
    return speech_recog_pipe
def create_vi2en_translator():
    current_directory = os.getcwd()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = AutoConfig.from_pretrained(f"{current_directory}/vinai-translate-vi2en-v2/config.json")
    vi2en_model = AutoModelForSeq2SeqLM.from_pretrained(f"{current_directory}/vinai-translate-vi2en-v2/pytorch_model.bin", config = config)
    tokenizer_vi2en = MBartTokenizer.from_pretrained(f"{current_directory}/vinai-translate-vi2en-v2/sentencepiece.bpe.model")
    vi2en_model.to(device)
    return tokenizer_vi2en, vi2en_model
def translate_vi2en(vi_text: str, tokenizer_vi2en, vi2en_model, device = 'cpu') -> str:
    input_ids = tokenizer_vi2en(vi_text, return_tensors="pt").input_ids.to(device)
    output_ids = vi2en_model.generate(
        input_ids,
        decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    en_text = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
    en_text = " ".join(en_text)
    return en_text



def create_txt2aud():
    
    current_directory = os.getcwd()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preset = 'fast'
    use_deepspeed = False
    kv_cache =True
    half = True
    model_dir = MODELS_DIR
    seed = None
    produce_debug_state = False
    cvvp_amount = 0

    if torch.backends.mps.is_available():
        use_deepspeed = False
    os.makedirs(f"{current_directory}/text2audio/results/", exist_ok=True)
    tts = TextToSpeech(models_dir=model_dir, use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=half, device = device)
    return tts

def create_diff2lip(diff2lip_model_path):
    old_config = tfg_model_and_diffusion_defaults()

    new_config = {
        "attention_resolutions": "32,16,8",
            "class_cond": False,
            "learn_sigma": True,
            "num_channels": 128,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_fp16": True,
            "use_scale_shift_norm": False,
            "predict_xstart": False,
            "diffusion_steps": 1000,
            "noise_schedule": "linear",
            "rescale_timesteps": False,
            "timestep_respacing": "ddim25",
            "nframes": 5,
            "nrefer": 1,
            "image_size": 128,

            "use_ref": True,
            "use_audio": True,
            "audio_as_style": True,
    }
    for key in new_config:
        old_config[key] = new_config[key]
    model, diffusion = tfg_create_model_and_diffusion(**old_config)
    model.load_state_dict(
            dist_util.load_state_dict(path = diff2lip_model_path, map_location='cpu')
    )
    model.to(dist_util.dev())
    if old_config['use_fp16']:
        model.convert_to_fp16()
    model.eval()

    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda' if torch.cuda.is_available() else 'cpu')

    return detector, model

