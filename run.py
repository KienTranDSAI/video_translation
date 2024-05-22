from utils import *
import os


if __name__ ==  "__main__":
    args = get_args()
    video_path = args.video_path
    
    whisper = create_whisper(device = 'cuda')
    vi2en_tokenizer,vi2en_translator = create_vi2en_translator(device = "cuda")
    tts = create_txt2aud(device = 'cuda')
    detector, model, diffusion = create_diff2lip(args.diff2lip_model_path)
    
    os.makedirs(f"{os.getcwd()}/audio")
    raw_audio_path = f"{os.getcwd()}/audio/inp_aud.mp3"

    #Get raw audio
    get_raw_aud(video_path, raw_audio_path)
    #Run speech recognition
    recog_text = whisper(raw_audio_path)
    #Tranlate text from vietnamese to english
    en_text = translate_vi2en(recog_text['text'],vi2en_tokenizer,vi2en_translator,"cuda")
    #Run text to audio
    run_txt2aud(en_text, f"{os.getcwd()}/audio", tts)
    #Run diff2lip
    translated_audio_path = f"{os.getcwd()}/text2audio/results/output_txt2aud.wav"
    output_path = "output.mp4"
    run_diff2lip(detector, model, diffusion, video_path,translated_audio_path, output_path)