import base64, json, tempfile, vosk, pathlib, os, torch, getpass
from channels.generic.websocket import AsyncWebsocketConsumer
from seamless_communication.inference import Translator
from fairseq2.assets import InProcAssetMetadataProvider, asset_store
from huggingface_hub import snapshot_download

model = vosk.Model(model_name="vosk-model-en-us-0.42-gigaspeech")
recognizer = vosk.KaldiRecognizer(model, 16000)

user = getpass.getuser()
CHECKPOINTS_PATH = pathlib.Path(os.getenv("CHECKPOINTS_PATH", f"/home/{user}/app/models"))
if not CHECKPOINTS_PATH.exists():
    snapshot_download(repo_id="facebook/seamless-m4t-v2-large", repo_type="model", local_dir=CHECKPOINTS_PATH)
asset_store.env_resolvers.clear()
asset_store.env_resolvers.append(lambda: "demo")
demo_metadata = [
    {
        "name": "seamlessM4T_v2_large@demo",
        "checkpoint": f"file://{CHECKPOINTS_PATH}/seamlessM4T_v2_large.pt",
        "char_tokenizer": f"file://{CHECKPOINTS_PATH}/spm_char_lang38_tc.model",
    },
    {
        "name": "vocoder_v2@demo",
        "checkpoint": f"file://{CHECKPOINTS_PATH}/vocoder_v2.pt",
    },
]
asset_store.metadata_providers.append(InProcAssetMetadataProvider(demo_metadata))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

    DEFAULT_SOURCE_LANGUAGE = "English"
    DEFAULT_TARGET_LANGUAGE = "Portuguese"

translator = Translator(
    model_name_or_card="seamlessM4T_v2_large",
    vocoder_name_or_card="vocoder_v2",
    device=device,
    dtype=dtype,
    apply_mintox=True,
)

language_code_to_name = {
    "afr": "Afrikaans",
    "amh": "Amharic",
    "arb": "Modern Standard Arabic",
    "ary": "Moroccan Arabic",
    "arz": "Egyptian Arabic",
    "asm": "Assamese",
    "ast": "Asturian",
    "azj": "North Azerbaijani",
    "bel": "Belarusian",
    "ben": "Bengali",
    "bos": "Bosnian",
    "bul": "Bulgarian",
    "cat": "Catalan",
    "ceb": "Cebuano",
    "ces": "Czech",
    "ckb": "Central Kurdish",
    "cmn": "Mandarin Chinese",
    "cym": "Welsh",
    "dan": "Danish",
    "deu": "German",
    "ell": "Greek",
    "eng": "English",
    "est": "Estonian",
    "eus": "Basque",
    "fin": "Finnish",
    "fra": "French",
    "gaz": "West Central Oromo",
    "gle": "Irish",
    "glg": "Galician",
    "guj": "Gujarati",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hrv": "Croatian",
    "hun": "Hungarian",
    "hye": "Armenian",
    "ibo": "Igbo",
    "ind": "Indonesian",
    "isl": "Icelandic",
    "ita": "Italian",
    "jav": "Javanese",
    "jpn": "Japanese",
    "kam": "Kamba",
    "kan": "Kannada",
    "kat": "Georgian",
    "kaz": "Kazakh",
    "kea": "Kabuverdianu",
    "khk": "Halh Mongolian",
    "khm": "Khmer",
    "kir": "Kyrgyz",
    "kor": "Korean",
    "lao": "Lao",
    "lit": "Lithuanian",
    "ltz": "Luxembourgish",
    "lug": "Ganda",
    "luo": "Luo",
    "lvs": "Standard Latvian",
    "mai": "Maithili",
    "mal": "Malayalam",
    "mar": "Marathi",
    "mkd": "Macedonian",
    "mlt": "Maltese",
    "mni": "Meitei",
    "mya": "Burmese",
    "nld": "Dutch",
    "nno": "Norwegian Nynorsk",
    "nob": "Norwegian Bokm\u00e5l",
    "npi": "Nepali",
    "nya": "Nyanja",
    "oci": "Occitan",
    "ory": "Odia",
    "pan": "Punjabi",
    "pbt": "Southern Pashto",
    "pes": "Western Persian",
    "pol": "Polish",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "slk": "Slovak",
    "slv": "Slovenian",
    "sna": "Shona",
    "snd": "Sindhi",
    "som": "Somali",
    "spa": "Spanish",
    "srp": "Serbian",
    "swe": "Swedish",
    "swh": "Swahili",
    "tam": "Tamil",
    "tel": "Telugu",
    "tgk": "Tajik",
    "tgl": "Tagalog",
    "tha": "Thai",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzn": "Northern Uzbek",
    "vie": "Vietnamese",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "yue": "Cantonese",
    "zlm": "Colloquial Malay",
    "zsm": "Standard Malay",
    "zul": "Zulu",
}
LANGUAGE_NAME_TO_CODE = {v: k for k, v in language_code_to_name.items()}

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        bytes_data = base64.b64decode(text_data)
        transcription = None

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
            temp_file.write(bytes_data)
            temp_file.flush()
            with open(temp_file.name, 'rb') as wav_file:
                wav_data = wav_file.read()
                if len(wav_data) == 0:
                    return None
                if recognizer.AcceptWaveform(wav_data):
                    result = json.loads(recognizer.Result())
                    if len(result["text"]) >= 1:
                            source_language_code = LANGUAGE_NAME_TO_CODE[ DEFAULT_SOURCE_LANGUAGE]
                            target_language_code = LANGUAGE_NAME_TO_CODE[DEFAULT_TARGET_LANGUAGE]
                            out_texts, _ = translator.predict(
                                input=result["text"],
                                task_str="T2TT",
                                src_lang=source_language_code,
                                tgt_lang=target_language_code,
                            )
                            transcription = str(out_texts[0])
        if transcription:
            await self.send(text_data=json.dumps(transcription))