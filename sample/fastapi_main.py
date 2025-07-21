from fastapi import FastAPI, HTTPException, Body, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from transformers import pipeline
from gtts import gTTS
import speech_recognition as sr
import os
import uuid
import requests
import time

app = FastAPI()

# 1) 챗봇 (Polyglot-ko)
chatbot_model = pipeline(
    "text-generation",
    model="EleutherAI/polyglot-ko-1.3b",
    tokenizer="EleutherAI/polyglot-ko-1.3b",
    framework="pt"
)

# 2) 감정분석 (예: KoBERT, KLUE/bert-base, etc. Huggingface API 사용)
emotion_model = pipeline(
    "sentiment-analysis",
    model="WhitePeak/bert-base-cased-Korean-sentiment",
    tokenizer="WhitePeak/bert-base-cased-Korean-sentiment"
)

def make_prompt(user_input, medicine_time: bool):
    # 기본 예시 + 공감·위로 강화 예시 추가
    example1 = (
        "어르신: 오늘은 많이 더워서 힘드네.\n"
        "챗봇: 오늘 날씨가 무척 덥죠? 물 많이 드시고 시원하게 쉬시면서 건강 챙기세요.\n"
    )
    example2 = (
        "어르신: 오늘 너무 외로워서 힘들어.\n"
        "챗봇: 많이 외로우셨죠? 마음이 힘드실 땐 저와 이야기를 나누셔도 좋고, 좋아하시는 음악을 들어보는 것도 도움이 될 거예요. 언제나 어르신 곁에 있어드릴게요.\n"
    )
    example3 = (
        "어르신: 기분이 너무 우울해.\n"
        "챗봇: 우울한 기분이 드는 날에는 잠깐 산책을 해보시거나, 따뜻한 차 한 잔을 드셔 보는 것도 좋아요. 제가 힘이 되어드릴 수 있어 기쁩니다.\n"
    )
    example4 = (
        "어르신: 오늘 아무도 연락이 없어서 쓸쓸했어.\n"
        "챗봇: 혼자 계시다 보면 쓸쓸할 수 있죠. 언제든지 저에게 마음을 털어놓으셔도 돼요. 항상 어르신 말씀에 귀 기울일게요.\n"
    )
    base_prompt = (
        "아래는 다정하게 위로하고 공감하는 챗봇과 어르신의 대화입니다.\n"
        f"{example1}{example2}{example3}{example4}어르신: {user_input}\n챗봇:"
    )
    if medicine_time:
        medicine_prompt = "지금 약 드실 시간이예요! 건강을 위해 꼭 잊지 말고 챙겨드셔야 해요.\n"
        return medicine_prompt + base_prompt
    else:
        return base_prompt

def chatbot_reply(prompt, max_length):
    responses = chatbot_model(
        prompt,
        max_length=max_length,  # 인자로 받음!
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        temperature=0.8,
        top_p=0.85,
        eos_token_id=chatbot_model.tokenizer.eos_token_id
    )
    generated = responses[0]['generated_text'][len(prompt):].strip()
    result = generated.split('\n')[0].strip()
    if not result.endswith('.'):
        result += '.'
    return result


# 1️⃣ 텍스트 챗봇 (텍스트 입력/출력)
@app.post("/chat")
async def chat(
    user_input: str = Body(..., embed=True),
    medicine_time: bool = Body(False, embed=True)
):
    try:
        prompt = make_prompt(user_input, medicine_time)
        reply = chatbot_reply(prompt,80)
        return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/emotion")
async def emotion_api(
    user_input: str = Body(..., embed=True)
):
    try:
        result = emotion_model(user_input)
        label_map = {
            "LABEL_0": "negative",
            "LABEL_1": "positive"
        }
        # label 치환
        emotion_label = label_map.get(result[0]['label'], result[0]['label'])
        return {
            "emotion": emotion_label,
            "confidence": result[0]['score']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2️⃣ 텍스트 챗봇 + TTS (텍스트 입력, mp3 음성 답변)
@app.post("/chat-tts")
async def chat_tts(
    user_input: str = Body(..., embed=True),
    medicine_time: bool = Body(False, embed=True)
):
    try:
        prompt = make_prompt(user_input, medicine_time)
        # 최대 2~3문장 반환 가공
        generated = chatbot_reply(prompt, 40)  # 2~3문장 용도
        # 한글에서 '다.'로 최대 3문장까지만 추출
        sentences = generated.split('다.')
        tts_reply = '다.'.join(sentences[:3]).strip()
        if not tts_reply.endswith('다.') and len(sentences) >= 1:
            tts_reply += '다.'
        # TTS로 음성 생성
        tts = gTTS(tts_reply, lang="ko")
        tts_fp = "chatbot_reply.mp3"
        tts.save(tts_fp)
        return FileResponse(tts_fp, media_type="audio/mpeg", filename="chatbot_reply.mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3️⃣ 음성 업로드 → 챗봇 답변 → TTS 음성 반환
from fastapi import UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydub import AudioSegment
from io import BytesIO
import os, time, speech_recognition as sr
from gtts import gTTS
import gc

@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):
    try:
        # 🔹 1. 오디오 읽기 및 mp3 → wav 변환 (BytesIO로 메모리 처리)
        audio_bytes = await file.read()
        mp3_data = BytesIO(audio_bytes)
        audio = AudioSegment.from_file(mp3_data, format="mp3")

        # wav 파일로 저장
        wav_path = f"temp_input_{file.filename.rsplit('.', 1)[0]}.wav"
        audio.export(wav_path, format="wav")

        # 메모리 정리
        del audio
        gc.collect()

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"mp3→wav 변환 오류: {e}"})

    # 🔹 2. STT (wav 파일에서 텍스트 추출)
    r = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = r.record(source)
            user_text = r.recognize_google(audio_data, language='ko-KR')
    except Exception as e:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return JSONResponse(status_code=400, content={"error": f"STT 오류(wav): {e}"})

    # 🔹 3. 챗봇 응답 생성 (임의 함수 사용: make_prompt, chatbot_reply)
    try:
        prompt = make_prompt(user_text, False)
        generated = chatbot_reply(prompt, max_tokens=35)

        # 응답 너무 길면 앞 3문장까지만
        sentences = generated.split('다.')
        tts_reply = '다.'.join(sentences[:3]).strip()
        if not tts_reply.endswith('다.') and len(sentences) >= 1:
            tts_reply += '다.'
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"챗봇 생성 실패: {e}"})

    # 🔹 4. TTS 음성 생성
    try:
        tts = gTTS(tts_reply, lang="ko")
        tts_fp = f"output_voice_{int(time.time())}.mp3"
        tts.save(tts_fp)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"TTS 오류: {e}"})

    # 🔹 5. 임시 wav 파일 삭제
    try:
        if os.path.exists(wav_path):
            os.remove(wav_path)
    except Exception as e:
        print(f"⚠️ wav 삭제 실패: {e}")

    # 🔹 6. mp3 응답 반환
    return FileResponse(tts_fp, media_type="audio/mpeg", filename="chatbot_reply.mp3")
    

UPLOAD_DIR  = "uploads"
RESTORE_DIR = "restored"
os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(RESTORE_DIR, exist_ok=True)

# ⛔️ 환경변수 대신 직접 토큰 넣기
REPLICATE_TOKEN = "r8_HfX8uCVT2VFKontYqWt8XYS66cl0QrD0bLJoj"
GFPGAN_VERSION  = "0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c"

HEADERS = {
    "Authorization": f"Token {REPLICATE_TOKEN}",
    "Content-Type": "application/json"
}

PREDICT_API = "https://api.replicate.com/v1/predictions"
UPLOAD_API  = "https://api.replicate.com/v1/uploads"

@app.post("/photo-restore")
async def photo_restore_api(file: UploadFile = File(...)):
    # 1. 파일 저장
    uid_name  = f"{uuid.uuid4().hex}_{file.filename}"
    orig_path = os.path.join(UPLOAD_DIR, uid_name)
    with open(orig_path, "wb") as buf:
        buf.write(await file.read())

    # 2. presigned URL 생성
    up_req = requests.post(UPLOAD_API, headers={"Authorization": f"Token {REPLICATE_TOKEN}"},
                           json={"filename": uid_name, "content_type": file.content_type})
    if up_req.status_code != 201:
        return JSONResponse(status_code=500, content={"error": "Upload URL 생성 실패", "detail": up_req.text})


    upload_url   = up_req.json()["upload_url"]
    download_url = up_req.json()["download_url"]

    # 3. PUT 업로드
    with open(orig_path, "rb") as f:
        put_res = requests.put(upload_url, data=f, headers={"Content-Type": file.content_type})
    if put_res.status_code not in (200, 201):
        return JSONResponse(500, {"error": "이미지 업로드 실패", "detail": put_res.text})

    # 4. 복원 요청
    payload = {
        "version": GFPGAN_VERSION,
        "input": {"img": download_url}
    }
    pred_res = requests.post(PREDICT_API, headers=HEADERS, json=payload)
    if pred_res.status_code != 201:
        return JSONResponse(500, {"error": "Replicate 호출 실패", "detail": pred_res.text})

    prediction_url = pred_res.json()["urls"]["get"]

    # 5. 복원 결과 폴링
    restored_url = None
    for _ in range(60):
        poll = requests.get(prediction_url, headers=HEADERS)
        pr   = poll.json()
        if pr["status"] == "succeeded":
            restored_url = pr["output"][0] if isinstance(pr["output"], list) else pr["output"]
            break
        elif pr["status"] == "failed":
            return JSONResponse(500, {"error": "AI 복원 실패", "detail": pr})
        time.sleep(1)

    if not restored_url:
        return JSONResponse(500, {"error": "AI 복원 대기 시간 초과"})

    # 6. 결과 저장
    resp = requests.get(restored_url)
    result_file = os.path.join(RESTORE_DIR, f"restored_{uid_name}")
    with open(result_file, "wb") as fw:
        fw.write(resp.content)

    return {"restored_url": f"/avatar/{os.path.basename(result_file)}"}

@app.get("/avatar/{filename}")
async def get_avatar(filename: str):
    path = os.path.join(RESTORE_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/jpeg")
    return JSONResponse(404, {"error": "Image not found"})



    # cd "C:\Users\gunhu\Desktop\공모전\2025 오픈소스 대회"
    # .\chatbotenv\Scripts\activate
    # uvicorn chatbot_api:app --reload