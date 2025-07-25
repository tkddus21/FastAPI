# main.py
from fastapi import FastAPI, File, UploadFile
import whisper
import tempfile

app = FastAPI()
model = whisper.load_model("base")  # tiny, base, small, medium, large 중 선택

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # 업로드된 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    # Whisper 모델로 변환
    result = model.transcribe(tmp_path)
    return {"text": result["text"]}
