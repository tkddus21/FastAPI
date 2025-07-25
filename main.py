from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from domain.answer import answer_router
from domain.question import question_router
from fastapi import FastAPI, File, UploadFile
import whisper

import tempfile

app = FastAPI()

origins = [
    "http://127.0.0.1:5173",    # 또는 "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.get("/hello")
# def hello():
#     return {"message": "안녕하세요 파이보!!212!"}

app.include_router(question_router.router)
app.include_router(answer_router.router)

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
