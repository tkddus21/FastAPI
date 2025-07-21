# 답변라우터를 작성하여, 답변 등록 API(답변 등록을 처리할 answer_create 라우터 함수)
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from starlette import status

from database import get_db
from domain.answer import answer_schema, answer_crud
from domain.question import question_crud

router = APIRouter(
    prefix="/api/answer",
)

@router.post("/create/{question_id}", status_code=status.HTTP_204_NO_CONTENT)
def answer_create(question_id: int,
                  _answer_create: answer_schema.AnswerCreate,
                  db: Session = Depends(get_db)):
    

    #create answer
    question = question_crud.get_questiton(db, question_id=question_id)
    if not question:
        raise HTTPException(status_code=404, detail= "Question not found")
    
    answer_crud.create_answer(db, question=question,
                              answer_create=_answer_create)