from typing import Optional
from fastapi import FastAPI, File, Query
from starlette.responses import Response
import io
from model import get_model, get_result
import logging
# from nlp import NLP
# import uvicorn

logging.basicConfig(level=logging.DEBUG)

app = FastAPI(title="Question Answering",
              description=''' El objetivo es encontrar el espacio 
              de texto en el párrafo que responde a la pregunta 
              planteada.''',
              version="0.1.0",
              )
# nlp = NLP()
model = get_model()

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/qas/")
async def get_qas(context: str = Query(..., min_length=3), question: str = Query(..., min_length=3)):
    '''Get question answering'''
    logging.debug("ejecutar modelo...")

    if context and question:
        result = get_result(model, context, question)
        logging.debug("modelo ejecutado...")
        print("**********************")
        print(result)
        print("**********************")
        # return Response(result)
        return result
    return {"items": "Null"}


# if __name__ == "__main__":
#     uvicorn.run("server:app", host="0.0.0.0", port=8008)