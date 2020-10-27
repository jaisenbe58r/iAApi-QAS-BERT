from typing import Optional
from fastapi import FastAPI, File, Query
from starlette.responses import Response
import io
from model import get_model, get_result
import logging


logging.basicConfig(level=logging.DEBUG)

model = get_model()

app = FastAPI(title="Question Answering",
              description=''' El objetivo es encontrar el espacio 
              de texto en el párrafo que responde a la pregunta 
              planteada.''',
              version="0.1.0",
              )

@app.post("/qas/")
async def get_segmentation_map(context: str = Query(..., min_length=3), question: str = Query(..., min_length=3)):
    '''Get quention answering'''
    logging.debug("ejecutar modelo...")
    if context and question:
        result = get_result(model, context, question)
        logging.debug("modelo ejecutado...")
        return Response(result)
    return {"items": "Null"}

