from PIL import Image
import io
from transformers import pipeline


def get_model():

    # qa = pipeline('question-answering', 
    #           model="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es", 
    #           tokenizer="mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es")
    def qa(context: str, question: str):
        return "Esto es una prueba"
    return qa

def get_result(qa, context, question, max_size=512):

    r = qa(context=context, question=question)

    return r
