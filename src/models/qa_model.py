from transformers import AutoModelForQuestionAnswering
from src.utils.mapper import configmapper

@configmapper.map("models","qa_model")
class AutoModelForQuestionAnswering(AutoModelForQuestionAnswering):
    def __init__(self, *args):
        super(AutoModelForQuestionAnswering, self).__init__()