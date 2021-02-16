from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering
from src.utils.mapper import configmapper

configmapper.map("models", "autotoken")(AutoModelForTokenClassification)
configmapper.map("models", "autospans")(AutoModelForQuestionAnswering)
