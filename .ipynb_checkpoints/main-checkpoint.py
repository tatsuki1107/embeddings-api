import openai
import os

openai.organization = "org-qu4lCRmyoAcUXUOigj76QGnv"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list()


