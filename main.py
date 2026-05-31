import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_skills(skills):
    skills_str = ""
    try:
        # Convert string representation of list to an actual list
        skill_list = ast.literal_eval(skills)
        skills_str = ", ".join(
            [str(s).lower().replace(" ", "_") for s in skill_list])
    except:
        skills_str = skills.lower()

    return skills_str
