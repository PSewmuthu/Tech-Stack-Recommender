import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class JobRecommender:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        # Clean skills column
        self.df['clean_skills'] = self.df['job_skill_set'].apply(
            self.clean_skills)

        # Initialize TF-IDF Vectorizer
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['clean_skills'])

    def clean_skills(self, skills):
        skills_str = ""
        try:
            # Convert string representation of list to an actual list
            skill_list = ast.literal_eval(skills)
            skills_str = ", ".join(
                [str(s).lower().replace(" ", "_") for s in skill_list])
        except:
            skills_str = skills.lower().strip()

        return skills_str
