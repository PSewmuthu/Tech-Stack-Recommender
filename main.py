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

    def recommend_jobs(self, user_skills, top_n=5):
        user_input = ", ".join([s.lower().replace(" ", "_")
                               for s in user_skills])
        user_vec = self.tfidf.transform([user_input])

        # Calculate similarity
        scores = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
        self.df['similarity_score'] = scores

        # Get top N recommendations
        recommendations = self.df.sort_values(by='similarity_score', ascending=False)[
            ['job_title', 'category', 'similarity_score']].head(top_n)

        return recommendations
