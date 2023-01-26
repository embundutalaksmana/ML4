import streamlit as st
import pandas as pd
import re
from collections import defaultdict
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Prediksi Anime", page_icon=":tada:", layout="wide")
def create_page(content, page_title=""):
    st.title(page_title)
    st.write(content)
st.header("Prediksi Kemiripan Anime")
st.write("Oleh Kelompok\n"+
"- T.ilham Alsyamahdar - 2055301134\n - Nanda Kurniawan - 2055301104\n - Hidayatul Mukmin - 2055301134")

def create_page(content):
    st.write(content)

content = """Bagian ini berfungsi untuk melakukan Prediksi terhadap kemiripan data anime
        Pada aplikasi ini kami menggunakan algoritma linear kernel

        """
create_page(content)
st.write("---")

uploaded_file = st.file_uploader("File Anime csv", type=["csv"])
if uploaded_file:
    anime_df = pd.read_csv(uploaded_file)
    null_features = anime_df.columns[anime_df.isna().any()]
    anime_df[null_features].isna().sum()
    anime_df.dropna(inplace=True)
    st.dataframe(anime_df)
    def text_cleaning(text):
        text = re.sub(r'&quot;', '', text)
        text = re.sub(r'.hack//', '', text)
        text = re.sub(r'&#039;', '', text)
        text = re.sub(r'A&#039;s', '', text)
        text = re.sub(r'I&#039;', 'I\'', text)
        text = re.sub(r'&amp;', 'and', text)
        
        return text
    #EDA
    anime_df['name'] = anime_df['name'].apply(text_cleaning)
    st.title('Anime Types')
    type_count = anime_df['type'].value_counts()
    st.bar_chart(type_count)
    
    st.write("---")
    #heatmap korelasi dataset
    dataset_corr = anime_df.corr()
    fig = plt.figure()
    plt.title("Correlation Heatmap", fontsize=20)
    sns.heatmap(anime_df.corr(), annot=True)
    st.pyplot(fig)

    #menghitung jumlah genre dalam data anime
    all_genres = defaultdict(int)
    for genres in anime_df['genre']:
        for genre in genres.split(','):
            all_genres[genre.strip()] += 1


    
     #mengubah data genre dari dataframe anime_df menjadi representasi numerik yang dapat digunakan dalam proses analisis data.
    genres_str = anime_df['genre'].str.split(',').astype(str)
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 4), min_df=0)
    tfidf_matrix = tfidf.fit_transform(genres_str)

    #menghitung matriks similarity antara setiap pasangan anime dalam dataframe anime_df berdasarkan representasi numerik yang dihasilkan dari TfidfVectorizer
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(anime_df.index, index=anime_df['name'])

    def genre_recommendations(title, highest_rating=False, similarity=False):
     if highest_rating == False:
        if similarity == False:
        
            idx = indices[title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
        
            anime_indices = [i[0] for i in sim_scores]
        
            return pd.DataFrame({'Anime name': anime_df['name'].iloc[anime_indices].values,
                                 'Type': anime_df['type'].iloc[anime_indices].values})
    
        elif similarity == True:
        
            idx = indices[title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
        
            anime_indices = [i[0] for i in sim_scores]
            similarity_ = [i[1] for i in sim_scores]
        
            return pd.DataFrame({'Anime name': anime_df['name'].iloc[anime_indices].values,
                                 'Similarity': similarity_,
                                 'Type': anime_df['type'].iloc[anime_indices].values})
        
     elif highest_rating == True:
        if similarity == False:
        
            idx = indices[title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
        
            anime_indices = [i[0] for i in sim_scores]
        
            result_df = pd.DataFrame({'Anime name': anime_df['name'].iloc[anime_indices].values,
                                 'Type': anime_df['type'].iloc[anime_indices].values,
                                 'Rating': anime_df['rating'].iloc[anime_indices].values})
            
            return result_df.sort_values('Rating', ascending=False)
    
        elif similarity == True:
        
            idx = indices[title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
        
            anime_indices = [i[0] for i in sim_scores]
            similarity_ = [i[1] for i in sim_scores]
        
            result_df = pd.DataFrame({'Anime name': anime_df['name'].iloc[anime_indices].values,
                                 'Similarity': similarity_,
                                 'Type': anime_df['type'].iloc[anime_indices].values,
                                 'Rating': anime_df['rating'].iloc[anime_indices].values})
            
            return result_df.sort_values('Rating', ascending=False)

    anime_title = st.text_input('Enter anime title:').capitalize()

    if st.button('Submit'):
        try:
            st.text("Hasil")
            result = genre_recommendations(anime_title, highest_rating=True, similarity=True)
            st.dataframe(result)
        except:
            st.error("Anime not found in the database ()")
