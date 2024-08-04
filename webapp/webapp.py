import streamlit as st
import arxiv
import datetime as dt
import pandas as pd 

import plotly.express as px
import plotly.graph_objects as go
import colorsys

from transformers import BertTokenizer, BertModel
import torch
from sklearn.decomposition import PCA


import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title= "Arvix Paper Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state= "collapsed"
)

# Session state and functions
# add doc s 
#
#
if "selected_paper" not in st.session_state:
    st.session_state.selected_paper = None

if "search_results" not in st.session_state:
    st.session_state.search_results = []


# Functions
#
#
def create_search_results_df(search_results) -> pd.DataFrame:
    """
    Creates df from list of search results.

    :params:    - search_results (list): list of search results with fixed attributes (title, authors, published, summary, primary_category, categories, links)

    :returns:    - pd.Dataframe: df, each row represents one search result

    :example:   
        >>> search_results = [Suchergebnis1, Suchergebnis2, ...]
        >>> df = create_search_results_df(search_results)
        >>> print(df.head())
    """

    data = []
    for r in search_results:
        data.append({
            'title': r.title,
            'authors': ', '.join([author.name for author in r.authors]),
            'published': r.published.date(),
            'summary': r.summary,
            'primary_category': r.primary_category,
            'categories': ', '.join(r.categories),
            "links": [link.href for link in r.links]
        })
    return pd.DataFrame(data)

def select_paper(title: str, auth: str, sum: str, pub: dt.date, links: list):
    """
    On click of a paper button, the session state "selected paper" gets updated with the val of the selected paper

    :params:    - title (str): title of selected paper
                - auth (str): authors "
                - sum (str): abstract "
                - pub (str): publish date
                - links (list): commonly a list of 2 links (1. abstract/download page of paper, 2. pdf)

    :returns:   - dict: update the attributes of the dict in st.session_state with those of the selected paper
    """
    st.session_state.selected_paper = {"title": title, "authors": auth, "summary": sum, "published": pub, "links": links}

def generate_unique_colors(num_colors: int) -> list:
    """
    Generates a list of unique colors in hex format.

    :params:    - num_colors (int): number of colors to generate
    :returns:   - list: list of unique colors in hex format
    :example:
        >>> colors = generate_unique_colors(5)
        >>> print(colors)
    """        
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        lightness = 0.5
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors

def select_row():
    pass
    

# Page title
st.title("Arxiv Paper Search Engine")

# Add welcome message for user
st.write("Welcome!") # add this to sidebar 

# Search bar
# add doc
# 
# 
col_searchbar, col_searchbutton = st.columns([3, 1])

with st.container():
    with col_searchbar:
        input_searchbar = st.text_input(label= "searchbar", placeholder= "Your query", key="searchbar_input", label_visibility= "collapsed", help= "Enter your search query here.")

        col_max_results, col_sort_by, col_sort_order = st.columns(3)
        with col_max_results:
            input_max_results = st.slider(label= "max_results", min_value= 50, max_value= 2000, value= 100, step= 50, key= "max_results_slider", label_visibility= "collapsed", help= "Choose the number of search results to display.")
        with col_sort_by:
            input_sort_by = st.radio(label= "sort_by", options= ["Relevance", "Last Updated Date", "Submitted Date"], key= "sort_by_selectbox", label_visibility= "collapsed", horizontal= True, help= "Choose the sorting criterion.")
        with col_sort_order:
            input_sort_order = st.radio(label= "sort_order", options= ["Descending", "Ascending"], key= "sort_order_selectbox", label_visibility= "collapsed", horizontal= True, help= "Choose the sorting order.")

        with st.expander("INFO: Query construction"):
            st.write("For a detailed query constructions guide see the official arxiv api documentation 5.1 Details of Query Constructions https://info.arxiv.org/help/api/user-manual.html#query_details")

    with col_searchbutton:
        if st.button(label="Search!", use_container_width= False, key="searchbutton", type= "primary"):
            client = arxiv.Client()
            search = arxiv.Search(
                query= input_searchbar, 
                max_results= input_max_results, 
                sort_by= arxiv.SortCriterion.Relevance if input_sort_by == "Relevance" else arxiv.SortCriterion.UpdatedDate if input_sort_by == "Last Updated Date" else arxiv.SortCriterion.SubmittedDate, 
                sort_order= arxiv.SortOrder.Descending if input_sort_order == "Descending" else arxiv.SortOrder.Ascending
                )                
            search_results = list(client.results(search))
            st.session_state.search_results_df = create_search_results_df(search_results)

# Display results 
#
#
#
results_tab, vec_space_tab = st.tabs(["Search results", "Vector space model"])

with results_tab:
    col_title, col_info = st.columns(2)

    if "search_results_df" in st.session_state and not st.session_state.search_results_df.empty:
        max_title_length = max(st.session_state.search_results_df['title'].apply(len))
        max_button_label = 106

        with col_title:
            con_title = st.container(height=600, border=False)
            with con_title:
                for idx, row in st.session_state.search_results_df.iterrows():
                    display_title = (f"{idx+1}." + row['title'][:max_button_label] + "...") if len(row['title']) > max_button_label else f"{idx+1}." + row["title"]
                    st.button(label= display_title, key= display_title, on_click= select_paper, args= (row["title"], row["authors"], row["summary"], str(row["published"]), row["links"]), use_container_width= False, help= "Click to see description.")

    with col_info:
        con_info = st.container(border= False)
        with con_info:
            if st.session_state.selected_paper:
                st.write(f"{st.session_state.selected_paper["title"]}, {st.session_state.selected_paper["published"][:10]}")
                st.write(f"Authors: {st.session_state.selected_paper["authors"]}")
                st.write(st.session_state.selected_paper["summary"])

                if len(st.session_state.selected_paper["links"]) == 3:
                    st.write(f"Link to doi: {st.session_state.selected_paper["links"][0]}")
                    st.write(f"Link to abs: {st.session_state.selected_paper["links"][1]}")
                    st.write(f"Link to pdf: {st.session_state.selected_paper["links"][2]}")
                else:
                    st.write(f"Link to abs: {st.session_state.selected_paper["links"][0]}")
                    st.write(f"Link to pdf: {st.session_state.selected_paper["links"][1]}")
            
with vec_space_tab:

    if "search_results_df" in st.session_state and not st.session_state.search_results_df.empty:
        
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        st.session_state.search_results_df["emb_ti_auth_sum"] = None

        for idx, row in st.session_state.search_results_df.iterrows():
            paper_info = row["authors"] + ", " + row["title"] + ", " + row["summary"]
            inputs = tokenizer(paper_info, return_tensors= "pt", max_length= 512, truncation= True, padding= True)

            with torch.no_grad():
                outputs = model(**inputs)

            output_cls_token_emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            st.session_state.search_results_df.at[idx, "emb_ti_auth_sum"] = output_cls_token_emb

        pca = PCA(n_components= 3)
        embeddings_reduced = pca.fit_transform(st.session_state.search_results_df["emb_ti_auth_sum"].tolist())
        st.session_state.search_results_df["emb_ti_auth_sum_reduced"] = embeddings_reduced.tolist()

        unique_categories = st.session_state.search_results_df["primary_category"].unique()
        colors = generate_unique_colors(len(unique_categories))
        color_mapping = {category: color for category, color in zip(unique_categories, colors)}

        traces = []
        for category in unique_categories:
            category_mask = st.session_state.search_results_df["primary_category"] == category
            trace = go.Scatter3d(
                x=embeddings_reduced[category_mask, 0],
                y=embeddings_reduced[category_mask, 1],
                z=embeddings_reduced[category_mask, 2],
                mode="markers+text",
                marker=dict(
                    size=5,
                    color= color_mapping.get(category, "black"),
                ),
                text= st.session_state.search_results_df[category_mask].index.values,
                name= category
            )
            traces.append(trace)

        fig = go.Figure(data= traces, layout= dict(width= 2000, height= 600, title= "3D PCA of paper embeddings", legend_title= dict(text= "Categories"), scene= dict(camera= dict(eye= dict(x= .7, y= .7, z= .7)))))

        st.plotly_chart(fig, selection_mode= "points")
        st.write(st.session_state.search_results_df)

# To do's:
#
# - Implement feat to show embeddings by category (now) or by none (all in one color an click on point to highlight paper in df) 
# - Implement feat to cluster paper in emebdding space (on-off-switch to activate color by cluster)
# - Implement feat for cos sim between selected data points in vec space tab (only available if 2 points are selected -> all in one color mode -> calc for every point in df an give option to sort -> see two papers that are most sim)

# - Fix warning occuring when embedding paper info with bert

# - Fine tune own model with arxiv data (lama model), model should be able to generate abstracts from papers (user selects paper via search, paper gets downlaoded and fed into model, model generates abstract, user can compare generated abstract with original abstract and vote which is better)
# - Implement tab for user to compare generated abs with original abs 

