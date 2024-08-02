import streamlit as st
import arxiv
import datetime as dt

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

def select_paper(title: str, auth: str, sum: str, pub: dt.date, links: list):
    """
    On click of a paper button, the session state "selected paper" gets updated with the val of the selected paper

    :params:    - title (title of selected paper)
                - auth (authors ")
                - sum (abstract ")
                - pub (publish date)
                - links (commonly a list of 2 links: 1. abstract/download page of paper, 2. pdf)
    """
    st.session_state.selected_paper = {"title": title, "authors": auth, "summary": sum, "published": pub, "links": links}

# Page title
st.title("Arxiv Paper Search Engine")

# Add welcome message for user
st.write("Welcome!")


# Search bar
# add doc
# 
# 
col_searchbar, col_searchbutton = st.columns([3, 1])

with st.container():
    with col_searchbar:
        input_searchbar = st.text_input(label= "searchbar", placeholder= "Your query", key="searchbar_input", label_visibility= "collapsed")

        with st.expander("INFO: Query construction"):
            st.write("For a detailed query constructions guide see the official arxiv api documentation 5.1 Details of Query Constructions https://info.arxiv.org/help/api/user-manual.html#query_details")

    with col_searchbutton:
        if st.button(label="Search!", use_container_width= False, key="searchbutton", type= "primary"):
            client = arxiv.Client()
            search = arxiv.Search(query= input_searchbar, max_results= 50, sort_by= arxiv.SortCriterion.SubmittedDate)          # To do: implement feat to limit num of res and sort by 
            st.session_state.search_results = list(client.results(search))


# Display results 
#
#
#
results_tab, vec_space_tab = st.tabs(["Search results", "Vector space model"])

with results_tab:

    col_title, col_info = st.columns(2)

    if st.session_state.search_results:
        max_title_length = max(len(r.title) for r in st.session_state.search_results)
        max_button_label = 106

    with col_title:
        con_title = st.container(height= 600, border= False) 
        with con_title:
            for idx, r in enumerate(st.session_state.search_results):
                display_title = (f"{idx+1}." + r.title[:max_button_label] + "...") if len(r.title) > max_button_label else f"{idx+1}." + r.title
                st.button(label= display_title, key=r.title, on_click= select_paper, args= (r.title, ', '.join([author.name for author in r.authors]), r.summary, str(r.published), r.links), use_container_width= False, help= "Click to see description.")

    with col_info:
        con_info = st.container(border= False)
        with con_info:
            if st.session_state.selected_paper:
                st.write(f"{st.session_state.selected_paper["title"]}, {st.session_state.selected_paper["published"][:10]}")
                st.write(f"Authors: {st.session_state.selected_paper["authors"]}")
                st.write(st.session_state.selected_paper["summary"])
                st.write(f"Link to abs: {st.session_state.selected_paper["links"][0]}")
                st.write(f"Link to pdf: {st.session_state.selected_paper["links"][1]}")

with vec_space_tab:

    st.write("Placeholder for vec space graphic for the embedded feat")