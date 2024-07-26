import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import nltk
from nltk.tokenize import sent_tokenize
from anthropic import Anthropic

# Ensure you have the necessary NLTK data files
nltk.download('punkt')

# Function to initialize or update clients
def initialize_clients():
    if 'openai_client' not in st.session_state or st.session_state.openai_key != st.session_state.get('prev_openai_key'):
        st.session_state.openai_client = OpenAI(api_key=st.session_state.openai_key)
        st.session_state.prev_openai_key = st.session_state.openai_key

    if 'anthropic_client' not in st.session_state or st.session_state.anthropic_key != st.session_state.get('prev_anthropic_key'):
        st.session_state.anthropic_client = Anthropic(api_key=st.session_state.anthropic_key)
        st.session_state.prev_anthropic_key = st.session_state.anthropic_key

# Sidebar for API configuration
with st.sidebar:
    st.title("Configurazione API")
    st.session_state.openai_key = st.text_input("OpenAI Key", type="password", key="openai_key_input")
    st.session_state.anthropic_key = st.text_input("Claude Key", type="password", key="anthropic_key_input")
    
    if st.session_state.openai_key and st.session_state.anthropic_key:
        initialize_clients()
        st.success("Chiavi API impostate con successo!")
    else:
        st.warning("Per favore, inserisci entrambe le chiavi API per continuare.")

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL {url}: {e}")
        return ""

def generate_image_prompt(keyword, tone, article_content):
    try:
        prompt = f"""
        Based on the following information, create a detailed and creative prompt for generating an image using DALL-E:

        Keyword: {keyword}
        Tone: {tone}
        Article content: {article_content[:500]}  # Using first 500 characters of the article

        The prompt should:
        1. Be vivid and descriptive
        2. Reflect the keyword and tone
        3. Relate to the content of the article
        4. Be suitable for DALL-E image generation
        5. Be about 50-100 words long

        Generate the image prompt:
        """

        response = st.session_state.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative prompt engineer for image generation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating image prompt: {e}")
        return ""

def generate_image(prompt):
    try:
        response = st.session_state.openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        return image_url
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return ""

def generate_article_with_claude(combined_text, keyword, target_audience, tone):
    try:
        prompt = f"""
        You are an expert SEO content writer. Your task is to create a high-quality, SEO-optimized blog article based on the following information:

        Keyword: {keyword}
        Target Audience: {target_audience}
        Tone: {tone}

        Use the following text as a source of information, but do not copy it directly. Instead, use it as inspiration to create original content:

        {combined_text}

        Your article should:
        1. Have an engaging title that includes the keyword
        2. Include an SEO-optimized meta description
        3. Be well-structured with headings and subheadings
        4. Naturally incorporate the keyword throughout the text
        5. Be informative and valuable to the target audience
        6. Maintain the specified tone throughout
        7. Be between 800-1000 words long

        Please format the article in Markdown, including placeholders for three images: ![Image 1](image_placeholder_1), ![Image 2](image_placeholder_2), ![Image 3](image_placeholder_3).
        """

        response = st.session_state.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    except Exception as e:
        st.error(f"Error generating article with Claude: {e}")
        return ""

def main():
    st.title("SEO-Optimized Blog Article Generator")
    
    # Check if API keys are set
    if 'openai_client' not in st.session_state or 'anthropic_client' not in st.session_state:
        st.warning("Please set your API keys in the sidebar to continue.")
        return

    # Input fields
    url1 = st.text_input("Enter the first URL")
    url2 = st.text_input("Enter the second URL")
    url3 = st.text_input("Enter the third URL")
    keyword = st.text_input("Enter the keyword")
    target_audience = st.text_input("Enter the target audience")
    tone = st.text_input("Enter the tone")
    
    if st.button("Generate Article"):
        st.write("Extracting text from URLs...")
        
        # Extract text from URLs
        text1 = extract_text_from_url(url1)
        text2 = extract_text_from_url(url2)
        text3 = extract_text_from_url(url3)
        
        if not text1 or not text2 or not text3:
            st.error("Failed to extract text from one or more URLs.")
            return
        
        st.write("Combining and processing text...")
        
        # Combine texts
        combined_text = f"{text1}\n\n{text2}\n\n{text3}"
        
        st.write("Generating article with Claude 3.5 Sonnet...")
        
        # Generate article with Claude
        article = generate_article_with_claude(combined_text, keyword, target_audience, tone)
        
        if not article:
            st.error("Failed to generate article with Claude.")
            return
        
        st.write("Generating image prompts with ChatGPT...")
        
        # Generate image prompts
        prompt1 = generate_image_prompt(keyword, tone, article)
        prompt2 = generate_image_prompt(keyword, tone, article)
        prompt3 = generate_image_prompt(keyword, tone, article)
        
        st.write("Generating images with DALL-E...")
        
        # Generate images
        image1 = generate_image(prompt1)
        image2 = generate_image(prompt2)
        image3 = generate_image(prompt3)
        
        if not image1 or not image2 or not image3:
            st.error("Failed to generate one or more images.")
            return
        
        # Replace image placeholders in the article
        article = article.replace("![Image 1](image_placeholder_1)", f"![Image 1]({image1})")
        article = article.replace("![Image 2](image_placeholder_2)", f"![Image 2]({image2})")
        article = article.replace("![Image 3](image_placeholder_3)", f"![Image 3]({image3})")
        
        # Display the article
        st.markdown(article)
        
        # Display the generated prompts
        st.subheader("Generated Image Prompts:")
        st.write(f"Prompt 1: {prompt1}")
        st.write(f"Prompt 2: {prompt2}")
        st.write(f"Prompt 3: {prompt3}")

if __name__ == "__main__":
    main()