import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import nltk
from nltk.tokenize import sent_tokenize
from anthropic import Anthropic

nltk.download('punkt')

def initialize_clients():
    if 'openai_client' not in st.session_state or st.session_state.openai_key != st.session_state.get('prev_openai_key'):
        st.session_state.openai_client = OpenAI(api_key=st.session_state.openai_key)
        st.session_state.prev_openai_key = st.session_state.openai_key

    if 'anthropic_client' not in st.session_state or st.session_state.anthropic_key != st.session_state.get('prev_anthropic_key'):
        st.session_state.anthropic_client = Anthropic(api_key=st.session_state.anthropic_key)
        st.session_state.prev_anthropic_key = st.session_state.anthropic_key

# Sidebar per la configurazione API
with st.sidebar:
    st.title("Configurazione API")
    st.session_state.openai_key = st.text_input("Chiave OpenAI", type="password", key="openai_key_input")
    st.session_state.anthropic_key = st.text_input("Chiave Claude", type="password", key="anthropic_key_input")
    
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
        st.error(f"Errore nel recupero dell'URL {url}: {e}")
        return ""

def generate_image_prompt(keyword, tone, article_content):
    try:
        prompt = f"""
        Basandoti sulle seguenti informazioni, crea un prompt dettagliato e creativo per generare un'immagine usando DALL-E:

        Parola chiave: {keyword}
        Tono: {tone}
        Contenuto dell'articolo: {article_content[:500]}  # Usando i primi 500 caratteri dell'articolo

        Il prompt dovrebbe:
        1. Essere vivido e descrittivo
        2. Riflettere la parola chiave e il tono
        3. Essere correlato al contenuto dell'articolo
        4. Essere adatto alla generazione di immagini con DALL-E
        5. Essere lungo circa 50-100 parole

        Genera il prompt per l'immagine:
        """

        response = st.session_state.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sei un ingegnere creativo di prompt per la generazione di immagini."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Errore nella generazione del prompt per l'immagine: {e}")
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
        st.error(f"Errore nella generazione dell'immagine: {e}")
        return ""

def generate_article_with_claude(combined_text, keyword, target_audience, tone):
    try:
        prompt = f"""
        Sei un esperto scrittore di contenuti SEO e specialista di marketing digitale. Il tuo compito è creare un articolo di blog di alta qualità, originale e ottimizzato per SEO in italiano, basato sulle seguenti informazioni:

        Parola chiave: {keyword}
        Pubblico target: {target_audience}
        Tono: {tone}

        Usa il seguente testo come fonte di informazioni e ispirazione, ma non copiarlo direttamente. Invece, usalo come punto di partenza per creare contenuti completamente originali:

        {combined_text}

        Il tuo articolo dovrebbe:
        1. Avere un titolo accattivante che includa la parola chiave principale in modo naturale
        2. Includere una meta descrizione ottimizzata per SEO di circa 155-160 caratteri
        3. Essere ben strutturato con una chiara introduzione, corpo e conclusione
        4. Utilizzare titoli H2 e H3 per organizzare il contenuto, incorporando parole chiave secondarie dove appropriato
        5. Incorporare naturalmente la parola chiave principale in tutto il testo, mirando a una densità di parole chiave di circa 1-2%
        6. Includere almeno un elenco numerato o puntato per migliorare la leggibilità
        7. Essere informativo, prezioso e pratico per il pubblico target
        8. Mantenere il tono specificato in tutto il contenuto
        9. Essere lungo almeno 1000 parole, preferibilmente tra 1200-1500 parole per una copertura completa
        10. Includere segnaposto per link interni ed esterni dove appropriato (usa [LINK INTERNO] e [LINK ESTERNO] come segnaposto)
        11. Concludere con una forte call-to-action (CTA) rilevante per l'argomento e il pubblico

        Altre best practice SEO da implementare:
        - Usa variazioni della parola chiave principale e termini correlati per migliorare il SEO semantico
        - Includi almeno una statistica o un dato rilevante, citando la fonte
        - Scrivi in voce attiva e usa parole di transizione per migliorare il flusso
        - Ottimizza per i featured snippet includendo una breve risposta diretta a una domanda comune relativa all'argomento
        - Includi una sezione di domande frequenti (FAQ) alla fine con 3-5 domande pertinenti e risposte concise

        Formatta l'articolo in Markdown, includendo segnaposto per tre immagini: ![Immagine 1](segnaposto_immagine_1), ![Immagine 2](segnaposto_immagine_2), ![Immagine 3](segnaposto_immagine_3).

        Ricorda, l'obiettivo è creare un articolo altamente originale, coinvolgente e ottimizzato per SEO che fornisca un valore reale al lettore mentre mira efficacemente alla parola chiave e al pubblico specificati.
        """

        response = st.session_state.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,  # Aumentato per consentire contenuti più lunghi
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    except Exception as e:
        st.error(f"Errore nella generazione dell'articolo con Claude: {e}")
        return ""

def main():
    st.title("Generatore di Articoli di Blog Ottimizzati per SEO")
    
    # Controlla se le chiavi API sono impostate
    if 'openai_client' not in st.session_state or 'anthropic_client' not in st.session_state:
        st.warning("Per favore, imposta le tue chiavi API nella barra laterale per continuare.")
        return

    # Campi di input
    url1 = st.text_input("Inserisci il primo URL")
    url2 = st.text_input("Inserisci il secondo URL")
    url3 = st.text_input("Inserisci il terzo URL")
    keyword = st.text_input("Inserisci la parola chiave")
    target_audience = st.text_input("Inserisci il pubblico target")
    tone = st.text_input("Inserisci il tono")
    
    if st.button("Genera Articolo"):
        with st.spinner("Estrazione del testo dagli URL..."):
            # Estrai il testo dagli URL
            text1 = extract_text_from_url(url1)
            text2 = extract_text_from_url(url2)
            text3 = extract_text_from_url(url3)
            
            if not text1 or not text2 or not text3:
                st.error("Impossibile estrarre il testo da uno o più URL.")
                return
        
        with st.spinner("Combinazione e elaborazione del testo..."):
            # Combina i testi
            combined_text = f"{text1}\n\n{text2}\n\n{text3}"
        
        with st.spinner("Generazione dell'articolo con Claude 3.5 Sonnet..."):
            # Genera l'articolo con Claude
            article = generate_article_with_claude(combined_text, keyword, target_audience, tone)
            
            if not article:
                st.error("Impossibile generare l'articolo con Claude.")
                return
        
        with st.spinner("Generazione dei prompt per le immagini con ChatGPT..."):
            # Genera i prompt per le immagini
            prompt1 = generate_image_prompt(keyword, tone, article)
            prompt2 = generate_image_prompt(keyword, tone, article)
            prompt3 = generate_image_prompt(keyword, tone, article)
        
        with st.spinner("Generazione delle immagini con DALL-E..."):
            # Genera le immagini
            image1 = generate_image(prompt1)
            image2 = generate_image(prompt2)
            image3 = generate_image(prompt3)
            
            if not image1 or not image2 or not image3:
                st.error("Impossibile generare una o più immagini.")
                return
        
        # Sostituisci i segnaposto delle immagini nell'articolo
        article = article.replace("![Immagine 1](segnaposto_immagine_1)", f"![Immagine 1]({image1})")
        article = article.replace("![Immagine 2](segnaposto_immagine_2)", f"![Immagine 2]({image2})")
        article = article.replace("![Immagine 3](segnaposto_immagine_3)", f"![Immagine 3]({image3})")
        
        # Visualizza l'articolo
        st.subheader("Articolo Generato:")
        st.markdown(article)
        
        # Visualizza i prompt generati
        st.subheader("Prompt Generati per le Immagini:")
        st.write(f"Prompt 1: {prompt1}")
        st.write(f"Prompt 2: {prompt2}")
        st.write(f"Prompt 3: {prompt3}")

        # Opzione per scaricare l'articolo
        st.download_button(
            label="Scarica Articolo",
            data=article,
            file_name="articolo_seo.md",
            mime="text/markdown",
        )

if __name__ == "__main__":
    main()
