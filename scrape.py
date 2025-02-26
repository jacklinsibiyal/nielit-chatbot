import os
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from urllib.parse import urljoin
from webdriver_manager.chrome import ChromeDriverManager
import re
import pdfplumber
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

options = webdriver.ChromeOptions()
options.add_argument('--headless')  
options.add_argument('--no-sandbox')
options.add_argument("--disable-gpu")
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
if not os.getenv("GOOGLE_API_KEY"):
    load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
url = 'https://nva.nielit.gov.in/'

os.makedirs('doc', exist_ok=True)
os.makedirs('vector_store', exist_ok=True)

def scroll_down_page():
    """ Scrolls down the page to load dynamic content """
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(10)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def download_pdf(pdf_url):
    """ Downloads the PDF from the given URL , replaces existing file if already exists """
    if pdf_url and pdf_url.endswith('.pdf'):
        pdf_name = os.path.join('doc', pdf_url.split('/')[-1])
        print(f"Downloading: {pdf_name}")
        try:
            response = requests.get(pdf_url)
            response.raise_for_status()
            with open(pdf_name, 'wb') as pdf_file:
                pdf_file.write(response.content)
            print(f"Downloaded: {pdf_name}")

        except Exception as e:
            print(f"Failed to download {pdf_url}: {e}")

def scrape(url):
    driver.get(url)
    time.sleep(15) 
    
    scroll_down_page() 
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    text_data = []
    for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li', 'a', 'span', 'div']):
        text_content = tag.get_text(strip=True)
        if text_content:
            text_data.append(text_content)
    text_file = f'doc/data.txt'

    with open(text_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\n=== {url} ===\n\n")
        f.write("\n".join(text_data))
        f.write("\n" + "="*50 + "\n")
    
    links = soup.find_all('a')
    print(f"Found {len(links)} anchor tags on {url}")

    pattern = r"window\.open\(['\"](.*?)['\"]\)"
    
    for index, link in enumerate(links):
        try:
            onclick_attr = link.get('onclick')
            if onclick_attr:
                matches = re.findall(pattern, onclick_attr)
                if matches:
                    for relative_url in matches:
                        full_url = urljoin(url, relative_url)
                        print(f"Constructed PDF URL: {full_url}")
                        if full_url.lower().endswith('.pdf'):
                            download_pdf(full_url)
                        else:
                            print(f"Constructed URL is not a PDF: {full_url}")
                else:
                    print("No matching URL found in 'onclick' attribute.")
            else:
                print("No 'onclick' attribute found.")
        
        except Exception as e:
            print(f"Error processing link {index + 1}: {e}")

def vector_embedding():
    """ Vectorizes documents (PDF and TXT) and saves vectors to disk """
    try:
        print("🚀 Starting Vector Embedding Process...")

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        docs = []

        pdf_files = os.listdir("./doc")
        for file in pdf_files:
            file_path = f"./doc/{file}"
            
            text = ""
            if file.lower().endswith('.pdf'):
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
            
            elif file.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    text = txt_file.read()
            
            if text:
                text_document = Document(page_content=text, metadata={"source": file})
                docs.append(text_document)
            else:
                print(f"⚠️ No text extracted from: {file}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        vectors = FAISS.from_documents(final_documents, embeddings)
        vectors.save_local("vector_store/faiss_index")
        print("✅ Vector Store DB Is Ready and Saved Locally!")

    except Exception as e:
        print(f"❌ Error in Vector Embedding: {e}")

    except Exception as e:
        print(f"Error in vector embedding: {e}")

scrape(url)
vector_embedding()
driver.quit()
print("Scraping and Vector Embedding completed.")