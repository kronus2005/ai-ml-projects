from sentence_transformers import SentenceTransformer, util
import os
import pdfplumber
import pandas as pd

# Load BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight & fast

def read_resume_text(file_path):
    text = ""
    if file_path.endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    return text

def load_resumes(folder_path):
    resumes = []
    filenames = []
    for fname in os.listdir(folder_path):
        if fname.endswith(('.pdf', '.txt')):
            full_path = os.path.join(folder_path, fname)
            resumes.append(read_resume_text(full_path))
            filenames.append(fname)
    return resumes, filenames

def main():
    # Load job description
    with open('job_description.txt', 'r', encoding='utf-8') as f:
        job_desc = f.read()

    # Embed job description
    job_embedding = model.encode(job_desc, convert_to_tensor=True)

    # Load and embed resumes
    resumes, filenames = load_resumes('resumes')
    resume_embeddings = model.encode(resumes, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.cos_sim(job_embedding, resume_embeddings)[0]

    # Create results DataFrame
    results = pd.DataFrame({
        'Resume': filenames,
        'Similarity': similarities.cpu().numpy()
    })

    results.sort_values(by='Similarity', ascending=False, inplace=True)
    print(results)

if __name__ == "__main__":
    main()
