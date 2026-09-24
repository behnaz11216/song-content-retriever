import torch
import torch.nn.functional as F
import re
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import warnings

warnings.filterwarnings('ignore')

# ----------SETUP ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- مدل BERT ----------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.to(DEVICE)
bert_model.eval()

# ---------- FUNCTIONS----------

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = text.replace("\'", "").strip()
    return [text]

def encode_texts(list_of_texts, max_input=512):
    tokenized = tokenizer.batch_encode_plus(list_of_texts, add_special_tokens=True, padding=True, truncation=True, max_length=max_input)
    input_ids = tokenized['input_ids']
    attention_masks = tokenized['attention_mask']

    mean_embeddings = []
    for ids, mask in tqdm(zip(input_ids, attention_masks), total=len(input_ids), desc="Encoding"):
        ids_tensor = torch.tensor([ids]).to(DEVICE)
        mask_tensor = torch.tensor([mask]).to(DEVICE)
        with torch.no_grad():
            outputs = bert_model(ids_tensor, attention_mask=mask_tensor)
            hidden_states = outputs[0].squeeze(0)
            valid_embeddings = hidden_states[mask_tensor[0] != 0]
            mean_embedding = valid_embeddings.mean(dim=0)
            mean_embeddings.append(mean_embedding.unsqueeze(0))

    return F.normalize(torch.cat(mean_embeddings), p=2, dim=1)


def retrieve_and_classify(song_embedding, question_embeddings, safe_answer_embeddings, unsafe_answer_embeddings, threshold=0.5):
    similarity = torch.matmul(question_embeddings, song_embedding.T).reshape(-1)
    top_question_indices = torch.topk(similarity, k=5).indices.tolist()

    safe_score = 0
    unsafe_score = 0

    print("\nRelevant Questions Retrieved:")
    for idx in top_question_indices:
        safe_sim = torch.matmul(safe_answer_embeddings[idx], song_embedding.T).item()
        unsafe_sim = torch.matmul(unsafe_answer_embeddings[idx], song_embedding.T).item()

        print(f"- Q: {questions[idx]}")
        print(f"  Safe Match: {safe_sim:.2f}, Unsafe Match: {unsafe_sim:.2f}")

        if unsafe_sim > safe_sim and unsafe_sim > threshold:
            unsafe_score += 1
        elif safe_sim > unsafe_sim and safe_sim > threshold:
            safe_score += 1

    print(f"\nSafe score: {safe_score}, Unsafe score: {unsafe_score}")

    if unsafe_score > safe_score:
       return "❌ Not Child-Safe"
    elif unsafe_score == 0 and safe_score > 0:
       return "✅ Child-Safe"
    else:
       return "⚠️ Unclear — manual review needed"

# ----------ANSWER AND QUESTION DATA ----------

questions = [
    "Does this song contain violence?",
    "Does this song have explicit language?",
    "Is the song's theme appropriate for children?",
    "Does the song reference weapons?",
    "Does the song promote positive values?",
    "Is there sexual content in the song?",
    "Is the song educational?",
    "Does the song promote emotional resilience?"
]

safe_answers = [
    "No, the song does not contain violence.",
    "No, the song does not have explicit language.",
    "Yes, the song's theme is appropriate for children.",
    "No, the song does not reference weapons.",
    "Yes, the song promotes positive values.",
    "No, the song does not include sexual content.",
    "Yes, the song has educational value.",
    "Yes, the song promotes emotional resilience."
]

unsafe_answers = [
    "Yes, the song contains violence.",
    "Yes, the song has explicit language.",
    "No, the song's theme is not appropriate for children.",
    "Yes, the song references weapons.",
    "No, the song promotes negative or inappropriate values.",
    "Yes, the song includes sexual content.",
    "No, the song does not have educational value.",
    "No, the song does not promote emotional resilience."
]

print("\nEmbedding questions and answers...")
question_embeddings = encode_texts(questions)
safe_answer_embeddings = encode_texts(safe_answers)
unsafe_answer_embeddings = encode_texts(unsafe_answers)

# ---------- SONGS ----------

songs = {
    "Demo Input 1": """
    A person describes friends helping each other and learning together.
    """,
    "Demo Input 2": """
    A person describes an argument and says they feel angry and want to hurt someone.
    """
}

# ---------- PROCESSING AND EVALUATING ----------

print("\nProcessing and evaluating songs...")

for song_name, lyrics in songs.items():
    clean_lyrics = clean_text(lyrics)
    song_embedding = encode_texts(clean_lyrics)

    print(f"\n🔎 Evaluating Song: {song_name}")
    result = retrieve_and_classify(song_embedding, question_embeddings, safe_answer_embeddings, unsafe_answer_embeddings)
    print(f"Final Verdict: {result}")
