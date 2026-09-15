from sentence_transformers import SentenceTransformer, util

# 1. Load the BGE-M3 model
print("Loading BGE-M3 Model...")
model = SentenceTransformer('BAAI/bge-m3')

# 2. Define two labels that are slightly different
old_label = "fac_idle_corr"
new_label = "factor_idle_correction"

# 3. Convert both labels into AI coordinates (embeddings)
vector1 = model.encode(old_label, convert_to_tensor=True)
vector2 = model.encode(new_label, convert_to_tensor=True)

# 4. Calculate similarity
similarity = util.cos_sim(vector1, vector2)

print("\n--- AI MATCHING RESULTS ---")
print(f"Old Label: {old_label}")
print(f"New Label: {new_label}")
print(f"Matching Score: {similarity.item():.4f} (Close to 1.0 means highly similar)")
