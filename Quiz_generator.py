import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Read the contents of the TXT file
with open('[English] Cybersecurity IDR_ Incident Detection & Response _ Google Cybersecurity Certificate [DownSub.com].txt', 'r') as file:
    text = file.read()

# Replace line breaks and paragraphs with spaces
text = re.sub(r'[\n\r]+', ' ', text)

# Save the cleaned text to a new file or use it for further analysis
with open('cleaned_text.txt', 'w') as file:
    file.write(text)

# Read the contents of the TXT file
with open('cleaned_text.txt', 'r') as file:
    subtitle_string = file.read()

tokenizer = AutoTokenizer.from_pretrained("Sujithanumala/QuizBot.AI-base")
model = AutoModelForSeq2SeqLM.from_pretrained("Sujithanumala/QuizBot.AI-base")
max_token_limit = 500
num_quizzes = 20

sentences = subtitle_string.split(". ")
chunked_segments = []
temp_segment = ""

for sentence in sentences:
    # Tokenize the current sentence
    # Check if adding the current sentence exceeds the maximum token limit
    if len(temp_segment.split()) + len(tokenizer.encode(sentence, add_special_tokens=False)) > max_token_limit:
        # Add the current chunked segment to the list of chunked segments
        chunked_segments.append(temp_segment.strip())
        temp_segment = ""

    temp_segment += sentence + ". "

# Add the last segment to the list of chunked segments, if it exists
if temp_segment:
    chunked_segments.append(temp_segment.strip())

# Now, chunked_segments contains the smaller segments of the subtitle string

generated_quizzes = []
for segment in chunked_segments:
    # Tokenize the segment
    tokenized_segment = tokenizer(segment, return_tensors="pt", truncation=True, padding=True)
    # Forward pass through the model
    with torch.no_grad():
        outputs = model.generate(
            input_ids=tokenized_segment.input_ids,
            attention_mask=tokenized_segment.attention_mask,
            max_length=100,  # Adjust the max_length as desired
            num_return_sequences=1,  # Adjust the num_return_sequences as desired
            num_beams=5,  # Adjust the num_beams as desired
            early_stopping=True
        )

    # Decode the generated quizzes
    generated_quizzes.extend([tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
