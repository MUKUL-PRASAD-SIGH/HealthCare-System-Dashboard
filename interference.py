from transformers import AutoTokenizer, AutoModelForCausalLM

def get_diagnosis(symptoms):
    tokenizer = AutoTokenizer.from_pretrained('./llm_model')
    model = AutoModelForCausalLM.from_pretrained('./llm_model')

    input_text = f"Symptoms: {symptoms}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    diagnosis = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return diagnosis
