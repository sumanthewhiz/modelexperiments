import onnxruntime_genai as onnxgenai

model = onnxgenai.Model("../models/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4")

tokenizer = onnxgenai.Tokenizer(model)

tokenizer_stream = tokenizer.create_stream()

#The chat prompt
prompttext = "Who is Mukesh Ambani?"

#search_options = ["max_length=2048"]

chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
prompt = f'{chat_template.format(input=prompttext)}'
input_tokens = tokenizer.encode(prompt)

params = onnxgenai.GeneratorParams(model)
#params.set_search_options(**search_options)
params.input_ids = input_tokens

#Generate the output
generator = onnxgenai.Generator(model, params)   

print("\nPrompt: "+prompttext+"\n")

try:
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(new_token), end='', flush=True)
        
except KeyboardInterrupt:
    print("--Ctrl+C pressed, aborting generation--")
        
print("\n")
