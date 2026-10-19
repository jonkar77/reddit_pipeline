from groq import Groq

client = Groq()
completion = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=[
      {
        "role": "user",
        "content": ""
      }
    ],
    temperature=0.6,
    max_completion_tokens=2048,
    top_p=0.95,
    reasoning_effort="default",
    stream=True,
    stop=None
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
