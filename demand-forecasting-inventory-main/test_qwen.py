
import os
from openai import OpenAI


# ============================================================
# CHECK API KEY
# ============================================================

api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:

    print("❌ DASHSCOPE_API_KEY 没有读取到")
    print("请先在 PowerShell 中设置 API Key")
    exit()

print("✅ DASHSCOPE_API_KEY 已读取")


# ============================================================
# QWEN CLIENT
# ============================================================

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "qwen3.8-max"


# ============================================================
# TEST API
# ============================================================

try:

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": "请用中文简单解释什么是 Safety Stock。"
            }
        ],

        temperature=0.3
    )

    print("\n==============================")
    print("QWEN RESPONSE")
    print("==============================\n")

    print(
        response.choices[0].message.content
    )

except Exception as e:
    print("\n==============================")
    print("❌ API CALL FAILED")
    print("==============================\n")
    print("Error Type:", type(e).__name__)
    print("Error:", repr(e))
    print("Cause:", repr(e.__cause__))
    print("Context:", repr(e.__context__))
