import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ handles messages that contain the /start command."""
    await context.bot.send_message(chat_id=update.effective_chat.id, 
                                   text="Hi there, I am the animal facts bot, ask me about any animal of your choice!")
def llm_response(user_input):
    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable librarian specializing in animal facts. Share your findings in an informative way",
        },
        {"role": "user", "content": user_input},
    ] 
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    rep = outputs[0]["generated_text"].split("<|assistant|>")[-1]
    return rep

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles user messages"""
    user_message = update.message.text.lower() 
    answer = llm_response(user_message) 

    await context.bot.send_message(chat_id=update.effective_chat.id, text= answer)

if __name__ == '__main__':
    tok = "token"
    application = ApplicationBuilder().token(tok).build()
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    application.add_handler(message_handler)
    application.run_polling()