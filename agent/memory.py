from langchain.memory import ConversationBufferMemory

def get_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key = "input",  # 显式声明输入键
        output_key = "output"  # 显式声明输出键
    )