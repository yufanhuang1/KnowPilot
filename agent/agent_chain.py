from langchain.agents import AgentExecutor, ConversationalChatAgent, initialize_agent, AgentType
from langchain.agents.agent import AgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_core.agents import AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain.agents.conversational.output_parser import ConvoOutputParser

from models.dashscope_model import get_llm
from agent.memory import get_memory
from tools.tools import get_tools

MAX_RETRIES = 3

class CustomAgentExecutor:
    def __init__(self):
        self.llm = get_llm()
        self.tools = get_tools()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # 自定义中文 Prompt
        tool_names = ", ".join([tool.name for tool in self.tools])
        system_prompt = f"""
        你是一个 AI Agent，能调用以下工具解决问题：
        {tool_names}
        
        请严格使用以下格式：
        
        思考: 你现在要做什么
        操作: 工具名称
        操作输入: 输入内容
        
        当你知道答案时，用：
        思考: 我知道了
        最终答案: 答案文本
        
         只能使用以上格式，否则会报错。"""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = ConversationalChatAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        self.output_parser = ConvoOutputParser()

    def run(self, input_text: str) -> str:
        intermediate_steps = []
        user_input = {"input": input_text, "chat_history": []}

        for attempt in range(MAX_RETRIES):
            try:
                # 1. 生成 Agent 输出
                agent_output = self.agent.plan(intermediate_steps, **user_input)

                # 2. AgentFinish 表示已经回答完毕
                if isinstance(agent_output, AgentFinish):
                    return agent_output.return_values["output"]

                # 3. AgentAction -> 调用工具
                tool = next((t for t in self.tools if t.name == agent_output.tool), None)
                if not tool:
                    raise ValueError(f"未知工具: {agent_output.tool}")
                observation = tool.run(agent_output.tool_input)

                intermediate_steps.append((agent_output, observation))

            except OutputParserException as e:
                print(f"[WARN] 第 {attempt + 1} 次解析失败，错误：{str(e)}")
                if attempt >= MAX_RETRIES - 1:
                    return "❌ 很抱歉，我无法理解模型的输出格式。"
                continue  # retry

        return "⚠️ 未知错误"



'''
def create_agent():
    llm = get_llm()
    memory = get_memory()
    tools = get_tools()

    # 获取工具名称列表
    tool_names = ", ".join([tool.name for tool in tools])

    system_prompt = f"""You are a helpful AI assistant. You have access to the following tools:

    {tool_names}

    Use the following format:

    Thought: What do you want to do
    Action: The action to take, must be one of [{tool_names}]
    Action Input: The input to the action

    ... (this can repeat several times)

    When you have the final answer, use:

    Thought: I have the final answer
    Final Answer: [your answer]
    """

    # 构造 prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 构造 Agent
    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        prompt=prompt,
        handle_parsing_errors=True,  # 处理解析错误
        verbose=True  # 输出中间步骤
    )

    # Agent 执行器
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        handle_parsing_errors=True,
        verbose=True
    )

    return agent_executor
'''