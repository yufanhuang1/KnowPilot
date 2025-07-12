from langchain.agents import AgentExecutor, ConversationalChatAgent, initialize_agent, AgentType
from langchain.agents.agent import AgentOutputParser
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_core.agents import AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain.agents.conversational.output_parser import ConvoOutputParser
from agent.model_manager import get_llm
from agent.rag_qa import list_knowledge_bases, load_retriever
from tools.tools import get_tools
from langchain.tools import Tool

class CustomAgentExecutor:
    def __init__(self,llm=None):
        self.llm = llm or get_llm("deepseek")  # 默认模型
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        base_tools = get_tools()
        self.tools = base_tools

        # 添加所有已有知识库为独立工具
        for kb in list_knowledge_bases():
            self.tools.append(self._create_rag_tool(kb))

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
        
         """

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
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            handle_parsing_errors=True,
            verbose=True
        )

    def _create_rag_tool(self, kb_name):
        retriever = load_retriever(kb_name)
        qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)
        return Tool(
            name=f"知识库：{kb_name}",
            func=lambda q: qa_chain.run(q),
            description=f"从知识库 {kb_name} 中检索答案。适用于相关文档的问题。"
        )

    def run(self, input_text: str, max_retries=3) -> str:
        for attempt in range(max_retries):
            try:
                result = self.agent_executor.invoke({"input": input_text})
                # result 是一个 dict，可能包含 intermediate_steps
                #steps = result.get("intermediate_steps", [])
                #used_tools = [action.tool for action, _ in steps]
                #print(f"[INFO] 本次调用使用的工具：{used_tools}")
                return result  # 保留结构以便前端处理
            except OutputParserException as e:
                print(f"[WARN] 第 {attempt + 1} 次 LLM 输出解析失败：{e}")
                if attempt == max_retries - 1:
                    return "❌ 抱歉，我无法理解模型的输出格式。"
        return "⚠️ 未知错误"
