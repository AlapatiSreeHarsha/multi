import os
import json
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_groq import ChatGroq
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import GoogleSearchAPIWrapper

# API Key
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Set up language models
llama_3_model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
mixtral_model = ChatGroq(model_name="mixtral-8x7b", temperature=0.1)

# Set up search tools
wikipedia_tool = Tool(
    name="Wikipedia",
    func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
    description="Useful for searching for information on Wikipedia."
)

duckduckgo_tool = Tool(
    name="DuckDuckGo Search",
    func=DuckDuckGoSearchAPIWrapper().run,
    description="Useful for searching the web for current information."
)

google_search_tool = Tool(
    name="Google Search",
    func=GoogleSearchAPIWrapper().run,
    description="Useful for searching the web for current information."
)

# Define the agents with specific prompt templates

# 1. Content Analyzer Agent
content_analyzer = Agent(
    role="Linguistic Analyst",
    goal="Analyze linguistic features and patterns in text to identify stance markers",
    backstory="You are an expert linguist with years of experience analyzing text for stance detection.",
    verbose=True,
    llm=llama_3_model,
    tools=[duckduckgo_tool, google_search_tool],
    instructions="""
    Analyze the linguistic features of the following text in relation to [TARGET]. Focus on explicit stance markers, sentiment analysis, key phrase extraction, and linguistic pattern recognition.
    
    Your output must be in the following JSON format:
    {
        "reasoning": "Your detailed reasoning process",
        "evidence": "Specific linguistic features and patterns found in the text",
        "confidence": "Numeric value between 0-1",
        "stance_assessment": "Your assessment of the stance (positive, negative, neutral, mixed)",
        "alternative_interpretations": "Possible alternative readings of the text"
    }
    """
)

# 2. Context Researcher Agent
context_researcher = Agent(
    role="Context Researcher",
    goal="Provide factual background information about targets and contexts",
    backstory="You are a meticulous researcher providing factual and historical context for stance analysis.",
    verbose=True,
    llm=mixtral_model,
    tools=[wikipedia_tool, duckduckgo_tool],
    instructions="""
    Provide relevant factual context about [TARGET] that helps contextualize the stance in the text.
    Focus on retrieval of relevant factual information, identification of common argumentative patterns,
    analysis of terminology usage in context of the domain, and recognition of historical or cultural references.
    
    Your output must be in the following JSON format:
    {
        "reasoning": "Your detailed reasoning process",
        "evidence": "Relevant factual information about the target",
        "confidence": "Numeric value between 0-1",
        "stance_assessment": "How contextual factors influence stance interpretation",
        "alternative_interpretations": "How different contexts might alter interpretation"
    }
    """
)

# 3. Perspective Agent
perspective_analyzer = Agent(
    role="Perspective Analyst",
    goal="Evaluate text from multiple ideological perspectives",
    backstory="You are a media analyst trained to identify biases and evaluate content from various viewpoints.",
    verbose=True,
    llm=llama_3_model,
    tools=[duckduckgo_tool, google_search_tool],
    instructions="""
    Interpret the text from multiple ideological perspectives (liberal, conservative, academic).
    Focus on identification of potential assumptions, analysis of framing and value-based language,
    and detection of audience-targeting features.
    
    Your output must be in the following JSON format:
    {
        "reasoning": "Your detailed reasoning process across multiple perspectives",
        "evidence": "Specific indicators of perspective bias in the text",
        "confidence": "Numeric value between 0-1",
        "stance_assessment": "Stance assessment from each perspective considered",
        "alternative_interpretations": "How different audiences might interpret the text"
    }
    """
)

# 4. Devil's Advocate Agent
devils_advocate = Agent(
    role="Devil's Advocate",
    goal="Critically examine preliminary stance assessments",
    backstory="You are an analytical critic who challenges preliminary analyses to ensure thoroughness and reduce confirmation bias.",
    verbose=True,
    llm=llama_3_model,
    tools=[duckduckgo_tool],
    instructions="""
    Challenge the preliminary stance assessment by considering alternative interpretations.
    Focus on identification of confirmation biases in previous analyses, generation of counter-arguments,
    testing of edge case interpretations, and evaluation of confidence levels in previous assessments.
    
    Your output must be in the following JSON format:
    {
        "reasoning": "Your critical examination of previous analyses",
        "evidence": "Facts or interpretations that contradict preliminary assessments",
        "confidence": "Numeric value between 0-1 representing confidence in counter-arguments",
        "stance_assessment": "Alternative stance assessment based on counter-analysis",
        "alternative_interpretations": "Edge cases and minority viewpoints worth considering"
    }
    """
)

# 5. Synthesis Agent
synthesis_agent = Agent(
    role="Synthesis Specialist",
    goal="Integrate all analyses to produce the final stance determination",
    backstory="You are an expert in integrating multiple analytical perspectives into coherent, balanced final assessments.",
    verbose=True,
    llm=llama_3_model,
    tools=[],  # No external tools needed for synthesis
    instructions="""
    Integrate all analyses to determine the final stance of the text toward [TARGET].
    Focus on weighted integration of all agent inputs, resolution of conflicting assessments,
    calibration of confidence scores, and production of final stance label with supporting evidence.
    
    Your output must be in the following JSON format:
    {
        "reasoning": "Your detailed integration process and final reasoning",
        "evidence": "Key evidence from all previous analyses that supports final determination",
        "confidence": "Final calibrated confidence score between 0-1",
        "stance_assessment": "Final stance label (positive, negative, neutral, mixed) with explanation",
        "alternative_interpretations": "Significant minority viewpoints worth noting"
    }
    """
)

# Define tasks
def create_tasks(text, target):
    # Replace [TARGET] with actual target in agent instructions
    for agent in [content_analyzer, context_researcher, perspective_analyzer, devils_advocate, synthesis_agent]:
        agent.instructions = agent.instructions.replace("[TARGET]", target)
    
    # 1. Content Analysis Task
    linguistic_analysis = Task(
        description=f"Analyze the linguistic features of '{text}' in relation to '{target}'. Identify explicit stance markers, sentiment, key phrases, and linguistic patterns.",
        agent=content_analyzer,
        expected_output="JSON output with linguistic analysis results"
    )
    
    # 2. Context Research Task
    contextual_research = Task(
        description=f"Research factual information about '{target}' to provide context for stance analysis. Identify relevant historical or cultural references and analyze common argumentative patterns.",
        agent=context_researcher,
        expected_output="JSON output with contextual research results"
    )
    
    # 3. Perspective Analysis Task
    perspective_analysis = Task(
        description=f"Analyze '{text}' regarding '{target}' from multiple ideological perspectives. Identify potential assumptions, biases, framing and value-based language, and audience-targeting features.",
        agent=perspective_analyzer,
        expected_output="JSON output with perspective analysis results"
    )
    
    # 4. Devil's Advocate Task
    devils_advocate_analysis = Task(
        description=f"Critically examine the preliminary stance assessments for '{text}' regarding '{target}'. Identify confirmation biases, generate counter-arguments, test edge case interpretations, and evaluate confidence levels.",
        agent=devils_advocate,
        expected_output="JSON output with devil's advocate analysis",
        context=[linguistic_analysis, contextual_research, perspective_analysis]
    )
    
    # 5. Synthesis Task
    final_synthesis = Task(
        description=f"Integrate all analyses to determine the final stance of '{text}' toward '{target}'. Provide weighted integration, resolve conflicts, calibrate confidence, and produce the final stance label with supporting evidence.",
        agent=synthesis_agent,
        expected_output="JSON output with final stance determination",
        context=[linguistic_analysis, contextual_research, perspective_analysis, devils_advocate_analysis]
    )
    
    return [linguistic_analysis, contextual_research, perspective_analysis, devils_advocate_analysis, final_synthesis]

# Create crew
def create_crew(text, target):
    tasks = create_tasks(text, target)
    return Crew(
        agents=[content_analyzer, context_researcher, perspective_analyzer, devils_advocate, synthesis_agent],
        tasks=tasks,
        verbose=2,
        process=Process.sequential  # Ensure tasks run in the specific order defined
    )

# Function to format the final output
def format_final_output(crew_result, text, target):
    try:
        # Try to parse the crew result as JSON if it's not already
        if isinstance(crew_result, str):
            result_data = json.loads(crew_result)
        else:
            result_data = crew_result
            
        # Create the final output structure
        final_output = {
            "text": text,
            "target": target,
            "stance_analysis": result_data
        }
        
        return final_output
    except json.JSONDecodeError:
        # If the result is not valid JSON, return it as is with minimal structure
        return {
            "text": text,
            "target": target,
            "stance_analysis": {
                "raw_result": crew_result
            }
        }

# Streamlit UI
if __name__ == "__main__":
    st.title("Advanced Stance Detection Using Multi-Agent System")
    
    st.markdown("""
    This system analyzes the stance of text toward a target topic using multiple specialized agents:
    
    1. **Linguistic Analyst**: Examines language patterns and explicit stance markers
    2. **Context Researcher**: Provides factual background on the target topic
    3. **Perspective Analyst**: Evaluates from multiple ideological viewpoints
    4. **Devil's Advocate**: Challenges preliminary assessments with counter-arguments
    5. **Synthesis Specialist**: Integrates all analyses into a final stance determination
    """)
    
    text_input = st.text_area("Enter the text for stance detection:", height=150)
    target_input = st.text_input("Enter the target topic:")
    
    if st.button("Analyze Stance"):
        if text_input and target_input:
            with st.spinner("Performing comprehensive stance analysis... This may take a few minutes."):
                # Create and run the crew
                crew = create_crew(text_input, target_input)
                result = crew.kickoff()
                
                # Format the final output
                final_output = format_final_output(result, text_input, target_input)
                
                # Save to file
                output_path = "data/output/final_stance.json"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(final_output, f, indent=4)
                
                # Display results
                st.subheader("Stance Analysis Results")
                st.json(final_output)
                st.success(f"Stance analysis completed and saved to {output_path}")
        else:
            st.error("Please enter both text and target topic.")