from context import Context, llm, tools, logger
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage


# ---------- NODES ---------- #
def intentParserNode(cxt: Context):
    logger.info("BEGIN INTENT PARSING")
    """Uses the LLM to rewrite and optimize the user's original message."""
    
    # Safely get the latest message text
    if not cxt["messages"]:
        raise ValueError("No messages found in context.")
    user_message = cxt["messages"][-1].content
    PROMPT_INTENT_PARSER = f"""
        You are an expert workflow architect tasked with understanding user intent and improving clarity.

        Given an input message from a user describing a desired process, rewrite the message to be:
        - Clear, concise, and professional
        - Grammatically correct and easy to interpret
        - Structured in a way suitable for automated process generation

        Instructions:
        1. Identify what type of process the user is requesting (e.g., onboarding, training, SOP, approval, etc.).
        2. Rewrite the message in a way that makes the intent explicit.
        3. Preserve all relevant details but remove filler, slang, or ambiguous phrasing.
        4. Output only the improved message.

        Example:
        Input: "hey can u make me an onboarding thing for new hires"
        Output: "Create an onboarding workflow for new employees, outlining required steps and resources."

        ---

        Now process the following message:
        {user_message}
        """
    
    result = llm.invoke(PROMPT_INTENT_PARSER)
    optimized_message = result.content.strip()

    # Store result inside context's message state (not overwriting other fields)
    cxt["messages"].append(AIMessage(content=optimized_message))

    return cxt
def generateProcessDiagramNode(cxt: Context):
    """Generates a high-level process diagram outline from the optimized message."""
    logger.info("GENERATING PROCESS DIAGRAM")

    # --- 1. Retrieve the latest optimized message ---
    if not cxt["messages"]:
        raise ValueError("No messages found in context.")
    optimized_message = cxt["messages"][-1].content

    # --- 2. Define the system prompt ---
    PROMPT_PROCESS_DIAGRAM = f"""
        You are a workflow visualization expert. 
        Your task is to generate a structured **process diagram outline** 
        that clearly represents the following workflow description.

        The diagram should:
        - Identify key **stages**, **decisions**, and **actions**
        - Be structured as labeled nodes and arrows
        - Use concise, professional naming
        - Use clear formatting that can be easily converted into a visual flowchart
          (e.g., Mermaid, PlantUML, or bullet-step hierarchy)

        Example (Mermaid syntax):
        ```mermaid
        flowchart TD
            Start --> Gather_Requirements
            Gather_Requirements --> Approval
            Approval -->|Approved| Implementation
            Approval -->|Rejected| Revision
            Implementation --> Review
            Review --> End
        ```

        ---
        Workflow description:
        {optimized_message}
        """
    result = llm.invoke(PROMPT_PROCESS_DIAGRAM)

    diagram_outline = result.content.strip()

    cxt["diagram"] = diagram_outline

    cxt["messages"].append(AIMessage(content=diagram_outline))
    return cxt
def generateDocumentNode(cxt: Context):
    """Generates a written process document from the optimized message and its diagram."""
    logger.info("GENERATING DOCUMENT")

    # --- 1. Extract inputs from context ---
    if not cxt["messages"]:
        raise ValueError("No messages found in context.")
    optimized_message = cxt["messages"][-2].content
    mermaid_diagram = cxt["diagram"]

    if not mermaid_diagram:
        raise ValueError("No diagram found in context. Ensure generateProcessDiagramNode ran first.")

    # --- 2. Define the document generation prompt ---
    PROMPT_DOCUMENT_WRITER = f"""
        You are a technical writer tasked with producing a professional process document
        from both a workflow description and its MermaidJS diagram.

        Your goals:
        - Transform the optimized workflow prompt and diagram into a structured, written process guide.
        - Include section headings, step-by-step explanations, and clear transition logic.
        - Keep it factual and actionable, suitable for documentation or training manuals.
        - Use the diagram as reference for step order and decision branches.
        - Avoid repeating the Mermaid code verbatim — interpret it into prose.

        ---
        Optimized Workflow Description:
        {optimized_message}

        MermaidJS Diagram:
        ```mermaid
        {mermaid_diagram}
        ```

        ---
        Write the full document below:
        """

    result = llm.invoke(PROMPT_DOCUMENT_WRITER)
    document_text = result.content.strip()

    cxt["document"] = document_text
    cxt["messages"].append(AIMessage(content=document_text))

    return cxt
def validationNode(cxt: Context):
    """Validates the quality and consistency of the generated document and Mermaid diagram."""
    logger.info("VALIDATING PROCESS")

    # --- 1. Extract inputs ---
    document = cxt["document"]
    diagram = cxt["diagram"]

    if not document or not diagram:
        raise ValueError("Missing document or diagram for validation.")

    # --- 2. Define validation prompt ---
    PROMPT_VALIDATION = f"""
        You are an expert workflow auditor and documentation reviewer.

        Review the following **process document** and **MermaidJS diagram** for quality and alignment.

        Evaluate on these criteria:
        1. **Consistency:** Do both describe the same workflow logically?
        2. **Completeness:** Are all key steps and decisions accounted for?
        3. **Clarity:** Is the document professional, well-structured, and understandable?
        4. **Diagram quality:** Does the Mermaid syntax appear valid and interpretable?

        For each category, provide:
        - A rating from 1–5
        - A short justification

        Finally, decide:
        - **Overall Verdict:** "Pass" if the content is clear and consistent enough for automation, otherwise "Fail".
        - **Recommendations:** How to improve if needed.

        ---
        Document:
        {document}

        ---
        MermaidJS Diagram:
        ```mermaid
        {diagram}
        ```
        """

    result = llm.invoke(PROMPT_VALIDATION)
    review = result.content.strip()

    # --- 4. Post-process verdict ---
    is_pass = "pass" in review.lower() and "fail" not in review.lower()

    # --- 5. Store results in context ---
    cxt["is_satisfied"] = is_pass
    cxt["messages"].append(AIMessage(content=review))

    return cxt
def processIterationNode(cxt: Context):
    """If validation fails, use recommendations to refine the Mermaid diagram for better consistency."""
    logger.info("ITERATING ON PROCESS")

    # --- 1. Get validation results ---
    if not cxt["messages"]:
        raise ValueError("No messages found in context.")
    last_message = cxt["messages"][-1].content

    # If validation passed, skip iteration
    if cxt["is_satisfied"]:
        cxt["messages"].append(AIMessage(content="Validation passed. No iteration required."))
        return cxt

    # --- 2. Extract existing artifacts ---
    document = cxt["document"]
    diagram = cxt["diagram"]
    if not document or not diagram:
        raise ValueError("Missing document or diagram for process iteration.")

    # --- 3. Build the improvement prompt ---
    PROMPT_PROCESS_ITERATION = f"""
        You are a workflow correction assistant.

        Your goal is to **revise the existing MermaidJS process diagram** so that it aligns more
        closely with the written document and follows the recommendations below.

        Guidelines:
        - Only modify nodes, labels, or connections that address inconsistencies or omissions.
        - Keep valid structure and logic intact.
        - Maintain valid Mermaid syntax.
        - Ensure the new diagram accurately reflects all key stages and decisions in the document.

        ---
        Current Document:
        {document}

        ---
        Current Mermaid Diagram:
        ```mermaid
        {diagram}
        ```

        ---
        Validation Feedback and Recommendations:
        {last_message}

        ---
        Output only the **revised MermaidJS diagram**.
        """

    result = llm.invoke(PROMPT_PROCESS_ITERATION)
    revised_diagram = result.content.strip()
    
    cxt["diagram"] = revised_diagram
    cxt["messages"].append(AIMessage(content=revised_diagram))

    return cxt
def docIterationNode(cxt: Context):
    """If validation fails, use recommendations to refine the written document so it aligns with the diagram."""
    logger.info("ITERATING ON DOCUMENT")

    # --- 1. Get validation results ---
    if not cxt["messages"]:
        raise ValueError("No messages found in context.")
    last_message = cxt["messages"][-1].content

    # Skip iteration if validation passed
    if cxt["is_satisfied"]:
        cxt["messages"].append(AIMessage(content="Validation passed. No document iteration required."))
        return cxt

    # --- 2. Extract current document and diagram ---
    document = cxt["document"]
    diagram = cxt["diagram"]
    if not document or not diagram:
        raise ValueError("Missing document or diagram for document iteration.")

    # --- 3. Build the revision prompt ---
    PROMPT_DOC_ITERATION = f"""
        You are a process documentation editor.

        Your task is to **revise the existing written document** so that it aligns perfectly
        with the provided MermaidJS diagram and addresses the feedback below.

        Guidelines:
        - Preserve the overall meaning and structure of the original document.
        - Ensure every step, decision, or branch in the diagram is clearly reflected in the text.
        - Maintain professional tone, grammar, and clarity.
        - Incorporate any missing details mentioned in the recommendations.
        - Output only the revised document in markdown format.

        ---
        Current Document:
        {document}

        ---
        Reference Mermaid Diagram:
        ```mermaid
        {diagram}
        ```

        ---
        Validation Feedback and Recommendations:
        {last_message}

        ---
        Revised Document:
        """

    result = llm.invoke(PROMPT_DOC_ITERATION)
    revised_document = result.content.strip()

    # --- 5. Update context ---
    cxt["document"] = revised_document
    cxt["messages"].append(AIMessage(content=revised_document))

    return cxt
def toolNode(cxt: Context):
    logger.info("SENDING PROCESS")
    document = cxt["document"]
    diagram = cxt["diagram"]
    
    if not document or not diagram:
        raise ValueError("Missing document or diagram for toolNode execution.")
    
    system_message = f"""
    You are a process automation agent responsible for delivering final workflow outputs.
    You have access to a Gmail-sending tool (GMAIL_SEND_EMAIL) that can send emails with
    subject and body content.

    Your task:
    1. Compose a professional summary email to the operations team at **opexteam.codelab@gmail.com**.
    2. Include the **validated document** and the **Mermaid diagram** within the email body.
    3. Use the Gmail tool to send the message.
    4. Once sent, confirm that the email has been dispatched successfully.

    Document:
    {document}

    Mermaid Diagram:
    ```mermaid
    {diagram}
    ```
    """
    
    # Create the react agent
    agent_executor = create_agent(llm, tools)
    
    # Run with the system message
    result = agent_executor.invoke({
        "messages": [HumanMessage(content=system_message)]
    })
    
    # Get the final AI message
    final_message = result["messages"][-1]
    final_output = final_message.content if hasattr(final_message, 'content') else str(final_message)
    
    # Save to context
    cxt["messages"].append(AIMessage(content=final_output))
    return cxt
