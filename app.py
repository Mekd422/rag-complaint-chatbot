"""
Gradio UI for RAG Complaint Chatbot

This module provides a user-friendly web interface for querying the complaint analysis system.
"""

import gradio as gr
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline


# Global variables for the pipeline (initialized lazily)
vector_store_manager = None
rag_pipeline = None


def initialize_pipeline():
    """
    Initialize the vector store and RAG pipeline.
    For Task 3, this loads the pre-built embeddings from the parquet file.
    """
    global vector_store_manager, rag_pipeline
    
    if vector_store_manager is None:
        print("Initializing vector store manager...")
        vector_store_manager = VectorStoreManager(
            persist_directory="vector_store/chroma",
            collection_name="complaint_chunks"
        )
        
        # Check if vector store already has data
        count = vector_store_manager.get_collection_count()
        
        if count == 0:
            # Try to load pre-built embeddings first (Task 3 requirement)
            prebuilt_path = "src/complaint_embeddings.parquet"
            if os.path.exists(prebuilt_path):
                print(f"Loading pre-built embeddings from {prebuilt_path}...")
                success = vector_store_manager.load_prebuilt_embeddings(prebuilt_path)
                if not success:
                    print("Failed to load pre-built embeddings. Falling back to building from filtered data...")
                    build_from_data()
            else:
                print("Pre-built embeddings not found. Building from filtered data...")
                build_from_data()
        else:
            print(f"Vector store already contains {count:,} chunks")
    
    if rag_pipeline is None:
        print("Initializing RAG pipeline...")
        # Use GPT-2 for now (lightweight, runs locally)
        # You can change this to a different model
        rag_pipeline = RAGPipeline(
            vector_store_manager=vector_store_manager,
            model_name="gpt2",  # Change to "gpt2-medium" or other models if needed
            use_cuda=False  # Set to True if you have GPU
        )
    
    return vector_store_manager, rag_pipeline


def build_from_data():
    """Build vector store from filtered complaints data."""
    global vector_store_manager
    
    try:
        import pandas as pd
        data_path = "data/processed/filtered_complaints.csv"
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Filtered data not found: {data_path}")
        
        print(f"Loading data from {data_path}...")
        # Load a subset for faster testing (you can load full dataset)
        df = pd.read_csv(data_path, nrows=5000)  # Using first 5000 for testing
        
        print(f"Building vector store from {len(df)} complaints...")
        vector_store_manager.build_from_dataframe(df, text_column="clean_narrative")
        print("Vector store built successfully!")
        
    except Exception as e:
        print(f"Error building vector store: {e}")
        raise


def query_complaints(question: str, product_filter: str = "All"):
    """
    Query the RAG system with a user question.
    
    Args:
        question: User's question
        product_filter: Product category filter
        
    Returns:
        Formatted answer and sources
    """
    global rag_pipeline
    
    if not question or question.strip() == "":
        return "Please enter a question.", ""
    
    try:
        # Initialize pipeline if not already done
        if rag_pipeline is None:
            initialize_pipeline()
        
        # Prepare filter dictionary
        filter_dict = None
        if product_filter != "All":
            filter_dict = {"product_category": product_filter}
        
        # Query the RAG pipeline
        response = rag_pipeline.query(
            question=question,
            top_k=5,
            filter_dict=filter_dict,
            max_answer_length=250,
            return_sources=True
        )
        
        answer = response.get("answer", "No answer generated.")
        sources = response.get("sources", [])
        
        # Format sources
        sources_text = "## Retrieved Sources:\n\n"
        for i, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            product = metadata.get("product_category", "Unknown")
            issue = metadata.get("issue", "Unknown")
            complaint_id = metadata.get("complaint_id", "N/A")
            
            sources_text += f"### Source {i}\n"
            sources_text += f"**Product:** {product}  \n"
            sources_text += f"**Issue:** {issue}  \n"
            sources_text += f"**Complaint ID:** {complaint_id}  \n"
            sources_text += f"**Text:** {source.get('text', '')}  \n\n"
            sources_text += "---\n\n"
        
        return answer, sources_text
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        return error_msg, ""


def clear_chat():
    """Clear the chat interface."""
    return "", ""


# Create Gradio interface
def create_interface():
    """Create and return the Gradio interface."""
    
    # Product categories for filtering
    product_categories = [
        "All",
        "Credit card",
        "Personal loan",
        "Savings account",
        "Money transfers"
    ]
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title="CrediTrust Complaint Analysis") as demo:
        gr.Markdown(
            """
            # 🏦 CrediTrust Financial - Complaint Analysis Chatbot
            
            Ask questions about customer complaints to get actionable insights. 
            This tool uses AI to analyze thousands of customer complaints and provide evidence-backed answers.
            
            **Example questions:**
            - "Why are people unhappy with credit cards?"
            - "What are the main issues with money transfers?"
            - "What complaints are related to billing disputes?"
            """,
            elem_classes=["main-header"]
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Enter your question about customer complaints...",
                    lines=3
                )
                
                with gr.Row():
                    product_filter = gr.Dropdown(
                        choices=product_categories,
                        value="All",
                        label="Filter by Product Category (Optional)"
                    )
                    submit_btn = gr.Button("Ask Question", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
            
        with gr.Row():
            with gr.Column():
                answer_output = gr.Textbox(
                    label="AI-Generated Answer",
                    lines=8,
                    interactive=False
                )
        
        with gr.Row():
            with gr.Column():
                sources_output = gr.Markdown(
                    label="Retrieved Sources (Evidence)"
                )
        
        # Event handlers
        submit_btn.click(
            fn=query_complaints,
            inputs=[question_input, product_filter],
            outputs=[answer_output, sources_output]
        )
        
        question_input.submit(
            fn=query_complaints,
            inputs=[question_input, product_filter],
            outputs=[answer_output, sources_output]
        )
        
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[question_input, answer_output]
        )
        
        # Note: Pipeline initialization happens on first query
    
    return demo


def main():
    """Main function to launch the Gradio app."""
    print("Starting CrediTrust Complaint Analysis Chatbot...")
    
    # Initialize pipeline
    try:
        initialize_pipeline()
        print("Pipeline initialized successfully!")
    except Exception as e:
        print(f"Warning: Could not initialize pipeline: {e}")
        print("The UI will still launch, but queries may fail until the vector store is built.")
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,  # Default Gradio port
        share=False  # Set to True to create a public link
    )


if __name__ == "__main__":
    main()
