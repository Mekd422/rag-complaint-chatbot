"""
RAG Pipeline Evaluation Script

This script evaluates the RAG pipeline with representative questions
and generates an evaluation report.
"""

import sys
import os
import pandas as pd
from typing import List, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline

# Fix path for pre-built embeddings
_script_dir = os.path.dirname(os.path.abspath(__file__))
_prebuilt_in_src = os.path.join(_script_dir, "complaint_embeddings.parquet")
_prebuilt_in_parent = os.path.join(os.path.dirname(_script_dir), "src", "complaint_embeddings.parquet")

if os.path.exists(_prebuilt_in_src):
    PREBUILT_PATH = _prebuilt_in_src
elif os.path.exists(_prebuilt_in_parent):
    PREBUILT_PATH = _prebuilt_in_parent
else:
    PREBUILT_PATH = None


class RAGEvaluator:
    """Evaluates the RAG pipeline with test questions."""
    
    def __init__(self, vector_store_manager: VectorStoreManager, rag_pipeline: RAGPipeline):
        """
        Initialize the evaluator.
        
        Args:
            vector_store_manager: Initialized VectorStoreManager
            rag_pipeline: Initialized RAGPipeline
        """
        self.vector_store = vector_store_manager
        self.rag_pipeline = rag_pipeline
    
    def evaluate_questions(self, questions: List[str]) -> pd.DataFrame:
        """
        Evaluate the RAG pipeline with a list of questions.
        
        Args:
            questions: List of test questions
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Evaluating: {question}")
            
            try:
                # Query the RAG pipeline
                response = self.rag_pipeline.query(
                    question=question,
                    top_k=5,
                    filter_dict=None,
                    max_answer_length=250,
                    return_sources=True
                )
                
                answer = response.get("answer", "No answer generated")
                sources = response.get("sources", [])
                
                # Format top 2 sources
                source_texts = []
                for j, source in enumerate(sources[:2], 1):
                    metadata = source.get("metadata", {})
                    product = metadata.get("product_category", "Unknown")
                    issue = metadata.get("issue", "Unknown")
                    text_preview = source.get("text", "")[:150] + "..." if len(source.get("text", "")) > 150 else source.get("text", "")
                    source_texts.append(f"Source {j}: [{product}] {issue}\n{text_preview}")
                
                sources_str = "\n\n".join(source_texts) if source_texts else "No sources retrieved"
                
                # For now, we'll leave quality score empty (user should fill manually)
                # In a real evaluation, you might use LLM-based scoring or human evaluation
                quality_score = None
                comments = "Generated successfully"
                
                results.append({
                    "Question": question,
                    "Generated Answer": answer,
                    "Retrieved Sources": sources_str,
                    "Quality Score (1-5)": quality_score,
                    "Comments/Analysis": comments
                })
                
            except Exception as e:
                print(f"Error evaluating question: {e}")
                results.append({
                    "Question": question,
                    "Generated Answer": f"Error: {str(e)}",
                    "Retrieved Sources": "Error",
                    "Quality Score (1-5)": None,
                    "Comments/Analysis": f"Error occurred: {str(e)}"
                })
        
        return pd.DataFrame(results)
    
    def generate_report(self, df: pd.DataFrame, output_path: str = "evaluation_report.md"):
        """
        Generate a markdown evaluation report.
        
        Args:
            df: DataFrame with evaluation results
            output_path: Path to save the report
        """
        report = "# RAG Pipeline Evaluation Report\n\n"
        report += "This report contains qualitative evaluation results for the RAG complaint analysis system.\n\n"
        report += "## Evaluation Questions and Results\n\n"
        
        for idx, row in df.iterrows():
            report += f"### Question {idx + 1}\n\n"
            report += f"**Question:** {row['Question']}\n\n"
            report += f"**Generated Answer:**\n{row['Generated Answer']}\n\n"
            report += f"**Retrieved Sources:**\n{row['Retrieved Sources']}\n\n"
            report += f"**Quality Score:** {row['Quality Score (1-5)'] or 'Not rated'}\n\n"
            report += f"**Comments/Analysis:**\n{row['Comments/Analysis']}\n\n"
            report += "---\n\n"
        
        # Save report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"\nEvaluation report saved to: {output_path}")
        
        # Also save as CSV
        csv_path = output_path.replace(".md", ".csv")
        df.to_csv(csv_path, index=False)
        print(f"Evaluation results (CSV) saved to: {csv_path}")


def main():
    """Main evaluation function."""
    
    # Test questions based on the requirements
    test_questions = [
        "Why are people unhappy with credit cards?",
        "What are the main issues with personal loans?",
        "What complaints are related to billing disputes?",
        "Why do customers complain about money transfers?",
        "What are the common problems with savings accounts?",
        "What issues do customers face with late fees?",
        "Why are customers complaining about customer service?",
        "What problems are related to account management?",
    ]
    
    print("=" * 60)
    print("RAG Pipeline Evaluation")
    print("=" * 60)
    
    # Initialize pipeline
    print("\n1. Initializing vector store manager...")
    vector_store_manager = VectorStoreManager(
        persist_directory="vector_store/chroma",
        collection_name="complaint_chunks"
    )
    
    count = vector_store_manager.get_collection_count()
    if count == 0:
        # Try to load pre-built embeddings (Task 3 requirement)
        if PREBUILT_PATH and os.path.exists(PREBUILT_PATH):
            print(f"Loading pre-built embeddings from {PREBUILT_PATH}...")
            success = vector_store_manager.load_prebuilt_embeddings(PREBUILT_PATH)
            if not success:
                print("ERROR: Failed to load pre-built embeddings!")
                print("Please ensure the complaint_embeddings.parquet file exists in the src/ directory")
                return
            count = vector_store_manager.get_collection_count()
        else:
            print("ERROR: Vector store is empty and pre-built embeddings not found!")
            print("Please ensure the complaint_embeddings.parquet file exists in the src/ directory")
            return
    
    print(f"Vector store contains {count} chunks")
    
    print("\n2. Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline(
        vector_store_manager=vector_store_manager,
        model_name="gpt2",
        use_cuda=False
    )
    
    print("\n3. Running evaluation with test questions...")
    evaluator = RAGEvaluator(vector_store_manager, rag_pipeline)
    results_df = evaluator.evaluate_questions(test_questions)
    
    print("\n4. Generating evaluation report...")
    evaluator.generate_report(results_df, output_path="evaluation_report.md")
    
    # Display summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Total questions evaluated: {len(test_questions)}")
    print(f"Successful evaluations: {len(results_df[results_df['Generated Answer'].str.contains('Error') == False])}")
    print(f"Failed evaluations: {len(results_df[results_df['Generated Answer'].str.contains('Error')])}")
    print("\nEvaluation complete! Check evaluation_report.md for details.")


if __name__ == "__main__":
    main()
