"""
RAG Pipeline Module

This module implements the core RAG (Retrieval-Augmented Generation) pipeline:
1. Retriever: Uses vector similarity search to find relevant complaint chunks
2. Generator: Uses an LLM to generate answers based on retrieved context
"""

from typing import List, Dict, Optional
from src.vector_store import VectorStoreManager
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


class RAGPipeline:
    """
    RAG Pipeline that combines retrieval and generation for answering questions
    about customer complaints.
    """
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str = "gpt2",  # Using GPT-2 as a default, can be changed
        use_cuda: bool = False
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store_manager: Initialized VectorStoreManager instance
            model_name: Name of the language model to use for generation
            use_cuda: Whether to use GPU (if available)
        """
        self.vector_store = vector_store_manager
        self.model_name = model_name
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        
        # Initialize the language model
        print(f"Loading language model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to a simpler generation approach")
            self.generator = None
            self.tokenizer = None
            self.model = None
    
    def create_prompt(self, context: str, question: str) -> str:
        """
        Create a prompt template for the LLM.
        
        Args:
            context: Retrieved complaint chunks as context
            question: User's question
            
        Returns:
            Formatted prompt string
        """
        prompt_template = """You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints based on the provided context from real customer complaint narratives.

Use the following retrieved complaint excerpts to formulate your answer. Focus on providing actionable insights and summarizing the main issues. If the context doesn't contain enough information to answer the question, state that clearly.

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt_template.format(context=context, question=question)
    
    def format_context(self, retrieved_chunks: List[Dict], max_context_length: int = 2000) -> str:
        """
        Format retrieved chunks into a context string.
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            max_context_length: Maximum length of context in characters
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk["text"]
            metadata = chunk.get("metadata", {})
            
            # Add metadata info
            product = metadata.get("product_category", "Unknown")
            issue = metadata.get("issue", "Unknown")
            
            chunk_text = f"[Complaint {i}] Product: {product}, Issue: {issue}\n{text}\n\n"
            
            if current_length + len(chunk_text) > max_context_length:
                break
                
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "".join(context_parts)
    
    def retrieve(self, question: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve relevant complaint chunks for a given question.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            filter_dict: Optional metadata filters
            
        Returns:
            List of retrieved chunks with metadata
        """
        return self.vector_store.query(
            query_text=question,
            n_results=top_k,
            filter_dict=filter_dict
        )
    
    def generate_answer(
        self,
        question: str,
        context: str,
        max_length: int = 300,
        temperature: float = 0.7
    ) -> str:
        """
        Generate an answer using the LLM.
        
        Args:
            question: User's question
            context: Formatted context from retrieved chunks
            max_length: Maximum length of generated response
            temperature: Sampling temperature (higher = more creative)
            
        Returns:
            Generated answer string
        """
        prompt = self.create_prompt(context=context, question=question)
        
        if self.generator is None:
            # Fallback: return a simple response
            return "I apologize, but the language model is not available. Please check the model configuration."
        
        try:
            # Generate response
            outputs = self.generator(
                prompt,
                max_length=len(prompt.split()) + max_length,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            
            # Extract only the answer part (after "Answer:")
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                # If prompt format changed, try to extract the new part
                answer = generated_text[len(prompt):].strip()
            
            # Clean up the answer
            answer = answer.split("\n")[0] if "\n" in answer else answer
            
            return answer if answer else "I couldn't generate a satisfactory answer. Please try rephrasing your question."
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return f"Error generating answer: {str(e)}"
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None,
        max_answer_length: int = 200,
        return_sources: bool = True
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve relevant chunks and generate an answer.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            filter_dict: Optional metadata filters
            max_answer_length: Maximum length of generated answer
            return_sources: Whether to return source chunks
            
        Returns:
            Dictionary containing answer and optionally source chunks
        """
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retrieve(
            question=question,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        if not retrieved_chunks:
            return {
                "answer": "I couldn't find any relevant complaint data to answer your question. Please try rephrasing or asking about a different topic.",
                "sources": []
            }
        
        # Step 2: Format context
        context = self.format_context(retrieved_chunks)
        
        # Step 3: Generate answer
        answer = self.generate_answer(
            question=question,
            context=context,
            max_length=max_answer_length
        )
        
        # Step 4: Prepare response
        response = {
            "answer": answer,
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "text": chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "relevance_score": 1 - chunk.get("distance", 0) if chunk.get("distance") else None
                }
                for chunk in retrieved_chunks
            ]
        
        return response


def create_simple_generator(prompt: str, max_tokens: int = 200) -> str:
    """
    Simple fallback generator using rule-based extraction.
    This is used when LLM models are not available.
    
    Args:
        prompt: The full prompt
        max_tokens: Maximum tokens to extract
        
    Returns:
        Generated answer
    """
    # Extract context and question
    if "Context:" in prompt and "Question:" in prompt:
        parts = prompt.split("Question:")
        if len(parts) == 2:
            context = parts[0].replace("Context:", "").strip()
            question = parts[1].replace("Answer:", "").strip()
            
            # Simple rule-based answer
            if "why" in question.lower() or "reason" in question.lower():
                return f"Based on the complaint data, here are the main issues: {context[:200]}..."
            elif "what" in question.lower():
                return f"The complaints indicate: {context[:200]}..."
            else:
                return f"Based on the retrieved complaints: {context[:200]}..."
    
    return "Unable to generate an answer. Please ensure the language model is properly configured."
