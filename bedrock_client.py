"""
AWS Bedrock Client for Claude Sonnet/Haiku integration.

This module provides a clean interface to AWS Bedrock's Claude models
for the Stock Perfect Model trading algorithm.
"""

import os
import json
import boto3
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class ClaudeModel(Enum):
    """Available Claude models on AWS Bedrock (using inference profiles)."""
    # Use inference profile IDs for cross-region routing
    SONNET = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"  # Complex reasoning
    HAIKU = "us.anthropic.claude-3-5-haiku-20241022-v1:0"    # Fast, cost-effective
    # Latest models
    SONNET_4 = "us.anthropic.claude-sonnet-4-20250514-v1:0"  # Claude 4 Sonnet
    HAIKU_4 = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # Claude 4.5 Haiku


class BedrockClient:
    """
    AWS Bedrock client for Claude models.
    
    Usage:
        client = BedrockClient()
        response = client.invoke("What is the market sentiment?", model=ClaudeModel.HAIKU)
    """
    
    def __init__(
        self,
        region: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
    ):
        """
        Initialize the Bedrock client.
        
        Args:
            region: AWS region (defaults to env var or us-east-1)
            aws_access_key: AWS access key (defaults to env var)
            aws_secret_key: AWS secret key (defaults to env var)
        """
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        
        # Build session with explicit credentials or default chain
        session_kwargs = {}
        if aws_access_key and aws_secret_key:
            session_kwargs["aws_access_key_id"] = aws_access_key
            session_kwargs["aws_secret_access_key"] = aws_secret_key
        elif os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
            session_kwargs["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
            session_kwargs["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        session = boto3.Session(**session_kwargs)
        self.client = session.client(
            service_name="bedrock-runtime",
            region_name=self.region
        )
        
    def invoke(
        self,
        prompt: str,
        model: ClaudeModel = ClaudeModel.HAIKU,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """
        Invoke a Claude model with the given prompt.
        
        Args:
            prompt: The user message/prompt
            model: Which Claude model to use (SONNET for complex, HAIKU for fast)
            system_prompt: Optional system prompt for context
            max_tokens: Maximum response length
            temperature: Creativity (0=deterministic, 1=creative)
            
        Returns:
            The model's text response
        """
        messages = [{"role": "user", "content": prompt}]
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        
        if system_prompt:
            body["system"] = system_prompt
            
        response = self.client.invoke_model(
            modelId=model.value,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]
    
    def invoke_for_json(
        self,
        prompt: str,
        model: ClaudeModel = ClaudeModel.HAIKU,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Invoke Claude and parse the response as JSON.
        
        Args:
            prompt: The user message (should request JSON output)
            model: Which Claude model to use
            system_prompt: Optional system prompt
            max_tokens: Maximum response length
            
        Returns:
            Parsed JSON dictionary
        """
        # Add JSON instruction to system prompt
        json_system = (system_prompt or "") + "\n\nRespond ONLY with valid JSON, no other text."
        
        response = self.invoke(
            prompt=prompt,
            model=model,
            system_prompt=json_system.strip(),
            max_tokens=max_tokens,
            temperature=0.1,  # Lower temperature for structured output
        )
        
        # Clean response (sometimes models wrap in markdown)
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
            
        return json.loads(response.strip())


# Convenience functions for common use cases
def analyze_with_haiku(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Quick analysis using Haiku (fast, cheap)."""
    client = BedrockClient()
    return client.invoke(prompt, model=ClaudeModel.HAIKU, system_prompt=system_prompt)


def analyze_with_sonnet(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Deep analysis using Sonnet (slower, smarter)."""
    client = BedrockClient()
    return client.invoke(prompt, model=ClaudeModel.SONNET, system_prompt=system_prompt)


if __name__ == "__main__":
    # Test the client
    print("Testing Bedrock Client...")
    
    try:
        client = BedrockClient()
        response = client.invoke(
            "What are the three most important factors in stock market analysis? Be brief.",
            model=ClaudeModel.HAIKU
        )
        print(f"Haiku Response:\n{response}")
    except Exception as e:
        print(f"Error (ensure AWS credentials are configured): {e}")

