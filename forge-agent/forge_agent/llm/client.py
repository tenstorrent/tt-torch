"""
LLM client for interacting with language model APIs.
"""
from typing import Dict, Any, Optional, List
import os
import json
import requests
from loguru import logger


class LLMClient:
    """Client for interacting with language model APIs."""
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "openai"):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key for the LLM service
            provider: LLM provider name (openai, anthropic, etc.)
        """
        self.provider = provider
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
        
        if not self.api_key:
            logger.warning(f"No API key provided for {provider}. "
                          f"Set the {provider.upper()}_API_KEY environment variable.")
    
    def generate_adaptation_code(
        self,
        model_info: Dict[str, Any],
        error_message: Optional[str] = None,
        model_code: Optional[str] = None,
        model_class_name: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate adaptation code for a model using an LLM.
        
        Args:
            model_info: Dictionary containing model info (type, structure, etc.)
            error_message: Error message from failed compilation attempt
            model_code: Source code of the model if available
            model_class_name: Class name of the model
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing:
            - success: Whether generation was successful
            - code: Generated adaptation code
            - adaptation_level: Suggested adaptation level
            - explanation: Explanation of the adaptation
        """
        if not self.api_key:
            return {
                "success": False,
                "code": None,
                "adaptation_level": None,
                "explanation": "No API key available for LLM service"
            }
        
        # Construct prompt for the LLM
        prompt = self._construct_adaptation_prompt(
            model_info, error_message, model_code, model_class_name
        )
        
        # Call the appropriate provider's API
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    result = self._call_openai_api(prompt)
                elif self.provider == "anthropic":
                    result = self._call_anthropic_api(prompt)
                else:
                    logger.error(f"Unsupported LLM provider: {self.provider}")
                    return {
                        "success": False,
                        "code": None,
                        "adaptation_level": None,
                        "explanation": f"Unsupported LLM provider: {self.provider}"
                    }
                
                return result
                
            except Exception as e:
                logger.error(f"Error calling LLM API (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "code": None,
                        "adaptation_level": None,
                        "explanation": f"Error calling LLM API: {str(e)}"
                    }
    
    def _construct_adaptation_prompt(
        self,
        model_info: Dict[str, Any],
        error_message: Optional[str] = None,
        model_code: Optional[str] = None,
        model_class_name: Optional[str] = None
    ) -> str:
        """
        Construct a prompt for the LLM to generate adaptation code.
        
        Args:
            model_info: Dictionary containing model info
            error_message: Error message from failed compilation attempt
            model_code: Source code of the model if available
            model_class_name: Class name of the model
            
        Returns:
            Formatted prompt string
        """
        # Extract model structure
        model_type = model_info.get("model_type", "unknown")
        model_structure = model_info.get("model_structure", "")
        
        # Base prompt
        prompt = f"""
You are an AI assistant specialized in adapting PyTorch models to work with the tt-torch compiler.
Your task is to generate adaptation code for a PyTorch model that will make it compatible with tt-torch.

## Model Information
- Model Type: {model_type}
- Model Class: {model_class_name or 'Unknown'}

## Model Structure
{model_structure}

"""

        # Add error message if available
        if error_message:
            prompt += f"""
## Error Message
The following error occurred when trying to compile the model with tt-torch:
```
{error_message}
```

"""

        # Add model code if available
        if model_code:
            prompt += f"""
## Model Code
```python
{model_code}
```

"""

        # Add instructions for adaptation
        prompt += """
## Instructions for Adaptation
1. Analyze the model structure and error message.
2. Generate Python code that adapts the model to be compatible with tt-torch.
3. Consider these common adaptations:
   - Replace unsupported operations with supported ones
   - Modify model architecture to use only operations supported by tt-torch
   - Add custom forward functions that are compatible with tt-torch
   - Convert dynamic shapes to static shapes
   - Handle any custom CUDA operations

## Response Format
Provide your response in the following JSON format:
```json
{
    "adaptation_level": "level_1|level_2|level_3",
    "code": "# Your Python adaptation code here",
    "explanation": "Explanation of what the adaptation does and why it's necessary"
}
```

Where adaptation_level is one of:
- level_1: Minor changes (parameter adjustments, mode switching)
- level_2: Moderate changes (operation replacements, small architectural changes)
- level_3: Major changes (significant architectural modifications, custom implementations)

Focus on generating working code that solves the specific compatibility issues.
"""
        
        return prompt
    
    def _call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call the OpenAI API to generate adaptation code.
        
        Args:
            prompt: Formatted prompt for the API
            
        Returns:
            Dictionary with generation results
        """
        api_url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "gpt-4-turbo", # Using the latest GPT-4 model
            "messages": [
                {"role": "system", "content": "You are a PyTorch expert specialized in model compilation and adaptation."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2, # Lower temperature for more deterministic responses
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Extract JSON from the response
            try:
                # Look for JSON object in the response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    adaptation_data = json.loads(json_str)
                    
                    return {
                        "success": True,
                        "code": adaptation_data.get("code", ""),
                        "adaptation_level": adaptation_data.get("adaptation_level", "level_2"),
                        "explanation": adaptation_data.get("explanation", "")
                    }
                else:
                    # Fallback to extracting code blocks
                    code_blocks = self._extract_code_blocks(content)
                    if code_blocks:
                        return {
                            "success": True,
                            "code": code_blocks[0],
                            "adaptation_level": "level_2",
                            "explanation": content
                        }
            except Exception as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                pass
            
            # Fallback return the raw response
            return {
                "success": True,
                "code": content,
                "adaptation_level": "level_2",
                "explanation": "Raw response from LLM"
            }
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return {
                "success": False,
                "code": None,
                "adaptation_level": None,
                "explanation": f"Error calling OpenAI API: {str(e)}"
            }
    
    def _call_anthropic_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call the Anthropic API to generate adaptation code.
        
        Args:
            prompt: Formatted prompt for the API
            
        Returns:
            Dictionary with generation results
        """
        api_url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.2
        }
        
        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            
            result = response.json()
            content = result["content"][0]["text"]
            
            # Extract JSON from the response
            try:
                # Look for JSON object in the response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    adaptation_data = json.loads(json_str)
                    
                    return {
                        "success": True,
                        "code": adaptation_data.get("code", ""),
                        "adaptation_level": adaptation_data.get("adaptation_level", "level_2"),
                        "explanation": adaptation_data.get("explanation", "")
                    }
                else:
                    # Fallback to extracting code blocks
                    code_blocks = self._extract_code_blocks(content)
                    if code_blocks:
                        return {
                            "success": True,
                            "code": code_blocks[0],
                            "adaptation_level": "level_2",
                            "explanation": content
                        }
            except Exception as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                pass
            
            # Fallback return the raw response
            return {
                "success": True,
                "code": content,
                "adaptation_level": "level_2",
                "explanation": "Raw response from LLM"
            }
            
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}")
            return {
                "success": False,
                "code": None,
                "adaptation_level": None,
                "explanation": f"Error calling Anthropic API: {str(e)}"
            }
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """
        Extract code blocks from text.
        
        Args:
            text: Text that may contain code blocks
            
        Returns:
            List of extracted code blocks
        """
        import re
        
        # Match code blocks with triple backticks
        pattern = r"```(?:python)?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        # Clean up the code blocks
        code_blocks = [block.strip() for block in matches]
        
        return code_blocks
