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
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "anthropic"):
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
        logger.info(f"🤖 LLM ADAPTATION: Starting model adaptation")
        logger.info(f"   - Model: {model_class_name or 'Unknown'}")
        logger.info(f"   - Model type: {model_info.get('model_type', 'Unknown')}")
        if error_message:
            logger.info(f"   - Error: {error_message}")
        if model_code:
            logger.info(f"   - Original code length: {len(model_code)} characters")
        
        if not self.api_key:
            logger.warning(f"❌ LLM ADAPTATION: No API key available for {self.provider}")
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
        
        logger.info(f"📝 LLM ADAPTATION: Sending prompt to {self.provider}")
        logger.debug(f"   - Prompt length: {len(prompt)} characters")
        
        # Call the appropriate provider's API
        for attempt in range(max_retries):
            try:
                logger.info(f"📞 LLM ADAPTATION: API call attempt {attempt + 1}/{max_retries}")
                
                if self.provider == "openai":
                    result = self._call_openai_api(prompt)
                elif self.provider == "anthropic":
                    result = self._call_anthropic_api(prompt)
                else:
                    logger.error(f"❌ LLM ADAPTATION: Unsupported provider: {self.provider}")
                    return {
                        "success": False,
                        "code": None,
                        "adaptation_level": None,
                        "explanation": f"Unsupported LLM provider: {self.provider}"
                    }
                
                if result.get("success"):
                    logger.info(f"✅ LLM ADAPTATION: Successfully generated adaptation code")
                    logger.info(f"   - Adaptation level: {result.get('adaptation_level', 'Unknown')}")
                    
                    # Log the adaptation details
                    self._log_adaptation_details(result, model_code, model_class_name)
                else:
                    logger.warning(f"⚠️ LLM ADAPTATION: Failed - {result.get('explanation', 'Unknown error')}")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ LLM ADAPTATION: API error (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "code": None,
                        "adaptation_level": None,
                        "explanation": f"Error calling LLM API: {str(e)}"
                    }
    
    def fix_input_arguments(
        self,
        model_info: Dict[str, Any],
        error_message: str,
        current_inputs: Dict[str, Any],
        model_class_name: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Fix input arguments for a model using an LLM when there are argument mismatches.
        
        Args:
            model_info: Dictionary containing model info (type, structure, etc.)
            error_message: Error message from failed model execution
            current_inputs: Current sample inputs that failed
            model_class_name: Class name of the model
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing:
            - success: Whether input fixing was successful
            - inputs: Fixed sample inputs as dictionary
            - explanation: Explanation of the input fixes
        """
        logger.info(f"🤖 LLM INPUT FIX: Starting input argument fixing")
        logger.info(f"   - Model: {model_class_name or 'Unknown'}")
        logger.info(f"   - Error: {error_message}")
        logger.info(f"   - Current inputs: {list(current_inputs.keys()) if current_inputs else 'None'}")
        
        if not self.api_key:
            logger.warning(f"❌ LLM INPUT FIX: No API key available for {self.provider}")
            return {
                "success": False,
                "inputs": None,
                "explanation": "No API key available for LLM service"
            }
        
        # Construct prompt for the LLM
        prompt = self._construct_input_fix_prompt(
            model_info, error_message, current_inputs, model_class_name
        )
        
        logger.info(f"📝 LLM INPUT FIX: Sending prompt to {self.provider}")
        logger.debug(f"   - Prompt length: {len(prompt)} characters")
        
        # Call the appropriate provider's API
        for attempt in range(max_retries):
            try:
                logger.info(f"📞 LLM INPUT FIX: API call attempt {attempt + 1}/{max_retries}")
                
                if self.provider == "openai":
                    result = self._call_openai_input_fix_api(prompt)
                elif self.provider == "anthropic":
                    result = self._call_anthropic_input_fix_api(prompt)
                else:
                    logger.error(f"❌ LLM INPUT FIX: Unsupported provider: {self.provider}")
                    return {
                        "success": False,
                        "inputs": None,
                        "explanation": f"Unsupported LLM provider: {self.provider}"
                    }
                
                if result.get("success"):
                    logger.info(f"✅ LLM INPUT FIX: Successfully generated input fixes")
                    logger.info(f"   - Fixed inputs: {list(result.get('inputs', {}).keys())}")
                else:
                    logger.warning(f"⚠️ LLM INPUT FIX: Failed - {result.get('explanation', 'Unknown error')}")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ LLM INPUT FIX: API error (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "inputs": None,
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

    def _construct_input_fix_prompt(
        self,
        model_info: Dict[str, Any],
        error_message: str,
        current_inputs: Dict[str, Any],
        model_class_name: Optional[str] = None
    ) -> str:
        """
        Construct a prompt for the LLM to fix input arguments.
        
        Args:
            model_info: Dictionary containing model info
            error_message: Error message from failed model execution
            current_inputs: Current sample inputs that failed
            model_class_name: Class name of the model
            
        Returns:
            Formatted prompt string
        """
        # Extract model structure
        model_type = model_info.get("model_type", "unknown")
        model_structure = model_info.get("model_structure", "")
        
        # Format current inputs for display
        current_inputs_str = ""
        if current_inputs:
            current_inputs_str = "\n".join([f"- {k}: {type(v).__name__} {getattr(v, 'shape', 'unknown shape')}" 
                                          for k, v in current_inputs.items()])
        
        # Base prompt
        prompt = f"""
You are an AI assistant specialized in fixing PyTorch model input arguments.
A model is failing to execute due to input argument mismatches. Your task is to analyze the error and generate the correct input arguments.

## Model Information
- Model Type: {model_type}
- Model Class: {model_class_name or 'Unknown'}

## Model Structure
{model_structure}

## Error Message
The following error occurred when trying to run the model:
```
{error_message}
```

## Current Inputs (that failed)
{current_inputs_str}

## Task
Based on the error message and model information, generate the correct input arguments for this model.
The error likely indicates that the model expects different parameter names or shapes.

For example:
- ViT models typically expect "pixel_values" instead of "hidden_states"
- BERT models expect "input_ids", "attention_mask", "token_type_ids"
- ResNet models expect "pixel_values" or just positional tensor arguments
- GPT models expect "input_ids" and "attention_mask"

## Output Format
Respond with a JSON object containing the corrected inputs:

```json
{{
    "success": true,
    "inputs": {{
        "parameter_name": {{
            "shape": [1, 3, 224, 224],
            "dtype": "float32",
            "description": "RGB image tensor"
        }}
    }},
    "explanation": "The model expects 'pixel_values' instead of 'hidden_states' because..."
}}
```

Make sure to:
1. Analyze the error message to understand what inputs are expected
2. Provide appropriate tensor shapes for the model type
3. Use correct parameter names based on the model architecture
4. Include clear explanation of why these inputs are correct

Generate the corrected inputs now:
"""
        return prompt

    def _call_openai_input_fix_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call the OpenAI API to fix input arguments.
        
        Args:
            prompt: Formatted prompt for the API
            
        Returns:
            Dictionary with input fix results
        """
        api_url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "gpt-4-turbo",
            "messages": [
                {"role": "system", "content": "You are a PyTorch expert specialized in model input arguments and tensor shapes."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1500
        }
        
        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return self._parse_input_fix_response(content)
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API for input fix: {str(e)}")
            return {
                "success": False,
                "inputs": None,
                "explanation": f"Error calling OpenAI API: {str(e)}"
            }

    def _call_anthropic_input_fix_api(self, prompt: str) -> Dict[str, Any]:
        """
        Call the Anthropic API to fix input arguments.
        
        Args:
            prompt: Formatted prompt for the API
            
        Returns:
            Dictionary with input fix results
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
            "max_tokens": 1500,
            "temperature": 0.2
        }
        
        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            
            result = response.json()
            content = result["content"][0]["text"]
            
            return self._parse_input_fix_response(content)
            
        except Exception as e:
            logger.error(f"Error calling Anthropic API for input fix: {str(e)}")
            return {
                "success": False,
                "inputs": None,
                "explanation": f"Error calling Anthropic API: {str(e)}"
            }

    def _parse_input_fix_response(self, content: str) -> Dict[str, Any]:
        """
        Parse the LLM response for input fixes.
        
        Args:
            content: Raw response content from LLM
            
        Returns:
            Dictionary with parsed input fix results
        """
        try:
            # Look for JSON object in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                fix_data = json.loads(json_str)
                
                # Convert the input specifications to actual tensors
                if fix_data.get("success") and "inputs" in fix_data:
                    import torch
                    actual_inputs = {}
                    
                    for param_name, param_spec in fix_data["inputs"].items():
                        shape = param_spec.get("shape", [1, 3, 224, 224])
                        dtype_str = param_spec.get("dtype", "float32")
                        
                        # Convert string dtype to torch dtype
                        if dtype_str == "float32":
                            dtype = torch.float32
                        elif dtype_str == "int64" or dtype_str == "long":
                            dtype = torch.long
                        elif dtype_str == "int32":
                            dtype = torch.int32
                        else:
                            dtype = torch.float32  # fallback
                        
                        # Create appropriate tensor
                        if "input_ids" in param_name or "token_type_ids" in param_name or "attention_mask" in param_name:
                            # For text tokens, use random integers
                            if "attention_mask" in param_name:
                                actual_inputs[param_name] = torch.ones(shape, dtype=dtype)
                            else:
                                actual_inputs[param_name] = torch.randint(0, 1000, shape, dtype=dtype)
                        else:
                            # For image/float tensors, use random floats
                            actual_inputs[param_name] = torch.rand(shape, dtype=dtype)
                    
                    return {
                        "success": True,
                        "inputs": actual_inputs,
                        "explanation": fix_data.get("explanation", "Inputs fixed by LLM")
                    }
                
            # If JSON parsing fails, try to extract common patterns
            if "pixel_values" in content.lower():
                import torch
                return {
                    "success": True,
                    "inputs": {"pixel_values": torch.rand(1, 3, 224, 224)},
                    "explanation": "Extracted pixel_values from LLM response"
                }
            elif "input_ids" in content.lower():
                import torch
                return {
                    "success": True,
                    "inputs": {
                        "input_ids": torch.randint(0, 1000, (1, 16), dtype=torch.long),
                        "attention_mask": torch.ones(1, 16, dtype=torch.long)
                    },
                    "explanation": "Extracted text input format from LLM response"
                }
                
        except Exception as e:
            logger.error(f"Error parsing input fix response: {str(e)}")
        
        # Fallback
        return {
            "success": False,
            "inputs": None,
            "explanation": f"Could not parse LLM response for input fixes"
        }

    def _log_adaptation_details(
        self,
        result: Dict[str, Any],
        original_code: Optional[str],
        model_class_name: Optional[str]
    ) -> None:
        """
        Log detailed information about what changes the LLM made during adaptation.
        
        Args:
            result: The result dictionary from LLM adaptation
            original_code: The original model code before adaptation
            model_class_name: Name of the model class being adapted
        """
        logger.info(f"🔄 CODE ADAPTATION CHANGES MADE BY LLM:")
        
        adaptation_level = result.get('adaptation_level', 'Unknown')
        explanation = result.get('explanation', 'No explanation provided')
        generated_code = result.get('code', '')
        
        # Log adaptation level with emoji indicators
        level_indicators = {
            'level_1': '🟢 MINOR',
            'level_2': '🟡 MODERATE', 
            'level_3': '🔴 MAJOR'
        }
        level_indicator = level_indicators.get(adaptation_level, '⚪ UNKNOWN')
        
        logger.info(f"   📊 ADAPTATION LEVEL: {level_indicator} ({adaptation_level})")
        logger.info(f"   🎯 MODEL: {model_class_name or 'Unknown'}")
        
        # Log explanation with proper formatting
        logger.info(f"   💡 LLM EXPLANATION:")
        for line in explanation.split('\n'):
            if line.strip():
                logger.info(f"      {line.strip()}")
        
        # Show original vs adapted code comparison
        if original_code and generated_code:
            logger.info(f"   📋 CODE CHANGES:")
            logger.info(f"      📥 ORIGINAL CODE ({len(original_code)} chars):")
            
            # Show first few lines of original code
            original_lines = original_code.split('\n')
            for i, line in enumerate(original_lines[:10]):
                logger.info(f"         {i+1:2d} | {line}")
            if len(original_lines) > 10:
                remaining_lines = len(original_lines) - 10
                logger.info(f"         ... ({remaining_lines} more lines)")
            
            logger.info(f"      📤 ADAPTED CODE ({len(generated_code)} chars):")
            
            # Show first few lines of generated code
            generated_lines = generated_code.split('\n')
            for i, line in enumerate(generated_lines[:10]):
                logger.info(f"         {i+1:2d} | {line}")
            if len(generated_lines) > 10:
                remaining_lines = len(generated_lines) - 10
                logger.info(f"         ... ({remaining_lines} more lines)")
                
            # Highlight key differences
            logger.info(f"   🔍 KEY CHANGES DETECTED:")
            self._analyze_code_differences(original_code, generated_code)
            
        elif generated_code:
            logger.info(f"   📤 GENERATED ADAPTATION CODE ({len(generated_code)} chars):")
            
            # Show the generated code
            generated_lines = generated_code.split('\n')
            for i, line in enumerate(generated_lines[:15]):
                logger.info(f"         {i+1:2d} | {line}")
            if len(generated_lines) > 15:
                remaining_lines = len(generated_lines) - 15
                logger.info(f"         ... ({remaining_lines} more lines)")
        
        logger.info(f"   ✅ ADAPTATION COMPLETE")

    def _analyze_code_differences(self, original: str, adapted: str) -> None:
        """
        Analyze and log key differences between original and adapted code.
        
        Args:
            original: Original model code
            adapted: LLM-adapted model code
        """
        import re
        
        # Look for common adaptation patterns
        patterns_to_check = [
            (r'class\s+(\w+)', 'Class definitions'),
            (r'def\s+(\w+)', 'Method definitions'),
            (r'import\s+(.+)', 'Import statements'),
            (r'torch\.(\w+)', 'PyTorch operations'),
            (r'nn\.(\w+)', 'Neural network layers'),
            (r'F\.(\w+)', 'Functional operations'),
            (r'\.to\((.+?)\)', 'Device/dtype conversions'),
            (r'\.view\((.+?)\)', 'Tensor reshaping'),
            (r'\.reshape\((.+?)\)', 'Tensor reshaping'),
        ]
        
        original_matches = {}
        adapted_matches = {}
        
        for pattern, description in patterns_to_check:
            original_matches[description] = set(re.findall(pattern, original, re.IGNORECASE))
            adapted_matches[description] = set(re.findall(pattern, adapted, re.IGNORECASE))
        
        changes_found = False
        
        for description in original_matches:
            orig_set = original_matches[description]
            adapted_set = adapted_matches[description]
            
            added = adapted_set - orig_set
            removed = orig_set - adapted_set
            
            if added or removed:
                changes_found = True
                logger.info(f"      🔧 {description}:")
                
                if removed:
                    logger.info(f"         ❌ REMOVED: {', '.join(sorted(removed))}")
                if added:
                    logger.info(f"         ✅ ADDED: {', '.join(sorted(added))}")
        
        if not changes_found:
            logger.info(f"      📝 Code structure appears similar - changes may be in logic/parameters")
        
        # Check for specific adaptation keywords
        adaptation_keywords = [
            'tt_torch', 'tenstorrent', 'compile', 'backend', 
            'unsupported', 'replace', 'workaround', 'fallback'
        ]
        
        found_keywords = []
        for keyword in adaptation_keywords:
            if keyword.lower() in adapted.lower() and keyword.lower() not in original.lower():
                found_keywords.append(keyword)
        
        if found_keywords:
            logger.info(f"      🎯 ADAPTATION KEYWORDS FOUND: {', '.join(found_keywords)}")
