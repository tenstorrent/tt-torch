"""
Model adaptation engine for making models compatible with tt-torch.
"""
from typing import Dict, Any, Optional, Tuple, List
import os
import importlib.util
import tempfile
import inspect
import re
import sys

import torch
from loguru import logger

from forge_agent.test_pipeline.models import (
    TestConfig, 
    AdaptationLevel, 
    FailureReason,
    TestResult
)
from forge_agent.llm.client import LLMClient


class AdaptationEngine:
    """
    Engine for adapting models to be compatible with tt-torch.
    Applies both standard templates and LLM-guided adaptations.
    """
    
    def __init__(self, templates_dir: Optional[str] = None, llm_provider: str = "openai"):
        """
        Initialize the adaptation engine.
        
        Args:
            templates_dir: Directory containing adaptation templates
            llm_provider: LLM provider to use for adaptation (openai, anthropic, etc.)
        """
        self.templates_dir = templates_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "adaptation_templates"
        )
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Ensure the template directory exists and has an __init__.py file
        init_file = os.path.join(self.templates_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# Adaptation templates for tt-torch compatibility\n")
        
        # Initialize LLM client
        self.llm_client = LLMClient(provider=llm_provider)
    
    def adapt_model(
        self, 
        model_data: Dict[str, Any], 
        test_config: TestConfig
    ) -> Tuple[bool, Optional[Dict[str, Any]], AdaptationLevel, Optional[FailureReason], Optional[str]]:
        """
        Adapt a model for tt-torch compatibility.
        
        Args:
            model_data: Dictionary containing model, config, and other model data
            test_config: Configuration for the model test
            
        Returns:
            Tuple containing:
            - Success flag (True if successful)
            - Dictionary with adapted model if successful
            - Adaptation level required
            - Failure reason enum if failed
            - Error message if failed
        """
        model_id = test_config.model_id
        logger.info(f"Adapting model: {model_id}")
        
        # Try standard adaptation templates first
        success, adapted_model, adaptation_level, failure_reason, error_message = self._apply_standard_templates(
            model_data, test_config
        )
        
        # If standard adaptation failed and LLM adaptation is enabled, try that
        if not success and test_config.use_llm_adaptation:
            logger.info(f"Standard adaptation failed for {model_id}, trying LLM-guided adaptation")
            success, adapted_model, adaptation_level, failure_reason, error_message = self._apply_llm_adaptation(
                model_data, test_config, error_message
            )
        
        if success:
            logger.info(f"Successfully adapted model {model_id} with adaptation level {adaptation_level}")
        else:
            logger.error(f"Failed to adapt model {model_id}: {error_message}")
            
        return success, adapted_model, adaptation_level, failure_reason, error_message
    
    def _apply_standard_templates(
        self, 
        model_data: Dict[str, Any], 
        test_config: TestConfig
    ) -> Tuple[bool, Optional[Dict[str, Any]], AdaptationLevel, Optional[FailureReason], Optional[str]]:
        """
        Apply standard adaptation templates based on model architecture.
        
        Args:
            model_data: Dictionary containing model data
            test_config: Configuration for the model test
            
        Returns:
            Tuple containing:
            - Success flag (True if successful)
            - Dictionary with adapted model if successful
            - Adaptation level required
            - Failure reason enum if failed
            - Error message if failed
        """
        model_id = test_config.model_id
        model = model_data.get("model")
        config = model_data.get("config")
        
        if model is None:
            return False, None, AdaptationLevel.BEYOND, FailureReason.ADAPTATION_ERROR, "No model provided"
        
        # Identify model architecture
        architecture_type = self._identify_architecture(model, model_id, config)
        
        # Try to find a matching template
        template_path = self._find_template_for_architecture(architecture_type)
        
        if template_path:
            try:
                # Load the template module
                spec = importlib.util.spec_from_file_location("template", template_path)
                template_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(template_module)
                
                # Apply the adaptation
                if hasattr(template_module, "adapt"):
                    adapted_model_data = template_module.adapt(model_data, test_config)
                    
                    # Determine adaptation level
                    adaptation_level = getattr(template_module, "ADAPTATION_LEVEL", AdaptationLevel.LEVEL_1)
                    
                    return True, adapted_model_data, adaptation_level, None, None
                else:
                    return False, None, AdaptationLevel.LEVEL_1, FailureReason.ADAPTATION_ERROR, "Template does not have adapt function"
            except Exception as e:
                logger.error(f"Error applying template {template_path}: {str(e)}")
                return False, None, AdaptationLevel.LEVEL_1, FailureReason.ADAPTATION_ERROR, f"Template error: {str(e)}"
        else:
            # No matching template found
            logger.warning(f"No matching template found for architecture {architecture_type}")
            return False, None, AdaptationLevel.LEVEL_2, FailureReason.UNSUPPORTED_ARCHITECTURE, f"No template for {architecture_type}"
    
    def _identify_architecture(self, model: Any, model_id: str, config: Any) -> str:
        """
        Identify the architecture type of a model.
        
        Args:
            model: The model object
            model_id: Hugging Face model ID
            config: Model configuration
            
        Returns:
            String identifier for the architecture type
        """
        # Check model class name
        class_name = model.__class__.__name__.lower()
        
        # Common architecture identifiers
        architectures = [
            "bert", "gpt", "t5", "llama", "roberta", "bart", "distilbert", 
            "vit", "resnet", "efficientnet", "swin", "transformer"
        ]
        
        # Check class name
        for arch in architectures:
            if arch in class_name:
                return arch
        
        # Check model ID
        for arch in architectures:
            if arch in model_id.lower():
                return arch
        
        # Check config
        if config and hasattr(config, "model_type"):
            return config.model_type.lower()
        
        # Default fallback
        return "generic"
    
    def _find_template_for_architecture(self, architecture_type: str) -> Optional[str]:
        """
        Find an adaptation template for the given architecture.
        
        Args:
            architecture_type: Type of model architecture
            
        Returns:
            Path to the template file, or None if not found
        """
        # Check for exact match
        template_path = os.path.join(self.templates_dir, f"{architecture_type}_template.py")
        if os.path.exists(template_path):
            return template_path
        
        # Check for generic template
        generic_path = os.path.join(self.templates_dir, "generic_template.py")
        if os.path.exists(generic_path):
            return generic_path
            
        return None
    
    def _apply_llm_adaptation(
        self, 
        model_data: Dict[str, Any], 
        test_config: TestConfig,
        error_message: Optional[str]
    ) -> Tuple[bool, Optional[Dict[str, Any]], AdaptationLevel, Optional[FailureReason], Optional[str]]:
        """
        Apply LLM-guided adaptation.
        
        Args:
            model_data: Dictionary containing model data
            test_config: Configuration for the model test
            error_message: Error message from standard adaptation attempt
            
        Returns:
            Tuple containing:
            - Success flag (True if successful)
            - Dictionary with adapted model if successful
            - Adaptation level required
            - Failure reason enum if failed
            - Error message if failed
        """
        logger.info(f"Applying LLM-guided adaptation for {test_config.model_id}")
        
        try:
            # Extract model and other important information
            model = model_data.get("model")
            if model is None:
                return False, None, AdaptationLevel.LEVEL_3, FailureReason.ADAPTATION_ERROR, "Model not available for LLM adaptation"
                
            # Extract model structure information
            model_info = self._extract_model_info(model, test_config.model_id)
            
            # Try to extract model source code if available
            model_code = None
            model_class_name = model.__class__.__name__
            try:
                model_source = inspect.getsource(model.__class__)
                model_code = model_source
            except Exception as e:
                logger.warning(f"Could not extract model source code: {str(e)}")
            
            # Call LLM to generate adaptation code
            logger.info("Calling LLM for adaptation code generation")
            llm_result = self.llm_client.generate_adaptation_code(
                model_info=model_info,
                error_message=error_message,
                model_code=model_code,
                model_class_name=model_class_name
            )
            
            if not llm_result["success"]:
                return False, None, AdaptationLevel.LEVEL_3, FailureReason.ADAPTATION_ERROR, \
                       f"LLM adaptation failed: {llm_result['explanation']}"
            
            # Extract adaptation information
            adaptation_code = llm_result["code"]
            adaptation_level_str = llm_result.get("adaptation_level", "level_2")
            explanation = llm_result.get("explanation", "")
            
            # Convert adaptation level string to enum
            adaptation_level = AdaptationLevel.LEVEL_2  # Default
            if adaptation_level_str == "level_1":
                adaptation_level = AdaptationLevel.LEVEL_1
            elif adaptation_level_str == "level_2":
                adaptation_level = AdaptationLevel.LEVEL_2
            elif adaptation_level_str == "level_3":
                adaptation_level = AdaptationLevel.LEVEL_3
            
            # Execute the adaptation code
            adapted_model_data = self._execute_adaptation_code(
                model_data=model_data,
                test_config=test_config,
                adaptation_code=adaptation_code
            )
            
            if adapted_model_data:
                # Successfully adapted
                logger.info(f"LLM adaptation successful with level {adaptation_level}")
                logger.info(f"Adaptation explanation: {explanation}")
                
                # Store the adaptation template for future use if it was successful
                architecture_type = self._identify_architecture(
                    model, test_config.model_id, model_data.get("config")
                )
                self.create_adaptation_template(
                    model_data, test_config, adaptation_code, architecture_type
                )
                
                return True, adapted_model_data, adaptation_level, None, None
            else:
                # Adaptation failed
                return False, None, adaptation_level, FailureReason.ADAPTATION_ERROR, \
                       "LLM adaptation code execution failed"
                
        except Exception as e:
            logger.error(f"Error during LLM adaptation: {str(e)}")
            return False, None, AdaptationLevel.LEVEL_3, FailureReason.ADAPTATION_ERROR, \
                   f"LLM adaptation error: {str(e)}"
        
    def _extract_model_info(self, model: Any, model_id: str) -> Dict[str, Any]:
        """
        Extract information about a model for LLM adaptation.
        
        Args:
            model: The model object
            model_id: Hugging Face model ID
            
        Returns:
            Dictionary with model information
        """
        model_info = {
            "model_id": model_id,
            "model_type": model.__class__.__name__
        }
        
        # Extract model structure
        try:
            # Get model structure as string representation
            model_structure = str(model)
            model_info["model_structure"] = model_structure
            
            # Try to get named modules
            named_modules = {}
            for name, module in model.named_modules():
                if name and not name.isdigit() and name != "":
                    named_modules[name] = str(module.__class__.__name__)
            model_info["named_modules"] = named_modules
            
            # Get list of parameters
            param_shapes = {}
            for name, param in model.named_parameters():
                param_shapes[name] = list(param.shape)
            model_info["parameter_shapes"] = param_shapes
            
        except Exception as e:
            logger.warning(f"Error extracting model structure: {str(e)}")
        
        return model_info
    
    def _execute_adaptation_code(self, model_data: Dict[str, Any], test_config: TestConfig, adaptation_code: str) -> Optional[Dict[str, Any]]:
        """
        Execute adaptation code generated by LLM.
        
        Args:
            model_data: Dictionary containing model data
            test_config: Configuration for the model test
            adaptation_code: Python code string generated by LLM
            
        Returns:
            Dictionary with adapted model data, or None if failed
        """
        try:
            # Create a temporary module to execute the adaptation code
            module_name = f"forge_agent_adaptation_{test_config.model_id.replace('/', '_').replace('-', '_')}"
            
            # Clean the code to ensure it defines an adapt function
            adaptation_code = self._clean_adaptation_code(adaptation_code)
            
            # Create a temporary file for the adaptation code
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
                tmp_file.write(adaptation_code.encode('utf-8'))
            
            try:
                # Import the temporary module
                spec = importlib.util.spec_from_file_location(module_name, tmp_file_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load module from {tmp_file_path}")
                    
                adaptation_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = adaptation_module
                spec.loader.exec_module(adaptation_module)
                
                # Check if the module has an adapt function
                if not hasattr(adaptation_module, "adapt"):
                    logger.error("Adaptation code does not define an 'adapt' function")
                    return None
                
                # Convert test_config to a dictionary for the adaptation function
                test_config_dict = {
                    "model_id": test_config.model_id,
                    "use_llm_adaptation": test_config.use_llm_adaptation
                }
                
                # Call the adapt function
                adapted_data = adaptation_module.adapt(model_data, test_config_dict)
                return adapted_data
                
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(tmp_file_path)
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                except Exception as e:
                    logger.warning(f"Error cleaning up temporary adaptation module: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error executing adaptation code: {str(e)}")
            return None
    
    def _clean_adaptation_code(self, code: str) -> str:
        """
        Clean and standardize adaptation code.
        
        Args:
            code: Raw adaptation code from LLM
            
        Returns:
            Cleaned code that defines an adapt function
        """
        # If the code already has an adapt function, use it as is
        if "def adapt(" in code:
            return code
        
        # If the code is just a code snippet, wrap it in an adapt function
        clean_code = """
from typing import Dict, Any
import torch

def adapt(model_data: Dict[str, Any], test_config: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt the model for tt-torch compatibility."""
    # Get the model and other data
    model = model_data.get("model")
    if model is None:
        return model_data
    
    # Make a copy of the input data
    adapted_data = {**model_data}
    
    # Apply adaptations
    try:
{}
    except Exception as e:
        print(f"Error in adaptation: {{str(e)}}")
    
    return adapted_data
"""
        
        # Indent the code to fit inside the function
        indented_code = "\n".join(f"        {line}" for line in code.strip().split("\n"))
        return clean_code.format(indented_code)

    def create_adaptation_template(
        self, 
        model_data: Dict[str, Any], 
        test_config: TestConfig,
        adaptation_code: str,
        architecture_type: str
    ) -> bool:
        """
        Create a new adaptation template from successful adaptation code.
        
        Args:
            model_data: Dictionary containing model data
            test_config: Configuration for the model test
            adaptation_code: Successful adaptation code
            architecture_type: Type of model architecture
            
        Returns:
            True if template was created successfully
        """
        try:
            template_path = os.path.join(self.templates_dir, f"{architecture_type}_template.py")
            
            # Check if template already exists
            if os.path.exists(template_path):
                logger.warning(f"Template for {architecture_type} already exists, not overwriting")
                return False
            
            # Create the template file
            with open(template_path, "w") as f:
                f.write(f'''"""
Adaptation template for {architecture_type} models.
"""
from typing import Dict, Any, Optional

# Level of adaptation required for this template
ADAPTATION_LEVEL = "level_2"

def adapt(model_data: Dict[str, Any], test_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt a {architecture_type} model for tt-torch compatibility.
    
    Args:
        model_data: Dictionary containing model, config, and other model data
        test_config: Configuration for the model test
        
    Returns:
        Dictionary with adapted model data
    """
    # Generated adaptation code
{adaptation_code}
''')
            
            return True
        except Exception as e:
            logger.error(f"Error creating adaptation template: {str(e)}")
            return False


# Create default templates directory with a generic template
def create_default_templates():
    """Create default adaptation templates."""
    templates_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "adaptation_templates"
    )
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create __init__.py
    with open(os.path.join(templates_dir, "__init__.py"), "w") as f:
        f.write("# Adaptation templates for tt-torch compatibility\n")
    
    # Create generic template
    generic_path = os.path.join(templates_dir, "generic_template.py")
    if not os.path.exists(generic_path):
        with open(generic_path, "w") as f:
            f.write('''"""
Generic adaptation template for models without specific templates.
"""
from typing import Dict, Any, Optional
import torch

# Level of adaptation required for this template
ADAPTATION_LEVEL = "level_1"

def adapt(model_data: Dict[str, Any], test_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply generic adaptations to make the model compatible with tt-torch.
    
    Args:
        model_data: Dictionary containing model, config, and other model data
        test_config: Configuration for the model test
        
    Returns:
        Dictionary with adapted model data
    """
    # Get the model
    model = model_data.get("model")
    
    # Set model to eval mode for inference
    if model is not None:
        model.eval()
    
    # Make a copy of the input data
    adapted_data = {**model_data}
    
    return adapted_data
''')


# Create default templates on import
create_default_templates()
