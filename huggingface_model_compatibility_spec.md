# Hugging Face Model Compatibility Pipeline Specification

## Overview
This specification outlines a system to evaluate the compatibility of the top 10,000 Hugging Face models with the Tenstorrent compiler and hardware. The goal is to create an agentic pipeline that automatically tests these models, documents their compatibility status, and provides insights into the success rate and areas for improvement.

## 1. Project Goals

- Test the top 10,000 Hugging Face models based on popularity/download count
- Create an automated pipeline to attempt compilation and execution on Tenstorrent hardware using the tt-torch frontend
- Document which models successfully run and which encounter errors
- Provide statistical insights on model compatibility with tt-torch
- Establish a baseline for future tt-torch improvements and PyTorch integration enhancements

## 2. System Architecture

### 2.1 Components

#### 2.1.1 Model Selection and Metadata Service
- Query the Hugging Face API to get the top 10,000 models by download count
- Extract relevant metadata (model architecture, framework, size, etc.)
- Filter models by criteria (size, framework, etc.)
- Store metadata in a structured database

#### 2.1.2 Agentic Test Pipeline
- Autonomous system that:
  - Retrieves model information from metadata service
  - Downloads model weights and architecture
  - Performs automated adaptation using LLM-guided code generation
  - Attempts compilation using tt-torch's PyTorch integration
  - Executes inference with sample inputs
  - Records execution metrics and success/failure status
  - Logs detailed error information if failures occur

#### 2.1.3 Result Tracking Database
- Store detailed information about each test attempt:
  - Model identifier and metadata
  - Compilation success/failure status
  - Execution success/failure status
  - Error messages and stack traces
  - Performance metrics (compilation time, execution time, memory usage)
  - Logs of attempted adaptations

#### 2.1.4 Analysis and Reporting Dashboard
- Visual dashboard showing:
  - Overall compatibility rate
  - Success rates by model architecture
  - Success rates by framework
  - Common failure patterns
  - Performance metrics across model types

### 2.2 Workflow

1. **Model Selection**
   - Query Hugging Face API for top models
   - Filter models based on predefined criteria
   - Queue models for testing

2. **Testing Lifecycle**
   - For each model:
     1. Download model and sample inputs
     2. Apply standard adaptation templates based on model architecture for tt-torch compatibility
     3. If generic adaptation fails, engage LLM-guided adaptation
     4. Attempt compilation with tt-torch frontend
     5. Execute inference with sample inputs
     6. Validate outputs (when possible)
     7. Record results

3. **Result Analysis**
   - Aggregate success/failure statistics
   - Identify patterns in failures
   - Categorize models by compatibility status
   - Generate recommendations for compiler improvements

## 3. Model Testing Process

### 3.1 Model Qualification Criteria

A model is considered "qualified" for testing if:
- It is one of the top 10,000 models by download count on Hugging Face
- It uses a framework supported by Tenstorrent (PyTorch, TensorFlow/JAX via XLA)
- It has public weights available for download
- It has a documented input/output format

### 3.2 Testing Steps

1. **Model Preparation**
   - Download model weights and configuration
   - Create a standardized test harness based on model type
   - Prepare sample inputs (synthetic or from datasets)

2. **Adaptation Attempt**
   - Apply standard adaptation patterns based on model architecture
   - Generate model-specific adapter code using templates
   - If standard adaptation fails, use LLM to generate custom adaptations

3. **Compilation Process**
   - Attempt compilation using tt-torch's workflow:
     ```python
     import tt_torch
     from tt_torch.tools.utils import CompilerConfig
     from tt_torch.dynamo.backend import BackendOptions
     
     # Configure compiler options
     cc = CompilerConfig()
     cc.enable_consteval = True
     cc.consteval_parameters = True
     
     # Set backend options
     options = BackendOptions()
     options.compiler_config = cc
     
     # Compile the model
     tt_model = torch.compile(model, backend="tt", dynamic=False, options=options)
     ```
   - Record compilation time, memory usage, and any warnings/errors
   - If compilation fails, attempt progressively more aggressive fallback strategies (e.g., adjusting CompilerConfig settings)

4. **Execution Verification**
   - Run inference with sample inputs
   - Compare outputs against expected format (not necessarily expected values)
   - Measure inference performance metrics

### 3.3 Success Criteria

A model is considered to "run successfully" if:
1. It compiles without errors using tt-torch's `torch.compile()` with the "tt" backend
2. It executes inference on sample inputs without crashing
3. It produces outputs in the expected format
4. (Optional) Outputs match reference outputs within acceptable tolerance

## 4. LLM-Assisted Adaptation Approach

The agentic system will use LLMs to assist in adapting models that don't compile or execute correctly on first attempt:

### 4.1 LLM Agent Responsibilities
- Analyze model architecture and compilation errors
- Generate adaptation code based on successful patterns for tt-torch compatibility
- Modify model code to be compatible with Tenstorrent compiler requirements using tt-torch
- Document adaptation strategies for future reference

### 4.2 Adaptation Workflow
1. Attempt compilation with standard tt-torch approach
2. If compilation fails, extract error information
3. Present errors and model information to LLM agent
4. LLM suggests code modifications to resolve issues (e.g., modifying model architecture or adjusting tt-torch CompilerConfig parameters)
5. Apply modifications and retry compilation
6. If successful, record adaptation pattern for future use
7. If unsuccessful after N attempts, mark model as incompatible with current tt-torch version

### 4.3 Test Case Generation
For each successfully adapted model:
1. Create a unit test in the tt-torch `tests` folder that validates the model compilation and execution
2. The test should include:
   - Model initialization code
   - Any required adaptation code
   - Sample inputs for validation
   - Expected output format validation
3. Tests should be categorized by model type and added to the CI pipeline to ensure continued compatibility
4. Documentation within the test should explain any special adaptations or considerations

### 4.4 Adaptation Effort Levels
- **Level 1 (Low effort)**: Apply standard templates and minimal changes
- **Level 2 (Medium effort)**: Apply targeted modifications to specific model components
- **Level 3 (High effort)**: Significant restructuring of model architecture
- **Beyond Level 3**: Mark as incompatible with current compiler version

## 5. Data Collection and Analysis

### 5.1 Per-Model Data Points
- Model identifier and basic metadata (architecture, framework, size)
- Compilation status (success/failure)
- Compilation time and resource usage
- Execution status (success/failure)
- Inference performance metrics
- Adaptation level required (if applicable)
- Specific error messages or failure points

### 5.2 Aggregate Analytics
- Overall compatibility rate
- Compatibility by model architecture
- Compatibility by framework
- Common failure patterns and their frequency
- Adaptation success rate
- Performance comparison across compatible models

## 6. Estimated Timeline and Resource Requirements

### 6.1 Development Phase (4-6 weeks)
- Week 1-2: Build metadata service and database infrastructure
- Week 2-3: Develop standard adaptation templates
- Week 3-4: Implement LLM-based adaptation system
- Week 4-5: Create testing harness and result tracking
- Week 5-6: Develop analytics dashboard

### 6.2 Testing Phase (8-12 weeks)
- Models processed in batches of 500-1000 per week
- Initial focus on most popular model architectures
- Progressive testing of less common architectures
- Weekly analysis and compiler optimization suggestions

### 6.3 Resource Requirements
- Compute:
  - Testing cluster with Tenstorrent hardware (minimum 4-8 devices)
  - CPU/GPU servers for model preprocessing
  - Storage for model weights and test results (estimated 5-10TB)
  
- Personnel:
  - 2-3 ML engineers for pipeline development
  - 1-2 DevOps engineers for infrastructure
  - 1 Data scientist for analytics
  
- External Dependencies:
  - Hugging Face API access
  - LLM API for adaptation assistance

## 7. Expected Outcomes and Success Metrics

### 7.1 Key Performance Indicators
- Percentage of models successfully running (target: >50%)
- Average adaptation time per model (target: <30 minutes)
- Categorized failure patterns with frequency analysis
- Compiler enhancement recommendations

### 7.2 Deliverables
- Fully automated testing pipeline
- Comprehensive database of model compatibility status
- Dashboard with compatibility analytics
- Documentation of common adaptation patterns
- Recommendations for compiler improvements

### 7.3 Future Expansion
- Continuous testing of new models added to Hugging Face
- Integration with CI/CD for compiler updates
- Expansion to include non-Hugging Face models
- Automated adaptation system for user-submitted models

## 8. Time Estimate for Full Execution

Based on initial testing of similar workflows, we estimate the following time requirements for full execution of the pipeline on 10,000 models:

- **Development time**: 4-6 weeks
- **Testing preparation**: 1-2 weeks
- **Model testing**: 8-12 weeks (processing ~1,000 models per week)
- **Analysis and reporting**: 2-3 weeks (overlapping with testing)

**Total time estimate**: 15-23 weeks for complete pipeline execution, with preliminary results available from week 7 onwards.

### 8.1 Factors Affecting Timeline
- Availability of Tenstorrent hardware resources
- Complexity of model architectures encountered
- Success rate of automated adaptations with tt-torch
- API rate limits for Hugging Face
- Optimization of parallel testing capabilities
- Compatibility issues between PyTorch models and tt-torch requirements

## 9. Conclusion

This specification outlines an agentic pipeline for testing Hugging Face model compatibility with the Tenstorrent tt-torch frontend at scale. By systematically evaluating 10,000 popular models, we will gain valuable insights into tt-torch performance, identify common adaptation patterns, and establish a clear roadmap for tt-torch improvements. The automated nature of the system allows for efficient testing and reduces manual effort while maintaining comprehensive documentation of results and guiding the evolution of the tt-torch PyTorch integration.
