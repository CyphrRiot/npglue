# NPGlue Progress Log - INT4 to INT8 Migration

**DO NOT ADD THIS FILE TO GITHUB** - This is for crash recovery only

## Project Overview
NPGlue is a local AI development assistant using DeepSeek-R1 with OpenVINO optimization.

## Current Status (2025-06-22 20:15)

### Issue Identified
- Model was using INT4-AWQ quantization which was causing poor response quality
- Responses were often incomplete, rambling, or not properly formed
- Need to migrate from INT4 to INT8 for better reasoning and response quality

### Current State Analysis

#### Git Status
- Branch: main
- Uncommitted changes in:
  - `install` script (modified)
  - `server_production.py` (modified)

#### Model Files Present
- Currently have INT4-AWQ model: `models/deepseek-r1-int4-awq/DeepSeek-R1-0528-Qwen3-8B-int4_asym-awq-se-ov/`
- Files: openvino_model.bin, openvino_model.xml, tokenizer files

#### Code Changes Made
1. **Install Script (`install`)**: 
   - Changed download path from INT4-AWQ to INT8
   - Updated from `models/deepseek-r1-int4-awq` to `models/deepseek-r1-int8`
   - Updated allow_patterns for INT8 model variant

2. **Server Code (`server_production.py`)**:
   - Updated model path to INT8: `"models/deepseek-r1-int8/DeepSeek-R1-0528-Qwen3-8B-int8_asym-ov"`
   - Memory management and response length controls in place
   - Multiple API endpoints: OpenAI-compatible, Ollama-compatible, native
   - Aggressive memory protection with 2GB minimum free requirement

## Migration Plan

### Phase 1: Download INT8 Model ⏳ WAITING FOR USER
- [x] Update install script to point to INT8 variant
- [ ] **USER ACTION NEEDED**: Run `./install` to download INT8 model (large download!)
- [ ] Verify INT8 model downloads correctly
- [ ] Test model loading without generation

### Phase 2: Update Server Configuration ✅ SWITCHING TO BETTER MODEL
- [x] Update server_production.py model path
- [x] **ISSUE FOUND**: Memory checks too aggressive (1.5GB -> 0.8GB)
- [x] **FIXED**: Reduced memory requirements for generation
- [x] Test server startup with INT8 model
- [❌] **MAJOR ISSUE**: DeepSeek-R1 model itself is designed for reasoning, NOT direct answers
- [x] **TRIED**: Multiple prompt formats and post-processing attempts
- [❌] **CONCLUSION**: DeepSeek-R1 fundamentally wrong model type for direct Q&A
- [✅] **NEW PLAN**: Switch to Qwen3 models - designed for general tasks & direct answers
- [x] **COMPLETED**: Update install script with model choice menu
- [x] **COMPLETED**: Support both Qwen3-8B-INT8 (~6-8GB) and Qwen3-0.6B-FP16 (~1-2GB)
- [x] **COMPLETED**: Server auto-detects chosen model from config file
- [x] **COMPLETED**: Simplified generation parameters (removed aggressive post-processing)
- [x] **COMPLETED**: Updated app title and descriptions
- [x] **COMPLETED**: Updated README.md to reflect Qwen3 models and new features
- [x] **MEMORY OPTIONS**: 8B for quality OR 0.6B for speed/low memory
- [x] **CRITICAL FIX**: Install script now uses CPU-only PyTorch (no NVIDIA dependencies)
- [x] **CRITICAL FIX v2**: Use --no-deps and manual dependency installation to avoid CUDA
- [x] **NAMING FIX**: Changed environment name from "openvino-env" to "npglue-env" (consistent branding)
- [x] **USER CHOICE**: Install script now offers model selection menu
- [ ] **USER ACTION**: Run updated `./install` and choose model size
- [ ] **NEXT**: Test new model with "What is 2+2" question

### Phase 3: Quality Validation
- [ ] Test response quality and completeness
- [ ] Verify memory usage is acceptable
- [ ] Test with different prompt types
- [ ] Document performance differences

### Phase 4: Deployment
- [ ] Commit changes to git
- [ ] Update documentation if needed
- [ ] Remove old INT4 model files to save space

## Technical Details

### Model Variants
- **INT4-AWQ**: `DeepSeek-R1-0528-Qwen3-8B-int4_asym-awq-se-ov`
  - Smaller size, faster inference
  - Lower quality responses, reasoning issues
- **INT8**: `DeepSeek-R1-0528-Qwen3-8B-int8_asym-ov` 
  - Larger size, better quality
  - Should improve response coherence

### Memory Considerations
- INT8 model will use more RAM than INT4
- Current system has aggressive memory protection (2GB minimum free)
- May need to adjust memory thresholds

### API Endpoints
- `/v1/chat/completions` - OpenAI compatible
- `/api/chat` - Ollama compatible  
- `/generate` - Native endpoint
- All endpoints have memory monitoring

## Next Steps
1. **USER ACTION**: Run `./install` to download the INT8 model (this is a large download!)
2. After download completes, test server startup and basic functionality
3. Compare response quality with previous version
4. Adjust memory limits if needed

---
*Last updated: 2025-06-22 20:40 - Waiting for user to run install script*
