# GPT-4 Vision with Preprocessed Images Test

This directory contains scripts to test and compare GPT-4 Vision performance with raw images vs preprocessed images for smart meter reading.

## Files

- **`gpt_preprocessed_ocr.py`** - GPT-4 Vision script that uses preprocessed images
- **`compare_approaches.py`** - Comparison script to test both approaches side by side
- **`README.md`** - This file

## Setup

1. **Install dependencies:**
   ```bash
   pip install openai python-dotenv
   ```

2. **Set up your OpenAI API key:**
   Create a `.env` file in the parent directory with:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Option 1: Test Preprocessed Images Only

Run the preprocessed version:
```bash
cd gpt_preprocessed_test
python gpt_preprocessed_ocr.py
```

This will:
- Process all images in the `../images/` directory
- Apply preprocessing (resolution, noise removal, contrast, etc.)
- Send preprocessed images to GPT-4 Vision
- Save results to `output/gpt4_preprocessed_extractions.json`
- Validate against database and save to `output/gpt4_preprocessed_validation.json`

### Option 2: Compare Both Approaches

Run the comparison test:
```bash
cd gpt_preprocessed_test
python compare_approaches.py
```

This will:
- Test each image with both raw and preprocessed approaches
- Compare success rates, confidence scores, and accuracy
- Provide recommendations on which approach to use
- Save detailed comparison results to `output/comparison_results_*.json`

## What Gets Tested

### Raw Images Approach
- Sends original images directly to GPT-4 Vision
- No preprocessing applied
- Faster processing time

### Preprocessed Images Approach
- Applies full preprocessing pipeline:
  - Resolution enhancement
  - Noise removal
  - Contrast enhancement
  - Adaptive thresholding
  - Deskewing correction
- Sends enhanced images to GPT-4 Vision
- Slower processing time but potentially better results

## Output Files

All results are saved in the `output/` directory:

- **`gpt4_preprocessed_extractions.json`** - Raw extraction results
- **`gpt4_preprocessed_validation.json`** - Database validation results
- **`comparison_results_*.json`** - Side-by-side comparison analysis

## Key Metrics Compared

1. **Success Rate** - Percentage of images where readings were successfully extracted
2. **Confidence Score** - GPT-4's confidence in the extracted reading (1-10)
3. **Preprocessing Effectiveness** - How much preprocessing helped (1-10)
4. **Database Match Rate** - How well results match your existing database

## Expected Results

The comparison will tell you:
- ✅ If preprocessing improves results
- ❌ If preprocessing reduces results  
- ➖ If preprocessing has no significant impact

## Recommendations

Based on the results, you'll get specific recommendations:
- Whether to use preprocessing for your specific images
- If preprocessing is introducing artifacts
- Which approach provides higher confidence
- Processing speed vs accuracy trade-offs

## Notes

- Both scripts use the same preprocessing pipeline as `img_preprocessing.py`
- The comparison requires OpenAI API credits for both approaches
- Results may vary depending on image quality and content
- Preprocessing effectiveness is rated by GPT-4 itself 