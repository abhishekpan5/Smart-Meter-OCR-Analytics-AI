import os
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt_ocr import GPT4VisionMeterReader
from gpt_preprocessed_ocr import GPT4VisionPreprocessedMeterReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComparisonTester:
    """
    Compare GPT-4 Vision performance with raw vs preprocessed images
    """
    
    def __init__(self):
        self.raw_reader = GPT4VisionMeterReader()
        self.preprocessed_reader = GPT4VisionPreprocessedMeterReader()
        
    def test_single_image(self, image_path: str) -> dict:
        """
        Test both approaches on a single image
        """
        print(f"\n=== Testing Image: {Path(image_path).name} ===")
        
        # Test with raw image
        print("1. Testing with RAW image...")
        try:
            raw_result = self.raw_reader.extract_meter_data(image_path)
            print(f"   Raw result: {raw_result.get('meter_reading', 'No reading')}")
            print(f"   Raw confidence: {raw_result.get('confidence', 0)}")
        except Exception as e:
            print(f"   Raw processing error: {e}")
            raw_result = {'error': str(e)}
        
        # Test with preprocessed image
        print("2. Testing with PREPROCESSED image...")
        try:
            preprocessed_result = self.preprocessed_reader.extract_meter_data(image_path)
            print(f"   Preprocessed result: {preprocessed_result.get('meter_reading', 'No reading')}")
            print(f"   Preprocessed confidence: {preprocessed_result.get('confidence', 0)}")
            print(f"   Preprocessing effectiveness: {preprocessed_result.get('preprocessing_effectiveness', 0)}")
        except Exception as e:
            print(f"   Preprocessed processing error: {e}")
            preprocessed_result = {'error': str(e)}
        
        return {
            'image_path': image_path,
            'raw_result': raw_result,
            'preprocessed_result': preprocessed_result,
            'timestamp': datetime.now().isoformat()
        }
    
    def compare_results(self, comparison_data: list) -> dict:
        """
        Analyze and compare results from both approaches
        """
        print("\n=== COMPARISON ANALYSIS ===")
        
        successful_raw = 0
        successful_preprocessed = 0
        raw_confidences = []
        preprocessed_confidences = []
        preprocessing_effectiveness = []
        
        for item in comparison_data:
            raw_result = item['raw_result']
            preprocessed_result = item['preprocessed_result']
            
            # Count successful extractions
            if raw_result.get('meter_reading') and not raw_result.get('error'):
                successful_raw += 1
                raw_confidences.append(raw_result.get('confidence', 0))
            
            if preprocessed_result.get('meter_reading') and not preprocessed_result.get('error'):
                successful_preprocessed += 1
                preprocessed_confidences.append(preprocessed_result.get('confidence', 0))
                preprocessing_effectiveness.append(preprocessed_result.get('preprocessing_effectiveness', 0))
        
        # Calculate statistics
        total_images = len(comparison_data)
        raw_success_rate = (successful_raw / total_images) * 100 if total_images > 0 else 0
        preprocessed_success_rate = (successful_preprocessed / total_images) * 100 if total_images > 0 else 0
        
        avg_raw_confidence = sum(raw_confidences) / len(raw_confidences) if raw_confidences else 0
        avg_preprocessed_confidence = sum(preprocessed_confidences) / len(preprocessed_confidences) if preprocessed_confidences else 0
        avg_preprocessing_effectiveness = sum(preprocessing_effectiveness) / len(preprocessing_effectiveness) if preprocessing_effectiveness else 0
        
        # Determine winner
        confidence_improvement = avg_preprocessed_confidence - avg_raw_confidence
        success_rate_improvement = preprocessed_success_rate - raw_success_rate
        
        analysis = {
            'summary': {
                'total_images_tested': total_images,
                'raw_success_rate': raw_success_rate,
                'preprocessed_success_rate': preprocessed_success_rate,
                'success_rate_improvement': success_rate_improvement,
                'avg_raw_confidence': avg_raw_confidence,
                'avg_preprocessed_confidence': avg_preprocessed_confidence,
                'confidence_improvement': confidence_improvement,
                'avg_preprocessing_effectiveness': avg_preprocessing_effectiveness
            },
            'recommendations': self._generate_recommendations(
                raw_success_rate, preprocessed_success_rate,
                avg_raw_confidence, avg_preprocessed_confidence,
                avg_preprocessing_effectiveness
            ),
            'detailed_results': comparison_data
        }
        
        # Print summary
        print(f"Total images tested: {total_images}")
        print(f"Raw success rate: {raw_success_rate:.1f}%")
        print(f"Preprocessed success rate: {preprocessed_success_rate:.1f}%")
        print(f"Success rate improvement: {success_rate_improvement:+.1f}%")
        print(f"Average raw confidence: {avg_raw_confidence:.2f}")
        print(f"Average preprocessed confidence: {avg_preprocessed_confidence:.2f}")
        print(f"Confidence improvement: {confidence_improvement:+.2f}")
        print(f"Average preprocessing effectiveness: {avg_preprocessing_effectiveness:.2f}")
        
        if confidence_improvement > 0:
            print(f"\n✅ PREPROCESSING IMPROVES RESULTS by {confidence_improvement:.2f} confidence points")
        elif confidence_improvement < 0:
            print(f"\n❌ PREPROCESSING REDUCES RESULTS by {abs(confidence_improvement):.2f} confidence points")
        else:
            print(f"\n➖ PREPROCESSING HAS NO SIGNIFICANT IMPACT")
        
        return analysis
    
    def _generate_recommendations(self, raw_success, preprocessed_success, 
                                raw_conf, preprocessed_conf, preprocessing_effectiveness):
        """Generate recommendations based on comparison results"""
        recommendations = []
        
        if preprocessed_success > raw_success:
            recommendations.append("Use preprocessing for better success rate")
        elif raw_success > preprocessed_success:
            recommendations.append("Raw images perform better - avoid preprocessing")
        
        if preprocessed_conf > raw_conf:
            recommendations.append("Use preprocessing for higher confidence readings")
        elif raw_conf > preprocessed_conf:
            recommendations.append("Raw images provide higher confidence - avoid preprocessing")
        
        if preprocessing_effectiveness < 5:
            recommendations.append("Preprocessing may be introducing artifacts - review preprocessing pipeline")
        elif preprocessing_effectiveness > 7:
            recommendations.append("Preprocessing is highly effective - continue using it")
        
        if abs(preprocessed_conf - raw_conf) < 0.5:
            recommendations.append("Both approaches perform similarly - choose based on processing speed")
        
        return recommendations
    
    def save_comparison_results(self, analysis: dict, filename: str = None):
        """Save comparison results to JSON file"""
        try:
            output_dir = "gpt_preprocessed_test/output"
            os.makedirs(output_dir, exist_ok=True)
            
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"comparison_results_{timestamp}.json"
            
            filepath = Path(output_dir) / filename
            
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            print(f"\nComparison results saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving comparison results: {e}")

def main():
    """
    Main function to run comparison test
    """
    try:
        print("=== GPT-4 Vision: Raw vs Preprocessed Images Comparison ===")
        
        # Initialize comparison tester
        tester = ComparisonTester()
        
        # Get images directory
        images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
        images_path = Path(images_dir)
        
        if not images_path.exists():
            print(f"Images directory not found: {images_dir}")
            return
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in images_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No image files found in {images_dir}")
            return
        
        print(f"Found {len(image_files)} images to test")
        
        # Test each image
        comparison_data = []
        for image_file in image_files:
            result = tester.test_single_image(str(image_file))
            comparison_data.append(result)
        
        # Analyze results
        analysis = tester.compare_results(comparison_data)
        
        # Save results
        tester.save_comparison_results(analysis)
        
        print("\n=== Comparison Complete ===")
        
    except Exception as e:
        logger.error(f"Comparison test error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 