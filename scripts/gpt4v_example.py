#!/usr/bin/env python3
"""
GPT-4V Integration Example & Test Script

Demonstrates how to use the production-ready GPT-4V integration
for cascade routing with zero-shot anomaly detection.

Usage:
    # Test with a single image
    python examples/gpt4v_example.py path/to/image.jpg
    
    # Test with cascade mode
    python examples/gpt4v_example.py path/to/image.jpg --cascade
    
    # Test with custom thresholds
    python examples/gpt4v_example.py path/to/image.jpg --cascade \\
        --normal-threshold 0.3 --anomaly-threshold 0.7
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional
import json
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.services.inference import InferenceService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gpt4v_test.log')
    ]
)
logger = logging.getLogger(__name__)


def check_prerequisites() -> bool:
    """Check that API key is set and OpenAI is available."""
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY environment variable not set")
        logger.info("Set it with: export OPENAI_API_KEY='sk-proj-...'")
        return False
    
    try:
        import openai
        logger.info(f"✅ OpenAI {openai.__version__} installed")
    except ImportError:
        logger.error("❌ OpenAI not installed. Run: pip install openai")
        return False
    
    return True


def test_direct_gpt4v(inference: InferenceService, image_path: str) -> dict:
    """
    Test 1: Direct GPT-4V zero-shot analysis.
    
    This tests the raw GPT-4V call without cascade routing.
    Useful for understanding what GPT-4V thinks about an image independently.
    """
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Direct GPT-4V Zero-Shot Analysis")
    logger.info("="*70)
    
    logger.info(f"Analyzing: {image_path}")
    
    result = inference.call_gpt4v_zero_shot(image_path)
    
    # Display results
    print("\n📊 RESULT:")
    print(f"  Status: {result['status']}")
    print(f"  Latency: {result['latency_ms']:.0f}ms")
    
    if result["status"] == "success":
        print(f"\n  Anomaly Detected: {result['is_anomaly']}")
        print(f"  Defect Type: {result['defect_type']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"\n  Reasoning:\n    {result['reasoning']}")
    else:
        print(f"\n  ⚠️  Analysis Failed")
        print(f"  Error: {result.get('api_error', 'Unknown error')}")
    
    # Save result
    with open("gpt4v_result_direct.json", "w") as f:
        # Convert to JSON-serializable
        json_result = {k: v for k, v in result.items() 
                      if isinstance(v, (str, int, float, bool, type(None)))}
        json.dump(json_result, f, indent=2)
    
    logger.info("✓ Result saved to gpt4v_result_direct.json")
    
    return result


def test_cascade_with_gpt4v(
    inference: InferenceService,
    image_path: str,
    normal_threshold: float = 0.2,
    anomaly_threshold: float = 0.8
) -> dict:
    """
    Test 2: Cascade prediction with GPT-4V fallback.
    
    This tests the full cascade routing system with 3-tier logic:
    - BGAD score < 0.2: Confident normal (fast, on edge)
    - BGAD score > 0.8: Confident anomaly (fast, on edge)
    - 0.2 <= BGAD score <= 0.8: Uncertain (routes to GPT-4V)
    """
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Cascade Prediction with GPT-4V Fallback")
    logger.info("="*70)
    
    logger.info(f"Analyzing: {image_path}")
    logger.info(f"Thresholds: Normal < {normal_threshold}, Anomaly > {anomaly_threshold}")
    
    result = inference.cascade_predict_with_gpt4v(
        image_path,
        normal_threshold=normal_threshold,
        anomaly_threshold=anomaly_threshold
    )
    
    # Display routing decision
    print("\n📊 CASCADE ROUTING DECISION:")
    print(f"  Routing Case: {result['routing_case']}")
    
    # Case A: Confident Normal
    if result["routing_case"] == "A_confident_normal":
        print(f"\n  ✅ CASE A: Confident Normal")
        print(f"     BGAD Score: {result['anomaly_score']:.4f} < {normal_threshold}")
        print(f"     Confidence: {result['confidence']:.1%}")
        print(f"     Action: ACCEPT (no labeling needed)")
        print(f"     Latency: <10ms (edge device)")
    
    # Case B: Confident Anomaly
    elif result["routing_case"] == "B_confident_anomaly":
        print(f"\n  ❌ CASE B: Confident Anomaly")
        print(f"     BGAD Score: {result['anomaly_score']:.4f} > {anomaly_threshold}")
        print(f"     Confidence: {result['confidence']:.1%}")
        print(f"     Action: REJECT (no expert labeling needed)")
        print(f"     Latency: <10ms (edge device)")
    
    # Case C: Uncertain - Routed to GPT-4V
    elif "gpt4v" in result["routing_case"]:
        print(f"\n  ❓ CASE C: Uncertain - Routed to GPT-4V")
        print(f"     BGAD Score: {result['bgad_score']:.4f}")
        print(f"     BGAD Uncertainty: {normal_threshold} < score < {anomaly_threshold}")
        
        if "gpt4v_routed" in result["routing_case"]:
            print(f"\n     GPT-4V Analysis:")
            vlm = result["vlm_result"]
            print(f"       Status: {vlm['status']}")
            print(f"       Anomaly: {vlm['is_anomaly']}")
            print(f"       Defect: {vlm['defect_type']}")
            print(f"       Confidence: {vlm['confidence']:.1%}")
            print(f"       Latency: {vlm['latency_ms']:.0f}ms")
            print(f"       Reasoning: {vlm['reasoning']}")
            
            print(f"\n     Ensemble Score: {result['anomaly_score']:.4f}")
            print(f"       = (BGAD {result['bgad_score']:.4f} + GPT-4V {result['gpt4v_score']:.4f}) / 2")
        
        elif "gpt4v_failed" in result["routing_case"]:
            print(f"\n     ⚠️  GPT-4V Call Failed: {result['vlm_result']['error']}")
            print(f"     Fallback: Using BGAD score with expert labeling flag")
    
    # Expert labeling decision
    print(f"\n  📋 EXPERT LABELING REQUIRED: {result['requires_expert_labeling']}")
    if result["requires_expert_labeling"]:
        print(f"     Reason: {result['routing_case']}")
        if result.get("vlm_result"):
            print(f"     Details: {result['vlm_result'].get('reasoning', 'uncertain')}")
    
    # Save result
    with open("gpt4v_result_cascade.json", "w") as f:
        # Convert to JSON-serializable (exclude non-serializable vlm_result)
        json_result = dict(result)
        if "vlm_result" in json_result and isinstance(json_result["vlm_result"], dict):
            json_result["vlm_result"] = {
                k: v for k, v in json_result["vlm_result"].items()
                if isinstance(v, (str, int, float, bool, type(None)))
            }
        json.dump(json_result, f, indent=2)
    
    logger.info("✓ Result saved to gpt4v_result_cascade.json")
    
    return result


def test_statistics(inference: InferenceService):
    """Display cascade routing statistics."""
    logger.info("\n" + "="*70)
    logger.info("STATISTICS: Cascade Routing Performance")
    logger.info("="*70)
    
    stats = inference.get_cascade_statistics()
    
    print("\n📈 CASCADE STATISTICS:")
    total = stats["total_inferences"]
    
    if total == 0:
        print("  No inferences yet")
        return
    
    print(f"  Case A (Confident Normal):  {stats['confident_normal']:4d} ({stats['confident_normal']/total*100:5.1f}%)")
    print(f"  Case B (Confident Anomaly): {stats['confident_anomaly']:4d} ({stats['confident_anomaly']/total*100:5.1f}%)")
    print(f"  Case C (Uncertain/VLM):     {stats['uncertain_routed_to_vlm']:4d} ({stats['uncertain_routed_to_vlm']/total*100:5.1f}%)")
    print(f"  ────────────────────────────────────")
    print(f"  Total Inferences:           {total:4d}")
    print(f"\n  Edge Model Utilization:     {stats['edge_model_utilization']:.1f}% (Cases A+B)")
    print(f"  VLM Catch Rate (of Case C): {stats['vlm_catch_rate']:.1f}%")
    print(f"  VLM Anomalies Detected:     {stats['vlm_flagged_anomaly']}")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="GPT-4V Integration Example & Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with single image (direct GPT-4V)
  python examples/gpt4v_example.py path/to/image.jpg

  # Test with cascade routing
  python examples/gpt4v_example.py path/to/image.jpg --cascade

  # Test with custom thresholds
  python examples/gpt4v_example.py path/to/image.jpg --cascade \\
    --normal-threshold 0.3 --anomaly-threshold 0.7

  # Run both tests
  python examples/gpt4v_example.py path/to/image.jpg --both
        """
    )
    
    parser.add_argument("image", help="Path to image file")
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="Test cascade mode (routes uncertain to GPT-4V)"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both direct and cascade tests"
    )
    parser.add_argument(
        "--normal-threshold",
        type=float,
        default=0.2,
        help="BGAD score below this = confident normal (default: 0.2)"
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=0.8,
        help="BGAD score above this = confident anomaly (default: 0.8)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.normal_threshold >= args.anomaly_threshold:
        parser.error("normal_threshold must be < anomaly_threshold")
    
    image_path = Path(args.image)
    if not image_path.exists():
        parser.error(f"Image not found: {image_path}")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Initialize inference service
    logger.info("Initializing InferenceService...")
    inference = InferenceService()
    
    if not inference.openai_client:
        logger.error("OpenAI client not initialized. Cannot proceed.")
        sys.exit(1)
    
    # Determine which tests to run
    run_direct = args.both or (not args.cascade and not args.both)
    run_cascade = args.cascade or args.both
    
    # Run tests
    try:
        if run_direct:
            result_direct = test_direct_gpt4v(inference, str(image_path))
        
        if run_cascade:
            result_cascade = test_cascade_with_gpt4v(
                inference,
                str(image_path),
                normal_threshold=args.normal_threshold,
                anomaly_threshold=args.anomaly_threshold
            )
        
        # Show statistics
        test_statistics(inference)
        
        logger.info("\n" + "="*70)
        logger.info("✅ Tests completed successfully")
        logger.info("="*70)
        logger.info("Results saved:")
        logger.info("  - gpt4v_result_direct.json (if direct test run)")
        logger.info("  - gpt4v_result_cascade.json (if cascade test run)")
        logger.info("  - gpt4v_test.log")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
